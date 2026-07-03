//! The 3D scene: wgpu pipelines, mesh/edge/gizmo/grid buffers, the per-frame
//! uniforms, and the render-pass encoding shared by the live window and the
//! offscreen screenshot path. Also hosts the small device/egui bring-up
//! helpers both entry points use.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use glam::Vec3;

use crate::camera::OrbitCamera;
use crate::controller::{Highlights, MeshData, Pick};
use crate::gizmo::{make_gizmo_buffer, GIZMO_MAX_VERTS};
use crate::grid::{make_grid_buffer, GRID_MAX_VERTS};
use crate::math::rgba;
use crate::theme::ViewportTheme;
use crate::{EdgeVertex, GizmoVertex, Vertex};

pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Pixel format of the offscreen face-id pick buffer.
pub(crate) const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
/// Second pick target: the front-surface depth (NDC z), for edge occlusion.
pub(crate) const PICK_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Max selected edges the shader can highlight at once (4 ids per `vec4`).
const SEL_EDGE_CAP: usize = 64;
const SEL_EDGE_VEC4: usize = SEL_EDGE_CAP / 4;
/// Max selected faces the shader can highlight at once (4 ids per `vec4`).
const SEL_FACE_CAP: usize = 64;
const SEL_FACE_VEC4: usize = SEL_FACE_CAP / 4;

/// GPU uniform block, mirrored by `Globals` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Globals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    /// `[selected_face_count, 0, hovered_face, has_hov]`.
    faces: [u32; 4],
    /// `[selected_edge_count, 0, hovered_edge, has_hov]`.
    edges: [u32; 4],
    /// The selected edge ids, packed 4 per `vec4`, `edges[0]` of them valid.
    sel_edges: [[u32; 4]; SEL_EDGE_VEC4],
    /// The selected face ids, packed 4 per `vec4`, `faces[0]` of them valid.
    sel_faces: [[u32; 4]; SEL_FACE_VEC4],
    /// Theme material color (unlit base), `rgb + _`. Set by `encode` per frame.
    face_base: [f32; 4],
    /// Theme feature-edge color, `rgb + _`. Set by `encode` per frame.
    edge_base: [f32; 4],
}

/// Uniform for the background gradient pass: top/bottom colors (linear RGB).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BgUniform {
    top: [f32; 4],
    bottom: [f32; 4],
}

impl BgUniform {
    fn new(theme: &ViewportTheme) -> Self {
        Self {
            top: rgba(theme.bg_top, 1.0),
            bottom: rgba(theme.bg_bottom, 1.0),
        }
    }
}

impl Globals {
    pub(crate) fn for_view(camera: &OrbitCamera, aspect: f32, highlights: &Highlights) -> Self {
        let eye = camera.eye();

        // Selection: a face set + an edge set (both multi-select).
        let mut faces = [0u32; 4];
        let mut edges = [0u32; 4];
        let mut sel_edges = [[0u32; 4]; SEL_EDGE_VEC4];
        let mut sel_faces = [[0u32; 4]; SEL_FACE_VEC4];
        let (mut n_edges, mut n_faces) = (0usize, 0usize);
        for pick in &highlights.selected {
            match pick {
                Pick::Face(id) => {
                    if n_faces < SEL_FACE_CAP {
                        sel_faces[n_faces / 4][n_faces % 4] = *id;
                        n_faces += 1;
                    }
                }
                Pick::Edge(id) => {
                    if n_edges < SEL_EDGE_CAP {
                        sel_edges[n_edges / 4][n_edges % 4] = *id;
                        n_edges += 1;
                    }
                }
            }
        }
        faces[0] = n_faces as u32;
        edges[0] = n_edges as u32;

        // Hover is always a single entity.
        match highlights.hovered {
            Some(Pick::Face(id)) => {
                faces[2] = id;
                faces[3] = 1;
            }
            Some(Pick::Edge(id)) => {
                edges[2] = id;
                edges[3] = 1;
            }
            None => {}
        }

        Globals {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [0.4, 0.5, 1.0, 0.0],
            faces,
            edges,
            sel_edges,
            sel_faces,
            // Colors default to the dark look; `encode` overrides them from the
            // active `ViewportTheme` (pick passes are color-agnostic).
            face_base: rgba(ViewportTheme::DARK.face_base, 1.0),
            edge_base: rgba(ViewportTheme::DARK.edge, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene: the 3D pipeline + mesh buffers, encodable into any color/depth target.
// ---------------------------------------------------------------------------

/// A persistent GPU buffer with grow-only capacity: rewritten in place via
/// `queue.write_buffer` while the data fits, reallocated (never shrunk) when
/// it doesn't. Steady-state geometry updates — e.g. a push/pull drag that
/// re-meshes every frame — therefore allocate nothing, matching the pattern
/// the gizmo/grid vertex buffers already use.
pub(crate) struct GrowBuffer {
    pub(crate) buffer: wgpu::Buffer,
    /// Allocated size in bytes (>= the live data, which the draw-count fields
    /// on [`Scene`] bound; stale bytes past the count are never read).
    capacity: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl GrowBuffer {
    fn new(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages, capacity: u64) -> Self {
        // Never zero-sized (an empty mesh still binds the buffer), and keep the
        // size copy-aligned.
        let capacity = capacity.max(4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, capacity, usage, label }
    }

    fn with_data(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        usage: wgpu::BufferUsages,
        data: &[u8],
    ) -> Self {
        let mut this = Self::new(device, label, usage, data.len() as u64);
        this.write(device, queue, data);
        this
    }

    /// Upload `data`, growing (never shrinking) the buffer first if needed.
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        if data.len() as u64 > self.capacity {
            *self = Self::new(device, self.label, self.usage, data.len() as u64);
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, data);
        }
    }
}

pub(crate) struct Scene {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    /// Offscreen pipeline that writes face ids for picking.
    pub(crate) pick_pipeline: wgpu::RenderPipeline,
    /// Line pipeline for crisp feature edges.
    edge_pipeline: wgpu::RenderPipeline,
    /// Line pipeline for the transform gizmo (drawn on top, depth test off).
    gizmo_pipeline: wgpu::RenderPipeline,
    /// Line pipeline for the workplane grid (depth-tested, drawn behind model).
    grid_pipeline: wgpu::RenderPipeline,
    /// Fullscreen gradient background pipeline (drawn first, no depth write).
    bg_pipeline: wgpu::RenderPipeline,
    bg_buffer: wgpu::Buffer,
    bg_bind_group: wgpu::BindGroup,
    pub(crate) vertex_buffer: GrowBuffer,
    pub(crate) index_buffer: GrowBuffer,
    pub(crate) index_count: u32,
    edge_vertex_buffer: GrowBuffer,
    edge_index_buffer: GrowBuffer,
    edge_index_count: u32,
    /// Gizmo line vertices (rebuilt each frame; empty when hidden).
    gizmo_vertex_buffer: wgpu::Buffer,
    gizmo_vertex_count: u32,
    /// Workplane grid line vertices (rebuilt when the grid setting changes).
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
    /// CPU copy of edge segments (world endpoints + edge id) for screen-space
    /// edge picking — edges are too thin for the GPU id buffer.
    pub(crate) edge_segments: Vec<(Vec3, Vec3, u32)>,
    pub(crate) globals_buffer: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
    /// Persistent pick-pass targets + readback buffers, created lazily on the
    /// first pick and recreated only when the surface size changes.
    pub(crate) pick_targets: Option<crate::picking::PickTargets>,
}

impl Scene {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
        mesh: &MeshData,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdt-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cdt-globals-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdt-globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cdt-globals-bind"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cdt-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32];
        let vertex_layout = || wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        };
        let depth_stencil = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdt-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Picking pipeline: same geometry, writes face ids to an R32Uint target.
        let pick_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdt-pick-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pick.wgsl").into()),
        });
        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdt-pick-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &pick_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &pick_shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: PICK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: PICK_DEPTH_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Edge pipeline: line list, dark, biased slightly toward the camera so
        // the lines sit crisply on the shaded surface without z-fighting.
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdt-edge-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("edges.wgsl").into()),
        });
        const EDGE_ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32];
        let edge_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<EdgeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &EDGE_ATTRS,
        };
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdt-edge-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_main"),
                buffers: &[edge_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            // Depth bias is illegal on line topology; the edge shader nudges
            // clip-space z toward the camera instead.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Gizmo pipeline: colored line list, always drawn on top (depth test
        // disabled) so handles stay visible and grabbable through the model.
        let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdt-gizmo-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gizmo.wgsl").into()),
        });
        const GIZMO_ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let gizmo_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GizmoVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &GIZMO_ATTRS,
        };
        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdt-gizmo-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gizmo_shader,
                entry_point: Some("vs_main"),
                buffers: &[gizmo_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gizmo_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // Alpha blend so the rotation protractor wedge is translucent.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // Filled arrows/handles; both windings visible (billboards) so no
                // back-face culling.
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Grid pipeline: same colored-vertex shader, but a depth-tested line
        // list so the model occludes the ground-plane grid behind it.
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdt-grid-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gizmo_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GizmoVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &GIZMO_ATTRS,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gizmo_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Background gradient pipeline: a fullscreen triangle (no vertex buffer)
        // reading its two colors from a small dedicated uniform. Drawn first with
        // depth writes off so the rest of the scene composites over it.
        let bg_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdt-bg-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("background.wgsl").into()),
        });
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cdt-bg-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdt-bg-uniform"),
            size: std::mem::size_of::<BgUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cdt-bg-bind"),
            layout: &bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bg_buffer.as_entire_binding(),
            }],
        });
        let bg_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cdt-bg-pipeline-layout"),
            bind_group_layouts: &[Some(&bg_layout)],
            immediate_size: 0,
        });
        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdt-bg-pipeline"),
            layout: Some(&bg_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bg_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bg_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Present a depth attachment (the pass has one) but never test or
            // write it — the gradient sits behind everything.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = GrowBuffer::with_data(
            &device,
            &queue,
            "cdt-vertices",
            wgpu::BufferUsages::VERTEX,
            bytemuck::cast_slice(&mesh.vertices),
        );
        let index_buffer = GrowBuffer::with_data(
            &device,
            &queue,
            "cdt-indices",
            wgpu::BufferUsages::INDEX,
            bytemuck::cast_slice(&mesh.indices),
        );
        let index_count = mesh_draw_count(mesh);
        let edge_vertex_buffer = GrowBuffer::with_data(
            &device,
            &queue,
            "cdt-edge-vertices",
            wgpu::BufferUsages::VERTEX,
            bytemuck::cast_slice(&mesh.edge_vertices),
        );
        let edge_index_buffer = GrowBuffer::with_data(
            &device,
            &queue,
            "cdt-edge-indices",
            wgpu::BufferUsages::INDEX,
            bytemuck::cast_slice(&mesh.edge_indices),
        );
        let edge_index_count = edge_draw_count(mesh);
        let gizmo_vertex_buffer = make_gizmo_buffer(&device);
        let gizmo_vertex_count = 0u32;
        let grid_vertex_buffer = make_grid_buffer(&device);
        let grid_vertex_count = 0u32;

        Self {
            device,
            queue,
            pipeline,
            pick_pipeline,
            edge_pipeline,
            gizmo_pipeline,
            grid_pipeline,
            bg_pipeline,
            bg_buffer,
            bg_bind_group,
            vertex_buffer,
            index_buffer,
            index_count,
            edge_vertex_buffer,
            edge_index_buffer,
            edge_index_count,
            gizmo_vertex_buffer,
            gizmo_vertex_count,
            grid_vertex_buffer,
            grid_vertex_count,
            edge_segments: edge_segments(mesh),
            globals_buffer,
            bind_group,
            pick_targets: None,
        }
    }

    /// Replace the displayed mesh. Rewrites the persistent buffers in place
    /// (growing them only when the mesh outgrows their capacity) — this runs
    /// per frame during a push/pull drag, so it must not allocate in steady
    /// state.
    pub(crate) fn upload_mesh(&mut self, mesh: &MeshData) {
        self.vertex_buffer
            .write(&self.device, &self.queue, bytemuck::cast_slice(&mesh.vertices));
        self.index_buffer
            .write(&self.device, &self.queue, bytemuck::cast_slice(&mesh.indices));
        self.index_count = mesh_draw_count(mesh);
        self.edge_vertex_buffer
            .write(&self.device, &self.queue, bytemuck::cast_slice(&mesh.edge_vertices));
        self.edge_index_buffer
            .write(&self.device, &self.queue, bytemuck::cast_slice(&mesh.edge_indices));
        self.edge_index_count = edge_draw_count(mesh);
        self.edge_segments = edge_segments(mesh);
    }

    /// Replace the gizmo line geometry (empty `verts` hides it).
    pub(crate) fn upload_gizmo(&mut self, verts: &[GizmoVertex]) {
        // Rewrite the persistent buffer in place — no per-frame allocation.
        let n = verts.len().min(GIZMO_MAX_VERTS as usize);
        if n > 0 {
            self.queue
                .write_buffer(&self.gizmo_vertex_buffer, 0, bytemuck::cast_slice(&verts[..n]));
        }
        self.gizmo_vertex_count = n as u32;
    }

    /// Replace the workplane grid line geometry (empty `verts` hides it).
    pub(crate) fn upload_grid(&mut self, verts: &[GizmoVertex]) {
        let n = verts.len().min(GRID_MAX_VERTS as usize);
        if n > 0 {
            self.queue
                .write_buffer(&self.grid_vertex_buffer, 0, bytemuck::cast_slice(&verts[..n]));
        }
        self.grid_vertex_count = n as u32;
    }

    /// Record the 3D pass (clearing color + depth) into `encoder`. `selected`
    /// highlights that face id, if any.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
        highlights: &Highlights,
        theme: &ViewportTheme,
    ) {
        let aspect = width as f32 / height.max(1) as f32;
        let mut globals = Globals::for_view(camera, aspect, highlights);
        globals.face_base = rgba(theme.face_base, 1.0);
        globals.edge_base = rgba(theme.edge, 1.0);
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
        self.queue.write_buffer(
            &self.bg_buffer,
            0,
            bytemuck::bytes_of(&BgUniform::new(theme)),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cdt-3d-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The gradient quad covers every pixel; this clear is just a
                    // fallback, matched to the gradient's top so any gap blends in.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: theme.bg_top[0] as f64,
                        g: theme.bg_top[1] as f64,
                        b: theme.bg_top[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // Gradient background fills every pixel first (no depth write).
        pass.set_pipeline(&self.bg_pipeline);
        pass.set_bind_group(0, &self.bg_bind_group, &[]);
        pass.draw(0..3, 0..1);
        // Grid next (behind the model — it's depth-tested so the model occludes it).
        if self.grid_vertex_count > 0 {
            pass.set_pipeline(&self.grid_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            pass.draw(0..self.grid_vertex_count, 0..1);
        }
        if self.index_count > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        if self.edge_index_count > 0 {
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.edge_vertex_buffer.buffer.slice(..));
            pass.set_index_buffer(
                self.edge_index_buffer.buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..self.edge_index_count, 0, 0..1);
        }
        // The gizmo draws last, on top of everything (its pipeline ignores depth).
        if self.gizmo_vertex_count > 0 {
            pass.set_pipeline(&self.gizmo_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.gizmo_vertex_buffer.slice(..));
            pass.draw(0..self.gizmo_vertex_count, 0..1);
        }
    }
}

/// CPU edge segments (world endpoints + edge id) from the mesh's edge lines.
fn edge_segments(mesh: &MeshData) -> Vec<(Vec3, Vec3, u32)> {
    mesh.edge_indices
        .chunks_exact(2)
        .filter_map(|pair| {
            let a = mesh.edge_vertices.get(pair[0] as usize)?;
            let b = mesh.edge_vertices.get(pair[1] as usize)?;
            Some((Vec3::from(a.position), Vec3::from(b.position), a.edge_id))
        })
        .collect()
}

/// Indices to draw for the face mesh — 0 when either array is empty (a failed
/// regen), which leaves the persistent buffers bound but unused.
fn mesh_draw_count(mesh: &MeshData) -> u32 {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        0
    } else {
        mesh.indices.len() as u32
    }
}

/// Indices to draw for the feature-edge lines (0 when there are none).
fn edge_draw_count(mesh: &MeshData) -> u32 {
    if mesh.edge_vertices.is_empty() || mesh.edge_indices.is_empty() {
        0
    } else {
        mesh.edge_indices.len() as u32
    }
}

pub(crate) fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cdt-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub(crate) fn frame_camera(vertices: &[Vertex]) -> OrbitCamera {
    if vertices.is_empty() {
        return OrbitCamera::framing(Vec3::ZERO, 1.0);
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        let p = Vec3::from_array(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    let center = (min + max) * 0.5;
    let radius = (max - center).length();
    OrbitCamera::framing(center, radius)
}

pub(crate) async fn request_device(
    instance: &wgpu::Instance,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("cdt-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request device");
    (adapter, device, queue)
}

pub(crate) fn egui_renderer(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> EguiRenderer {
    EguiRenderer::new(
        device,
        color_format,
        RendererOptions {
            msaa_samples: 1,
            depth_stencil_format: None,
            ..Default::default()
        },
    )
}

/// Record the egui pass (loading the existing color, no depth) into `encoder`.
/// Caller must have run `update_texture`/`update_buffers` beforehand.
pub(crate) fn encode_egui(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &EguiRenderer,
    color_view: &wgpu::TextureView,
    jobs: &[egui::ClippedPrimitive],
    screen: &ScreenDescriptor,
) {
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cdt-egui-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    renderer.render(&mut pass.forget_lifetime(), jobs, screen);
}

/// Register the Phosphor icon font on a fresh egui context so the UI can use
/// `egui_phosphor::regular::*` glyphs (egui's default fonts render most symbols
/// as tofu). Added as a fallback in the proportional family.
pub(crate) fn install_phosphor(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}
