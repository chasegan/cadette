//! The interactive viewport: wgpu 3D scene + egui overlay, driven by a
//! [`Controller`] the application implements.
//!
//! Two entry points share the same render path:
//! - [`run`] — a live winit window (orbit camera + egui panels).
//! - [`screenshot`] — one composited frame (3D + egui) to a PNG, for headless
//!   verification.
//!
//! Layering: this module knows nothing about documents or the kernel. The
//! application supplies a [`Controller`] that draws egui UI and produces the
//! mesh to display; re-meshing happens whenever `ui` reports a change.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::camera::OrbitCamera;
use crate::view::ViewContext;
use crate::{EdgeVertex, Vertex};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.09,
    g: 0.10,
    b: 0.12,
    a: 1.0,
};

/// A mesh ready for display: interleaved face vertices + triangle indices, plus
/// crisp edge lines.
#[derive(Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub edge_vertices: Vec<EdgeVertex>,
    pub edge_indices: Vec<u32>,
}

/// The application's hook into the viewport.
///
/// `ui` draws egui for the frame and returns `true` if the displayed model
/// needs rebuilding; `mesh` then produces the new geometry. `mesh` is also
/// called once at startup for the initial display.
pub trait Controller {
    /// Draw this frame's egui UI. `view` projects between the sketch plane and
    /// the screen for overlay drawing. Return true if the model changed.
    fn ui(&mut self, ctx: &egui::Context, view: &ViewContext) -> bool;
    /// Produce the mesh to display.
    fn mesh(&mut self) -> MeshData;
    /// The faces to emphasize this frame (selected + hovered).
    fn highlights(&self) -> Highlights {
        Highlights::default()
    }
    /// A viewport click resolved to this entity (or `None` for empty space).
    fn on_pick(&mut self, _pick: Option<Pick>) {}
    /// The entity currently under the cursor (or `None`), updated as it moves.
    fn on_hover(&mut self, _pick: Option<Pick>) {}
    /// Whether the viewport should resolve picks right now. (Disabled e.g.
    /// while drawing a sketch, where clicks mean something else.)
    fn wants_picking(&self) -> bool {
        false
    }
}

/// Pixel format of the offscreen face-id pick buffer.
const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

pub use rmf_core::Pick;

/// How close (pixels) the cursor must be to an edge to pick it over a face.
const EDGE_PICK_PX: f32 = 6.0;

/// GPU uniform block, mirrored by `Globals` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    /// `[selected_face, has_sel, hovered_face, has_hov]`.
    faces: [u32; 4],
    /// `[selected_edge, has_sel, hovered_edge, has_hov]`.
    edges: [u32; 4],
}

/// Split a pick into ([face_id, has_face], [edge_id, has_edge]).
fn pick_slots(pick: Option<Pick>) -> ([u32; 2], [u32; 2]) {
    match pick {
        Some(Pick::Face(id)) => ([id, 1], [0, 0]),
        Some(Pick::Edge(id)) => ([0, 0], [id, 1]),
        None => ([0, 0], [0, 0]),
    }
}

impl Globals {
    fn for_view(camera: &OrbitCamera, aspect: f32, highlights: Highlights) -> Self {
        let eye = camera.eye();
        let (sel_face, sel_edge) = pick_slots(highlights.selected);
        let (hov_face, hov_edge) = pick_slots(highlights.hovered);
        Globals {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [0.4, 0.5, 1.0, 0.0],
            faces: [sel_face[0], sel_face[1], hov_face[0], hov_face[1]],
            edges: [sel_edge[0], sel_edge[1], hov_edge[0], hov_edge[1]],
        }
    }
}

/// Entities to emphasize this frame: a clicked selection (strong) and a hover
/// pre-highlight (subtle). Each may be a face or an edge.
#[derive(Clone, Copy, Default)]
pub struct Highlights {
    pub selected: Option<Pick>,
    pub hovered: Option<Pick>,
}

// ---------------------------------------------------------------------------
// Scene: the 3D pipeline + mesh buffers, encodable into any color/depth target.
// ---------------------------------------------------------------------------

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    /// Offscreen pipeline that writes face ids for picking.
    pick_pipeline: wgpu::RenderPipeline,
    /// Line pipeline for crisp feature edges.
    edge_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    edge_vertex_buffer: wgpu::Buffer,
    edge_index_buffer: wgpu::Buffer,
    edge_index_count: u32,
    /// CPU copy of edge segments (world endpoints + edge id) for screen-space
    /// edge picking — edges are too thin for the GPU id buffer.
    edge_segments: Vec<(Vec3, Vec3, u32)>,
    globals_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Scene {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
        mesh: &MeshData,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmf-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rmf-globals-layout"),
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
            label: Some("rmf-globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmf-globals-bind"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rmf-pipeline-layout"),
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
            label: Some("rmf-pipeline"),
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
            label: Some("rmf-pick-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pick.wgsl").into()),
        });
        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rmf-pick-pipeline"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: PICK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            label: Some("rmf-edge-shader"),
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
            label: Some("rmf-edge-pipeline"),
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

        let (vertex_buffer, index_buffer, index_count) = make_mesh_buffers(&device, mesh);
        let (edge_vertex_buffer, edge_index_buffer, edge_index_count) =
            make_edge_buffers(&device, mesh);

        Self {
            device,
            queue,
            pipeline,
            pick_pipeline,
            edge_pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            edge_vertex_buffer,
            edge_index_buffer,
            edge_index_count,
            edge_segments: edge_segments(mesh),
            globals_buffer,
            bind_group,
        }
    }

    /// Replace the displayed mesh.
    fn upload_mesh(&mut self, mesh: &MeshData) {
        let (v, i, n) = make_mesh_buffers(&self.device, mesh);
        self.vertex_buffer = v;
        self.index_buffer = i;
        self.index_count = n;
        let (ev, ei, en) = make_edge_buffers(&self.device, mesh);
        self.edge_vertex_buffer = ev;
        self.edge_index_buffer = ei;
        self.edge_index_count = en;
        self.edge_segments = edge_segments(mesh);
    }

    /// Record the 3D pass (clearing color + depth) into `encoder`. `selected`
    /// highlights that face id, if any.
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
        highlights: Highlights,
    ) {
        let aspect = width as f32 / height.max(1) as f32;
        let globals = Globals::for_view(camera, aspect, highlights);
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rmf-3d-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR),
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
        if self.index_count > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        if self.edge_index_count > 0 {
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.edge_vertex_buffer.slice(..));
            pass.set_index_buffer(self.edge_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.edge_index_count, 0, 0..1);
        }
    }

    /// Render the face-id pass and read back the face under pixel `(px, py)`.
    /// Returns the face id, or `None` for background. Synchronous — intended
    /// for discrete clicks, not per-frame hover.
    fn pick_face(
        &self,
        px: u32,
        py: u32,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
    ) -> Option<u32> {
        if self.index_count == 0 || px >= width || py >= height {
            return None;
        }

        let aspect = width as f32 / height.max(1) as f32;
        let globals = Globals::for_view(camera, aspect, Highlights::default());
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        let id_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rmf-pick-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let id_view = id_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = create_depth(&self.device, width, height);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rmf-pick") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rmf-pick-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &id_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // 0 = no geometry (the pick shader writes face_id + 1).
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
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
            pass.set_pipeline(&self.pick_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        // Copy the single pixel under the cursor to a readback buffer.
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmf-pick-readback"),
            size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &id_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map pick readback"));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        let data = slice.get_mapped_range();
        let raw = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        readback.unmap();

        (raw != 0).then(|| raw - 1)
    }

    /// Pick the entity under pixel `(px, py)`: a nearby edge takes priority
    /// over the face behind it; otherwise the face (or nothing).
    fn pick_at(&self, px: u32, py: u32, camera: &OrbitCamera, w: u32, h: u32) -> Option<Pick> {
        if let Some(edge) = self.pick_edge(px as f32, py as f32, camera, w, h) {
            return Some(Pick::Edge(edge));
        }
        self.pick_face(px, py, camera, w, h).map(Pick::Face)
    }

    /// Nearest edge to `(px, py)` in screen space, within [`EDGE_PICK_PX`].
    fn pick_edge(&self, px: f32, py: f32, camera: &OrbitCamera, w: u32, h: u32) -> Option<u32> {
        let view_proj = camera.view_proj(w as f32 / h.max(1) as f32);
        let (wf, hf) = (w as f32, h as f32);
        let project = |p: Vec3| -> Option<(f32, f32)> {
            let clip = view_proj * p.extend(1.0);
            if clip.w <= 1e-6 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            Some(((ndc.x * 0.5 + 0.5) * wf, (0.5 - ndc.y * 0.5) * hf))
        };

        let mut best = (EDGE_PICK_PX, None);
        for &(a, b, id) in &self.edge_segments {
            if let (Some(pa), Some(pb)) = (project(a), project(b)) {
                let d = point_segment_distance(px, py, pa, pb);
                if d < best.0 {
                    best = (d, Some(id));
                }
            }
        }
        best.1
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

/// Distance from a point to a line segment, all in screen pixels.
fn point_segment_distance(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let (apx, apy) = (px - a.0, py - a.1);
    let len_sq = abx * abx + aby * aby;
    let t = if len_sq > 0.0 {
        ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (a.0 + abx * t, a.1 + aby * t);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

fn make_mesh_buffers(
    device: &wgpu::Device,
    mesh: &MeshData,
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    use wgpu::util::DeviceExt;
    // Avoid zero-sized buffers when the mesh is empty (failed regen): use a
    // single dummy vertex/index and draw nothing (index_count = 0).
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-vertices-empty"),
            contents: bytemuck::bytes_of(&Vertex::default()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-indices-empty"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::INDEX,
        });
        return (vbuf, ibuf, 0);
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmf-vertices"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmf-indices"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, mesh.indices.len() as u32)
}

fn make_edge_buffers(device: &wgpu::Device, mesh: &MeshData) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    use wgpu::util::DeviceExt;
    if mesh.edge_vertices.is_empty() || mesh.edge_indices.is_empty() {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-edge-vertices-empty"),
            contents: bytemuck::bytes_of(&EdgeVertex::default()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-edge-indices-empty"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::INDEX,
        });
        return (vbuf, ibuf, 0);
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmf-edge-vertices"),
        contents: bytemuck::cast_slice(&mesh.edge_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmf-edge-indices"),
        contents: bytemuck::cast_slice(&mesh.edge_indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, mesh.edge_indices.len() as u32)
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rmf-depth"),
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

fn frame_camera(vertices: &[Vertex]) -> OrbitCamera {
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

async fn request_device(
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
            label: Some("rmf-device"),
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

fn egui_renderer(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> EguiRenderer {
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
fn encode_egui(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &EguiRenderer,
    color_view: &wgpu::TextureView,
    jobs: &[egui::ClippedPrimitive],
    screen: &ScreenDescriptor,
) {
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("rmf-egui-pass"),
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

// ---------------------------------------------------------------------------
// Live window
// ---------------------------------------------------------------------------

/// Open the interactive viewport, driven by `controller`, until closed.
pub fn run(controller: impl Controller + 'static) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = WindowApp::new(controller);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct MouseState {
    last: Option<PhysicalPosition<f64>>,
    orbiting: bool,
    panning: bool,
    /// Latest cursor position (physical pixels), tracked every move.
    cursor: PhysicalPosition<f64>,
    /// Where the left button went down, and whether it has since dragged far
    /// enough to count as a drag rather than a click.
    left_press: Option<PhysicalPosition<f64>>,
    dragged: bool,
    /// Set when something changed that could move the hovered face (cursor or
    /// camera); consumed by the per-frame hover pick.
    hover_dirty: bool,
}

struct WindowApp<C: Controller> {
    controller: C,
    camera: OrbitCamera,
    mouse: MouseState,
    state: Option<WindowState>,
}

struct WindowState {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    scene: Scene,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: EguiRenderer,
}

impl<C: Controller> WindowApp<C> {
    fn new(controller: C) -> Self {
        Self {
            controller,
            camera: OrbitCamera::framing(Vec3::ZERO, 1.0),
            mouse: MouseState::default(),
            state: None,
        }
    }
}

impl<C: Controller> ApplicationHandler for WindowApp<C> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = winit::window::Window::default_attributes()
            .with_title("Riemanifold")
            .with_inner_size(PhysicalSize::new(1280, 820));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let (adapter, device, queue) =
            pollster::block_on(request_device(&instance, Some(&surface)));

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mesh = self.controller.mesh();
        self.camera = frame_camera(&mesh.vertices);
        let scene = Scene::new(device.clone(), queue, format, &mesh);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_renderer(&device, format);

        window.request_redraw();
        self.state = Some(WindowState {
            window,
            surface,
            config,
            depth_view: create_depth(&device, size.width, size.height),
            scene,
            egui_ctx,
            egui_state,
            egui_renderer,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // egui gets first look; if it consumes the event, the viewport ignores it.
        let response = state.egui_state.on_window_event(&state.window, &event);
        let egui_used = response.consumed;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.config.width = size.width;
                state.config.height = size.height;
                state.surface.configure(&state.scene.device, &state.config);
                state.depth_view = create_depth(&state.scene.device, size.width, size.height);
                state.window.request_redraw();
            }

            WindowEvent::MouseInput { button, state: pressed, .. } => {
                let down = pressed == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if down && !egui_used {
                            self.mouse.orbiting = true;
                            self.mouse.left_press = Some(self.mouse.cursor);
                            self.mouse.dragged = false;
                        } else if !down {
                            self.mouse.orbiting = false;
                            self.mouse.last = None;
                            self.mouse.hover_dirty = true; // re-evaluate after a drag
                            // A left click that didn't drag, in the viewport, is
                            // a selection pick (if the controller wants one).
                            if self.mouse.left_press.take().is_some()
                                && !self.mouse.dragged
                                && self.controller.wants_picking()
                            {
                                self.pick_at_cursor();
                            }
                            self.mouse.dragged = false; // ready to hover again
                        }
                    }
                    MouseButton::Right => {
                        // Right-drag orbits — always available, even over
                        // geometry (so left-drag can become manipulation).
                        if down && !egui_used {
                            self.mouse.orbiting = true;
                        } else if !down {
                            self.mouse.orbiting = false;
                            self.mouse.last = None;
                            self.mouse.hover_dirty = true;
                        }
                    }
                    MouseButton::Middle => {
                        if !egui_used {
                            self.mouse.panning = down;
                        }
                        if !down {
                            self.mouse.last = None;
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse.cursor = position;
                self.mouse.hover_dirty = true; // cursor (and maybe camera) moved
                // Promote to a drag once the pointer leaves a small dead zone.
                if let Some(press) = self.mouse.left_press {
                    let d = (position.x - press.x).hypot(position.y - press.y);
                    if d > 4.0 {
                        self.mouse.dragged = true;
                    }
                }
                if !egui_used {
                    if let Some(last) = self.mouse.last {
                        let dx = (position.x - last.x) as f32;
                        let dy = (position.y - last.y) as f32;
                        if self.mouse.orbiting {
                            self.camera.orbit(dx * 0.01, dy * 0.01);
                            state.window.request_redraw();
                        } else if self.mouse.panning {
                            self.camera.pan(dx, dy);
                            state.window.request_redraw();
                        }
                    }
                    self.mouse.last = if self.mouse.orbiting || self.mouse.panning {
                        Some(position)
                    } else {
                        None
                    };
                }
            }

            WindowEvent::MouseWheel { delta, .. } if !egui_used => {
                match delta {
                    // Mouse wheel (discrete lines): zoom.
                    MouseScrollDelta::LineDelta(_, y) => self.camera.dolly(y),
                    // Trackpad two-finger scroll (smooth pixels): pan.
                    MouseScrollDelta::PixelDelta(p) => {
                        self.camera.pan(p.x as f32, p.y as f32)
                    }
                }
                self.mouse.hover_dirty = true;
                state.window.request_redraw();
            }

            // Trackpad pinch: zoom.
            WindowEvent::PinchGesture { delta, .. } if !egui_used => {
                self.camera.dolly(delta as f32 * 6.0);
                self.mouse.hover_dirty = true;
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                self.redraw();
            }

            _ => {}
        }

        // Keep redrawing while egui wants animation/repaint (hover, drags, etc.).
        if let Some(state) = self.state.as_ref() {
            if egui_used || state.egui_ctx.has_requested_repaint() {
                state.window.request_redraw();
            }
        }
    }

    /// Drive continuous redraws. Without this, `ControlFlow::Wait` can present a
    /// single frame and then idle forever if the user never interacts — which
    /// left the egui panels unpainted after the first frame.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

impl<C: Controller> WindowApp<C> {
    /// Resolve the entity under the cursor and hand it to the controller.
    fn pick_at_cursor(&mut self) {
        let pick = match self.state.as_ref() {
            Some(state) => {
                let (w, h) = (state.config.width, state.config.height);
                let px = (self.mouse.cursor.x as i64).clamp(0, w as i64 - 1) as u32;
                let py = (self.mouse.cursor.y as i64).clamp(0, h as i64 - 1) as u32;
                state.scene.pick_at(px, py, &self.camera, w, h)
            }
            None => return,
        };
        self.controller.on_pick(pick);
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    fn redraw(&mut self) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        let ppp = state.window.scale_factor() as f32;
        let size = egui::vec2(
            state.config.width as f32 / ppp,
            state.config.height as f32 / ppp,
        );
        let aspect = state.config.width as f32 / state.config.height.max(1) as f32;
        let view = ViewContext::new(self.camera.view_proj(aspect), size);

        let raw_input = state.egui_state.take_egui_input(&state.window);
        state.egui_ctx.begin_pass(raw_input);
        let changed = self.controller.ui(&state.egui_ctx, &view);
        let full_output = state.egui_ctx.end_pass();
        if changed {
            let mesh = self.controller.mesh();
            state.scene.upload_mesh(&mesh);
        }
        state
            .egui_state
            .handle_platform_output(&state.window, full_output.platform_output);

        // Hover pre-highlight: pick the face under the cursor at most once per
        // frame, and only when the cursor/camera actually moved.
        //
        // The pick is a synchronous GPU readback (it stalls until the pass
        // finishes). Measured fine for these scene sizes; if it ever stutters
        // on heavier models, switch to an async readback (issue the copy, read
        // the mapped result a frame or two later) to remove the stall.
        if self.mouse.hover_dirty {
            self.mouse.hover_dirty = false;
            let hovered = if self.controller.wants_picking()
                && !self.mouse.dragged
                && !state.egui_ctx.is_pointer_over_egui()
            {
                let (w, h) = (state.config.width, state.config.height);
                let px = (self.mouse.cursor.x as i64).clamp(0, w as i64 - 1) as u32;
                let py = (self.mouse.cursor.y as i64).clamp(0, h as i64 - 1) as u32;
                state.scene.pick_at(px, py, &self.camera, w, h)
            } else {
                None
            };
            self.controller.on_hover(hovered);
        }

        let jobs = state
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = ScreenDescriptor {
            size_in_pixels: [state.config.width, state.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let device = &state.scene.device;
        let queue = &state.scene.queue;

        // Apply egui's texture updates (notably the font atlas) BEFORE acquiring
        // the surface. egui hands each delta to us exactly once via end_pass();
        // if we skipped a frame after consuming it, the atlas would be lost for
        // good and egui-wgpu would drop every primitive ("Missing texture
        // Managed(0)") — the UI would never paint.
        for (id, delta) in &full_output.textures_delta.set {
            state.egui_renderer.update_texture(device, queue, *id, delta);
        }

        let frame = match state.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                state.surface.configure(&state.scene.device, &state.config);
                for id in &full_output.textures_delta.free {
                    state.egui_renderer.free_texture(id);
                }
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let egui_cmds =
            state
                .egui_renderer
                .update_buffers(device, queue, &mut encoder, &jobs, &screen);

        state.scene.encode(
            &mut encoder,
            &view,
            &state.depth_view,
            &self.camera,
            state.config.width,
            state.config.height,
            self.controller.highlights(),
        );
        encode_egui(&mut encoder, &state.egui_renderer, &view, &jobs, &screen);

        queue.submit(egui_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();

        for id in &full_output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }
    }
}

// ---------------------------------------------------------------------------
// Offscreen composited screenshot (3D + egui), for verification
// ---------------------------------------------------------------------------

/// Render one composited frame (3D scene + egui UI) to a PNG at `path`.
pub fn screenshot(
    mut controller: impl Controller,
    width: u32,
    height: u32,
    path: &str,
) -> anyhow::Result<()> {
    const PPP: f32 = 1.5;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None));

    let mesh = controller.mesh();
    let camera = frame_camera(&mesh.vertices);
    let mut scene = Scene::new(device.clone(), queue.clone(), format, &mesh);
    let _ = &mut scene;

    let egui_ctx = egui::Context::default();
    egui_ctx.set_pixels_per_point(PPP);
    let mut egui_renderer = egui_renderer(&device, format);

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width as f32 / PPP, height as f32 / PPP),
        )),
        ..Default::default()
    };
    let view = ViewContext::new(
        camera.view_proj(width as f32 / height as f32),
        egui::vec2(width as f32 / PPP, height as f32 / PPP),
    );
    egui_ctx.begin_pass(raw_input);
    controller.ui(&egui_ctx, &view);
    let full_output = egui_ctx.end_pass();
    let jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen = ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: full_output.pixels_per_point,
    };

    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rmf-offscreen-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = create_depth(&device, width, height);

    for (id, delta) in &full_output.textures_delta.set {
        egui_renderer.update_texture(&device, &queue, *id, delta);
    }
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let egui_cmds = egui_renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &screen);
    let highlights = controller.highlights();
    scene.encode(&mut encoder, &color_view, &depth_view, &camera, width, height, highlights);
    encode_egui(&mut encoder, &egui_renderer, &color_view, &jobs, &screen);
    queue.submit(egui_cmds.into_iter().chain(std::iter::once(encoder.finish())));
    for id in &full_output.textures_delta.free {
        egui_renderer.free_texture(id);
    }

    save_texture_png(&device, &queue, &color, width, height, path)
}

fn save_texture_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    path: &str,
) -> anyhow::Result<()> {
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmf-readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback buffer"));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    buffer.unmap();

    let img: image::RgbaImage =
        image::ImageBuffer::from_raw(width, height, pixels).expect("image buffer size");
    img.save(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a device, render a single +Z-facing quad (face id 0), and confirm
    /// the picker resolves the center pixel to that face — the GPU pick path.
    #[test]
    fn picks_the_face_under_the_center_pixel() {
        let instance = wgpu::Instance::default();
        let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None));

        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            face_id: 0,
        };
        let mesh = MeshData {
            vertices: vec![v(-10.0, -10.0), v(10.0, -10.0), v(10.0, 10.0), v(-10.0, 10.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        };
        let scene = Scene::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, &mesh);
        let camera = OrbitCamera::framing(Vec3::ZERO, 15.0);

        // The quad straddles the origin (the camera target), so the center ray
        // hits it.
        assert_eq!(scene.pick_face(32, 32, &camera, 64, 64), Some(0));
    }

    /// An edge passing through the cursor is picked in preference to the face
    /// behind it.
    #[test]
    fn pick_prefers_a_nearby_edge_over_the_face() {
        let instance = wgpu::Instance::default();
        let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None));

        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            face_id: 0,
        };
        let mesh = MeshData {
            vertices: vec![v(-10.0, -10.0), v(10.0, -10.0), v(10.0, 10.0), v(-10.0, 10.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            // A horizontal edge through the origin (the screen center).
            edge_vertices: vec![
                EdgeVertex { position: [-10.0, 0.0, 0.0], edge_id: 0 },
                EdgeVertex { position: [10.0, 0.0, 0.0], edge_id: 0 },
            ],
            edge_indices: vec![0, 1],
        };
        let scene = Scene::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, &mesh);
        let camera = OrbitCamera::framing(Vec3::ZERO, 15.0);

        assert_eq!(scene.pick_at(32, 32, &camera, 64, 64), Some(Pick::Edge(0)));
    }
}
