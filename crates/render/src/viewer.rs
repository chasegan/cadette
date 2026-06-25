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
use crate::{EdgeVertex, GizmoVertex, Vertex};

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
    /// `point` is the clicked world point on it (a face's surface or a point on
    /// an edge), used to build durable anchors; `None` if unavailable.
    /// `additive` is true when ⇧/⌘ is held — add/remove from a multi-selection.
    fn on_pick(&mut self, _pick: Option<Pick>, _point: Option<[f64; 3]>, _additive: bool) {}
    /// The entity currently under the cursor (or `None`), updated as it moves.
    fn on_hover(&mut self, _pick: Option<Pick>) {}
    /// Whether the viewport should resolve picks right now. (Disabled e.g.
    /// while drawing a sketch, where clicks mean something else.)
    fn wants_picking(&self) -> bool {
        false
    }

    /// Begin a push/pull on `pick` if it's manipulable. `point` is the clicked
    /// world point (on the face); the controller captures the anchor (orienting
    /// the normal toward `eye`) and returns the drag axis `(point, outward
    /// normal)`. `None` means not manipulable, so the viewport orbits instead.
    fn start_manipulation(
        &mut self,
        _pick: Pick,
        _point: [f64; 3],
        _eye: [f64; 3],
    ) -> Option<([f64; 3], [f64; 3])> {
        None
    }
    /// Update the active push/pull to `distance` along the axis. Returns whether
    /// geometry changed (so the host re-meshes).
    fn update_manipulation(&mut self, _distance: f64) -> bool {
        false
    }
    /// Finish the active push/pull: commit it, or cancel and discard.
    fn finish_manipulation(&mut self, _commit: bool) {}

    /// The transform gizmo to show this frame, or `None` to hide it. Shown at a
    /// selected body (gumball-style).
    fn gizmo(&self) -> Option<Gizmo> {
        None
    }
    /// Begin a transform on `handle` (a gizmo handle was grabbed).
    fn start_transform(&mut self, _handle: GizmoHandle) {}
    /// Update the active transform by `delta` (an absolute translation/rotation
    /// from the grab point). Returns whether geometry changed (so we re-mesh).
    fn update_transform(&mut self, _delta: TransformDelta) -> bool {
        false
    }
    /// Finish the active transform: commit it, or cancel and discard.
    fn finish_transform(&mut self, _commit: bool) {}
}

/// A transform gizmo shown at a selected body.
pub struct Gizmo {
    /// World-space origin — the body's bounding-box center.
    pub origin: [f64; 3],
}

/// A draggable gizmo handle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GizmoHandle {
    /// Translate along a world axis (an arrow).
    TranslateAxis(Axis3),
    /// Translate within a world plane (a corner square).
    TranslatePlane(Plane3),
    /// Rotate about a world axis (a ring).
    RotateAxis(Axis3),
}

/// An absolute transform from the grab point, applied by the controller.
#[derive(Clone, Copy, Debug)]
pub enum TransformDelta {
    /// World-space translation offset.
    Translate([f64; 3]),
    /// Rotation by `angle` radians about `axis` (through the gizmo pivot).
    Rotate { axis: [f64; 3], angle: f64 },
}

/// One of the three world axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis3 {
    X,
    Y,
    Z,
}

impl Axis3 {
    const ALL: [Axis3; 3] = [Axis3::X, Axis3::Y, Axis3::Z];

    /// Unit direction.
    fn dir(self) -> Vec3 {
        match self {
            Axis3::X => Vec3::X,
            Axis3::Y => Vec3::Y,
            Axis3::Z => Vec3::Z,
        }
    }

    /// The two unit vectors spanning the plane perpendicular to this axis (the
    /// in-plane basis for the rotation ring).
    fn perps(self) -> (Vec3, Vec3) {
        match self {
            Axis3::X => (Vec3::Y, Vec3::Z),
            Axis3::Y => (Vec3::Z, Vec3::X),
            Axis3::Z => (Vec3::X, Vec3::Y),
        }
    }

    /// Base color (X red, Y green, Z blue).
    fn color(self) -> [f32; 3] {
        match self {
            Axis3::X => [0.90, 0.25, 0.28],
            Axis3::Y => [0.35, 0.78, 0.30],
            Axis3::Z => [0.30, 0.45, 0.95],
        }
    }
}

/// One of the three world planes (for plane-constrained translation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane3 {
    Xy,
    Yz,
    Zx,
}

impl Plane3 {
    const ALL: [Plane3; 3] = [Plane3::Xy, Plane3::Yz, Plane3::Zx];

    /// The two in-plane axes.
    fn axes(self) -> (Axis3, Axis3) {
        match self {
            Plane3::Xy => (Axis3::X, Axis3::Y),
            Plane3::Yz => (Axis3::Y, Axis3::Z),
            Plane3::Zx => (Axis3::Z, Axis3::X),
        }
    }

    /// The axis normal to the plane.
    fn normal(self) -> Axis3 {
        match self {
            Plane3::Xy => Axis3::Z,
            Plane3::Yz => Axis3::X,
            Plane3::Zx => Axis3::Y,
        }
    }

    /// Handle color — the normal axis's color, lightened so a plane reads as a
    /// plane rather than an axis.
    fn color(self) -> [f32; 3] {
        let c = self.normal().color();
        [
            c[0] + (1.0 - c[0]) * 0.35,
            c[1] + (1.0 - c[1]) * 0.35,
            c[2] + (1.0 - c[2]) * 0.35,
        ]
    }
}

/// Pixel format of the offscreen face-id pick buffer.
const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

pub use rmf_core::Pick;

/// How close (pixels) the cursor must be to an edge to pick it over a face.
const EDGE_PICK_PX: f32 = 6.0;

/// Max selected edges the shader can highlight at once (4 ids per `vec4`).
const SEL_EDGE_CAP: usize = 64;
const SEL_EDGE_VEC4: usize = SEL_EDGE_CAP / 4;

/// GPU uniform block, mirrored by `Globals` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    /// `[selected_face, has_sel, hovered_face, has_hov]` (face select is single).
    faces: [u32; 4],
    /// `[selected_edge_count, 0, hovered_edge, has_hov]`.
    edges: [u32; 4],
    /// The selected edge ids, packed 4 per `vec4`, `edges[0]` of them valid.
    sel_edges: [[u32; 4]; SEL_EDGE_VEC4],
}

impl Globals {
    fn for_view(camera: &OrbitCamera, aspect: f32, highlights: &Highlights) -> Self {
        let eye = camera.eye();

        // Selection: one face highlight (sketch-on-face is single) + an edge set.
        let mut faces = [0u32; 4];
        let mut edges = [0u32; 4];
        let mut sel_edges = [[0u32; 4]; SEL_EDGE_VEC4];
        let mut n_edges = 0usize;
        for pick in &highlights.selected {
            match pick {
                Pick::Face(id) => {
                    if faces[1] == 0 {
                        faces[0] = *id;
                        faces[1] = 1;
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
        }
    }
}

/// Entities to emphasize this frame: a clicked selection set (strong) and a
/// single hover pre-highlight (subtle). Faces highlight one at a time; edges
/// highlight as a set (for multi-edge fillet).
#[derive(Clone, Default)]
pub struct Highlights {
    pub selected: Vec<Pick>,
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
    /// Line pipeline for the transform gizmo (drawn on top, depth test off).
    gizmo_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    edge_vertex_buffer: wgpu::Buffer,
    edge_index_buffer: wgpu::Buffer,
    edge_index_count: u32,
    /// Gizmo line vertices (rebuilt each frame; empty when hidden).
    gizmo_vertex_buffer: wgpu::Buffer,
    gizmo_vertex_count: u32,
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

        // Gizmo pipeline: colored line list, always drawn on top (depth test
        // disabled) so handles stay visible and grabbable through the model.
        let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmf-gizmo-shader"),
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
            label: Some("rmf-gizmo-pipeline"),
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

        let (vertex_buffer, index_buffer, index_count) = make_mesh_buffers(&device, mesh);
        let (edge_vertex_buffer, edge_index_buffer, edge_index_count) =
            make_edge_buffers(&device, mesh);
        let (gizmo_vertex_buffer, gizmo_vertex_count) = make_gizmo_buffer(&device, &[]);

        Self {
            device,
            queue,
            pipeline,
            pick_pipeline,
            edge_pipeline,
            gizmo_pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            edge_vertex_buffer,
            edge_index_buffer,
            edge_index_count,
            gizmo_vertex_buffer,
            gizmo_vertex_count,
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

    /// Replace the gizmo line geometry (empty `verts` hides it).
    fn upload_gizmo(&mut self, verts: &[GizmoVertex]) {
        let (buf, n) = make_gizmo_buffer(&self.device, verts);
        self.gizmo_vertex_buffer = buf;
        self.gizmo_vertex_count = n;
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
        highlights: &Highlights,
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
        // The gizmo draws last, on top of everything (its pipeline ignores depth).
        if self.gizmo_vertex_count > 0 {
            pass.set_pipeline(&self.gizmo_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.gizmo_vertex_buffer.slice(..));
            pass.draw(0..self.gizmo_vertex_count, 0..1);
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
        let globals = Globals::for_view(camera, aspect, &Highlights::default());
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

    /// Like [`Self::pick_at`], plus the world-space point under the cursor (from
    /// the depth buffer) when a face is hit — used as a push/pull anchor that's
    /// guaranteed to lie on the face (avoids holes/odd shapes).
    fn pick_with_point(
        &self,
        px: u32,
        py: u32,
        camera: &OrbitCamera,
        w: u32,
        h: u32,
    ) -> (Option<Pick>, Option<Vec3>) {
        if let Some(edge) = self.pick_edge(px as f32, py as f32, camera, w, h) {
            return (Some(Pick::Edge(edge)), self.edge_point(edge, px, py, camera, w, h));
        }
        match self.pick_face(px, py, camera, w, h) {
            Some(id) => (Some(Pick::Face(id)), self.world_under_cursor(px, py, camera, w, h)),
            None => (None, None),
        }
    }

    /// The world-space point under pixel `(px, py)`, reconstructed from a depth
    /// render. `None` if nothing is there.
    fn world_under_cursor(
        &self,
        px: u32,
        py: u32,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
    ) -> Option<Vec3> {
        if self.index_count == 0 || px >= width || py >= height {
            return None;
        }
        let aspect = width as f32 / height.max(1) as f32;
        let globals = Globals::for_view(camera, aspect, &Highlights::default());
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        let make = |format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("rmf-depth-pick"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = make(PICK_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = make(
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rmf-depth-pick-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
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

        // Depth textures only allow full copies, so copy the whole texture and
        // index the pixel at `(px, py)`.
        let padded = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmf-depth-readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &depth,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
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
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map depth readback"));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        let data = slice.get_mapped_range();
        let off = (py * padded + px * 4) as usize;
        let z = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        drop(data);
        readback.unmap();

        if z >= 1.0 - 1e-6 {
            return None; // background
        }
        // Unproject the pixel CENTRE (+0.5) so the NDC matches the depth sample.
        let nx = (px as f32 + 0.5) / width as f32 * 2.0 - 1.0;
        let ny = 1.0 - (py as f32 + 0.5) / height as f32 * 2.0;
        let inv = camera.view_proj(aspect).inverse();
        let p = inv * glam::Vec4::new(nx, ny, z, 1.0);
        Some(p.truncate() / p.w)
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

    /// A world-space point on edge `edge_id` nearest the cursor ray — a durable
    /// anchor for fillet/chamfer. Picks the closest point among the edge's
    /// segments to the ray cast through pixel `(px, py)`.
    fn edge_point(
        &self,
        edge_id: u32,
        px: u32,
        py: u32,
        camera: &OrbitCamera,
        w: u32,
        h: u32,
    ) -> Option<Vec3> {
        let cursor = PhysicalPosition::new(px as f64 + 0.5, py as f64 + 0.5);
        let (origin, ray) = cursor_ray(camera, cursor, w, h);
        let mut best = (f32::MAX, None);
        for &(a, b, id) in &self.edge_segments {
            if id != edge_id {
                continue;
            }
            let (pt, d) = closest_point_on_segment_to_ray(a, b, origin, ray);
            if d < best.0 {
                best = (d, Some(pt));
            }
        }
        best.1
    }
}

/// Closest point on segment `[a, b]` to the ray `(origin, dir)`, with the
/// distance between that point and the ray. Used to anchor edge picks.
fn closest_point_on_segment_to_ray(a: Vec3, b: Vec3, origin: Vec3, dir: Vec3) -> (Vec3, f32) {
    let ab = b - a;
    let w0 = a - origin;
    let (u, v) = (ab.dot(ab), ab.dot(dir));
    let (d, e) = (ab.dot(w0), dir.dot(w0));
    let denom = u - v * v; // dir is unit-length, so dir·dir = 1
    let t = if denom.abs() < 1e-6 {
        0.0
    } else {
        ((v * e - d) / denom).clamp(0.0, 1.0)
    };
    let pt = a + ab * t;
    // Distance from pt to the ray line.
    let to_pt = pt - origin;
    let proj = to_pt.dot(dir);
    (pt, (to_pt - dir * proj).length())
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

/// World-space ray (origin, unit dir) through `cursor` for the given camera.
fn cursor_ray(
    camera: &OrbitCamera,
    cursor: PhysicalPosition<f64>,
    width: u32,
    height: u32,
) -> (Vec3, Vec3) {
    let (w, h) = (width as f32, height as f32);
    let nx = cursor.x as f32 / w * 2.0 - 1.0;
    let ny = 1.0 - cursor.y as f32 / h * 2.0;
    let inv = camera.view_proj(w / h.max(1.0)).inverse();
    let unproject = |z: f32| {
        let p = inv * glam::Vec4::new(nx, ny, z, 1.0);
        p.truncate() / p.w
    };
    let near = unproject(0.0);
    (near, (unproject(1.0) - near).normalize_or_zero())
}

/// Signed distance along the manip axis `(point, dir)` to the closest point of
/// the cursor ray — how far the user has dragged the face.
fn manip_distance(
    camera: &OrbitCamera,
    cursor: PhysicalPosition<f64>,
    point: Vec3,
    dir: Vec3,
    width: u32,
    height: u32,
) -> f32 {
    let (origin, ray) = cursor_ray(camera, cursor, width, height);
    // Closest approach between line (point + t·dir) and ray (origin + s·ray).
    let w0 = point - origin;
    let (b, c) = (dir.dot(ray), ray.dot(ray));
    let (d, e) = (dir.dot(w0), ray.dot(w0));
    let denom = dir.dot(dir) * c - b * b;
    if denom.abs() < 1e-6 {
        return 0.0; // ray parallel to the axis
    }
    (b * e - c * d) / denom
}

// --- Transform gizmo -------------------------------------------------------

/// How close (pixels) the cursor must be to an arrow to grab it.
const GIZMO_PICK_PX: f32 = 10.0;
/// Hovered/active handle highlight color (gold).
const GIZMO_HILITE: [f32; 3] = [1.0, 0.80, 0.15];
/// Alpha for idle (non-hovered) handles — they recede until reached for.
const GIZMO_FADE: f32 = 0.5;
/// Inner edge of a planar handle, as a fraction of the axis length.
const GIZMO_PLANE_OFFSET: f32 = 0.38;
/// Side length of a planar handle, as a fraction of the axis length.
const GIZMO_PLANE_SIZE: f32 = 0.18;
/// Rotation-ring radius, as a fraction of the axis length (encircles the arrows).
const GIZMO_RING_RADIUS: f32 = 1.05;
/// Segments used to draw / hit-test a rotation ring.
const RING_SEGMENTS: usize = 64;

/// Ratio of the cursor's screen distance from the gizmo center to the ring's
/// screen radius — the input to the radial snap zones (~1.0 means on the ring).
fn gizmo_ring_ratio(
    center: Vec3,
    axis: Axis3,
    camera: &OrbitCamera,
    cursor: PhysicalPosition<f64>,
    w: u32,
    h: u32,
) -> f32 {
    let len = gizmo_axis_length(camera);
    let Some(cc) = project_point(center, camera, w, h) else {
        return 99.0;
    };
    let (u, _) = axis.perps();
    let Some(edge) = project_point(center + u * len * GIZMO_RING_RADIUS, camera, w, h) else {
        return 99.0;
    };
    let ring_px = (edge.0 - cc.0).hypot(edge.1 - cc.1).max(1.0);
    (cursor.x as f32 - cc.0).hypot(cursor.y as f32 - cc.1) / ring_px
}

/// World-space points evenly spaced around a rotation ring (axis `axis`).
fn ring_points(origin: Vec3, axis: Axis3, len: f32) -> [Vec3; RING_SEGMENTS] {
    let (u, v) = axis.perps();
    let r = len * GIZMO_RING_RADIUS;
    std::array::from_fn(|i| {
        let t = i as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        origin + (u * t.cos() + v * t.sin()) * r
    })
}

/// The signed angle (radians) of `p` (relative to the ring center) within the
/// rotation plane of `axis`, measured from the `u` basis vector.
fn ring_angle(p_minus_center: Vec3, axis: Axis3) -> f32 {
    let (u, v) = axis.perps();
    p_minus_center.dot(v).atan2(p_minus_center.dot(u))
}

/// Wrap an angle delta to `[-π, π]` (so accumulating around the ring is smooth
/// across the ±π seam).
fn wrap_pi(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut a = a % tau;
    if a > std::f32::consts::PI {
        a -= tau;
    } else if a < -std::f32::consts::PI {
        a += tau;
    }
    a
}

/// Tinkercad-style radial snap increment (radians): coarse near the ring, finer
/// as the cursor moves out, `0` (free) far out or when `free` is held. `ratio`
/// is the cursor distance from center over the ring radius (~1.0 = on the ring).
fn snap_increment(ratio: f32, free: bool) -> f32 {
    if free {
        return 0.0;
    }
    let deg = if ratio < 1.25 {
        45.0
    } else if ratio < 2.0 {
        15.0
    } else if ratio < 3.0 {
        5.0
    } else {
        return 0.0; // free zone, far out
    };
    deg * std::f32::consts::PI / 180.0
}

/// Snap `angle` to the increment for the given radial `ratio` (see
/// [`snap_increment`]). Returns the snapped angle in radians.
fn snap_rotation(angle: f32, ratio: f32, free: bool) -> f32 {
    let inc = snap_increment(ratio, free);
    if inc == 0.0 {
        angle
    } else {
        (angle / inc).round() * inc
    }
}

/// Wrap an RGB color to RGBA with the given alpha.
fn rgba(c: [f32; 3], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

/// Visual state for the rotation protractor during a ring drag.
#[derive(Clone, Copy)]
struct RotationViz {
    axis: Axis3,
    /// Angle (in the ring plane) where the drag began.
    start: f32,
    /// Snapped swept angle (signed) from `start`.
    swept: f32,
    /// Current snap increment (radians); `0` when free (no tick marks).
    inc: f32,
}

/// World length of the gizmo arrows — proportional to the orbit distance so the
/// gizmo stays a roughly constant size on screen.
fn gizmo_axis_length(camera: &OrbitCamera) -> f32 {
    camera.distance * 0.22
}

/// Project `p` to pixel coordinates, or `None` if behind the camera.
fn project_point(p: Vec3, camera: &OrbitCamera, w: u32, h: u32) -> Option<(f32, f32)> {
    let clip = camera.view_proj(w as f32 / h.max(1) as f32) * p.extend(1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(((ndc.x * 0.5 + 0.5) * w as f32, (0.5 - ndc.y * 0.5) * h as f32))
}

/// The four world-space corners of a planar handle square.
fn plane_corners(origin: Vec3, plane: Plane3, len: f32) -> [Vec3; 4] {
    let (a1, a2) = plane.axes();
    let (u, v) = (a1.dir(), a2.dir());
    let off = len * GIZMO_PLANE_OFFSET;
    let s = len * GIZMO_PLANE_SIZE;
    [
        origin + u * off + v * off,
        origin + u * (off + s) + v * off,
        origin + u * (off + s) + v * (off + s),
        origin + u * off + v * (off + s),
    ]
}

/// Intersection of ray `(ro, rd)` with the plane through `p` with normal `n`.
fn ray_plane_intersect(ro: Vec3, rd: Vec3, p: Vec3, n: Vec3) -> Option<Vec3> {
    let denom = rd.dot(n);
    if denom.abs() < 1e-6 {
        return None; // ray parallel to the plane
    }
    let t = (p - ro).dot(n) / denom;
    (t >= 0.0).then(|| ro + rd * t)
}

/// Is screen point `p` inside the triangle `(a, b, c)`?
fn point_in_tri(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let cross = |o: (f32, f32), u: (f32, f32), v: (f32, f32)| {
        (u.0 - o.0) * (v.1 - o.1) - (u.1 - o.1) * (v.0 - o.0)
    };
    let d1 = cross(p, a, b);
    let d2 = cross(p, b, c);
    let d3 = cross(p, c, a);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// The gizmo handle under `cursor`: a planar handle (filled square) takes
/// priority over an axis arrow it might overlap.
fn gizmo_hit_test(
    origin: Vec3,
    camera: &OrbitCamera,
    cursor: PhysicalPosition<f64>,
    w: u32,
    h: u32,
) -> Option<GizmoHandle> {
    let len = gizmo_axis_length(camera);
    let c = (cursor.x as f32, cursor.y as f32);

    // Planar handles first (a filled region in each axis corner).
    for plane in Plane3::ALL {
        let pts: Option<Vec<(f32, f32)>> = plane_corners(origin, plane, len)
            .iter()
            .map(|p| project_point(*p, camera, w, h))
            .collect();
        if let Some(s) = pts {
            if point_in_tri(c, s[0], s[1], s[2]) || point_in_tri(c, s[0], s[2], s[3]) {
                return Some(GizmoHandle::TranslatePlane(plane));
            }
        }
    }

    // Then axis arrows (nearest shaft within the pixel threshold).
    let po = project_point(origin, camera, w, h)?;
    let mut best = (GIZMO_PICK_PX, None);
    for axis in Axis3::ALL {
        if let Some(tip) = project_point(origin + axis.dir() * len, camera, w, h) {
            let d = point_segment_distance(c.0, c.1, po, tip);
            if d < best.0 {
                best = (d, Some(axis));
            }
        }
    }
    if let Some(axis) = best.1 {
        return Some(GizmoHandle::TranslateAxis(axis));
    }

    // Finally rotation rings (nearest ring outline within the threshold).
    let mut best_ring = (GIZMO_PICK_PX, None);
    for axis in Axis3::ALL {
        let pts = ring_points(origin, axis, len);
        let mut prev = project_point(pts[RING_SEGMENTS - 1], camera, w, h);
        for p in pts {
            let cur = project_point(p, camera, w, h);
            if let (Some(a), Some(b)) = (prev, cur) {
                let d = point_segment_distance(c.0, c.1, a, b);
                if d < best_ring.0 {
                    best_ring = (d, Some(axis));
                }
            }
            prev = cur;
        }
    }
    best_ring.1.map(GizmoHandle::RotateAxis)
}

/// Build the gizmo triangle geometry at `origin`: three camera-facing axis
/// arrows, three planar-translate squares, and three rotation rings. `active`
/// is highlighted gold; `rotation` (if a ring is being dragged) adds the
/// translucent protractor wedge and snap tick marks.
fn gizmo_geometry(
    origin: Vec3,
    camera: &OrbitCamera,
    active: Option<GizmoHandle>,
    rotation: Option<RotationViz>,
) -> Vec<GizmoVertex> {
    let eye = camera.eye();
    let len = gizmo_axis_length(camera);
    let shaft_w = len * 0.016; // half-width of the shaft ribbon
    let head = len * 0.20; // arrowhead length
    let head_r = len * 0.07; // arrowhead half-width
    let mut verts = Vec::new();

    let tri = |verts: &mut Vec<GizmoVertex>, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]| {
        verts.push(GizmoVertex { position: a.to_array(), color });
        verts.push(GizmoVertex { position: b.to_array(), color });
        verts.push(GizmoVertex { position: c.to_array(), color });
    };
    // A camera-facing ribbon (two triangles) from `a` to `b`.
    let ribbon = |verts: &mut Vec<GizmoVertex>, a: Vec3, b: Vec3, hw: f32, color: [f32; 4]| {
        let view = (eye - (a + b) * 0.5).normalize_or_zero();
        let perp = (b - a).cross(view).normalize_or_zero() * hw;
        tri(verts, a - perp, a + perp, b + perp, color);
        tri(verts, a - perp, b + perp, b - perp, color);
    };

    // While rotating, focus the view: show only the active ring + protractor
    // (the arrows, planes, and other rings would just be clutter mid-turn).
    if let Some(rv) = rotation {
        let (u, v) = rv.axis.perps();
        let r = len * GIZMO_RING_RADIUS;
        let at = |a: f32| origin + (u * a.cos() + v * a.sin()) * r;

        // Translucent swept wedge from the start angle to the snapped angle.
        let wedge = rgba(GIZMO_HILITE, 0.35);
        let steps = ((rv.swept.abs() / (std::f32::consts::PI / 45.0)).ceil() as usize).max(1);
        for i in 0..steps {
            let a0 = rv.start + rv.swept * (i as f32 / steps as f32);
            let a1 = rv.start + rv.swept * ((i + 1) as f32 / steps as f32);
            tri(&mut verts, origin, at(a0), at(a1), wedge);
        }

        // Snap tick marks straddling the ring at each increment — anchored at
        // the grab angle so a tick sits exactly at 0 sweep and at every snap.
        if rv.inc > 0.0 {
            let ticks = (std::f32::consts::TAU / rv.inc).round() as usize;
            let tick = rgba([1.0, 1.0, 1.0], 0.6);
            for k in 0..ticks {
                let a = rv.start + k as f32 * rv.inc;
                let d = u * a.cos() + v * a.sin();
                ribbon(&mut verts, origin + d * (r * 0.93), origin + d * (r * 1.07), len * 0.006, tick);
            }
        }

        // The active ring itself (gold), and the two radii bounding the wedge.
        let gold = rgba(GIZMO_HILITE, 1.0);
        let ring_w = len * 0.013;
        let pts = ring_points(origin, rv.axis, len);
        for i in 0..RING_SEGMENTS {
            ribbon(&mut verts, pts[i], pts[(i + 1) % RING_SEGMENTS], ring_w, gold);
        }
        ribbon(&mut verts, origin, at(rv.start), len * 0.008, gold);
        ribbon(&mut verts, origin, at(rv.start + rv.swept), len * 0.008, gold);
        return verts;
    }

    // The hovered handle is gold and opaque; the rest recede (faded) so the
    // gizmo stays quiet until you reach for a specific handle.
    let color_for = |handle: GizmoHandle, base: [f32; 3]| -> [f32; 4] {
        if active == Some(handle) {
            rgba(GIZMO_HILITE, 1.0)
        } else {
            rgba(base, GIZMO_FADE)
        }
    };

    for axis in Axis3::ALL {
        let color = color_for(GizmoHandle::TranslateAxis(axis), axis.color());
        let dir = axis.dir();
        let tip = origin + dir * len;
        let base = origin + dir * (len - head);
        // Shaft ribbon, then a camera-facing arrowhead triangle.
        ribbon(&mut verts, origin, base, shaft_w, color);
        let view = (eye - tip).normalize_or_zero();
        let perp = dir.cross(view).normalize_or_zero() * head_r;
        tri(&mut verts, tip, base + perp, base - perp, color);
    }

    for plane in Plane3::ALL {
        let color = color_for(GizmoHandle::TranslatePlane(plane), plane.color());
        let q = plane_corners(origin, plane, len);
        tri(&mut verts, q[0], q[1], q[2], color);
        tri(&mut verts, q[0], q[2], q[3], color);
    }

    // Rotation rings (camera-facing tube around each axis).
    let ring_w = len * 0.013;
    for axis in Axis3::ALL {
        let color = color_for(GizmoHandle::RotateAxis(axis), axis.color());
        let pts = ring_points(origin, axis, len);
        for i in 0..RING_SEGMENTS {
            ribbon(&mut verts, pts[i], pts[(i + 1) % RING_SEGMENTS], ring_w, color);
        }
    }
    verts
}

fn make_gizmo_buffer(device: &wgpu::Device, verts: &[GizmoVertex]) -> (wgpu::Buffer, u32) {
    use wgpu::util::DeviceExt;
    if verts.is_empty() {
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-gizmo-empty"),
            contents: bytemuck::bytes_of(&GizmoVertex::default()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        return (buf, 0);
    }
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmf-gizmo"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    (buf, verts.len() as u32)
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

/// An in-progress gizmo drag. The reference geometry (axis line / plane) is
/// captured at grab time and kept fixed, so the body translating under the
/// gizmo doesn't shift the drag reference.
#[derive(Clone, Copy)]
enum GizmoDrag {
    /// Translate along an axis: line `(origin, dir)` + drag distance at grab.
    Axis {
        axis: Axis3,
        origin: Vec3,
        dir: Vec3,
        start: f32,
    },
    /// Translate in a plane: plane `(point, normal)` + the grab intersection.
    Plane {
        plane: Plane3,
        point: Vec3,
        normal: Vec3,
        grab: Vec3,
    },
    /// Rotate about an axis: pivot `center`, plane normal, and accumulated angle.
    /// `last` is the previous cursor angle (for seam-free accumulation), `total`
    /// the raw accumulated angle, `snapped` the value last applied.
    Rotate {
        axis: Axis3,
        center: Vec3,
        normal: Vec3,
        /// Cursor angle where the drag began (the protractor's zero).
        start: f32,
        last: f32,
        total: f32,
        snapped: f32,
    },
}

impl GizmoDrag {
    /// The handle this drag corresponds to (for highlighting).
    fn handle(&self) -> GizmoHandle {
        match self {
            GizmoDrag::Axis { axis, .. } => GizmoHandle::TranslateAxis(*axis),
            GizmoDrag::Plane { plane, .. } => GizmoHandle::TranslatePlane(*plane),
            GizmoDrag::Rotate { axis, .. } => GizmoHandle::RotateAxis(*axis),
        }
    }
}

struct WindowApp<C: Controller> {
    controller: C,
    camera: OrbitCamera,
    mouse: MouseState,
    /// Active push/pull drag axis `(point, unit normal)`, set when a left-drag
    /// began on a manipulable face.
    manip_axis: Option<(Vec3, Vec3)>,
    /// Active gizmo arrow drag, set when a left-press grabbed a gizmo handle.
    gizmo_drag: Option<GizmoDrag>,
    /// Geometry changed outside `ui` (a manipulation drag) — re-mesh next frame.
    mesh_dirty: bool,
    /// Latest keyboard modifier state, for ⇧/⌘-click additive selection.
    modifiers: winit::keyboard::ModifiersState,
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
            manip_axis: None,
            gizmo_drag: None,
            mesh_dirty: false,
            modifiers: winit::keyboard::ModifiersState::empty(),
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
                            let (w, h) = self
                                .state
                                .as_ref()
                                .map(|s| (s.config.width, s.config.height))
                                .unwrap_or((1, 1));
                            // A gizmo handle grab takes priority over everything.
                            let gizmo_hit = self.controller.gizmo().and_then(|g| {
                                let origin = Vec3::new(
                                    g.origin[0] as f32,
                                    g.origin[1] as f32,
                                    g.origin[2] as f32,
                                );
                                gizmo_hit_test(origin, &self.camera, self.mouse.cursor, w, h)
                                    .map(|handle| (origin, handle))
                            });
                            if let Some((origin, handle)) = gizmo_hit {
                                let drag = match handle {
                                    GizmoHandle::TranslateAxis(axis) => {
                                        let dir = axis.dir();
                                        let start = manip_distance(
                                            &self.camera,
                                            self.mouse.cursor,
                                            origin,
                                            dir,
                                            w,
                                            h,
                                        );
                                        GizmoDrag::Axis { axis, origin, dir, start }
                                    }
                                    GizmoHandle::TranslatePlane(plane) => {
                                        let normal = plane.normal().dir();
                                        let (ro, rd) =
                                            cursor_ray(&self.camera, self.mouse.cursor, w, h);
                                        let grab = ray_plane_intersect(ro, rd, origin, normal)
                                            .unwrap_or(origin);
                                        GizmoDrag::Plane { plane, point: origin, normal, grab }
                                    }
                                    GizmoHandle::RotateAxis(axis) => {
                                        let normal = axis.dir();
                                        let (ro, rd) =
                                            cursor_ray(&self.camera, self.mouse.cursor, w, h);
                                        let last = ray_plane_intersect(ro, rd, origin, normal)
                                            .map(|p| ring_angle(p - origin, axis))
                                            .unwrap_or(0.0);
                                        GizmoDrag::Rotate {
                                            axis,
                                            center: origin,
                                            normal,
                                            start: last,
                                            last,
                                            total: 0.0,
                                            snapped: 0.0,
                                        }
                                    }
                                };
                                self.controller.start_transform(handle);
                                self.gizmo_drag = Some(drag);
                                self.manip_axis = None;
                                self.mouse.orbiting = false;
                            } else {
                                // The "smart left button": a drag that starts on a
                                // manipulable face push/pulls it; else orbits.
                                let (pick, world) = if self.controller.wants_picking() {
                                    self.pick_with_point_under_cursor()
                                } else {
                                    (None, None)
                                };
                                let eye = self.camera.eye();
                                let axis = match (pick, world) {
                                    (Some(p), Some(pt)) => self.controller.start_manipulation(
                                        p,
                                        [pt.x as f64, pt.y as f64, pt.z as f64],
                                        [eye.x as f64, eye.y as f64, eye.z as f64],
                                    ),
                                    _ => None,
                                };
                                match axis {
                                    Some((point, normal)) => {
                                        self.manip_axis = Some((
                                            Vec3::new(point[0] as f32, point[1] as f32, point[2] as f32),
                                            Vec3::new(normal[0] as f32, normal[1] as f32, normal[2] as f32)
                                                .normalize_or_zero(),
                                        ));
                                        self.mouse.orbiting = false;
                                    }
                                    None => {
                                        self.manip_axis = None;
                                        self.mouse.orbiting = true;
                                    }
                                }
                            }
                            self.mouse.left_press = Some(self.mouse.cursor);
                            self.mouse.dragged = false;
                        } else if !down {
                            self.mouse.orbiting = false;
                            self.mouse.last = None;
                            self.mouse.hover_dirty = true;
                            if self.gizmo_drag.take().is_some() {
                                // Commit only if it actually moved (a click on the
                                // arrow without a drag discards the no-op feature).
                                self.controller.finish_transform(self.mouse.dragged);
                                if self.mouse.dragged {
                                    self.mesh_dirty = true;
                                    if let Some(s) = self.state.as_ref() {
                                        s.window.request_redraw();
                                    }
                                }
                            } else if self.manip_axis.is_some() && self.mouse.dragged {
                                self.controller.finish_manipulation(true);
                                self.mesh_dirty = true;
                                if let Some(s) = self.state.as_ref() {
                                    s.window.request_redraw();
                                }
                            } else if self.mouse.left_press.is_some()
                                && !self.mouse.dragged
                                && self.controller.wants_picking()
                            {
                                // A click that didn't drag selects.
                                self.pick_at_cursor();
                            }
                            self.manip_axis = None;
                            self.mouse.left_press = None;
                            self.mouse.dragged = false;
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
                    if let Some(drag) = self.gizmo_drag.as_mut() {
                        // Gizmo: translate along the grabbed axis / within the
                        // plane, or rotate about the grabbed ring (with snap).
                        if self.mouse.dragged {
                            let (w, h) = (state.config.width, state.config.height);
                            let free = self.modifiers.alt_key();
                            let cursor = self.mouse.cursor;
                            let delta = match drag {
                                GizmoDrag::Axis { origin, dir, start, .. } => {
                                    let cur = manip_distance(&self.camera, cursor, *origin, *dir, w, h);
                                    let off = *dir * (cur - *start);
                                    TransformDelta::Translate([off.x as f64, off.y as f64, off.z as f64])
                                }
                                GizmoDrag::Plane { point, normal, grab, .. } => {
                                    let (ro, rd) = cursor_ray(&self.camera, cursor, w, h);
                                    let off = ray_plane_intersect(ro, rd, *point, *normal)
                                        .map(|p| p - *grab)
                                        .unwrap_or(Vec3::ZERO);
                                    TransformDelta::Translate([off.x as f64, off.y as f64, off.z as f64])
                                }
                                GizmoDrag::Rotate { axis, center, normal, last, total, snapped, .. } => {
                                    let (ro, rd) = cursor_ray(&self.camera, cursor, w, h);
                                    if let Some(p) = ray_plane_intersect(ro, rd, *center, *normal) {
                                        let cur = ring_angle(p - *center, *axis);
                                        *total += wrap_pi(cur - *last);
                                        *last = cur;
                                    }
                                    let ratio =
                                        gizmo_ring_ratio(*center, *axis, &self.camera, cursor, w, h);
                                    *snapped = snap_rotation(*total, ratio, free);
                                    let n = normal.normalize_or_zero();
                                    TransformDelta::Rotate {
                                        axis: [n.x as f64, n.y as f64, n.z as f64],
                                        angle: *snapped as f64,
                                    }
                                }
                            };
                            if self.controller.update_transform(delta) {
                                self.mesh_dirty = true;
                            }
                            state.window.request_redraw();
                        }
                        self.mouse.last = None;
                    } else if self.manip_axis.is_some() && self.mouse.dragged {
                        // Push/pull: distance = drag along the face normal.
                        let (point, dir) = self.manip_axis.unwrap();
                        let dist = manip_distance(
                            &self.camera,
                            self.mouse.cursor,
                            point,
                            dir,
                            state.config.width,
                            state.config.height,
                        );
                        if self.controller.update_manipulation(dist as f64) {
                            self.mesh_dirty = true;
                        }
                        state.window.request_redraw();
                        self.mouse.last = None;
                    } else {
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

            // Track ⇧/⌘ for additive (multi-) selection on the next click.
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
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
    /// The entity (and, for faces, the world point) under the cursor now.
    fn pick_with_point_under_cursor(&self) -> (Option<Pick>, Option<Vec3>) {
        let Some(state) = self.state.as_ref() else {
            return (None, None);
        };
        let (w, h) = (state.config.width, state.config.height);
        let px = (self.mouse.cursor.x as i64).clamp(0, w as i64 - 1) as u32;
        let py = (self.mouse.cursor.y as i64).clamp(0, h as i64 - 1) as u32;
        state.scene.pick_with_point(px, py, &self.camera, w, h)
    }

    /// Resolve the entity under the cursor and hand it to the controller.
    fn pick_at_cursor(&mut self) {
        let (pick, point) = self.pick_with_point_under_cursor();
        // ⇧ or ⌘ (super) held → add/remove from the current selection.
        let additive = self.modifiers.shift_key() || self.modifiers.super_key();
        self.controller.on_pick(
            pick,
            point.map(|p| [p.x as f64, p.y as f64, p.z as f64]),
            additive,
        );
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
        if changed || self.mesh_dirty {
            let mesh = self.controller.mesh();
            state.scene.upload_mesh(&mesh);
            self.mesh_dirty = false;
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

        // Build the transform gizmo for this frame, highlighting the axis under
        // the cursor (or the one being dragged).
        let gizmo_verts = if let Some(g) = self.controller.gizmo() {
            let origin = Vec3::new(g.origin[0] as f32, g.origin[1] as f32, g.origin[2] as f32);
            let (w, h) = (state.config.width, state.config.height);
            let active = if let Some(drag) = self.gizmo_drag {
                Some(drag.handle())
            } else if state.egui_ctx.is_pointer_over_egui() {
                None
            } else {
                gizmo_hit_test(origin, &self.camera, self.mouse.cursor, w, h)
            };
            // While dragging a ring, build the protractor (wedge + snap ticks).
            let rotation = match self.gizmo_drag {
                Some(GizmoDrag::Rotate { axis, center, start, snapped, .. }) if self.mouse.dragged => {
                    let ratio =
                        gizmo_ring_ratio(center, axis, &self.camera, self.mouse.cursor, w, h);
                    Some(RotationViz {
                        axis,
                        start,
                        swept: snapped,
                        inc: snap_increment(ratio, self.modifiers.alt_key()),
                    })
                }
                _ => None,
            };
            gizmo_geometry(origin, &self.camera, active, rotation)
        } else {
            Vec::new()
        };
        state.scene.upload_gizmo(&gizmo_verts);

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
            &self.controller.highlights(),
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
    if let Some(g) = controller.gizmo() {
        let origin = Vec3::new(g.origin[0] as f32, g.origin[1] as f32, g.origin[2] as f32);
        scene.upload_gizmo(&gizmo_geometry(origin, &camera, None, None));
    }
    scene.encode(&mut encoder, &color_view, &depth_view, &camera, width, height, &highlights);
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

    /// The depth-reconstructed world point under the cursor lands on the surface.
    #[test]
    fn world_under_cursor_recovers_the_surface_point() {
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

        // The center ray hits the quad at the origin (the camera target).
        let p = scene.world_under_cursor(32, 32, &camera, 64, 64).unwrap();
        assert!(p.length() < 1.0, "expected near origin, got {p:?}");
        assert!(p.z.abs() < 0.05, "should be on the z=0 plane, got z={}", p.z);
    }

    /// The gizmo hit-test grabs the arrow whose shaft the cursor sits on, and
    /// nothing when the cursor is away from all three. (Pure projection math, no
    /// GPU needed.)
    #[test]
    fn gizmo_hit_test_grabs_the_axis_under_the_cursor() {
        let camera = OrbitCamera::framing(Vec3::ZERO, 15.0);
        let (w, h) = (640u32, 480u32);
        let origin = Vec3::ZERO;
        let len = gizmo_axis_length(&camera);

        // A cursor sitting on each shaft (90% out, past the plane handles and
        // clear of the shared origin).
        for axis in Axis3::ALL {
            let on = project_point(origin + axis.dir() * len * 0.9, &camera, w, h).unwrap();
            let cursor = PhysicalPosition::new(on.0 as f64, on.1 as f64);
            assert_eq!(
                gizmo_hit_test(origin, &camera, cursor, w, h),
                Some(GizmoHandle::TranslateAxis(axis)),
                "axis {axis:?}"
            );
        }

        // A cursor in the middle of each planar handle grabs that plane.
        for plane in Plane3::ALL {
            let q = plane_corners(origin, plane, len);
            let center = (q[0] + q[1] + q[2] + q[3]) * 0.25;
            let on = project_point(center, &camera, w, h).unwrap();
            let cursor = PhysicalPosition::new(on.0 as f64, on.1 as f64);
            assert_eq!(
                gizmo_hit_test(origin, &camera, cursor, w, h),
                Some(GizmoHandle::TranslatePlane(plane))
            );
        }

        // A cursor on a rotation ring (sampled at 45°, clear of the arrows that
        // lie along the axes) grabs that axis's ring.
        for axis in Axis3::ALL {
            let (u, v) = axis.perps();
            let diag = (u + v).normalize() * len * GIZMO_RING_RADIUS;
            let on = project_point(origin + diag, &camera, w, h).unwrap();
            let cursor = PhysicalPosition::new(on.0 as f64, on.1 as f64);
            assert_eq!(
                gizmo_hit_test(origin, &camera, cursor, w, h),
                Some(GizmoHandle::RotateAxis(axis)),
                "ring {axis:?}"
            );
        }

        // Top-left corner is far from the centered gizmo → no grab.
        let far = PhysicalPosition::new(4.0, 4.0);
        assert_eq!(gizmo_hit_test(origin, &camera, far, w, h), None);
    }

    /// Radial snap zones: a 40° turn snaps to 45° near the ring (coarse), to 40°
    /// (5° steps) farther out, and stays free far out or with the modifier.
    #[test]
    fn rotation_snaps_by_radial_zone() {
        let deg = |d: f32| d * std::f32::consts::PI / 180.0;
        // Near the ring (ratio ~1) → coarse 45° snap.
        assert!((snap_rotation(deg(40.0), 1.0, false) - deg(45.0)).abs() < 1e-4);
        // Mid zone (ratio ~2.5) → 5° snap leaves 40° as-is.
        assert!((snap_rotation(deg(40.0), 2.5, false) - deg(40.0)).abs() < 1e-4);
        // Far out → free (no snap).
        assert!((snap_rotation(deg(41.3), 4.0, false) - deg(41.3)).abs() < 1e-4);
        // Modifier forces free even near the ring.
        assert!((snap_rotation(deg(40.0), 1.0, true) - deg(40.0)).abs() < 1e-4);
    }
}
