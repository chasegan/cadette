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

use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::camera::OrbitCamera;
use crate::controller::{Axis3, Controller, GizmoHandle, Highlights, Pick, TransformDelta};
use crate::gizmo::{
    gizmo_geometry, gizmo_hit_test, gizmo_ring_ratio, resize_geometry, resize_grip_pos,
    resize_hit_test, ring_angle, snap_distance, snap_increment, snap_rotation, snap_translation,
    GizmoDrag, RotationViz, MIN_RESIZE_DIM,
};
use crate::grid::grid_geometry;
use crate::math::{
    closest_point_on_segment_to_ray, cursor_ray, manip_distance, point_segment_distance,
    ray_plane_intersect, wrap_pi,
};
use crate::scene::{
    create_depth, egui_renderer, encode_egui, frame_camera, install_phosphor, request_device,
    Globals, Scene, DEPTH_FORMAT, PICK_DEPTH_FORMAT, PICK_FORMAT,
};
use crate::view::ViewContext;
use crate::view_cube::draw_view_cube;


/// The window icon (Windows/Linux taskbar + title bar). macOS ignores this and
/// uses the app bundle's `.icns`, so it's effectively a no-op there. `None` on
/// decode failure leaves winit's default.
fn app_icon() -> Option<winit::window::Icon> {
    let bytes = include_bytes!("../../../assets/icon/icon-256.png");
    let img = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}

/// How close (pixels) the cursor must be to an edge to pick it over a face.
const EDGE_PICK_PX: f32 = 6.0;
/// How far (NDC depth) behind the front surface an edge must be to count as
/// occluded — large enough to keep a face's own boundary edges, small enough to
/// drop edges on the far side of the body.
const PICK_DEPTH_TOL: f32 = 0.005;

impl Scene {
    /// Render the face-id + depth pass and read back, under pixel `(px, py)`,
    /// the face id (`None` = background) and the front-surface depth (NDC z;
    /// `1.0` where there's no geometry). Synchronous.
    fn pick_face(
        &self,
        px: u32,
        py: u32,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
    ) -> (Option<u32>, f32) {
        if self.index_count == 0 || px >= width || py >= height {
            return (None, 1.0);
        }

        let aspect = width as f32 / height.max(1) as f32;
        let globals = Globals::for_view(camera, aspect, &Highlights::default());
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        let make_target = |format, label| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let id_texture = make_target(PICK_FORMAT, "cdt-pick-id");
        let depth_texture = make_target(PICK_DEPTH_FORMAT, "cdt-pick-depth");
        let id_view = id_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_color_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = create_depth(&self.device, width, height);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cdt-pick") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cdt-pick-pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &id_view,
                        resolve_target: None,
                        // 0 = no geometry (the pick shader writes face_id + 1).
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &depth_color_view,
                        resolve_target: None,
                        // 1.0 = far / no geometry.
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
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

        // Copy both single pixels under the cursor to readback buffers.
        let make_readback = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let id_rb = make_readback("cdt-pick-id-readback");
        let depth_rb = make_readback("cdt-pick-depth-readback");
        let copy_pixel = |encoder: &mut wgpu::CommandEncoder, tex: &wgpu::Texture, buf: &wgpu::Buffer| {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        };
        copy_pixel(&mut encoder, &id_texture, &id_rb);
        copy_pixel(&mut encoder, &depth_texture, &depth_rb);
        self.queue.submit(Some(encoder.finish()));

        id_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map pick id"));
        depth_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map pick depth"));
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .ok();

        let id_data = id_rb.slice(..).get_mapped_range();
        let raw = u32::from_le_bytes([id_data[0], id_data[1], id_data[2], id_data[3]]);
        drop(id_data);
        id_rb.unmap();
        let depth_data = depth_rb.slice(..).get_mapped_range();
        let z = f32::from_le_bytes([depth_data[0], depth_data[1], depth_data[2], depth_data[3]]);
        drop(depth_data);
        depth_rb.unmap();

        ((raw != 0).then(|| raw - 1), z)
    }

    /// Whether the picked `edge` is occluded by the front surface (whose depth at
    /// the cursor is `face_depth`). Lets a nearby edge be ignored when it's
    /// really behind the face the cursor is over — the cause of hover flicker.
    fn edge_occluded(
        &self,
        edge: u32,
        face: Option<u32>,
        face_depth: f32,
        px: u32,
        py: u32,
        camera: &OrbitCamera,
        w: u32,
        h: u32,
    ) -> bool {
        if face.is_none() {
            return false; // nothing in front (silhouette / background)
        }
        let Some(p) = self.edge_point(edge, px, py, camera, w, h) else {
            return false;
        };
        let aspect = w as f32 / h.max(1) as f32;
        let clip = camera.view_proj(aspect) * p.extend(1.0);
        if clip.w <= 1e-6 {
            return false;
        }
        (clip.z / clip.w) > face_depth + PICK_DEPTH_TOL
    }

    /// Pick the entity under pixel `(px, py)`: a nearby *unoccluded* edge takes
    /// priority over the face; otherwise the face (or nothing). The occlusion
    /// check stops back-of-the-body edges that merely project near the cursor
    /// from stealing the hover off the face it's over.
    fn pick_at(&self, px: u32, py: u32, camera: &OrbitCamera, w: u32, h: u32) -> Option<Pick> {
        let (face, face_depth) = self.pick_face(px, py, camera, w, h);
        if let Some(edge) = self.pick_edge(px as f32, py as f32, camera, w, h) {
            if !self.edge_occluded(edge, face, face_depth, px, py, camera, w, h) {
                return Some(Pick::Edge(edge));
            }
        }
        face.map(Pick::Face)
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
        let (face, face_depth) = self.pick_face(px, py, camera, w, h);
        if let Some(edge) = self.pick_edge(px as f32, py as f32, camera, w, h) {
            if !self.edge_occluded(edge, face, face_depth, px, py, camera, w, h) {
                return (Some(Pick::Edge(edge)), self.edge_point(edge, px, py, camera, w, h));
            }
        }
        match face {
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
                label: Some("cdt-depth-pick"),
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
        // The pick pipeline writes two color targets (id + depth); this pass
        // only cares about the z-buffer, so both colors are throwaways.
        let color = make(PICK_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth_color = make(PICK_DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = make(
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_color_view = depth_color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cdt-depth-pick-pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &depth_color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
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
            label: Some("cdt-depth-readback"),
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
    /// Active push/pull drag axis `(point, unit normal)`, set when a left-drag
    /// began on a manipulable face.
    manip_axis: Option<(Vec3, Vec3)>,
    /// Active gizmo arrow drag, set when a left-press grabbed a gizmo handle.
    gizmo_drag: Option<GizmoDrag>,
    /// Geometry changed outside `ui` (a manipulation drag) — re-mesh next frame.
    mesh_dirty: bool,
    /// Latest keyboard modifier state, for ⇧/⌘-click additive selection.
    modifiers: winit::keyboard::ModifiersState,
    /// True while the window is hidden (another Space, minimized, fully covered).
    /// We stop redrawing then — vsync no longer paces the loop, so a continuous
    /// redraw would spin unthrottled and balloon GPU memory.
    occluded: bool,
    /// The window title we last applied, so we only call `set_title` when the
    /// controller's title (filename + dirty marker) actually changes.
    last_title: String,
    /// The central viewport rect (panels excluded), captured from the last
    /// frame's egui pass. Camera-nav routing in `window_event` runs outside a
    /// pass — where `available_rect()` is unavailable — so it hit-tests against
    /// this last-known layout (one frame of lag, imperceptible for input).
    central_rect: egui::Rect,
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
            occluded: false,
            last_title: String::new(),
            central_rect: egui::Rect::NOTHING,
            state: None,
        }
    }
}

impl<C: Controller> ApplicationHandler for WindowApp<C> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        // LogicalSize (DPI-independent): PhysicalSize would be only ~640×410
        // points on a 2× Retina display — hence a tiny window.
        let attrs = winit::window::Window::default_attributes()
            .with_title("Cadette")
            .with_window_icon(app_icon())
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
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
        install_phosphor(&egui_ctx);
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

        // Camera NAV (orbit/pan/zoom via right/middle/scroll) works over the
        // central viewport even when egui "consumed" the pointer: while sketching,
        // a full-window CentralPanel canvas claims it, but nav over that canvas
        // must still work. Left-clicks stay with egui there (they draw). Nav over
        // the side/top panels is still suppressed (pointer outside the central
        // area). We use the central rect cached from the last frame's pass —
        // `available_rect()` panics if called here, outside a pass.
        let over_central = state
            .egui_ctx
            .pointer_latest_pos()
            .is_some_and(|p| self.central_rect.contains(p));
        let nav_ok = !egui_used || over_central;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            // Hidden (another Space / minimized / fully covered): stop the
            // continuous redraw; resume — and refresh once — when shown again.
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    state.window.request_redraw();
                }
            }

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
                            // A resize grip or gizmo handle grab takes priority
                            // over picking/orbit. Resize grips (bbox handles)
                            // are the most specific target, so test them first.
                            let resize_hit = self.controller.resize_box().and_then(|rb| {
                                resize_hit_test(&rb, &self.camera, self.mouse.cursor, w, h)
                                    .map(|face| (rb, face))
                            });
                            let gizmo_hit = self.controller.gizmo().and_then(|g| {
                                let origin = Vec3::new(
                                    g.origin[0] as f32,
                                    g.origin[1] as f32,
                                    g.origin[2] as f32,
                                );
                                gizmo_hit_test(origin, &self.camera, self.mouse.cursor, w, h)
                                    .map(|handle| (origin, handle))
                            });
                            if let Some((rb, (axis, positive))) = resize_hit {
                                let origin = resize_grip_pos(&rb, axis, positive);
                                // `dir` points outward from the grabbed face, so a
                                // drag away from the body always grows the extent.
                                let dir = if positive { axis.dir() } else { -axis.dir() };
                                let start = manip_distance(
                                    &self.camera,
                                    self.mouse.cursor,
                                    origin,
                                    dir,
                                    w,
                                    h,
                                );
                                let i = match axis {
                                    Axis3::X => 0,
                                    Axis3::Y => 1,
                                    Axis3::Z => 2,
                                };
                                let base_extent = rb.max[i] - rb.min[i];
                                self.controller.start_resize(axis, positive);
                                self.gizmo_drag =
                                    Some(GizmoDrag::Resize { axis, origin, dir, start, base_extent });
                                self.manip_axis = None;
                                self.mouse.orbiting = false;
                            } else if let Some((origin, handle)) = gizmo_hit {
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
                                    // gizmo_hit_test only returns the center
                                    // handles; resize grips are handled above.
                                    GizmoHandle::Resize(_) => unreachable!(),
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
                            if let Some(drag) = self.gizmo_drag.take() {
                                // Commit only if it actually moved (a click on a
                                // handle without a drag discards the no-op edit).
                                match drag {
                                    GizmoDrag::Resize { .. } => {
                                        self.controller.finish_resize(self.mouse.dragged)
                                    }
                                    _ => self.controller.finish_transform(self.mouse.dragged),
                                }
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
                        if down && nav_ok {
                            self.mouse.orbiting = true;
                        } else if !down {
                            self.mouse.orbiting = false;
                            self.mouse.last = None;
                            self.mouse.hover_dirty = true;
                        }
                    }
                    MouseButton::Middle => {
                        if nav_ok {
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
                if nav_ok {
                    if let Some(drag) = self.gizmo_drag.as_mut() {
                        // Gizmo: translate along the grabbed axis / within the
                        // plane, or rotate about the grabbed ring (with snap).
                        if self.mouse.dragged {
                            let (w, h) = (state.config.width, state.config.height);
                            let free = self.modifiers.alt_key();
                            let grid = self.controller.grid().snap_step();
                            let cursor = self.mouse.cursor;
                            // Each arm drives the matching controller edit and
                            // reports whether geometry changed.
                            let changed = match drag {
                                GizmoDrag::Axis { origin, dir, start, .. } => {
                                    let cur = manip_distance(&self.camera, cursor, *origin, *dir, w, h);
                                    let off = snap_translation(*dir * (cur - *start), grid, free);
                                    self.controller.update_transform(TransformDelta::Translate([
                                        off.x as f64,
                                        off.y as f64,
                                        off.z as f64,
                                    ]))
                                }
                                GizmoDrag::Plane { point, normal, grab, .. } => {
                                    let (ro, rd) = cursor_ray(&self.camera, cursor, w, h);
                                    let off = ray_plane_intersect(ro, rd, *point, *normal)
                                        .map(|p| p - *grab)
                                        .unwrap_or(Vec3::ZERO);
                                    let off = snap_translation(off, grid, free);
                                    self.controller.update_transform(TransformDelta::Translate([
                                        off.x as f64,
                                        off.y as f64,
                                        off.z as f64,
                                    ]))
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
                                    self.controller.update_transform(TransformDelta::Rotate {
                                        axis: [n.x as f64, n.y as f64, n.z as f64],
                                        angle: *snapped as f64,
                                    })
                                }
                                GizmoDrag::Resize { axis, origin, dir, start, base_extent } => {
                                    let cur = manip_distance(&self.camera, cursor, *origin, *dir, w, h);
                                    let raw = *base_extent as f32 + (cur - *start);
                                    let extent = snap_distance(raw, grid, free).max(MIN_RESIZE_DIM);
                                    self.controller.update_resize(*axis, extent as f64)
                                }
                            };
                            if changed {
                                self.mesh_dirty = true;
                            }
                            state.window.request_redraw();
                        }
                        self.mouse.last = None;
                    } else if self.manip_axis.is_some() && self.mouse.dragged {
                        // Push/pull: distance = drag along the face normal, snapped
                        // to the grid (Alt for free) — a face push/pull is a resize.
                        let (point, dir) = self.manip_axis.unwrap();
                        let dist = manip_distance(
                            &self.camera,
                            self.mouse.cursor,
                            point,
                            dir,
                            state.config.width,
                            state.config.height,
                        );
                        let dist = snap_distance(dist, self.controller.grid().snap_step(), self.modifiers.alt_key());
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

            WindowEvent::MouseWheel { delta, .. } if nav_ok => {
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
            WindowEvent::PinchGesture { delta, .. } if nav_ok => {
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
    /// left the egui panels unpainted after the first frame. Skipped while the
    /// window is hidden: vsync no longer paces presentation there, so a
    /// continuous redraw spins unthrottled and balloons GPU memory.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.occluded {
            return;
        }
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
        let view = ViewContext::new(self.camera.view_proj(aspect), size, self.camera.eye());

        let raw_input = state.egui_state.take_egui_input(&state.window);
        state.egui_ctx.begin_pass(raw_input);
        let changed = self.controller.ui(&state.egui_ctx, &view);
        // Capture the central viewport rect now that the panels are laid out, for
        // next frame's out-of-pass nav routing in `window_event`.
        #[allow(deprecated)]
        {
            self.central_rect = state.egui_ctx.available_rect();
        }
        draw_view_cube(&state.egui_ctx, &mut self.camera, &self.controller.viewport_theme());
        let full_output = state.egui_ctx.end_pass();

        // Keep the OS window title in sync with the open file + dirty state
        // (the title carries the unsaved-changes mark itself).
        let title = self.controller.title();
        if title != self.last_title {
            state.window.set_title(&title);
            self.last_title = title;
        }
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
        let mut gizmo_verts = if let Some(g) = self.controller.gizmo() {
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
        // Resize grips on the selected primitive's bounding box. They sit with
        // the move/rotate gizmo, except mid-rotation (gizmo_geometry focuses the
        // view on the active ring, so grips would just be clutter).
        if !matches!(self.gizmo_drag, Some(GizmoDrag::Rotate { .. })) {
            if let Some(rb) = self.controller.resize_box() {
                let (w, h) = (state.config.width, state.config.height);
                let active = match self.gizmo_drag {
                    // `dir` points outward from the grabbed face → its sign is the
                    // face side.
                    Some(GizmoDrag::Resize { axis, dir, .. }) => {
                        Some((axis, dir.dot(axis.dir()) > 0.0))
                    }
                    _ if state.egui_ctx.is_pointer_over_egui() => None,
                    _ => resize_hit_test(&rb, &self.camera, self.mouse.cursor, w, h),
                };
                gizmo_verts.append(&mut resize_geometry(&rb, &self.camera, active));
            }
        }
        state.scene.upload_gizmo(&gizmo_verts);
        state
            .scene
            .upload_grid(&grid_geometry(self.controller.grid(), &self.controller.viewport_theme()));

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
            &self.controller.viewport_theme(),
        );
        encode_egui(&mut encoder, &state.egui_renderer, &view, &jobs, &screen);

        queue.submit(egui_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();

        for id in &full_output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }

        // Reclaim GPU memory from resources dropped this frame — the per-frame
        // command buffer, the rebuilt gizmo buffer, any replaced mesh buffers.
        // wgpu only frees on poll; this loop redraws continuously, so without a
        // non-blocking poll every frame the deferred-destruction queue grows
        // unbounded and exhausts (unified) memory until the OS freezes.
        device.poll(wgpu::PollType::Poll).ok();
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
    let mut camera = frame_camera(&mesh.vertices);
    let mut scene = Scene::new(device.clone(), queue.clone(), format, &mesh);
    let _ = &mut scene;

    let egui_ctx = egui::Context::default();
    install_phosphor(&egui_ctx);
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
        camera.eye(),
    );
    // Warm-up pass: auto-sized egui Areas (e.g. the gizmo readout) need a prior
    // frame to know their size before they paint. Keep its texture deltas — the
    // font atlas is emitted on this first pass — and apply them below.
    egui_ctx.begin_pass(raw_input.clone());
    controller.ui(&egui_ctx, &view);
    draw_view_cube(&egui_ctx, &mut camera, &controller.viewport_theme()); // also here so content_rect + the
    let warmup = egui_ctx.end_pass(); // Area size settle before the painted pass

    egui_ctx.begin_pass(raw_input);
    controller.ui(&egui_ctx, &view);
    draw_view_cube(&egui_ctx, &mut camera, &controller.viewport_theme());
    let full_output = egui_ctx.end_pass();
    let jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen = ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: full_output.pixels_per_point,
    };

    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cdt-offscreen-color"),
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

    for (id, delta) in warmup
        .textures_delta
        .set
        .iter()
        .chain(full_output.textures_delta.set.iter())
    {
        egui_renderer.update_texture(&device, &queue, *id, delta);
    }
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let egui_cmds = egui_renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &screen);
    let highlights = controller.highlights();
    let mut gizmo_verts = match controller.gizmo() {
        Some(g) => {
            let origin = Vec3::new(g.origin[0] as f32, g.origin[1] as f32, g.origin[2] as f32);
            gizmo_geometry(origin, &camera, None, None)
        }
        None => Vec::new(),
    };
    if let Some(rb) = controller.resize_box() {
        gizmo_verts.append(&mut resize_geometry(&rb, &camera, None));
    }
    let vp_theme = controller.viewport_theme();
    scene.upload_gizmo(&gizmo_verts);
    scene.upload_grid(&grid_geometry(controller.grid(), &vp_theme));
    scene.encode(
        &mut encoder,
        &color_view,
        &depth_view,
        &camera,
        width,
        height,
        &highlights,
        &vp_theme,
    );
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
        label: Some("cdt-readback"),
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
    use crate::controller::MeshData;
    use crate::{EdgeVertex, Vertex};

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
        assert_eq!(scene.pick_face(32, 32, &camera, 64, 64).0, Some(0));
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
}
