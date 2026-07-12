//! Ray-cast picking against the scene: the GPU face-id/depth readback, the
//! screen-space edge pick, and the combined edge-over-face resolution.

use glam::Vec3;
use winit::dpi::PhysicalPosition;

use crate::camera::OrbitCamera;
use crate::controller::{Highlights, Pick};
use crate::math::{
    closest_point_on_segment_to_ray, cursor_ray, point_segment_distance, unproject_pixel,
};
use crate::scene::{create_depth, Globals, Scene, PICK_DEPTH_FORMAT, PICK_FORMAT};

/// How close (pixels) the cursor must be to an edge to pick it over a face.
const EDGE_PICK_PX: f32 = 6.0;
/// How far (NDC depth) behind the front surface an edge must be to count as
/// occluded — large enough to keep a face's own boundary edges, small enough to
/// drop edges on the far side of the body.
const PICK_DEPTH_TOL: f32 = 0.005;

/// The pick pass's persistent GPU resources: the two full-window render
/// targets (face id + NDC depth), the pass's z-buffer, and the single-pixel
/// readback buffers. Hover picks run on every mouse move, so these are created
/// lazily on the first pick and recreated only when the surface size changes —
/// not allocated per query.
pub(crate) struct PickTargets {
    width: u32,
    height: u32,
    id_texture: wgpu::Texture,
    depth_texture: wgpu::Texture,
    id_view: wgpu::TextureView,
    depth_color_view: wgpu::TextureView,
    /// The z-buffer for depth testing during the pass; never read back.
    zbuffer_view: wgpu::TextureView,
    /// Single-pixel readbacks, each `COPY_BYTES_PER_ROW_ALIGNMENT` (256) bytes
    /// so the one-row copy satisfies wgpu's bytes-per-row alignment.
    id_readback: wgpu::Buffer,
    depth_readback: wgpu::Buffer,
}

impl PickTargets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let make_target = |format, label| {
            device.create_texture(&wgpu::TextureDescriptor {
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
        let zbuffer_view = create_depth(device, width, height);

        let make_readback = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        Self {
            width,
            height,
            id_texture,
            depth_texture,
            id_view,
            depth_color_view,
            zbuffer_view,
            id_readback: make_readback("cdt-pick-id-readback"),
            depth_readback: make_readback("cdt-pick-depth-readback"),
        }
    }
}

impl Scene {
    /// Render the face-id + depth pass and read back, under pixel `(px, py)`,
    /// the face id (`None` = background) and the front-surface depth (NDC z;
    /// `1.0` where there's no geometry). Synchronous.
    fn pick_face(
        &mut self,
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

        // (Re)create the persistent pick resources only when the size changes.
        if self
            .pick_targets
            .as_ref()
            .is_none_or(|t| t.width != width || t.height != height)
        {
            self.pick_targets = Some(PickTargets::new(&self.device, width, height));
        }
        let targets = self.pick_targets.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cdt-pick") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cdt-pick-pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.id_view,
                        resolve_target: None,
                        // 0 = no geometry (the pick shader writes face_id + 1).
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.depth_color_view,
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
                    view: &targets.zbuffer_view,
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
            pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        // Copy both single pixels under the cursor to the persistent readback
        // buffers.
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
        copy_pixel(&mut encoder, &targets.id_texture, &targets.id_readback);
        copy_pixel(&mut encoder, &targets.depth_texture, &targets.depth_readback);
        self.queue.submit(Some(encoder.finish()));

        targets
            .id_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map pick id"));
        targets
            .depth_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map pick depth"));
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .ok();

        let id_data = targets.id_readback.slice(..).get_mapped_range();
        let raw = u32::from_le_bytes([id_data[0], id_data[1], id_data[2], id_data[3]]);
        drop(id_data);
        targets.id_readback.unmap();
        let depth_data = targets.depth_readback.slice(..).get_mapped_range();
        let z = f32::from_le_bytes([depth_data[0], depth_data[1], depth_data[2], depth_data[3]]);
        drop(depth_data);
        targets.depth_readback.unmap();

        ((raw != 0).then(|| raw - 1), z)
    }

    /// Whether the picked `edge` is occluded by the front surface (whose depth at
    /// the cursor is `face_depth`). Lets a nearby edge be ignored when it's
    /// really behind the face the cursor is over — the cause of hover flicker.
    // Parameter-object refactor (bundling camera/cursor/size into a view struct)
    // is planned alongside the Controller-trait slimming.
    #[expect(clippy::too_many_arguments)]
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

    /// Pick the entity under pixel `(px, py)` — a nearby *unoccluded* edge
    /// takes priority over the face; otherwise the face (or nothing) — plus
    /// the world-space point under the cursor (unprojected from the pick
    /// pass's depth target) when a surface is hit. The occlusion check stops
    /// back-of-the-body edges that merely project near the cursor from
    /// stealing the hover off the face it's over. The point serves as a
    /// push/pull anchor guaranteed to lie on the face (avoids holes/odd
    /// shapes) and as the tape measure's pin/hover point. One GPU pass serves
    /// both the id and the point.
    pub(crate) fn pick_with_point(
        &mut self,
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
            Some(id) => {
                // A face hit implies geometry at the cursor, but keep the
                // background guard (depth 1.0) for safety.
                let point = (face_depth < 1.0 - 1e-6)
                    .then(|| unproject_pixel(px, py, face_depth, camera, w, h));
                (Some(Pick::Face(id)), point)
            }
            None => (None, None),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::controller::MeshData;
    use crate::scene::request_device;
    use crate::{EdgeVertex, Vertex, NO_COLOR};

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
            color: NO_COLOR,
        };
        let mesh = MeshData {
            vertices: vec![v(-10.0, -10.0), v(10.0, -10.0), v(10.0, 10.0), v(-10.0, 10.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        };
        let mut scene = Scene::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, &mesh);
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
            color: NO_COLOR,
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
        let mut scene = Scene::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, &mesh);
        let camera = OrbitCamera::framing(Vec3::ZERO, 15.0);

        let (pick, _) = scene.pick_with_point(32, 32, &camera, 64, 64);
        assert_eq!(pick, Some(Pick::Edge(0)));
    }

    /// The depth-reconstructed world point under the cursor lands on the surface.
    #[test]
    fn pick_with_point_recovers_the_surface_point() {
        let instance = wgpu::Instance::default();
        let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None));

        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            face_id: 0,
            color: NO_COLOR,
        };
        let mesh = MeshData {
            vertices: vec![v(-10.0, -10.0), v(10.0, -10.0), v(10.0, 10.0), v(-10.0, 10.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        };
        let mut scene = Scene::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, &mesh);
        let camera = OrbitCamera::framing(Vec3::ZERO, 15.0);

        // The center ray hits the quad at the origin (the camera target).
        let (pick, point) = scene.pick_with_point(32, 32, &camera, 64, 64);
        assert_eq!(pick, Some(Pick::Face(0)));
        let p = point.unwrap();
        assert!(p.length() < 1.0, "expected near origin, got {p:?}");
        assert!(p.z.abs() < 0.05, "should be on the z=0 plane, got z={}", p.z);
    }
}
