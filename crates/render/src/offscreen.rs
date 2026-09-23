//! The model-only offscreen renderer: the viewport's 3D pass (bodies, edges,
//! highlights, grid) rendered to an image with no window, no egui and no
//! ViewCube — what a CLI, a test, or an agent wants when it asks "what does
//! this look like?". The composited screenshot with the app's chrome is
//! [`crate::screenshot`], built on the same pieces.

use crate::camera::{CameraPose, OrbitCamera};
use crate::controller::{GridSpec, Highlights, MeshData};
use crate::grid::grid_geometry;
use crate::scene::{create_depth, frame_camera, request_device, Scene};
use crate::theme::ViewportTheme;

/// What to draw and how, for one [`HeadlessRenderer::render`] call.
#[derive(Clone, Debug)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub theme: ViewportTheme,
    /// `None` frames the whole mesh automatically from a three-quarter view.
    pub camera: Option<CameraPose>,
    /// Ground/origin planes; the default draws none.
    pub grid: GridSpec,
    /// Faces/edges to emphasise (e.g. the one an agent is asking about).
    pub highlights: Highlights,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 800,
            theme: ViewportTheme::LIGHT,
            camera: None,
            grid: GridSpec::default(),
            highlights: Highlights::default(),
        }
    }
}

/// A GPU device plus the scene buffers, kept between renders so repeated shots
/// (an agent iterating on a part) don't rebuild pipelines every time.
pub struct HeadlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    scene: Scene,
}

impl HeadlessRenderer {
    /// Open a device with no window. Errors (rather than panicking) when the
    /// machine has no usable GPU or software adapter.
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None))?;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let scene = Scene::new(device.clone(), queue.clone(), format, &MeshData::default());
        Ok(Self { device, queue, format, scene })
    }

    /// The camera a render with `opts` will use for `mesh`: the automatic
    /// framing, or the requested pose on top of it (so `znear`/`zfar` still
    /// suit the model's size).
    pub fn camera_for(mesh: &MeshData, opts: &RenderOptions) -> OrbitCamera {
        let mut camera = frame_camera(&mesh.vertices);
        if let Some(pose) = opts.camera {
            camera.set_pose(pose);
        }
        camera
    }

    /// Render `mesh` to an RGBA image.
    pub fn render(&mut self, mesh: &MeshData, opts: &RenderOptions) -> anyhow::Result<image::RgbaImage> {
        let (width, height) = (opts.width.max(1), opts.height.max(1));
        let camera = Self::camera_for(mesh, opts);
        self.scene.upload_mesh(mesh);
        self.scene.upload_ghost(None);
        self.scene.upload_gizmo(&[]);
        self.scene.upload_grid(&grid_geometry(opts.grid, &opts.theme));

        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cdt-offscreen-color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = create_depth(&self.device, width, height);
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.scene.encode(
            &mut encoder,
            &color_view,
            &depth_view,
            &camera,
            width,
            height,
            &opts.highlights,
            &opts.theme,
        );
        self.queue.submit(Some(encoder.finish()));
        readback_rgba(&self.device, &self.queue, &color, width, height)
    }
}

/// One-shot convenience: open a device, render, and drop it. Prefer a
/// [`HeadlessRenderer`] when rendering more than once.
pub fn render_model(mesh: &MeshData, opts: &RenderOptions) -> anyhow::Result<image::RgbaImage> {
    HeadlessRenderer::new()?.render(mesh, opts)
}

/// Copy a rendered texture back to the CPU as an RGBA image (handles wgpu's
/// row-padding requirement).
pub(crate) fn readback_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> anyhow::Result<image::RgbaImage> {
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
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(|e| anyhow::anyhow!("waiting for the GPU: {e:?}"))?;
    rx.recv()
        .map_err(|_| anyhow::anyhow!("readback mapping never completed"))?
        .map_err(|e| anyhow::anyhow!("mapping the readback buffer: {e}"))?;

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    buffer.unmap();

    image::ImageBuffer::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("readback size mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Vertex, NO_COLOR};

    fn quad() -> MeshData {
        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            face_id: 0,
            color: NO_COLOR,
        };
        MeshData {
            vertices: vec![v(-10.0, -10.0), v(10.0, -10.0), v(10.0, 10.0), v(-10.0, 10.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        }
    }

    /// The model-only render puts the model in the middle of the image and
    /// the background at the corners, with no chrome anywhere.
    #[test]
    fn renders_a_model_to_an_image_without_a_window() {
        let mut r = HeadlessRenderer::new().expect("a GPU or software adapter");
        let opts = RenderOptions { width: 64, height: 48, ..Default::default() };
        // An empty mesh is just the background: a vertical gradient, so every
        // pixel in a row matches.
        let empty = r.render(&MeshData::default(), &opts).unwrap();
        assert_eq!((empty.width(), empty.height()), (64, 48));
        assert_eq!(empty.get_pixel(32, 24).0, empty.get_pixel(1, 24).0);
        assert_eq!(empty.get_pixel(32, 24).0[3], 255);
        // The quad lands in the middle of the image, over the background.
        let img = r.render(&quad(), &opts).unwrap();
        let centre = img.get_pixel(32, 24);
        assert_ne!(centre.0[..3], empty.get_pixel(32, 24).0[..3], "the quad is drawn");
        assert_eq!(img.get_pixel(1, 1).0, empty.get_pixel(1, 1).0, "the corner is still background");
        // The renderer is reusable: same input, same pixels, no rebuild.
        let again = r.render(&quad(), &opts).unwrap();
        assert_eq!(again.get_pixel(32, 24), centre);
    }

    #[test]
    fn a_requested_pose_overrides_the_automatic_framing() {
        let mesh = quad();
        let auto = HeadlessRenderer::camera_for(&mesh, &RenderOptions::default());
        let pose = CameraPose { target: [1.0, 2.0, 3.0], yaw: 0.3, pitch: 0.2, distance: 99.0 };
        let posed = HeadlessRenderer::camera_for(&mesh, &RenderOptions { camera: Some(pose), ..Default::default() });
        assert_eq!(posed.pose(), pose);
        assert_ne!(auto.pose(), pose);
        assert_eq!(posed.fov_y, auto.fov_y, "projection parameters come from the framing");
    }
}
