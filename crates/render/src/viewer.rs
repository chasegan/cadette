//! The wgpu rendering core, exposed two ways:
//! - [`run`] — a live winit window with an orbit camera (the Milestone B goal).
//! - [`render_to_png`] — a one-frame offscreen render to an image file, for
//!   automated/visual verification without a display.
//!
//! Both share [`Scene`], which owns the pipeline, mesh buffers, and uniforms.
//! This is deliberately one self-contained viewer for Phase 0; the
//! scene/material/picking split from the project outline grows out of it later.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::camera::OrbitCamera;
use crate::Vertex;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.09,
    g: 0.10,
    b: 0.12,
    a: 1.0,
};

/// GPU uniform block, mirrored by `Globals` in `shader.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
}

impl Globals {
    fn for_view(camera: &OrbitCamera, aspect: f32) -> Self {
        let eye = camera.eye();
        Globals {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [0.4, 0.5, 1.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Scene: device + pipeline + mesh, drawable to any color/depth target.
// ---------------------------------------------------------------------------

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    globals_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Scene {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Self {
        use wgpu::util::DeviceExt;

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

        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rmf-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
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
                // Two-sided shading in the fragment stage means we can show the
                // inside of bored holes without per-face winding worries.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rmf-indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            device,
            queue,
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            globals_buffer,
            bind_group,
        }
    }

    /// Encode and submit a single frame into the given color/depth views.
    fn draw(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
    ) {
        let aspect = width as f32 / height.max(1) as f32;
        let globals = Globals::for_view(camera, aspect);
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rmf-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rmf-pass"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
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

/// Bounding-sphere fit so the model lands nicely framed.
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

// ---------------------------------------------------------------------------
// Live window
// ---------------------------------------------------------------------------

/// Open the viewport and display `vertices`/`indices` until the window closes.
pub fn run(vertices: Vec<Vertex>, indices: Vec<u32>) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(vertices, indices);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct MouseState {
    last: Option<PhysicalPosition<f64>>,
    orbiting: bool,
    panning: bool,
}

struct App {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    camera: OrbitCamera,
    mouse: MouseState,
    window: Option<Window>,
}

struct Window {
    handle: Arc<winit::window::Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    scene: Scene,
}

impl App {
    fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let camera = frame_camera(&vertices);
        Self {
            vertices,
            indices,
            camera,
            mouse: MouseState::default(),
            window: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = winit::window::Window::default_attributes()
            .with_title("Riemanifold — viewport")
            .with_inner_size(PhysicalSize::new(1100, 760));
        let handle = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let size = handle.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(handle.clone()).expect("create surface");
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

        let scene = Scene::new(device.clone(), queue, format, &self.vertices, &self.indices);
        let depth_view = create_depth(&device, config.width, config.height);

        self.window = Some(Window {
            handle,
            surface,
            config,
            depth_view,
            scene,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                window.config.width = size.width;
                window.config.height = size.height;
                window
                    .surface
                    .configure(&window.scene.device, &window.config);
                window.depth_view =
                    create_depth(&window.scene.device, size.width, size.height);
                window.handle.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.mouse.orbiting = pressed,
                    MouseButton::Right | MouseButton::Middle => self.mouse.panning = pressed,
                    _ => {}
                }
                if !pressed {
                    self.mouse.last = None;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(last) = self.mouse.last {
                    let dx = (position.x - last.x) as f32;
                    let dy = (position.y - last.y) as f32;
                    if self.mouse.orbiting {
                        self.camera.orbit(dx * 0.01, dy * 0.01);
                        window.handle.request_redraw();
                    } else if self.mouse.panning {
                        self.camera.pan(dx, dy);
                        window.handle.request_redraw();
                    }
                }
                self.mouse.last = if self.mouse.orbiting || self.mouse.panning {
                    Some(position)
                } else {
                    None
                };
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.02,
                };
                self.camera.dolly(amount);
                window.handle.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let frame = match window.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                    _ => {
                        window
                            .surface
                            .configure(&window.scene.device, &window.config);
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                window.scene.draw(
                    &view,
                    &window.depth_view,
                    &self.camera,
                    window.config.width,
                    window.config.height,
                );
                frame.present();
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Offscreen render to PNG (verification / headless)
// ---------------------------------------------------------------------------

/// Render a single framed view of the mesh to a PNG at `path`. No window or
/// display required — used to verify the render pipeline produces real pixels.
pub fn render_to_png(
    vertices: &[Vertex],
    indices: &[u32],
    width: u32,
    height: u32,
    path: &str,
) -> anyhow::Result<()> {
    let camera = frame_camera(vertices);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None));
    let scene = Scene::new(device.clone(), queue.clone(), format, vertices, indices);

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

    scene.draw(&color_view, &depth_view, &camera, width, height);

    // Copy the rendered texture into a CPU-readable buffer. Rows must be padded
    // to COPY_BYTES_PER_ROW_ALIGNMENT (256).
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
            texture: &color,
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
