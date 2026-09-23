//! Offscreen composited screenshot (3D + egui), for verification.

use egui_wgpu::ScreenDescriptor;
use glam::Vec3;

use crate::controller::Controller;
use crate::gizmo::{gizmo_geometry, resize_geometry};
use crate::grid::grid_geometry;
use crate::offscreen::readback_rgba;
use crate::scene::{
    create_depth, egui_renderer, encode_egui, frame_camera, install_phosphor, request_device,
    Scene,
};
use crate::view::ViewContext;
use crate::view_cube::draw_view_cube;

/// Render one composited frame (3D scene + egui UI + ViewCube) to a PNG at
/// `path` — the app as a user sees it. For the model alone, see
/// [`crate::render_model`].
pub fn screenshot(
    controller: &mut impl Controller,
    width: u32,
    height: u32,
    path: &str,
) -> anyhow::Result<()> {
    const PPP: f32 = 1.5;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(request_device(&instance, None))?;

    let mesh = controller.mesh();
    let mut camera = frame_camera(&mesh.vertices);
    // A controller-supplied pose (a loaded project's saved view, or a demo's
    // hand-framed composition) overrides the automatic framing.
    if let Some(pose) = controller.requested_camera() {
        camera.set_pose(pose);
    }
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
        &camera,
        egui::vec2(width as f32 / PPP, height as f32 / PPP),
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
    // The hover ghost, when a demo pre-arranged one — polled AFTER the ui()
    // passes above: mesh() invalidates ghost caches, and it's the ui pass
    // that rebuilds them.
    if let Some((_, m)) = controller.ghost() {
        scene.upload_ghost(Some(m));
    }
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

    readback_rgba(&device, &queue, &color, width, height)?.save(path)?;
    Ok(())
}
