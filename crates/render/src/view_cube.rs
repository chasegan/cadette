//! The ViewCube overlay: a small camera-tracking cube in the viewport's
//! top-right corner for orbiting and snapping to orthographic views.

use glam::Vec3;

use crate::camera::OrbitCamera;
use crate::math::point_in_poly;
use crate::theme::ViewportTheme;

/// A ViewCube overlay in the viewport's top-right corner: a small cube that
/// tracks the camera orientation. Drag it to orbit; click a face to snap to that
/// orthographic view (Top/Bottom/Front/Back/Left/Right, Z-up world).
pub(crate) fn draw_view_cube(ctx: &egui::Context, camera: &mut OrbitCamera, theme: &ViewportTheme) {
    const SIZE: f32 = 92.0;
    let face_c = egui::Color32::from_rgb(theme.view_cube_face[0], theme.view_cube_face[1], theme.view_cube_face[2]);
    let hover_c = egui::Color32::from_rgb(
        theme.view_cube_face_hover[0],
        theme.view_cube_face_hover[1],
        theme.view_cube_face_hover[2],
    );
    let edge_c = egui::Color32::from_rgb(theme.view_cube_edge[0], theme.view_cube_edge[1], theme.view_cube_edge[2]);
    let label_c = egui::Color32::from_rgb(
        theme.view_cube_label[0],
        theme.view_cube_label[1],
        theme.view_cube_label[2],
    );

    // Camera view basis (world → the cube's screen).
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();
    let dir = Vec3::new(cp * cy, cp * sy, sp); // target → eye
    let fwd = -dir; // into the screen
    let right = fwd.cross(Vec3::Z).normalize_or_zero();
    let up = right.cross(fwd);

    // Top-right of the central viewport (to the left of the right panel). The
    // app's panels use egui's deprecated `Panel` API, whose reservations show up
    // in `available_rect` (not the newer `content_rect`), so use that.
    #[allow(deprecated)]
    let central = ctx.available_rect();
    let pos = egui::pos2(central.right() - SIZE - 14.0, central.top() + 14.0);

    egui::Area::new(egui::Id::new("view_cube"))
        .order(egui::Order::Foreground)
        .fixed_pos(pos)
        .show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(SIZE, SIZE), egui::Sense::click_and_drag());
            let painter = ui.painter_at(rect);
            let center = rect.center();
            let scale = SIZE * 0.30;
            let project = |p: Vec3| -> egui::Pos2 {
                egui::pos2(center.x + p.dot(right) * scale, center.y - p.dot(up) * scale)
            };

            // Outward normal (= face center on the unit cube), label, 4 corners.
            let v = Vec3::new;
            let faces: [(Vec3, &str, [Vec3; 4]); 6] = [
                (Vec3::X, "Right", [v(1., -1., -1.), v(1., 1., -1.), v(1., 1., 1.), v(1., -1., 1.)]),
                (Vec3::NEG_X, "Left", [v(-1., 1., -1.), v(-1., -1., -1.), v(-1., -1., 1.), v(-1., 1., 1.)]),
                (Vec3::Y, "Back", [v(1., 1., -1.), v(-1., 1., -1.), v(-1., 1., 1.), v(1., 1., 1.)]),
                (Vec3::NEG_Y, "Front", [v(-1., -1., -1.), v(1., -1., -1.), v(1., -1., 1.), v(-1., -1., 1.)]),
                (Vec3::Z, "Top", [v(-1., -1., 1.), v(1., -1., 1.), v(1., 1., 1.), v(-1., 1., 1.)]),
                (Vec3::NEG_Z, "Bottom", [v(-1., 1., -1.), v(1., 1., -1.), v(1., -1., -1.), v(-1., -1., -1.)]),
            ];

            let hover = response.hover_pos();
            let mut clicked_axis: Option<Vec3> = None;

            // Front-facing faces (normal toward the camera), back-to-front so the
            // nearer ones paint on top.
            let mut front: Vec<usize> = (0..6).filter(|&i| faces[i].0.dot(fwd) < -1e-3).collect();
            front.sort_by(|&a, &b| faces[b].0.dot(fwd).total_cmp(&faces[a].0.dot(fwd)));

            for &i in &front {
                let (normal, label, corners) = faces[i];
                let pts: Vec<egui::Pos2> = corners.iter().map(|c| project(*c)).collect();
                let hovered = hover.is_some_and(|h| point_in_poly(h, &pts));
                if hovered && response.clicked() {
                    clicked_axis = Some(normal); // nearest hovered wins (drawn last)
                }
                // Conventional ViewCube look: light faces with dark labels; the
                // theme supplies colors so it reads on either background.
                let fill = if hovered { hover_c } else { face_c };
                painter.add(egui::Shape::convex_polygon(
                    pts,
                    fill,
                    egui::Stroke::new(1.2, edge_c),
                ));
                painter.text(
                    project(normal),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(11.0),
                    label_c,
                );
            }

            if let Some(axis) = clicked_axis {
                camera.look_along(axis);
            } else if response.dragged() {
                let d = response.drag_delta();
                camera.orbit(d.x * 0.01, d.y * 0.01);
            }
        });
}
