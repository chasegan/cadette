//! Free math helpers shared by picking, the gizmo, and the overlays:
//! projection/unprojection, ray casts, and small screen-space predicates.

use glam::Vec3;
use winit::dpi::PhysicalPosition;

use crate::camera::OrbitCamera;

/// Project `p` to pixel coordinates, or `None` if behind the camera.
pub(crate) fn project_point(p: Vec3, camera: &OrbitCamera, w: u32, h: u32) -> Option<(f32, f32)> {
    let clip = camera.view_proj(w as f32 / h.max(1) as f32) * p.extend(1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(((ndc.x * 0.5 + 0.5) * w as f32, (0.5 - ndc.y * 0.5) * h as f32))
}

/// World-space ray (origin, unit dir) through `cursor` for the given camera.
pub(crate) fn cursor_ray(
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

/// Unproject the CENTRE of pixel `(px, py)` at NDC depth `z` back to world
/// space. The `+0.5` matches where the rasterizer sampled that depth, so the
/// result lands on the rendered surface.
pub(crate) fn unproject_pixel(
    px: u32,
    py: u32,
    z: f32,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
) -> Vec3 {
    let nx = (px as f32 + 0.5) / width as f32 * 2.0 - 1.0;
    let ny = 1.0 - (py as f32 + 0.5) / height as f32 * 2.0;
    let aspect = width as f32 / height.max(1) as f32;
    let inv = camera.view_proj(aspect).inverse();
    let p = inv * glam::Vec4::new(nx, ny, z, 1.0);
    p.truncate() / p.w
}

/// Intersection of ray `(ro, rd)` with the plane through `p` with normal `n`.
pub(crate) fn ray_plane_intersect(ro: Vec3, rd: Vec3, p: Vec3, n: Vec3) -> Option<Vec3> {
    let denom = rd.dot(n);
    if denom.abs() < 1e-6 {
        return None; // ray parallel to the plane
    }
    let t = (p - ro).dot(n) / denom;
    (t >= 0.0).then(|| ro + rd * t)
}

/// Distance from a point to a line segment, all in screen pixels.
pub(crate) fn point_segment_distance(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
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

/// Closest point on segment `[a, b]` to the ray `(origin, dir)`, with the
/// distance between that point and the ray. Used to anchor edge picks.
pub(crate) fn closest_point_on_segment_to_ray(
    a: Vec3,
    b: Vec3,
    origin: Vec3,
    dir: Vec3,
) -> (Vec3, f32) {
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

/// Signed distance along the manip axis `(point, dir)` to the closest point of
/// the cursor ray — how far the user has dragged the face.
pub(crate) fn manip_distance(
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

/// Wrap an angle delta to `[-π, π]` (so accumulating around the ring is smooth
/// across the ±π seam).
pub(crate) fn wrap_pi(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut a = a % tau;
    if a > std::f32::consts::PI {
        a -= tau;
    } else if a < -std::f32::consts::PI {
        a += tau;
    }
    a
}

/// Is screen point `p` inside the triangle `(a, b, c)`?
pub(crate) fn point_in_tri(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
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

/// Wrap an RGB color to RGBA with the given alpha.
pub(crate) fn rgba(c: [f32; 3], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

/// Whether `p` is inside the convex polygon `poly` (same side of every edge).
pub(crate) fn point_in_poly(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    let mut sign = 0.0f32;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross.abs() > 1e-4 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if sign != cross.signum() {
                return false;
            }
        }
    }
    true
}
