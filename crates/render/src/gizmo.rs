//! The transform gizmo (gumball) and resize grips: geometry built as colored
//! triangles, screen-space hit-testing, snap rules, and the in-progress drag
//! state the window loop drives.

use glam::Vec3;

use crate::math::Px;

use crate::camera::OrbitCamera;
use crate::controller::{Axis3, GizmoHandle, Plane3, ResizeBox};
use crate::math::{point_in_tri, point_segment_distance, project_point, rgba};
use crate::GizmoVertex;

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
pub(crate) fn gizmo_ring_ratio(
    center: Vec3,
    axis: Axis3,
    camera: &OrbitCamera,
    cursor: impl Into<Px>,
    w: u32,
    h: u32,
) -> f32 {
    let cursor: Px = cursor.into();
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
pub(crate) fn ring_points(origin: Vec3, axis: Axis3, len: f32) -> [Vec3; RING_SEGMENTS] {
    let (u, v) = axis.perps();
    let r = len * GIZMO_RING_RADIUS;
    std::array::from_fn(|i| {
        let t = i as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        origin + (u * t.cos() + v * t.sin()) * r
    })
}

/// The signed angle (radians) of `p` (relative to the ring center) within the
/// rotation plane of `axis`, measured from the `u` basis vector.
pub(crate) fn ring_angle(p_minus_center: Vec3, axis: Axis3) -> f32 {
    let (u, v) = axis.perps();
    p_minus_center.dot(v).atan2(p_minus_center.dot(u))
}

/// Tinkercad-style radial snap increment (radians): coarse near the ring, finer
/// as the cursor moves out, `0` (free) far out or when `free` is held. `ratio`
/// is the cursor distance from center over the ring radius (~1.0 = on the ring).
pub(crate) fn snap_increment(ratio: f32, free: bool) -> f32 {
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
pub(crate) fn snap_rotation(angle: f32, ratio: f32, free: bool) -> f32 {
    let inc = snap_increment(ratio, free);
    if inc == 0.0 {
        angle
    } else {
        (angle / inc).round() * inc
    }
}

/// Snap a world-space translation `offset` to a `grid` (each axis independently,
/// landing on grid intersections). `free` (a modifier) or `grid <= 0` disables it.
pub(crate) fn snap_translation(offset: Vec3, grid: f32, free: bool) -> Vec3 {
    if free || grid <= 0.0 {
        return offset;
    }
    Vec3::new(
        (offset.x / grid).round() * grid,
        (offset.y / grid).round() * grid,
        (offset.z / grid).round() * grid,
    )
}

/// Snap a scalar distance (mm) to the `grid`. `free` or `grid <= 0` disables it.
pub(crate) fn snap_distance(d: f32, grid: f32, free: bool) -> f32 {
    if free || grid <= 0.0 {
        d
    } else {
        (d / grid).round() * grid
    }
}

/// Visual state for the rotation protractor during a ring drag.
#[derive(Clone, Copy)]
pub(crate) struct RotationViz {
    pub(crate) axis: Axis3,
    /// Angle (in the ring plane) where the drag began.
    pub(crate) start: f32,
    /// Snapped swept angle (signed) from `start`.
    pub(crate) swept: f32,
    /// Current snap increment (radians); `0` when free (no tick marks).
    pub(crate) inc: f32,
}

/// World length of the gizmo arrows — proportional to the orbit distance so the
/// gizmo stays a roughly constant size on screen.
pub(crate) fn gizmo_axis_length(camera: &OrbitCamera) -> f32 {
    camera.distance * 0.22
}

/// The four world-space corners of a planar handle square.
pub(crate) fn plane_corners(origin: Vec3, plane: Plane3, len: f32) -> [Vec3; 4] {
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

/// The gizmo handle under `cursor`: a planar handle (filled square) takes
/// priority over an axis arrow it might overlap.
pub(crate) fn gizmo_hit_test(
    origin: Vec3,
    camera: &OrbitCamera,
    cursor: impl Into<Px>,
    w: u32,
    h: u32,
) -> Option<GizmoHandle> {
    let cursor: Px = cursor.into();
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
pub(crate) fn gizmo_geometry(
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

/// Half-size of a resize grip cube, as a fraction of the gizmo axis length (so
/// the grips stay a roughly constant size on screen).
const RESIZE_GRIP_SIZE: f32 = 0.035;
/// Smallest dimension a resize drag may produce (mm), so a box can't collapse.
pub(crate) const MIN_RESIZE_DIM: f32 = 0.1;

/// The six bounding-box faces a body can be resized from: each axis on its
/// positive (`true`) and negative (`false`) side.
const RESIZE_FACES: [(Axis3, bool); 6] = [
    (Axis3::X, true),
    (Axis3::X, false),
    (Axis3::Y, true),
    (Axis3::Y, false),
    (Axis3::Z, true),
    (Axis3::Z, false),
];

/// World-space center of the resize grip on the given face of `rb` (the
/// `positive` or negative side along `axis`).
pub(crate) fn resize_grip_pos(rb: &ResizeBox, axis: Axis3, positive: bool) -> Vec3 {
    let min = Vec3::new(rb.min[0] as f32, rb.min[1] as f32, rb.min[2] as f32);
    let max = Vec3::new(rb.max[0] as f32, rb.max[1] as f32, rb.max[2] as f32);
    let c = (min + max) * 0.5;
    let face = if positive { max } else { min };
    match axis {
        Axis3::X => Vec3::new(face.x, c.y, c.z),
        Axis3::Y => Vec3::new(c.x, face.y, c.z),
        Axis3::Z => Vec3::new(c.x, c.y, face.z),
    }
}

/// The resize face (axis + side) whose grip is under `cursor`, if any (nearest
/// within the pixel threshold).
pub(crate) fn resize_hit_test(
    rb: &ResizeBox,
    camera: &OrbitCamera,
    cursor: impl Into<Px>,
    w: u32,
    h: u32,
) -> Option<(Axis3, bool)> {
    let cursor: Px = cursor.into();
    let c = (cursor.x as f32, cursor.y as f32);
    let mut best = (GIZMO_PICK_PX + 2.0, None);
    for (axis, positive) in RESIZE_FACES {
        if let Some(p) = project_point(resize_grip_pos(rb, axis, positive), camera, w, h) {
            let d = (c.0 - p.0).hypot(c.1 - p.1);
            if d < best.0 {
                best = (d, Some((axis, positive)));
            }
        }
    }
    best.1
}

/// Push an axis-aligned cube (12 triangles) centered at `c` with half-size `hs`.
pub(crate) fn push_cube(verts: &mut Vec<GizmoVertex>, c: Vec3, hs: f32, color: [f32; 4]) {
    let v = |dx: f32, dy: f32, dz: f32| (c + Vec3::new(dx, dy, dz) * hs).to_array();
    let p = [
        v(-1.0, -1.0, -1.0), v(1.0, -1.0, -1.0), v(1.0, 1.0, -1.0), v(-1.0, 1.0, -1.0),
        v(-1.0, -1.0, 1.0), v(1.0, -1.0, 1.0), v(1.0, 1.0, 1.0), v(-1.0, 1.0, 1.0),
    ];
    let mut quad = |a: usize, b: usize, cc: usize, d: usize| {
        for &i in &[a, b, cc, a, cc, d] {
            verts.push(GizmoVertex { position: p[i], color });
        }
    };
    quad(0, 1, 2, 3); // -Z
    quad(4, 5, 6, 7); // +Z
    quad(0, 1, 5, 4); // -Y
    quad(2, 3, 7, 6); // +Y
    quad(1, 2, 6, 5); // +X
    quad(0, 3, 7, 4); // -X
}

/// Resize grips for `rb`: a small cube on each of the six bounding-box faces,
/// the one under the cursor (or being dragged) highlighted gold.
pub(crate) fn resize_geometry(
    rb: &ResizeBox,
    camera: &OrbitCamera,
    active: Option<(Axis3, bool)>,
) -> Vec<GizmoVertex> {
    let hs = gizmo_axis_length(camera) * RESIZE_GRIP_SIZE;
    let mut verts = Vec::new();
    for (axis, positive) in RESIZE_FACES {
        let color = if active == Some((axis, positive)) {
            rgba(GIZMO_HILITE, 1.0)
        } else {
            rgba([0.85, 0.85, 0.88], 0.9)
        };
        push_cube(&mut verts, resize_grip_pos(rb, axis, positive), hs, color);
    }
    verts
}

/// Capacity of the persistent gizmo vertex buffer. The full gumball (arrows +
/// planes + three rings) and the rotation protractor are both well under this.
pub(crate) const GIZMO_MAX_VERTS: u64 = 8192;

/// Allocate the persistent gizmo vertex buffer once; [`Scene::upload_gizmo`]
/// rewrites it each frame via `write_buffer` (no per-frame allocation).
pub(crate) fn make_gizmo_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cdt-gizmo"),
        size: GIZMO_MAX_VERTS * std::mem::size_of::<GizmoVertex>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// An in-progress gizmo drag. The reference geometry (axis line / plane) is
/// captured at grab time and kept fixed, so the body translating under the
/// gizmo doesn't shift the drag reference.
#[derive(Clone, Copy)]
pub(crate) enum GizmoDrag {
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
    /// Resize a body from one bbox face: grip line `(origin, dir)` (dir points
    /// outward from the grabbed face), the drag distance at grab, and the body's
    /// extent along that axis at grab.
    Resize {
        axis: Axis3,
        origin: Vec3,
        dir: Vec3,
        start: f32,
        base_extent: f64,
    },
}

impl GizmoDrag {
    /// The handle this drag corresponds to (for highlighting).
    pub(crate) fn handle(&self) -> GizmoHandle {
        match self {
            GizmoDrag::Axis { axis, .. } => GizmoHandle::TranslateAxis(*axis),
            GizmoDrag::Plane { plane, .. } => GizmoHandle::TranslatePlane(*plane),
            GizmoDrag::Rotate { axis, .. } => GizmoHandle::RotateAxis(*axis),
            GizmoDrag::Resize { axis, .. } => GizmoHandle::Resize(*axis),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let cursor = Px::new(on.0 as f64, on.1 as f64);
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
            let cursor = Px::new(on.0 as f64, on.1 as f64);
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
            let cursor = Px::new(on.0 as f64, on.1 as f64);
            assert_eq!(
                gizmo_hit_test(origin, &camera, cursor, w, h),
                Some(GizmoHandle::RotateAxis(axis)),
                "ring {axis:?}"
            );
        }

        // Top-left corner is far from the centered gizmo → no grab.
        let far = Px::new(4.0, 4.0);
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

    /// Translations snap to the 1mm grid per axis; the modifier frees them.
    #[test]
    fn translation_snaps_to_the_grid() {
        let snapped = snap_translation(Vec3::new(3.4, -2.6, 0.1), 1.0, false);
        assert_eq!(snapped, Vec3::new(3.0, -3.0, 0.0));
        // A 5mm grid snaps to coarser steps.
        assert_eq!(snap_translation(Vec3::new(3.4, -2.6, 0.1), 5.0, false), Vec3::new(5.0, -5.0, 0.0));
        // Free (Alt) or a non-positive grid leaves it untouched.
        let raw = Vec3::new(3.4, -2.6, 0.1);
        assert_eq!(snap_translation(raw, 1.0, true), raw);
        assert_eq!(snap_translation(raw, 0.0, false), raw);
        // Scalar push/pull distance snaps too.
        assert!((snap_distance(7.7, 1.0, false) - 8.0).abs() < 1e-4);
        assert!((snap_distance(7.7, 1.0, true) - 7.7).abs() < 1e-4);
    }
}
