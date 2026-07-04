//! Sweep a profile along a path (corrected-Frenet: profile stays normal to the
//! path). A straight sweep must reproduce a cylinder; a bent sweep must still be
//! a valid solid.

use cdt_kernel::Solid;
use std::f64::consts::PI;

#[test]
fn sweeping_a_circle_up_a_straight_path_is_a_cylinder() {
    // Circle of radius 2 on the XY plane at the origin (normal +Z).
    let profile = Solid::circle_face([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0).unwrap();
    // A straight path from the origin going +Z by 10. The path lives on a frame
    // whose local +u is world +Z, so the polyline (0,0)->(10,0) is a vertical
    // line — and its start tangent (+Z) matches the profile normal.
    let path = Solid::path_wire(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0], // u -> +Z
        [1.0, 0.0, 0.0], // v -> +X
        &[0.0, 0.0, 10.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0], // one straight segment
    )
    .unwrap();

    let tube = profile.sweep(&path, [0.0, 1.0, 0.0]).unwrap();
    // Volume of a cylinder r=2, h=10 = π r² h.
    let expected = PI * 2.0 * 2.0 * 10.0;
    let v = tube.volume().unwrap();
    assert!(
        (v - expected).abs() < 1e-3,
        "swept tube volume {v} vs cylinder {expected}"
    );
    // A cylinder is three faces: the round side plus two end caps.
    assert_eq!(tube.face_count().unwrap(), 3, "round side + two caps");
}

#[test]
fn sweeping_around_a_bend_stays_a_valid_solid() {
    let profile = Solid::circle_face([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0).unwrap();
    // An L-path: up +Z by 10, then a bezier elbow turning toward +X.
    let path = Solid::path_wire(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0], // u -> +Z
        [1.0, 0.0, 0.0], // v -> +X
        &[0.0, 0.0, 10.0, 0.0, 15.0, 5.0],
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, // straight up
            1.0, 13.0, 0.0, 15.0, 2.0, // bezier elbow
        ],
    )
    .unwrap();

    let tube = profile.sweep(&path, [0.0, 1.0, 0.0]).unwrap();
    assert!(tube.volume().unwrap() > 0.0, "bent sweep encloses a volume");
    // Tessellation must succeed and produce geometry (no degenerate result).
    let mesh = tube.tessellate(0.2).unwrap();
    assert!(!mesh.positions.is_empty(), "swept solid meshes");
}

#[test]
fn an_asymmetric_profile_keeps_its_orientation_through_a_bend() {
    // A small circle OFFSET +2 in the sketch's X, swept along a path that bends
    // entirely within the XZ plane. With the roll locked to the path-plane normal
    // the offset stays IN that plane, so the solid's Y-extent never exceeds the
    // tube radius. A twisting (Frenet) frame would swing the +2 offset out of
    // plane and blow the Y-extent up toward ±2.5.
    let profile = Solid::circle_face([2.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5).unwrap();
    let path = Solid::path_wire(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], // u -> +X
        [0.0, 0.0, 1.0], // v -> +Z
        &[0.0, 0.0, 0.0, 10.0, 10.0, 10.0], // up Z 10, then over X 10
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let tube = profile.sweep(&path, [0.0, -1.0, 0.0]).unwrap();
    let mesh = tube.tessellate(0.2).unwrap();
    let max_y = mesh.positions.chunks(3).map(|p| p[1].abs()).fold(0.0_f32, f32::max);
    assert!(max_y < 1.0, "offset stayed in the path plane (no twist), max|y| = {max_y}");
}

#[test]
fn the_section_reorients_along_a_polyline_corner() {
    // The key fix: on a POLYLINE the section must stay normal to the path and
    // reorient at corners (a plain Frenet frame leaves it facing the first
    // segment). Sweep up +Z then over +X; the end cap must face +X.
    let profile = Solid::circle_face([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0).unwrap();
    let path = Solid::path_wire(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], // u -> +X
        [0.0, 0.0, 1.0], // v -> +Z
        &[0.0, 0.0, 0.0, 10.0, 10.0, 10.0], // origin -> up Z 10 -> over X 10
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let tube = profile.sweep(&path, [0.0, -1.0, 0.0]).unwrap();
    // A round profile swept the full length (≈20) → volume ≈ π·1²·20.
    assert!((tube.volume().unwrap() - PI * 20.0).abs() < 3.0, "full-length tube, got {}", tube.volume().unwrap());
    // Some planar end cap must face ~+X (the final segment direction).
    let faces_x = (0..tube.face_count().unwrap() as u32).any(|i| {
        tube.face_plane(i).unwrap().is_some_and(|(_, x, y)| {
            let nx = x[1] * y[2] - x[2] * y[1];
            nx.abs() > 0.9
        })
    });
    assert!(faces_x, "the section reoriented: an end cap faces +X");
}

fn wide_profile() -> Solid {
    // A ~40-wide line+bezier profile (a cable-elbow-ish section).
    Solid::profile_face(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        &[-22.0, -20.0, 10.0, -20.0, 10.0, 20.0, -22.0, 20.0],
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, // bottom line
            1.0, 14.0, -20.0, 14.0, 20.0, // bulged right side
            0.0, 0.0, 0.0, 0.0, 0.0, // top line
            1.0, -26.0, 20.0, -26.0, -20.0, // bulged left side
        ],
    )
    .unwrap()
}

#[test]
fn a_sweep_needing_healing_still_supports_booleans() {
    // Regression (the "both bodies disappear on subtract" bug): sweeping a
    // wide line+bezier profile through a moderate bezier bend makes
    // MakePipeShell emit a slightly INVALID solid (tolerance/pcurve defects).
    // It renders and measures fine, but OCCT booleans silently classify an
    // invalid solid as nothing: cut returned an empty shape and fuse dropped
    // the swept body entirely. The sweep must heal its output (ShapeFix) so
    // downstream booleans see a valid solid.
    let path = Solid::path_wire(
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        &[0.0, 0.0, 40.0, 0.0, 55.0, 25.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 50.0, 0.0, 55.0, 10.0],
    )
    .unwrap();
    let tube = wide_profile().sweep(&path, [0.0, 1.0, 0.0]).unwrap();

    let desc = tube.describe().unwrap();
    assert!(desc.contains("SOLID"), "swept result is a solid: {desc}");
    assert!(!desc.contains("INVALID"), "swept solid passes BRepCheck: {desc}");

    // A box overlapping the straight leg: the cut must remove SOME volume but
    // not all of it, and the fuse must keep the tube (an invalid input would
    // be silently dropped, which is exactly the regression).
    let tool = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(-5.0, -5.0, 5.0).unwrap();
    let v_tube = tube.volume().unwrap();
    let cut = tube.cut(&tool).unwrap();
    let v_cut = cut.volume().unwrap();
    assert!(
        v_cut > 0.0 && v_cut < v_tube - 1.0,
        "cut removes part of the tube: {v_cut} of {v_tube}"
    );
    let fuse = tube.fuse(&tool).unwrap();
    assert!(
        fuse.volume().unwrap() > v_tube - 1.0,
        "fuse keeps the tube's volume (an invalid input would be dropped)"
    );
}

#[test]
fn an_unhealable_self_intersecting_sweep_is_an_error() {
    // When the bend is far tighter than the profile is wide, the sweep
    // self-intersects and no healing can fix it. That must surface as a
    // per-feature error — not a silently broken body that renders fine and
    // then corrupts booleans and STL export.
    let path = Solid::path_wire(
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        &[0.0, 0.0, 25.0, 0.0, 33.0, 8.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 31.0, 0.0, 33.0, 2.0],
    )
    .unwrap();
    let err = wide_profile()
        .sweep(&path, [0.0, 1.0, 0.0])
        .err()
        .expect("a self-intersecting sweep must fail");
    assert!(
        err.to_string().contains("self-intersects"),
        "the error explains the cause: {err}"
    );
}
