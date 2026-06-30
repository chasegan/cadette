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

    let tube = profile.sweep(&path).unwrap();
    // Volume of a cylinder r=2, h=10 = π r² h.
    let expected = PI * 2.0 * 2.0 * 10.0;
    let v = tube.volume();
    assert!(
        (v - expected).abs() < 1e-3,
        "swept tube volume {v} vs cylinder {expected}"
    );
    // A cylinder is three faces: the round side plus two end caps.
    assert_eq!(tube.face_count(), 3, "round side + two caps");
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

    let tube = profile.sweep(&path).unwrap();
    assert!(tube.volume() > 0.0, "bent sweep encloses a volume");
    // Tessellation must succeed and produce geometry (no degenerate result).
    let mesh = tube.tessellate(0.2).unwrap();
    assert!(!mesh.positions.is_empty(), "swept solid meshes");
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
    let tube = profile.sweep(&path).unwrap();
    // A round profile swept the full length (≈20) → volume ≈ π·1²·20.
    assert!((tube.volume() - PI * 20.0).abs() < 3.0, "full-length tube, got {}", tube.volume());
    // Some planar end cap must face ~+X (the final segment direction).
    let faces_x = (0..tube.face_count() as u32).any(|i| {
        tube.face_plane(i).unwrap().is_some_and(|(_, x, y)| {
            let nx = x[1] * y[2] - x[2] * y[1];
            nx.abs() > 0.9
        })
    });
    assert!(faces_x, "the section reoriented: an end cap faces +X");
}
