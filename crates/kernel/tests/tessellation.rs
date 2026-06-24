//! Tessellation tagging: every vertex carries its source face id, used for
//! GPU picking.

use std::collections::BTreeSet;

use rmf_kernel::Solid;

#[test]
fn box_tessellation_tags_six_faces() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let mesh = cube.tessellate(0.5).unwrap();

    // One face id per vertex (per positions triple).
    assert_eq!(mesh.face_ids.len(), mesh.positions.len() / 3);
    assert!(!mesh.face_ids.is_empty());

    // A box has exactly six planar faces, numbered 0..6.
    let distinct: BTreeSet<u32> = mesh.face_ids.iter().copied().collect();
    assert_eq!(distinct.len(), 6, "a box has six faces, got {distinct:?}");
    assert_eq!(*distinct.iter().max().unwrap(), 5);

    // Every triangle's three vertices belong to one and the same face.
    for tri in mesh.indices.chunks_exact(3) {
        let f = mesh.face_ids[tri[0] as usize];
        assert!(tri.iter().all(|&v| mesh.face_ids[v as usize] == f));
    }

    // A box has twelve edges; each emits a polyline tagged with its edge id.
    assert_eq!(mesh.edge_ids.len(), mesh.edge_positions.len() / 3);
    assert!(!mesh.edge_indices.is_empty());
    let edges: BTreeSet<u32> = mesh.edge_ids.iter().copied().collect();
    assert_eq!(edges.len(), 12, "a box has twelve edges, got {edges:?}");
}

#[test]
fn box_faces_report_planar_frames() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();

    // Every one of the six faces is planar with an orthonormal frame.
    for index in 0..6 {
        let (_o, x, y) = cube
            .face_plane(index)
            .unwrap()
            .unwrap_or_else(|| panic!("face {index} should be planar"));
        let dot = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        assert!(dot.abs() < 1e-9, "axes not perpendicular on face {index}");
        let xlen = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        assert!((xlen - 1.0).abs() < 1e-9, "x axis not unit on face {index}");
    }
    // Out of range -> None, not an error.
    assert!(cube.face_plane(99).unwrap().is_none());
}

fn max_z(solid: &Solid) -> f32 {
    let mesh = solid.tessellate(0.5).unwrap();
    mesh.positions
        .chunks_exact(3)
        .map(|p| p[2])
        .fold(f32::MIN, f32::max)
}

#[test]
fn push_pull_offsets_the_anchored_face() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    assert!((max_z(&cube) - 10.0).abs() < 0.5);

    // Top face at z=10, normal +Z. Pushing out by 5 fuses material (z -> 15).
    let taller = cube
        .push_pull([5.0, 5.0, 10.0], [0.0, 0.0, 1.0], 5.0)
        .unwrap();
    assert!((max_z(&taller) - 15.0).abs() < 0.5, "pushed max z {}", max_z(&taller));

    // Pulling in by 3 cuts material (z -> 7).
    let shorter = cube
        .push_pull([5.0, 5.0, 10.0], [0.0, 0.0, 1.0], -3.0)
        .unwrap();
    assert!((max_z(&shorter) - 7.0).abs() < 0.5, "pulled max z {}", max_z(&shorter));
}
