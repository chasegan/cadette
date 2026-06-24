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
}
