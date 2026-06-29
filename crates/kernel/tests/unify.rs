//! Same-domain face merging: booleans should not leave coplanar neighbours
//! split by a seam edge.

use rmf_kernel::Solid;

#[test]
fn fusing_stacked_boxes_merges_the_coplanar_sides() {
    // Two 10×10×10 boxes stacked along Z (one at z=0, one lifted to z=10).
    // Their union is geometrically a single 10×10×20 box: every side face of the
    // lower box is coplanar with the matching side of the upper one.
    let lower = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let upper = Solid::cuboid(10.0, 10.0, 10.0)
        .unwrap()
        .translate(0.0, 0.0, 10.0)
        .unwrap();
    let fused = lower.fuse(&upper).unwrap();

    // With seam merging the result is a clean box: 6 faces. Without it the four
    // sides would each stay split into two halves (→ 10 faces).
    assert_eq!(fused.face_count(), 6, "coplanar sides should merge into one");
}

#[test]
fn repeated_push_with_noisy_normals_does_not_stack_segments() {
    // The interactive pick supplies a face normal reconstructed from the depth
    // buffer, so it carries a few 1e-4 rad of noise. push_pull must extrude along
    // the resolved face's OWN normal, not that noisy input — otherwise each push
    // tilts its new wall slightly, consecutive walls aren't coplanar, and they
    // refuse to merge, stacking up as "segments".
    let block = Solid::cuboid(40.0, 40.0, 40.0)
        .unwrap()
        .fillet_all_edges(4.0)
        .unwrap();
    // Pull the top up (a clean inset tower → 30 faces), then push it back down in
    // several steps, each with a slightly different noisy normal.
    let noisy = [
        [0.0003, -0.0002, 0.99999],
        [-0.0001, 0.0004, 0.99999],
        [0.0002, 0.0002, 0.99999],
    ];
    let mut s = block.push_pull([20.0, 20.0, 40.0], [0.0, 0.0, 1.0], 8.0).unwrap();
    assert_eq!(s.face_count(), 30, "one clean inset tower");
    let mut z = 48.0;
    for n in noisy {
        s = s.push_pull([20.0, 20.0, z], n, -1.5).unwrap();
        z -= 1.5;
    }
    // Still one tower, not a stack of unmerged bands.
    assert_eq!(s.face_count(), 30, "noisy normals must not split the wall");
}

#[test]
fn unify_is_idempotent_on_a_clean_box() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    assert_eq!(cube.face_count(), 6);
    // Refining a shape with nothing to merge leaves its six faces intact.
    let refined = cube.unify().unwrap();
    assert_eq!(refined.face_count(), 6);
}
