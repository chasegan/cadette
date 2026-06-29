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
fn unify_is_idempotent_on_a_clean_box() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    assert_eq!(cube.face_count(), 6);
    // Refining a shape with nothing to merge leaves its six faces intact.
    let refined = cube.unify().unwrap();
    assert_eq!(refined.face_count(), 6);
}
