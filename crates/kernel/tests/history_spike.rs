//! Evolution-history coverage for tracked booleans (originally the spike that
//! green-lit element maps; kept as the regression suite for the composed
//! `BRepAlgoAPI_*` + `ShapeUpgrade_UnifySameDomain` history).
//!
//! The properties element maps rest on:
//!
//!  1. Coverage — every result face is claimed by some input face's lineage.
//!     (An unclaimed face would be an unnameable orphan.)
//!  2. Identity through change — "the top face of the box" resolves to the
//!     holed top face after a cut, purely via lineage (no geometry matching).
//!  3. Robustness under motion — the SAME input face still resolves after the
//!     tool moves (the exact case that breaks geometric anchors).
//!  4. Splits — one input face reports BOTH children when an op divides it.
//!  5. Edges — the same machinery tracks edges (the fillet-anchor case), and
//!     the only result edges without edge lineage are the genuinely NEW
//!     section curves, each claimed by the face pair that generated it.

use cdt_kernel::{Lineage, Solid, TrackedBoolOp};

/// A 20mm cube at the origin.
fn box20() -> Solid {
    Solid::cuboid(20.0, 20.0, 20.0).unwrap()
}

/// A r=4 drill through the full box height, centered at (x, y).
fn bore(x: f64, y: f64) -> Solid {
    Solid::cylinder(4.0, 40.0)
        .unwrap()
        .translate(x, y, -10.0)
        .unwrap()
}

/// The input-face index (TopExp order) of `solid`'s planar face whose plane
/// origin and normal match — how the tests pin down "the top face" without
/// assuming OCCT's face ordering.
fn planar_face_index(solid: &Solid, pred: impl Fn([f64; 3], [f64; 3]) -> bool) -> u32 {
    let n = solid.face_count().unwrap();
    for i in 0..n as u32 {
        if let Some((origin, x_dir, y_dir)) = solid.face_plane(i).unwrap() {
            let normal = [
                x_dir[1] * y_dir[2] - x_dir[2] * y_dir[1],
                x_dir[2] * y_dir[0] - x_dir[0] * y_dir[2],
                x_dir[0] * y_dir[1] - x_dir[1] * y_dir[0],
            ];
            if pred(origin, normal) {
                return i;
            }
        }
    }
    panic!("no face matched the predicate");
}

/// The result-element images of input element `index` of operand `which`.
fn images(lineage: &[Lineage], which: u8, index: u32) -> Vec<u32> {
    lineage
        .iter()
        .filter(|l| l.input == which && l.index == index)
        .filter_map(|l| l.output)
        .collect()
}

#[test]
fn every_result_face_is_claimed_by_an_input_lineage() {
    let tracked = box20()
        .boolean_tracked(&bore(10.0, 10.0), TrackedBoolOp::Cut)
        .unwrap();

    let n = tracked.result.face_count().unwrap();
    let claimed: std::collections::HashSet<u32> =
        tracked.faces.iter().filter_map(|l| l.output).collect();
    let orphans: Vec<u32> = (0..n as u32).filter(|i| !claimed.contains(i)).collect();
    assert!(
        orphans.is_empty(),
        "unclaimed result faces (unnameable orphans): {orphans:?} of {n}"
    );

    // Every box face survives the bore; only the tool's end caps (fully
    // outside the cut) are consumed.
    for i in 0..6 {
        assert!(
            !images(&tracked.faces, 0, i).is_empty(),
            "box face {i} lost its lineage"
        );
    }
    let deleted: Vec<u32> = tracked
        .faces
        .iter()
        .filter(|l| l.input == 1 && l.output.is_none())
        .map(|l| l.index)
        .collect();
    assert_eq!(deleted.len(), 2, "exactly the two drill end caps vanish: {deleted:?}");
}

#[test]
fn the_top_face_resolves_through_the_cut_by_lineage_alone() {
    let a = box20();
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);

    let tracked = a
        .boolean_tracked(&bore(10.0, 10.0), TrackedBoolOp::Cut)
        .unwrap();

    let imgs = images(&tracked.faces, 0, top);
    assert_eq!(imgs.len(), 1, "the holed top is ONE face: {imgs:?}");
    let area = tracked.result.face_area(imgs[0]).unwrap();
    let expected = 400.0 - std::f64::consts::PI * 16.0; // 20² minus the bore
    assert!(
        (area - expected).abs() < 0.1,
        "lineage found the wrong face: area {area:.2}, expected {expected:.2}"
    );
}

#[test]
fn the_same_face_still_resolves_after_the_tool_moves() {
    // The case that breaks geometric anchors: rebuild with the drill moved.
    // The input face INDEX is the stable key (same box), and lineage must
    // carry it to the (differently-holed) top face both times.
    let a = box20();
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let expected = 400.0 - std::f64::consts::PI * 16.0;

    for (x, y) in [(10.0, 10.0), (5.0, 5.0), (14.0, 6.0)] {
        let tracked = a
            .boolean_tracked(&bore(x, y), TrackedBoolOp::Cut)
            .unwrap();
        let imgs = images(&tracked.faces, 0, top);
        assert_eq!(imgs.len(), 1, "bore at ({x},{y}): top split unexpectedly");
        let area = tracked.result.face_area(imgs[0]).unwrap();
        assert!(
            (area - expected).abs() < 0.1,
            "bore at ({x},{y}): resolved to the wrong face (area {area:.2})"
        );
    }
}

#[test]
fn a_split_face_reports_both_children() {
    // A slab wider than the box, cutting clean through the middle: the box
    // becomes two pieces and its top face splits in two. Lineage must report
    // BOTH children.
    let slab = Solid::cuboid(40.0, 8.0, 40.0)
        .unwrap()
        .translate(-10.0, 6.0, -10.0)
        .unwrap();
    let a = box20();
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);

    let tracked = a.boolean_tracked(&slab, TrackedBoolOp::Cut).unwrap();

    let imgs = images(&tracked.faces, 0, top);
    assert_eq!(imgs.len(), 2, "the split top must report two children: {imgs:?}");
    // The two bands: y ∈ [0,6] and y ∈ [14,20], each 20 × 6.
    let mut areas: Vec<f64> = imgs
        .iter()
        .map(|&i| tracked.result.face_area(i).unwrap())
        .collect();
    areas.sort_by(f64::total_cmp);
    assert!(
        areas.iter().all(|a| (a - 120.0).abs() < 0.1),
        "children should be the two 20×6 bands: {areas:?}"
    );
}

#[test]
fn coplanar_faces_merge_through_unify_and_both_lineages_agree() {
    // The N→1 merge case, and the one that exercises the unify stage's history
    // hardest: two overlapping boxes fused, tops coplanar. The boolean leaves
    // two top faces with a seam; unify merges them into ONE. Both inputs' top
    // faces must map to the SAME result face, or merge-naming is impossible.
    let a = box20();
    let b = box20().translate(10.0, 0.0, 0.0).unwrap();
    let top_a = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let top_b = planar_face_index(&b, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);

    let tracked = a.boolean_tracked(&b, TrackedBoolOp::Fuse).unwrap();

    // The union is a clean 30×20×20 cuboid: 6 faces, nothing orphaned.
    assert_eq!(tracked.result.face_count().unwrap(), 6, "unify left seams behind");
    let claimed: std::collections::HashSet<u32> =
        tracked.faces.iter().filter_map(|l| l.output).collect();
    assert_eq!(claimed.len(), 6, "some result face is unclaimed");

    let imgs_a = images(&tracked.faces, 0, top_a);
    let imgs_b = images(&tracked.faces, 1, top_b);
    assert_eq!(imgs_a.len(), 1, "top of A: {imgs_a:?}");
    assert_eq!(imgs_a, imgs_b, "both tops must converge on the merged face");
    let area = tracked.result.face_area(imgs_a[0]).unwrap();
    assert!(
        (area - 600.0).abs() < 0.1,
        "merged top should be 30×20: {area:.2}"
    );
}

#[test]
fn edges_track_too_and_section_curves_are_claimed_by_their_faces() {
    // The fillet-anchor case. Through-bore in the box: all 12 box edges
    // survive; the only result edges with NO edge lineage are the two circles
    // where the bore meets the top and bottom faces — genuinely NEW curves,
    // each of which the SECTION records must attribute to the faces that
    // crossed there (bore wall × box top/bottom).
    let tracked = box20()
        .boolean_tracked(&bore(10.0, 10.0), TrackedBoolOp::Cut)
        .unwrap();

    for i in 0..12 {
        assert!(
            !images(&tracked.edges, 0, i).is_empty(),
            "box edge {i} lost its lineage"
        );
    }

    let claimed: std::collections::HashSet<u32> =
        tracked.edges.iter().filter_map(|l| l.output).collect();
    let total = tracked.result.edge_count().unwrap() as u32;
    let orphans: Vec<u32> = (0..total).filter(|i| !claimed.contains(i)).collect();
    assert_eq!(
        orphans.len(),
        2,
        "expected exactly the two bore-rim section circles as new edges, got {orphans:?} of {total}"
    );
    for &i in &orphans {
        let len = tracked.result.edge_length(i).unwrap();
        let rim = 2.0 * std::f64::consts::PI * 4.0;
        assert!(
            (len - rim).abs() < 0.1,
            "orphan edge {i} is not a bore rim: length {len:.2}"
        );
        // The section records attribute this new edge to parent faces from
        // BOTH operands (box top/bottom × bore wall).
        let parents: Vec<u8> = tracked
            .sections
            .iter()
            .filter(|&&(_, _, out)| out == i)
            .map(|&(which, _, _)| which)
            .collect();
        assert!(
            parents.contains(&0) && parents.contains(&1),
            "section edge {i} should be claimed by faces of both operands: {parents:?}"
        );
    }
}
