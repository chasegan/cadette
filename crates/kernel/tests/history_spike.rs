//! SPIKE: is OCCT 7.9's boolean evolution history complete enough to build
//! element maps (persistent face/edge naming) on?
//!
//! The production boolean pipeline is `BRepAlgoAPI_*` followed by
//! `ShapeUpgrade_UnifySameDomain` (seam removal), so a usable history must
//! COMPOSE both stages. These tests probe the questions the element-map design
//! rests on:
//!
//!  1. Coverage — is every result face claimed by some input face's lineage?
//!     (An unclaimed face would be an unnameable orphan.)
//!  2. Identity through change — does "the top face of the box" resolve to the
//!     holed top face after a cut, purely via lineage (no geometry matching)?
//!  3. Robustness under motion — does the SAME input face still resolve after
//!     the tool moves (the exact case that breaks geometric anchors)?
//!  4. Splits — does one input face report BOTH children when an op divides it?
//!  5. Edges — does the same machinery track edges (the fillet-anchor case),
//!     and are the only unclaimed result edges the genuinely NEW section
//!     curves (which need face-intersection names, a known follow-up)?

use cdt_kernel::{ElemKind, Lineage, Solid, TrackedBoolOp};

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

/// The result-face images of input face `index` of operand `which`.
fn images(lineage: &[Lineage], which: u8, index: u32) -> Vec<u32> {
    lineage
        .iter()
        .filter(|l| l.input == which && l.index == index)
        .filter_map(|l| l.output)
        .collect()
}

#[test]
fn every_result_face_is_claimed_by_an_input_lineage() {
    let (result, lineage) = box20()
        .boolean_with_history(&bore(10.0, 10.0), TrackedBoolOp::Cut, ElemKind::Face)
        .unwrap();

    let n = result.face_count().unwrap();
    let claimed: std::collections::HashSet<u32> =
        lineage.iter().filter_map(|l| l.output).collect();
    let orphans: Vec<u32> = (0..n as u32).filter(|i| !claimed.contains(i)).collect();
    assert!(
        orphans.is_empty(),
        "unclaimed result faces (unnameable orphans): {orphans:?} of {n}"
    );

    // Every box face survives the bore; only the tool's end caps (fully
    // outside the cut) are consumed.
    for i in 0..6 {
        assert!(
            !images(&lineage, 0, i).is_empty(),
            "box face {i} lost its lineage"
        );
    }
    let deleted: Vec<u32> = lineage
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

    let (result, lineage) = a
        .boolean_with_history(&bore(10.0, 10.0), TrackedBoolOp::Cut, ElemKind::Face)
        .unwrap();

    let imgs = images(&lineage, 0, top);
    assert_eq!(imgs.len(), 1, "the holed top is ONE face: {imgs:?}");
    let area = result.face_area(imgs[0]).unwrap();
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
        let (result, lineage) = a
            .boolean_with_history(&bore(x, y), TrackedBoolOp::Cut, ElemKind::Face)
            .unwrap();
        let imgs = images(&lineage, 0, top);
        assert_eq!(imgs.len(), 1, "bore at ({x},{y}): top split unexpectedly");
        let area = result.face_area(imgs[0]).unwrap();
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
    // BOTH children (this is where the ordinal tie-break policy will live).
    let slab = Solid::cuboid(40.0, 8.0, 40.0)
        .unwrap()
        .translate(-10.0, 6.0, -10.0)
        .unwrap();
    let a = box20();
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);

    let (result, lineage) = a
        .boolean_with_history(&slab, TrackedBoolOp::Cut, ElemKind::Face)
        .unwrap();

    let imgs = images(&lineage, 0, top);
    assert_eq!(imgs.len(), 2, "the split top must report two children: {imgs:?}");
    // The two bands: y ∈ [0,6] and y ∈ [14,20], each 20 × 6.
    let mut areas: Vec<f64> = imgs
        .iter()
        .map(|&i| result.face_area(i).unwrap())
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

    let (result, lineage) = a
        .boolean_with_history(&b, TrackedBoolOp::Fuse, ElemKind::Face)
        .unwrap();

    // The union is a clean 30×20×20 cuboid: 6 faces, nothing orphaned.
    assert_eq!(result.face_count().unwrap(), 6, "unify left seams behind");
    let claimed: std::collections::HashSet<u32> =
        lineage.iter().filter_map(|l| l.output).collect();
    assert_eq!(claimed.len(), 6, "some result face is unclaimed");

    let imgs_a = images(&lineage, 0, top_a);
    let imgs_b = images(&lineage, 1, top_b);
    assert_eq!(imgs_a.len(), 1, "top of A: {imgs_a:?}");
    assert_eq!(imgs_a, imgs_b, "both tops must converge on the merged face");
    let area = result.face_area(imgs_a[0]).unwrap();
    assert!(
        (area - 600.0).abs() < 0.1,
        "merged top should be 30×20: {area:.2}"
    );
}

#[test]
fn edges_track_too_and_only_section_curves_are_new() {
    // The fillet-anchor case. Through-bore in the box: all 12 box edges
    // survive; the drill's end-cap rims (outside the cut) are consumed; the
    // only result edges with NO input lineage are the two circles where the
    // bore meets the top and bottom faces — genuinely NEW curves that element
    // naming must label as "intersection of face A and face B" (follow-up).
    let (result, lineage) = box20()
        .boolean_with_history(&bore(10.0, 10.0), TrackedBoolOp::Cut, ElemKind::Edge)
        .unwrap();

    for i in 0..12 {
        assert!(
            !images(&lineage, 0, i).is_empty(),
            "box edge {i} lost its lineage"
        );
    }

    let claimed: std::collections::HashSet<u32> =
        lineage.iter().filter_map(|l| l.output).collect();
    let total = {
        // Count result edges by probing edge_length until it errors.
        let mut n = 0u32;
        while result.edge_length(n).is_ok() {
            n += 1;
        }
        n
    };
    let orphans: Vec<u32> = (0..total).filter(|i| !claimed.contains(i)).collect();
    assert_eq!(
        orphans.len(),
        2,
        "expected exactly the two bore-rim section circles as new edges, got {orphans:?} of {total}"
    );
    // Both orphans really are the r=4 bore rims.
    for &i in &orphans {
        let len = result.edge_length(i).unwrap();
        let rim = 2.0 * std::f64::consts::PI * 4.0;
        assert!(
            (len - rim).abs() < 0.1,
            "orphan edge {i} is not a bore rim: length {len:.2}"
        );
    }
}
