//! Slice 2 of element maps: booleans compose names from their operands.
//!
//! The rules under test (see `ElementMap::compose_boolean`):
//! - a survivor keeps its name VERBATIM — the property that makes stored
//!   references immune to a moved tool;
//! - a divided face's pieces are `Split { parent, ordinal }`, ordinals ranked
//!   by centroid;
//! - same-domain faces unified by a fuse become ONE `Merged` face carrying
//!   both parents;
//! - new section edges are `Intersection` of the face names that crossed;
//! - an untracked operand contributes `None`s, never fabricated names.

use cdt_core::{BooleanOp, DVec3, FeatureId, GeometryBackend};
use cdt_kernel::{Body, ElemKind, ElementName, KernelBackend};

fn born(origin: u64, kind: ElemKind, index: u32) -> ElementName {
    ElementName::Born { origin, kind, index }
}

/// A 20mm cube minted under feature `id`.
fn tracked_box(backend: &mut KernelBackend, id: u64) -> Body {
    backend.begin_feature(FeatureId(id));
    backend.make_box(DVec3::splat(20.0)).unwrap()
}

/// A r=4 through-drill at (x, y) minted under feature `id` (translated under
/// `id + 100`, which passes the map through unchanged).
fn tracked_bore(backend: &mut KernelBackend, id: u64, x: f64, y: f64) -> Body {
    backend.begin_feature(FeatureId(id));
    let cyl = backend.make_cylinder(4.0, 40.0).unwrap();
    backend.begin_feature(FeatureId(id + 100));
    backend
        .translate(&cyl, DVec3::new(x, y, -10.0))
        .unwrap()
}

/// The face index of `body` whose plane matches (planar faces only).
fn planar_face_index(body: &Body, pred: impl Fn([f64; 3], [f64; 3]) -> bool) -> u32 {
    let n = body.face_count().unwrap();
    for i in 0..n as u32 {
        if let Some((origin, x_dir, y_dir)) = body.face_plane(i).unwrap() {
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

/// The one NON-planar face of a body (a cylinder's side wall).
fn side_face_index(body: &Body) -> u32 {
    let n = body.face_count().unwrap();
    (0..n as u32)
        .find(|&i| body.face_plane(i).unwrap().is_none())
        .expect("a non-planar face")
}

#[test]
fn survivors_keep_their_names_and_resolve_after_the_tool_moves() {
    // THE headline property: the box top's Born name — recorded once, as a
    // stored reference would be — resolves to the holed top face across
    // rebuilds with the drill in three different places. Geometric anchors
    // are exactly what this replaces.
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let top_name = a.elements.face(top).unwrap().clone();
    assert_eq!(top_name, born(1, ElemKind::Face, top));

    let expected = 400.0 - std::f64::consts::PI * 16.0;
    for (x, y) in [(10.0, 10.0), (5.0, 5.0), (14.0, 6.0)] {
        let tool = tracked_bore(&mut backend, 2, x, y);
        backend.begin_feature(FeatureId(3));
        let cut = backend.boolean(BooleanOp::Subtract, &a, &tool).unwrap();

        let i = cut
            .elements
            .find_face(&top_name)
            .expect("the stored name resolves");
        let area = cut.face_area(i).unwrap();
        assert!(
            (area - expected).abs() < 0.1,
            "bore at ({x},{y}): name resolved to the wrong face (area {area:.2})"
        );
    }
}

#[test]
fn the_bore_wall_keeps_the_cylinders_name() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let tool = tracked_bore(&mut backend, 2, 10.0, 10.0);
    let side_name = tool.elements.face(side_face_index(&tool)).unwrap().clone();
    assert!(matches!(side_name, ElementName::Born { origin: 2, .. }));

    backend.begin_feature(FeatureId(3));
    let cut = backend.boolean(BooleanOp::Subtract, &a, &tool).unwrap();
    let i = cut.elements.find_face(&side_name).expect("bore wall resolves");
    let area = cut.face_area(i).unwrap();
    let wall = 2.0 * std::f64::consts::PI * 4.0 * 20.0;
    assert!(
        (area - wall).abs() < 0.1,
        "expected the bore wall ({wall:.1}), got {area:.2}"
    );
}

#[test]
fn split_pieces_get_centroid_ranked_ordinals() {
    // A slab cuts the box in two; the top face's pieces become
    // Split { parent: top, ordinal } with ordinal 0 = the lower-coordinate
    // band (centroid rank), and both resolve to distinct faces.
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let top_name = a.elements.face(top).unwrap().clone();

    backend.begin_feature(FeatureId(2));
    let slab = backend.make_box(DVec3::new(40.0, 8.0, 40.0)).unwrap();
    backend.begin_feature(FeatureId(3));
    let slab = backend
        .translate(&slab, DVec3::new(-10.0, 6.0, -10.0))
        .unwrap();
    backend.begin_feature(FeatureId(4));
    let cut = backend.boolean(BooleanOp::Subtract, &a, &slab).unwrap();

    let piece = |ordinal: u32| ElementName::Split {
        parent: Box::new(top_name.clone()),
        ordinal,
    };
    let p0 = cut.elements.find_face(&piece(0)).expect("piece 0 resolves");
    let p1 = cut.elements.find_face(&piece(1)).expect("piece 1 resolves");
    assert_ne!(p0, p1, "the pieces are distinct faces");
    // Ordinal 0 is the band with the smaller centroid: y ∈ [0,6] (centroid
    // y=3) before y ∈ [14,20] (centroid y=17). Same x and z, so y decides.
    let y_of = |i: u32| cut.solid.elem_centroid(ElemKind::Face, i).unwrap()[1];
    assert!(
        y_of(p0) < y_of(p1),
        "ordinals must follow centroid order: y0 {} vs y1 {}",
        y_of(p0),
        y_of(p1)
    );
    // The parent name itself no longer resolves — it has no single referent.
    assert_eq!(cut.elements.find_face(&top_name), None);
}

#[test]
fn a_fuse_merges_coplanar_tops_into_one_named_face() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let b = backend.make_box(DVec3::splat(20.0)).unwrap();
    backend.begin_feature(FeatureId(3));
    let b = backend.translate(&b, DVec3::new(10.0, 0.0, 0.0)).unwrap();

    let top_a = a.elements
        .face(planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99))
        .unwrap()
        .clone();
    let top_b = b.elements
        .face(planar_face_index(&b, |o, n| o[2] > 19.9 && n[2].abs() > 0.99))
        .unwrap()
        .clone();

    backend.begin_feature(FeatureId(4));
    let fused = backend.boolean(BooleanOp::Union, &a, &b).unwrap();

    let mut parents = vec![top_a, top_b];
    parents.sort();
    let merged = ElementName::Merged { parents };
    let i = fused.elements.find_face(&merged).expect("merged top resolves");
    let area = fused.face_area(i).unwrap();
    assert!(
        (area - 600.0).abs() < 0.1,
        "the merged top is the full 30×20: {area:.2}"
    );
}

#[test]
fn section_edges_are_intersections_of_both_parent_faces() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let tool = tracked_bore(&mut backend, 2, 10.0, 10.0);
    backend.begin_feature(FeatureId(3));
    let cut = backend.boolean(BooleanOp::Subtract, &a, &tool).unwrap();

    let rims: Vec<&ElementName> = cut
        .elements
        .edges
        .iter()
        .flatten()
        .filter(|n| matches!(n, ElementName::Intersection { .. }))
        .collect();
    assert_eq!(rims.len(), 2, "two bore rims: {rims:?}");
    for rim in rims {
        let ElementName::Intersection { parents } = rim else {
            unreachable!()
        };
        let origins: Vec<u64> = parents
            .iter()
            .map(|p| match p {
                ElementName::Born { origin, .. } => *origin,
                other => panic!("rim parent should be a Born face: {other:?}"),
            })
            .collect();
        assert!(
            origins.contains(&1) && origins.contains(&2),
            "each rim is box-face × bore-wall: origins {origins:?}"
        );
    }
}

#[test]
fn an_untracked_operand_still_names_the_tracked_side() {
    // Fuse a tracked box with an untracked body (a fillet result, pre-slice-4):
    // the box's faces stay named, the filleted side's faces are honestly None.
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top_name = a.elements
        .face(planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99))
        .unwrap()
        .clone();

    backend.begin_feature(FeatureId(2));
    let post = backend.make_box(DVec3::new(8.0, 8.0, 30.0)).unwrap();
    backend.begin_feature(FeatureId(3));
    let post = backend.fillet_all(&post, 1.0).unwrap();
    assert!(post.elements.is_empty(), "fillet is untracked (slice 4)");
    // Plant the post overlapping the box so the fuse is one body.
    backend.begin_feature(FeatureId(4));
    let post = backend.translate(&post, DVec3::new(6.0, 6.0, 0.0)).unwrap();

    backend.begin_feature(FeatureId(5));
    let fused = backend.boolean(BooleanOp::Union, &a, &post).unwrap();

    assert!(
        fused.elements.find_face(&top_name).is_some(),
        "the tracked side's names survive"
    );
    assert!(
        fused.elements.faces.iter().any(|n| n.is_none()),
        "the untracked side's faces stay unnamed — never fabricated"
    );
}

#[test]
fn names_flow_end_to_end_through_regen() {
    use cdt_core::{regenerate, Document, FeatureKind};
    // The full production path: a Document replayed by regen, with
    // begin_feature scoping the names — the boolean body's map must name the
    // box's top under the box FEATURE's id.
    let mut doc = Document::new("names");
    let bx = doc.add("Box", FeatureKind::Box { size: DVec3::splat(20.0) });
    let cyl = doc.add("Drill", FeatureKind::Cylinder { radius: 4.0, height: 40.0 });
    let moved = doc.add(
        "Position",
        FeatureKind::Translate { source: cyl, offset: DVec3::new(10.0, 10.0, -10.0) },
    );
    let bore = doc.add(
        "Bore",
        FeatureKind::Boolean {
            op: BooleanOp::Subtract,
            target: bx,
            tool: moved,
        },
    );

    let mut backend = KernelBackend::default();
    let regen = regenerate(&doc, &mut backend);
    assert!(regen.is_ok(), "regen errors: {:?}", regen.errors().len());
    let body = regen.body(bore).expect("boolean body built");

    // Find the holed top by geometry once, then check its NAME is the box
    // feature's Born face — proof the whole pipeline (begin_feature → born →
    // pass-through translate → boolean composition) held together.
    let top = planar_face_index(body, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    match body.elements.face(top) {
        Some(ElementName::Born { origin, kind: ElemKind::Face, .. }) => {
            assert_eq!(*origin, bx.0, "the top face is named under the Box feature");
        }
        other => panic!("expected a Born face name from the box, got {other:?}"),
    }
}
