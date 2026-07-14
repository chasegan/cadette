//! Slice 4 of element maps: fillet and push/pull propagate names.
//!
//! Fillet: surviving (trimmed) faces keep their names VERBATIM, each rounded
//! edge's new face is `Rounded { parent: edge }`, and the fillet's new
//! boundary edges are named by adjacency. Push/pull: the MOVED face keeps the
//! pushed face's name (its identity survives the push — that is what the
//! PushPull feature itself will store in slice 5), and untouched/grown
//! neighbours keep theirs.

use cdt_core::{DVec3, FaceAnchor, FeatureId, GeometryBackend};
use cdt_kernel::{Body, ElemKind, ElementName, KernelBackend};

/// A 20mm cube minted under feature `id`.
fn tracked_box(backend: &mut KernelBackend, id: u64) -> Body {
    backend.begin_feature(FeatureId(id));
    backend.make_box(DVec3::splat(20.0)).unwrap()
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

/// The edge of `body` nearest `point` (the anchor scheme picks it too).
fn edge_index_near(body: &Body, point: [f64; 3]) -> u32 {
    let n = body.edge_count().unwrap();
    let mut best = (0u32, f64::INFINITY);
    for i in 0..n as u32 {
        let c = body.solid.elem_centroid(ElemKind::Edge, i).unwrap();
        let d2 = (0..3).map(|k| (c[k] - point[k]).powi(2)).sum::<f64>();
        if d2 < best.1 {
            best = (i, d2);
        }
    }
    best.0
}

#[test]
fn a_fillet_names_the_rounding_and_keeps_survivors_verbatim() {
    // Fillet the box's top-front edge (midpoint (10, 0, 20)): the two trimmed
    // faces keep their Born names, the new face is Rounded{that edge}, and
    // BOTH names keep resolving when the radius changes — the parametric
    // stability a coordinate anchor can't give.
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let top_name = a.elements.face(top).unwrap().clone();
    let edge = edge_index_near(&a, [10.0, 0.0, 20.0]);
    let rounded_name = ElementName::Rounded {
        parent: Box::new(a.elements.edge(edge).unwrap().clone()),
    };

    for radius in [1.0, 3.0] {
        backend.begin_feature(FeatureId(2));
        let filleted = backend
            .fillet_edges(&a, &[cdt_core::EdgeAnchor { point: DVec3::new(10.0, 0.0, 20.0) }], radius)
            .unwrap();

        let ti = filleted
            .elements
            .find_face(&top_name)
            .expect("trimmed top keeps its name");
        let area = filleted.face_area(ti).unwrap();
        assert!(
            (area - (400.0 - 20.0 * radius)).abs() < 0.1,
            "r={radius}: resolved top has the trimmed area, got {area:.2}"
        );
        let ri = filleted
            .elements
            .find_face(&rounded_name)
            .expect("the fillet face is Rounded{edge}");
        assert!(
            filleted.face_plane(ri).unwrap().is_none(),
            "the fillet face is the non-planar one"
        );
        // The fillet's new boundary edges got adjacency names that reference
        // the Rounded face — so a follow-up fillet could anchor to them.
        let boundary = filleted
            .elements
            .edges
            .iter()
            .flatten()
            .filter(|n| matches!(n, ElementName::Intersection { parents } if parents.contains(&rounded_name)))
            .count();
        assert!(
            boundary >= 2,
            "the rounding's boundary edges are named by adjacency (got {boundary})"
        );
    }
}

#[test]
fn fillet_all_names_every_rounding() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let filleted = backend.fillet_all(&a, 2.0).unwrap();

    let roundings = filleted
        .elements
        .faces
        .iter()
        .flatten()
        .filter(|n| matches!(n, ElementName::Rounded { .. }))
        .count();
    assert_eq!(roundings, 12, "one Rounded face per box edge");
    // All six original faces survive (shrunk) under their Born names.
    for i in 0..6 {
        let name = a.elements.face(i).unwrap();
        assert!(
            filleted.elements.find_face(name).is_some(),
            "face {i} survives the all-over fillet"
        );
    }
}

#[test]
fn a_pulled_face_keeps_its_identity() {
    // THE push/pull property: pull the top face out by d — the face that now
    // sits at z = 20 + d carries the SAME name as the original top. Repeat
    // with different distances; the stored name resolves every time. This is
    // exactly the reference the PushPull feature stores in slice 5.
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let top_name = a.elements.face(top).unwrap().clone();
    let side_name = a.elements
        .face(planar_face_index(&a, |o, n| o[0] > 19.9 && n[0].abs() > 0.99))
        .unwrap()
        .clone();

    for distance in [5.0, 12.0, -6.0] {
        backend.begin_feature(FeatureId(2));
        let pushed = backend
            .push_pull(
                &a,
                FaceAnchor { point: DVec3::new(10.0, 10.0, 20.0), normal: DVec3::Z },
                distance,
            )
            .unwrap();

        let i = pushed
            .elements
            .find_face(&top_name)
            .unwrap_or_else(|| panic!("top resolves after a {distance}mm push"));
        let (origin, _, _) = pushed.face_plane(i).unwrap().expect("planar top");
        assert!(
            (origin[2] - (20.0 + distance)).abs() < 1e-7,
            "d={distance}: the resolved face sits at the MOVED height, got z={}",
            origin[2]
        );
        // A grown side wall keeps its name (prism sides are unnamed, so the
        // survivor rule leaves the body's wall name intact through the merge).
        assert!(
            pushed.elements.find_face(&side_name).is_some(),
            "d={distance}: the grown side wall keeps its name"
        );
    }
}

#[test]
fn a_zero_distance_push_is_an_identity() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let pushed = backend
        .push_pull(
            &a,
            FaceAnchor { point: DVec3::new(10.0, 10.0, 20.0), normal: DVec3::Z },
            0.0,
        )
        .unwrap();
    assert_eq!(
        pushed.elements, a.elements,
        "a no-op push keeps the input's map verbatim"
    );
}

#[test]
fn names_chain_through_extrude_fillet_and_cut() {
    // The end-to-end a real model exercises: sketch → extrude → fillet a top
    // rim → bore a hole. The extrusion's far cap must still resolve through
    // BOTH downstream ops, with its area reflecting the fillet trim and bore.
    use cdt_core::{BooleanOp, Profile, SketchPlane};
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(1));
    let sketch = backend
        .sketch(SketchPlane::Xy, Profile::Rectangle { width: 30.0, height: 20.0 })
        .unwrap();
    let cap_name = ElementName::Cap {
        parent: Box::new(sketch.elements.face(0).unwrap().clone()),
    };

    backend.begin_feature(FeatureId(2));
    let solid = backend.extrude(&sketch, 15.0).unwrap();
    backend.begin_feature(FeatureId(3));
    // The rectangle is centred: the top-front rim sits at (0, -10, 15).
    let filleted = backend
        .fillet_edges(&solid, &[cdt_core::EdgeAnchor { point: DVec3::new(0.0, -10.0, 15.0) }], 2.0)
        .unwrap();
    backend.begin_feature(FeatureId(4));
    let drill = backend.make_cylinder(3.0, 40.0).unwrap();
    backend.begin_feature(FeatureId(5));
    let drill = backend.translate(&drill, DVec3::new(5.0, 2.0, -10.0)).unwrap();
    backend.begin_feature(FeatureId(6));
    let done = backend.boolean(BooleanOp::Subtract, &filleted, &drill).unwrap();

    let i = done
        .elements
        .find_face(&cap_name)
        .expect("the extrusion cap resolves through fillet + cut");
    let area = done.face_area(i).unwrap();
    let expected = 30.0 * 20.0 - 30.0 * 2.0 - std::f64::consts::PI * 9.0;
    assert!(
        (area - expected).abs() < 0.1,
        "cap area should show the trim and the bore: got {area:.2}, expected {expected:.2}"
    );
}
