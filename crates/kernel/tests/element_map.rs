//! Slice 1 of element maps: bodies are born with names, and transforms carry
//! them through unchanged.
//!
//! The load-bearing assumption behind the pass-through is that a transform
//! preserves TopExp sub-shape ORDER: face `i` of the result is face `i` of the
//! input, moved. `translate`/`rotate`/`mirror` relocate shared topology, but
//! `scale` is a `BRepBuilderAPI_GTransform` that REBUILDS the geometry — so
//! the order-preservation test here is what licenses `Body::transformed`.
//! If OCCT ever breaks this, these tests fail and the transforms must switch
//! to evolution-history composition like the booleans.

use cdt_core::{DVec3, FeatureId, GeometryBackend};
use cdt_kernel::{Body, ElemKind, ElementName, KernelBackend};

/// A distinct-sided cuboid so every face pair is tellable apart.
fn tracked_cuboid(backend: &mut KernelBackend, id: u64) -> Body {
    backend.begin_feature(FeatureId(id));
    backend
        .make_box(DVec3::new(20.0, 10.0, 5.0))
        .expect("box builds")
}

/// (origin, outward normal) of face `index` — the geometric fingerprint the
/// tests compare across a transform.
fn face_frame(body: &Body, index: u32) -> ([f64; 3], [f64; 3]) {
    let (origin, x_dir, y_dir) = body
        .face_plane(index)
        .expect("face_plane query")
        .expect("box faces are planar");
    let normal = [
        x_dir[1] * y_dir[2] - x_dir[2] * y_dir[1],
        x_dir[2] * y_dir[0] - x_dir[0] * y_dir[2],
        x_dir[0] * y_dir[1] - x_dir[1] * y_dir[0],
    ];
    (origin, normal)
}

/// Assert face `index` of `after` is face `index` of `before` pushed through
/// `map_point`/`map_normal`: the mapped origin lies ON the new face's plane
/// and the surface normals align. (Plane containment, not origin equality —
/// a rebuild may re-parameterize the plane's representative point.)
///
/// `reflection`: a mirror flips the surface frame's handedness (det −1, so
/// `(Mx)×(My) = −M(x×y)`) and the raw `face_plane` normal comes back negated;
/// which face it is stays pinned by plane position, so accept ±.
fn assert_face_carried(
    before: &Body,
    after: &Body,
    index: u32,
    map_point: impl Fn([f64; 3]) -> [f64; 3],
    map_normal: impl Fn([f64; 3]) -> [f64; 3],
    reflection: bool,
) {
    let (b_origin, b_normal) = face_frame(before, index);
    let (a_origin, a_normal) = face_frame(after, index);
    let expect_p = map_point(b_origin);
    let expect_n = map_normal(b_normal);

    let offset = [
        expect_p[0] - a_origin[0],
        expect_p[1] - a_origin[1],
        expect_p[2] - a_origin[2],
    ];
    let dist = (offset[0] * a_normal[0] + offset[1] * a_normal[1] + offset[2] * a_normal[2]).abs();
    assert!(
        dist < 1e-7,
        "face {index}: mapped origin is off the transformed plane by {dist}"
    );

    let n_len = (expect_n[0].powi(2) + expect_n[1].powi(2) + expect_n[2].powi(2)).sqrt();
    let dot = (expect_n[0] * a_normal[0] + expect_n[1] * a_normal[1] + expect_n[2] * a_normal[2])
        / n_len;
    let aligned = if reflection { dot.abs() } else { dot };
    assert!(
        aligned > 0.999,
        "face {index}: surface normal not carried (dot {dot})"
    );
}

#[test]
fn a_born_body_names_every_face_and_edge() {
    let mut backend = KernelBackend::default();
    let body = tracked_cuboid(&mut backend, 7);

    assert_eq!(body.elements.faces.len(), 6);
    assert_eq!(body.elements.edges.len(), 12);
    assert_eq!(
        body.elements.face(0),
        Some(&ElementName::Born { origin: 7, kind: ElemKind::Face, index: 0 })
    );
    assert_eq!(
        body.elements.edge(11),
        Some(&ElementName::Born { origin: 7, kind: ElemKind::Edge, index: 11 })
    );
    // Resolution is the inverse of naming.
    let name = body.elements.face(3).unwrap().clone();
    assert_eq!(body.elements.find_face(&name), Some(3));
}

#[test]
fn a_sketch_face_is_born_too() {
    use cdt_core::{Profile, SketchPlane};
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(3));
    let face = backend
        .sketch(SketchPlane::Xy, Profile::Rectangle { width: 30.0, height: 20.0 })
        .unwrap();
    assert_eq!(face.elements.faces.len(), 1, "one planar face");
    assert_eq!(face.elements.edges.len(), 4, "four boundary edges");
    assert!(matches!(
        face.elements.face(0),
        Some(ElementName::Born { origin: 3, .. })
    ));
}

#[test]
fn untracked_ops_return_the_empty_map_not_stale_names() {
    // Group/compound is the one op left outside name tracking (deferred:
    // member maps need face-index offsetting).
    let mut backend = KernelBackend::default();
    let a = tracked_cuboid(&mut backend, 1);
    let b = tracked_cuboid(&mut backend, 2);
    backend.begin_feature(FeatureId(3));
    let grouped = backend.compound(&[&a, &b]).unwrap();
    assert!(
        grouped.elements.is_empty(),
        "compounds don't propagate names yet — the map must say so"
    );
}

#[test]
fn translate_carries_every_face_in_place() {
    let mut backend = KernelBackend::default();
    let before = tracked_cuboid(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let after = backend
        .translate(&before, DVec3::new(100.0, 5.0, -3.0))
        .unwrap();

    assert_eq!(after.elements, before.elements, "the map passes through");
    for i in 0..6 {
        assert_face_carried(
            &before,
            &after,
            i,
            |p| [p[0] + 100.0, p[1] + 5.0, p[2] - 3.0],
            |n| n,
            false,
        );
    }
}

#[test]
fn rotate_carries_every_face_in_place() {
    // 90° about +Z through the origin: (x, y, z) → (−y, x, z).
    let mut backend = KernelBackend::default();
    let before = tracked_cuboid(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let after = backend
        .rotate(&before, DVec3::ZERO, DVec3::Z, std::f64::consts::FRAC_PI_2)
        .unwrap();

    assert_eq!(after.elements, before.elements);
    let map = |v: [f64; 3]| [-v[1], v[0], v[2]];
    for i in 0..6 {
        assert_face_carried(&before, &after, i, map, map, false);
    }
}

#[test]
fn mirror_carries_every_face_in_place() {
    // Across the z=0 plane: (x, y, z) → (x, y, −z); outward normals mirror too.
    let mut backend = KernelBackend::default();
    let before = tracked_cuboid(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let after = backend.mirror(&before, DVec3::ZERO, DVec3::Z).unwrap();

    assert_eq!(after.elements, before.elements);
    let map = |v: [f64; 3]| [v[0], v[1], -v[2]];
    for i in 0..6 {
        assert_face_carried(&before, &after, i, map, map, true);
    }
}

#[test]
fn scale_carries_every_face_in_place_despite_gtransform_rebuilding() {
    // THE risky transform: non-uniform scale is a GTransform that rebuilds
    // geometry rather than relocating shared topology. (2, 3, 4) about the
    // origin: points map per-axis; an axis-aligned outward normal keeps its
    // axis.
    let mut backend = KernelBackend::default();
    let before = tracked_cuboid(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let after = backend
        .scale(&before, DVec3::new(2.0, 3.0, 4.0), DVec3::ZERO)
        .unwrap();

    assert_eq!(after.elements, before.elements);
    for i in 0..6 {
        assert_face_carried(
            &before,
            &after,
            i,
            |p| [p[0] * 2.0, p[1] * 3.0, p[2] * 4.0],
            |n| n, // box faces are axis-aligned: direction survives per-axis scaling
            false,
        );
    }
    // Edges too: the 20/10/5 cuboid's distinct lengths scale to 40/30/20 per
    // index — a permutation of same-direction edges would show up here.
    for i in 0..12 {
        let b = before.edge_length(i).unwrap();
        let a = after.edge_length(i).unwrap();
        let expected = match b {
            l if (l - 20.0).abs() < 1e-9 => 40.0, // x edges × 2
            l if (l - 10.0).abs() < 1e-9 => 30.0, // y edges × 3
            l if (l - 5.0).abs() < 1e-9 => 20.0,  // z edges × 4
            other => panic!("unexpected input edge length {other}"),
        };
        assert!(
            (a - expected).abs() < 1e-7,
            "edge {i}: length {b} should map to {expected}, got {a}"
        );
    }
}

#[test]
fn names_scope_by_feature_so_two_boxes_never_alias() {
    let mut backend = KernelBackend::default();
    let a = tracked_cuboid(&mut backend, 1);
    let b = tracked_cuboid(&mut backend, 2);
    assert_ne!(
        a.elements.face(0),
        b.elements.face(0),
        "same geometry, different origin — the names must differ"
    );
}
