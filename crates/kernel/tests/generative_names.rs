//! Slice 3 of element maps: generative ops (extrude/revolve/sweep) name their
//! output from the profile.
//!
//! The rules under test (see `ElementMap::compose_generated`):
//! - the profile face's START image keeps its name verbatim (it IS the
//!   sketch face); the far cap becomes `Cap { parent: face }`;
//! - each profile edge sweeps a `Swept { parent: edge }` side wall;
//! - start rim edges stay verbatim, end rims become `Cap { parent: edge }`;
//! - lateral edges (from untracked vertices) are `Intersection`s of the named
//!   faces they separate;
//! - names survive PARAMETER edits — the headline: a top-rim reference
//!   recorded once resolves after the profile is resized, exactly where a
//!   coordinate anchor would grab the wrong edge.

use cdt_core::{DVec3, FeatureId, GeometryBackend, Profile, SketchPlane};
use cdt_kernel::{Body, ElementName, KernelBackend};

/// A w×h rectangle sketch on XY, minted under feature `id`.
fn rect_sketch(backend: &mut KernelBackend, id: u64, w: f64, h: f64) -> Body {
    backend.begin_feature(FeatureId(id));
    backend
        .sketch(SketchPlane::Xy, Profile::Rectangle { width: w, height: h })
        .unwrap()
}

fn count_matching(names: &[Option<ElementName>], pred: impl Fn(&ElementName) -> bool) -> usize {
    names.iter().flatten().filter(|n| pred(n)).count()
}

#[test]
fn an_extrusion_names_every_face_and_edge() {
    let mut backend = KernelBackend::default();
    let sketch = rect_sketch(&mut backend, 1, 30.0, 20.0);
    let face_name = sketch.elements.face(0).unwrap().clone();

    backend.begin_feature(FeatureId(2));
    let solid = backend.extrude(&sketch, 15.0).unwrap();

    // 6 faces: the profile-side cap keeps the sketch face's name VERBATIM,
    // the far cap is Cap{face}, the four walls are Swept{edge}.
    assert_eq!(solid.elements.faces.len(), 6);
    assert!(
        solid.elements.find_face(&face_name).is_some(),
        "the sketch face IS the near cap — name unchanged"
    );
    let far_cap = ElementName::Cap { parent: Box::new(face_name) };
    assert!(solid.elements.find_face(&far_cap).is_some(), "far cap named");
    let walls = count_matching(&solid.elements.faces, |n| {
        matches!(n, ElementName::Swept { .. })
    });
    assert_eq!(walls, 4, "each profile edge sweeps one named wall");

    // 12 edges: 4 verbatim rims, 4 Cap rims, 4 lateral Intersections — full
    // coverage, no unnamed elements anywhere.
    assert_eq!(solid.elements.edges.len(), 12);
    for i in 0..4 {
        let rim = sketch.elements.edge(i).unwrap().clone();
        assert!(solid.elements.find_edge(&rim).is_some(), "near rim {i} verbatim");
        let far = ElementName::Cap { parent: Box::new(rim) };
        assert!(solid.elements.find_edge(&far).is_some(), "far rim {i} named");
    }
    let lateral = count_matching(&solid.elements.edges, |n| {
        matches!(n, ElementName::Intersection { .. })
    });
    assert_eq!(lateral, 4, "the four vertical edges are named by adjacency");
    assert!(
        solid.elements.faces.iter().all(Option::is_some)
            && solid.elements.edges.iter().all(Option::is_some),
        "an extrusion is fully named"
    );
}

#[test]
fn a_top_rim_reference_survives_resizing_the_profile() {
    // THE headline for generative naming: record the far-rim edge's name once
    // (as a fillet reference would), then rebuild with a RESIZED profile —
    // the name still resolves to the corresponding edge. A coordinate anchor
    // at the old rim position would miss, or worse, grab a different edge.
    let mut backend = KernelBackend::default();
    let sketch = rect_sketch(&mut backend, 1, 30.0, 20.0);
    let rim_name = ElementName::Cap {
        parent: Box::new(sketch.elements.edge(0).unwrap().clone()),
    };

    for (w, h, depth) in [(30.0, 20.0, 15.0), (45.0, 10.0, 15.0), (30.0, 20.0, 40.0)] {
        let sketch = rect_sketch(&mut backend, 1, w, h);
        backend.begin_feature(FeatureId(2));
        let solid = backend.extrude(&sketch, depth).unwrap();
        let i = solid
            .elements
            .find_edge(&rim_name)
            .unwrap_or_else(|| panic!("rim resolves at {w}x{h}x{depth}"));
        // The resolved edge really is on the far cap: both its bounding faces
        // meet at height `depth` — cheap sanity via the edge's centroid.
        let [_, _, z] = solid
            .solid
            .elem_centroid(cdt_kernel::ElemKind::Edge, i)
            .unwrap();
        assert!(
            (z - depth).abs() < 1e-9,
            "rim edge should sit at z={depth}, got {z}"
        );
    }
}

#[test]
fn a_partial_revolve_names_caps_and_walls() {
    // A rectangle away from the axis edge, revolved 270° about its own left
    // edge: start face verbatim, end face Cap, side walls Swept.
    let mut backend = KernelBackend::default();
    let sketch = rect_sketch(&mut backend, 1, 10.0, 6.0);
    let face_name = sketch.elements.face(0).unwrap().clone();

    backend.begin_feature(FeatureId(2));
    // The rectangle spans x ∈ [-5, 5]; its left edge (x = -5) is the axis.
    let solid = backend
        .revolve(&sketch, DVec3::new(-5.0, 0.0, 0.0), 1.5 * std::f64::consts::PI)
        .unwrap();

    assert!(
        solid.elements.find_face(&face_name).is_some(),
        "the profile face is the start cap, name verbatim"
    );
    let end_cap = ElementName::Cap { parent: Box::new(face_name) };
    assert!(solid.elements.find_face(&end_cap).is_some(), "end cap named");
    let walls = count_matching(&solid.elements.faces, |n| {
        matches!(n, ElementName::Swept { .. })
    });
    assert!(
        walls >= 3,
        "the non-axis profile edges sweep named walls (got {walls})"
    );
}

#[test]
fn a_full_revolve_names_the_wall_and_admits_its_gaps() {
    // KNOWN OCCT GAP: for a FULL (2π) revolution MakeRevol only reports
    // Generated() for profile edges that don't touch the axis — the outer
    // edge's cylinder wall is named; the disk faces (from axis-touching
    // edges) come back unreported and stay honestly unnamed (anchor
    // fallback). Partial revolves report everything (test above). Also: the
    // seam edge is BOTH the start and end image of the outer profile edge —
    // first-write-wins keeps its verbatim name (the seam IS that edge).
    let mut backend = KernelBackend::default();
    let sketch = rect_sketch(&mut backend, 1, 10.0, 6.0);
    let outer_edge_names: Vec<ElementName> =
        (0..4).map(|i| sketch.elements.edge(i).unwrap().clone()).collect();

    backend.begin_feature(FeatureId(2));
    let solid = backend
        .revolve(&sketch, DVec3::new(-5.0, 0.0, 0.0), 2.0 * std::f64::consts::PI)
        .unwrap();

    let caps = count_matching(&solid.elements.faces, |n| {
        matches!(n, ElementName::Cap { .. })
    });
    assert_eq!(caps, 0, "a closed revolution has no caps");
    let walls = count_matching(&solid.elements.faces, |n| {
        matches!(n, ElementName::Swept { .. })
    });
    assert_eq!(walls, 1, "the outer edge's cylinder wall is named");
    // The seam keeps the profile edge's name verbatim, not a Cap.
    assert!(
        outer_edge_names
            .iter()
            .any(|n| solid.elements.find_edge(n).is_some()),
        "the seam edge carries the outer profile edge's verbatim name"
    );
    assert_eq!(
        count_matching(&solid.elements.edges, |n| matches!(n, ElementName::Cap { .. })),
        0,
        "no rim on a closed revolution is an end-cap image"
    );
}

#[test]
fn a_straight_sweep_names_its_walls() {
    use cdt_core::ProfileElem;
    // A circle profile swept along a straight open path (no healing needed on
    // a straight sweep) — the tube wall must carry the profile edge's name.
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(1));
    let profile = backend
        .sketch(SketchPlane::Xz, Profile::Circle { radius: 3.0 })
        .unwrap();

    // An open two-point path on XY: a straight run along +Y... the sweep API
    // takes resolved path elements directly.
    let path = [ProfileElem::Line { a: [0.0, 0.0], b: [0.0, 25.0] }];
    backend.begin_feature(FeatureId(2));
    let solid = backend.sweep(&profile, SketchPlane::Xy, &path).unwrap();

    let walls = count_matching(&solid.elements.faces, |n| {
        matches!(n, ElementName::Swept { .. })
    });
    assert!(
        walls >= 1,
        "an unhealed sweep names at least the swept wall(s)"
    );
}
