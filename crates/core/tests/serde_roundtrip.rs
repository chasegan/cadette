//! Save-format round-trip: a [`Document`] exercising every serialization-
//! sensitive corner must survive JSON serialize → deserialize unchanged, and
//! [`Document::sanitize`] must repair the invariants a corrupt or hand-edited
//! `.cdt` file can violate. The app's `.cdt` archive stores exactly this JSON.

use cdt_core::{
    BooleanOp, Constraint, Document, DVec3, EdgeAnchor, FaceAnchor, FeatureKind, Profile,
    Sketch2d, SketchPlane,
};

/// A constraint sketch with lines, a bezier, a circle, and several constraint
/// kinds — the richest serialized payload in the format.
fn constraint_sketch() -> Sketch2d {
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(20.0, 0.0);
    let p2 = s.add_point(20.0, 10.0);
    let p3 = s.add_point(0.0, 10.0);
    let bottom = s.add_line(p0, p1);
    let right = s.add_line(p1, p2);
    s.add_line(p2, p3);
    // Close the loop with a curved left edge.
    s.add_bezier(p3, p0, [-3.0, 7.0], [-3.0, 3.0]);
    let center = s.add_point(10.0, 5.0);
    let c = s.add_circle(center, 2.5);
    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Horizontal(bottom));
    s.add_constraint(Constraint::Vertical(right));
    s.add_constraint(Constraint::Distance(p0, p1, 20.0));
    s.add_constraint(Constraint::Radius(c, 2.5));
    s
}

/// A document touching primitives, both sketch kinds, and a spread of
/// operations (single-input, two-input boolean, anchors, group).
fn sample_document() -> Document {
    let mut doc = Document::new("roundtrip");
    let bx = doc.add("Box 1", FeatureKind::Box { size: DVec3::new(10.0, 20.0, 30.0) });
    let cyl = doc.add("Cylinder 1", FeatureKind::Cylinder { radius: 4.0, height: 12.0 });
    doc.add(
        "Sketch 1",
        FeatureKind::Sketch {
            plane: SketchPlane::Xz,
            profile: Profile::Rectangle { width: 8.0, height: 6.0 },
        },
    );
    let csk = doc.add(
        "Sketch 2",
        FeatureKind::ConstraintSketch {
            plane: SketchPlane::from_frame(DVec3::new(1.0, 2.0, 3.0), DVec3::X, DVec3::Z),
            sketch: constraint_sketch(),
        },
    );
    let pad = doc.add("Extrude 1", FeatureKind::Extrude { source: csk, distance: 5.0 });
    let moved = doc.add("Translate 1", FeatureKind::Translate {
        source: cyl,
        offset: DVec3::new(0.0, 0.0, 2.5),
    });
    let fused = doc.add("Boolean 1", FeatureKind::Boolean {
        op: BooleanOp::Union,
        target: bx,
        tool: moved,
    });
    doc.add("Fillet 1", FeatureKind::Fillet {
        source: fused,
        edges: vec![EdgeAnchor { point: DVec3::new(5.0, 0.0, 30.0) }],
        radius: 1.5,
    });
    doc.add("Push/Pull 1", FeatureKind::PushPull {
        source: pad,
        anchor: FaceAnchor { point: DVec3::new(1.0, 2.0, 8.0), normal: DVec3::Z },
        distance: -1.0,
    });
    let bevel = doc.add("Chamfer 1", FeatureKind::Chamfer {
        source: fused,
        edges: vec![EdgeAnchor { point: DVec3::new(0.0, 10.0, 30.0) }],
        distance: 0.5,
    });
    // `open` is skipped when empty and defaulted when absent: both forms.
    let hollow = doc.add("Shell 1", FeatureKind::Shell { source: bevel, thickness: 1.0, open: Vec::new() });
    doc.add("Shell 2", FeatureKind::Shell {
        source: hollow,
        thickness: 1.0,
        open: vec![FaceAnchor { point: DVec3::new(5.0, 10.0, 30.0), normal: DVec3::Z }],
    });
    doc.add("Group 1", FeatureKind::Group { members: vec![fused, pad] });
    // An Import names its asset; the bytes live outside document.json.
    let asset = doc.assets.insert("servo.step", b"ISO-10303-21;".to_vec());
    doc.add("servo", FeatureKind::Import { asset });
    doc
}

#[test]
fn document_round_trips_through_json() {
    let mut doc = sample_document();
    // A rolled-back bar is part of the saved state; make it non-trivial.
    doc.set_rollback(doc.history.len() - 2);

    let json = serde_json::to_string_pretty(&doc).expect("serialize");
    let loaded: Document = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(loaded.name, doc.name);
    assert_eq!(loaded.units, doc.units);
    assert_eq!(loaded.rollback(), doc.rollback());
    assert_eq!(loaded.history.features(), doc.history.features());
    loaded.history.validate().expect("references still backward-pointing");

    // Full fidelity, including private fields (next_id): the re-serialized
    // JSON is byte-identical to the original.
    let rejson = serde_json::to_string_pretty(&loaded).expect("re-serialize");
    assert_eq!(rejson, json, "no state lost or altered in the round trip");
}

#[test]
fn loaded_document_keeps_allocating_fresh_ids() {
    // After a round trip, the id allocator must resume past the loaded ids.
    let doc = sample_document();
    let json = serde_json::to_string(&doc).expect("serialize");
    let mut loaded: Document = serde_json::from_str(&json).expect("deserialize");
    assert!(!loaded.sanitize(), "a faithfully saved document needs no repair");
    let ids: Vec<_> = loaded.history.features().iter().map(|f| f.id).collect();
    let fresh = loaded.add("New", FeatureKind::Sphere { radius: 1.0 });
    assert!(!ids.contains(&fresh), "new feature gets an unused id");
}

#[test]
fn sanitize_repairs_a_corrupt_file() {
    // Tamper with the serialized form the way a hand edit or corruption
    // would: id allocator reset behind the ids, rollback beyond the history.
    let doc = sample_document();
    let mut value = serde_json::to_value(&doc).expect("serialize");
    value["history"]["next_id"] = 1.into();
    value["rollback"] = 9999.into();
    let mut loaded: Document = serde_json::from_value(value).expect("deserialize");

    assert!(loaded.sanitize(), "the corruption is detected and fixed");
    assert_eq!(loaded.rollback(), loaded.history.len(), "bar clamped into range");

    let ids: Vec<_> = loaded.history.features().iter().map(|f| f.id).collect();
    let fresh = loaded.add("New", FeatureKind::Sphere { radius: 1.0 });
    assert!(!ids.contains(&fresh), "repair prevents id collisions on the next add");
    assert!(!loaded.sanitize(), "a repaired document is consistent");
}

#[test]
fn a_hand_written_file_may_omit_the_allocator_and_the_rollback_bar() {
    // An external writer (a script, an agent) shouldn't have to know about
    // `next_id` or `rollback`: both are reconstructible.
    let json = r#"{
        "name": "hand",
        "units": "Millimeter",
        "history": { "features": [
            { "id": 3, "name": "Box", "kind": { "Box": { "size": [1.0, 2.0, 3.0] } }, "suppressed": false },
            { "id": 5, "name": "Move", "kind": { "Translate": { "source": 3, "offset": [1.0, 0.0, 0.0] } }, "suppressed": false }
        ] }
    }"#;
    let mut doc: Document = serde_json::from_str(json).expect("loads without next_id/rollback");
    doc.sanitize();
    assert_eq!(doc.rollback(), 2, "an absent bar means the tip");
    assert_eq!(doc.active_features().len(), 2);
    let fresh = doc.add("New", FeatureKind::Sphere { radius: 1.0 });
    assert!(fresh.0 > 5, "the allocator resumed past the hand-picked ids: {fresh:?}");
    doc.history.validate().expect("valid");
}

#[test]
fn validate_rejects_duplicate_ids() {
    let doc = sample_document();
    let mut value = serde_json::to_value(&doc).expect("serialize");
    let first = value["history"]["features"][0]["id"].clone();
    value["history"]["features"][1]["id"] = first;
    let loaded: Document = serde_json::from_value(value).expect("deserialize");
    let errors = loaded.history.validate().expect_err("duplicate id is an error");
    assert!(
        errors.iter().any(|e| matches!(e, cdt_core::DependencyError::DuplicateId { .. })),
        "{errors:?}"
    );
}
