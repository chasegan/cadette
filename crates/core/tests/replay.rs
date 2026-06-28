//! End-to-end tests for the parametric replay engine — with no kernel.
//!
//! A `Recording` backend renders each operation to a symbolic string, so we can
//! assert the exact expression the history reduces to. This is the payoff of
//! the data-first design: the entire parametric model is verifiable in plain
//! Rust, independent of OCCT.

use rmf_core::{
    regenerate, BooleanOp, Document, FeatureId, FeatureKind, GeometryBackend, Profile, RegenError,
    SketchPlane, DVec3,
};

/// A backend whose "body" is a symbolic string describing how it was built.
#[derive(Default)]
struct Recording;

impl GeometryBackend for Recording {
    type Body = String;
    type Error = String;

    fn make_box(&mut self, size: DVec3) -> Result<String, String> {
        Ok(format!("box({:.0},{:.0},{:.0})", size.x, size.y, size.z))
    }
    fn make_cylinder(&mut self, radius: f64, height: f64) -> Result<String, String> {
        Ok(format!("cyl({radius:.0},{height:.0})"))
    }
    fn make_sphere(&mut self, radius: f64) -> Result<String, String> {
        Ok(format!("sphere({radius:.0})"))
    }
    fn sketch(&mut self, plane: SketchPlane, profile: Profile) -> Result<String, String> {
        Ok(format!("sketch({},{})", plane.label(), profile.type_name()))
    }
    fn extrude(&mut self, profile: &String, distance: f64) -> Result<String, String> {
        Ok(format!("extrude({profile},{distance:.0})"))
    }
    fn sketch_loop(
        &mut self,
        plane: rmf_core::SketchPlane,
        points: &[[f64; 2]],
    ) -> Result<String, String> {
        Ok(format!("loop({},{}pts)", plane.label(), points.len()))
    }
    fn translate(&mut self, body: &String, offset: DVec3) -> Result<String, String> {
        Ok(format!("xlat({body},{:.0},{:.0},{:.0})", offset.x, offset.y, offset.z))
    }
    fn boolean(&mut self, op: BooleanOp, target: &String, tool: &String) -> Result<String, String> {
        let name = match op {
            BooleanOp::Union => "union",
            BooleanOp::Subtract => "sub",
            BooleanOp::Intersect => "isect",
        };
        Ok(format!("{name}({target},{tool})"))
    }
    fn fillet_all(&mut self, body: &String, radius: f64) -> Result<String, String> {
        if radius <= 0.0 {
            return Err(format!("infeasible fillet radius {radius}"));
        }
        Ok(format!("fillet({body},{radius:.0})"))
    }
    fn fillet_edges(
        &mut self,
        body: &String,
        anchors: &[rmf_core::EdgeAnchor],
        radius: f64,
    ) -> Result<String, String> {
        if radius <= 0.0 {
            return Err(format!("infeasible fillet radius {radius}"));
        }
        Ok(format!("fillet_edges({body},{},{radius:.0})", anchors.len()))
    }
    fn push_pull(
        &mut self,
        body: &String,
        _anchor: rmf_core::FaceAnchor,
        distance: f64,
    ) -> Result<String, String> {
        Ok(format!("pushpull({body},{distance:.0})"))
    }
    fn rotate(
        &mut self,
        body: &String,
        _center: DVec3,
        _axis: DVec3,
        angle: f64,
    ) -> Result<String, String> {
        Ok(format!("rotate({body},{angle:.2})"))
    }
    fn scale(&mut self, body: &String, factors: DVec3, _anchor: DVec3) -> Result<String, String> {
        Ok(format!("scale({body},{:.1},{:.1},{:.1})", factors.x, factors.y, factors.z))
    }
    fn revolve(&mut self, profile: &String, _axis: DVec3, angle: f64) -> Result<String, String> {
        Ok(format!("revolve({profile},{angle:.2})"))
    }
    fn mirror(&mut self, body: &String, _origin: DVec3, normal: DVec3) -> Result<String, String> {
        Ok(format!("mirror({body},{:.0},{:.0},{:.0})", normal.x, normal.y, normal.z))
    }
    fn compound(&mut self, members: &[&String]) -> Result<String, String> {
        let parts: Vec<&str> = members.iter().map(|m| m.as_str()).collect();
        Ok(format!("group({})", parts.join(",")))
    }
}

/// The canonical Phase-0 part, as parametric data: a filleted box with a bored
/// hole. Returns the document and the four feature ids [box, fillet, cyl, cut].
fn sample() -> (Document, [FeatureId; 4]) {
    let mut doc = Document::new("part");
    let b = doc.add(
        "Box",
        FeatureKind::Box {
            size: DVec3::new(40.0, 40.0, 40.0),
        },
    );
    let f = doc.add("Fillet edges", FeatureKind::FilletAll { source: b, radius: 4.0 });
    let c = doc.add("Drill", FeatureKind::Cylinder { radius: 6.0, height: 60.0 });
    let cut = doc.add(
        "Bore hole",
        FeatureKind::Boolean {
            op: BooleanOp::Subtract,
            target: f,
            tool: c,
        },
    );
    (doc, [b, f, c, cut])
}

#[test]
fn full_replay_reduces_to_one_visible_body() {
    let (doc, [_b, _f, _c, cut]) = sample();
    let regen = regenerate(&doc, &mut Recording);

    assert!(regen.is_ok(), "no feature should fail");
    assert_eq!(regen.visible(), &[cut], "only the final cut is visible");
    assert_eq!(
        regen.body(cut).unwrap(),
        "sub(fillet(box(40,40,40),4),cyl(6,60))",
        "history reduces to the expected nested expression"
    );
}

#[test]
fn sketch_extrude_reduces_to_a_prism() {
    let mut doc = Document::new("prism");
    let s = doc.add(
        "Sketch",
        FeatureKind::Sketch {
            plane: SketchPlane::Xy,
            profile: Profile::Rectangle {
                width: 30.0,
                height: 20.0,
            },
        },
    );
    let ext = doc.add("Extrude", FeatureKind::Extrude { source: s, distance: 15.0 });
    let regen = regenerate(&doc, &mut Recording);

    assert!(regen.is_ok());
    // The sketch is consumed by the extrude; only the solid is visible.
    assert_eq!(regen.visible(), &[ext]);
    assert_eq!(
        regen.body(ext).unwrap(),
        "extrude(sketch(XY,Rectangle),15)"
    );
}

#[test]
fn deleting_a_feature_heals_its_dependents() {
    // box -> fillet(box) -> cut(fillet, cyl). Deleting the fillet should rewire
    // the cut's target back to the box, not orphan it.
    let (mut doc, [b, f, _c, cut]) = sample();
    doc.history.remove(f);

    match doc.history.get(cut).unwrap().kind {
        FeatureKind::Boolean { target, .. } => assert_eq!(target, b, "cut rewired to the box"),
        _ => panic!("cut should still be a boolean"),
    }
    assert!(doc.history.validate().is_ok());
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "no dangling references after a healed delete");
}

#[test]
fn history_validates_clean() {
    let (doc, _) = sample();
    assert!(doc.history.validate().is_ok());
}

#[test]
fn rollback_shows_earlier_state_with_unconsumed_bodies() {
    let (mut doc, [_b, f, c, _cut]) = sample();
    // Roll the bar back to before the boolean: box, fillet, cylinder are active.
    doc.set_rollback(3);
    let regen = regenerate(&doc, &mut Recording);

    assert!(regen.is_ok());
    // The box was consumed by the fillet; the fillet and the cylinder remain.
    assert_eq!(regen.visible(), &[f, c]);
}

#[test]
fn suppressing_an_input_breaks_only_its_dependents() {
    let (mut doc, [b, f, c, cut]) = sample();
    doc.history.set_suppressed(f, true);
    let regen = regenerate(&doc, &mut Recording);

    assert!(!regen.is_ok());
    let errors = regen.errors();
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        errors[0],
        RegenError::MissingInput { feature, input } if feature == cut && input == f
    ));
    // The fillet was skipped, so the box is no longer consumed; the failed cut
    // consumes nothing. Box and cylinder remain visible.
    assert_eq!(regen.visible(), &[b, c]);
}

#[test]
fn editing_a_parameter_propagates_downstream() {
    let (mut doc, [b, _f, _c, cut]) = sample();
    doc.history.get_mut(b).unwrap().kind = FeatureKind::Box {
        size: DVec3::new(10.0, 20.0, 30.0),
    };
    let regen = regenerate(&doc, &mut Recording);

    assert_eq!(
        regen.body(cut).unwrap(),
        "sub(fillet(box(10,20,30),4),cyl(6,60))",
        "the edited box size flows through fillet and cut"
    );
}

#[test]
fn reorder_rejects_forward_reference_and_leaves_history_unchanged() {
    let (mut doc, [_b, _f, _c, cut]) = sample();
    // Moving the cut before its inputs would make references point forward.
    let err = doc.history.reorder(cut, 0).unwrap_err();
    assert!(!err.is_empty());
    assert_eq!(
        doc.history.features().last().unwrap().id,
        cut,
        "a rejected reorder must not mutate history"
    );
}

#[test]
fn reorder_independent_feature_is_allowed() {
    let (mut doc, [_b, _f, c, _cut]) = sample();
    // The cylinder has no inputs and nothing depends on it before the cut, so
    // moving it to the front keeps every reference backward-pointing.
    assert!(doc.history.reorder(c, 0).is_ok());
    assert!(doc.history.validate().is_ok());
    assert_eq!(doc.history.features()[0].id, c);
    // Geometry is unaffected.
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok());
}

#[test]
fn backend_failure_is_recorded_and_isolated() {
    let mut doc = Document::new("bad-fillet");
    let b = doc.add(
        "Box",
        FeatureKind::Box {
            size: DVec3::new(5.0, 5.0, 5.0),
        },
    );
    let bad = doc.add("Fillet", FeatureKind::FilletAll { source: b, radius: 0.0 });
    let regen = regenerate(&doc, &mut Recording);

    assert!(!regen.is_ok());
    assert!(matches!(
        regen.errors()[0],
        RegenError::Backend { feature, .. } if feature == bad
    ));
    // The fillet failed, so the box was never consumed and stays visible.
    assert_eq!(regen.visible(), &[b]);
}

#[test]
fn clone_subtree_duplicates_a_body_independently() {
    // A two-feature body: a Box moved by a Translate.
    let mut doc = Document::new("dup");
    let a = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
    let b = doc.add(
        "Move",
        FeatureKind::Translate { source: a, offset: DVec3::new(5.0, 0.0, 0.0) },
    );

    let b2 = doc.duplicate(b).expect("clone the subtree");
    assert_ne!(b2, b, "fresh tip id");
    assert_eq!(doc.history.len(), 4, "Box' + Move' appended");

    // The clone references its OWN copy of the box, not the original.
    let a2 = doc.history.get(b2).unwrap().kind.inputs()[0];
    assert_ne!(a2, a, "clone rewired to its own Box copy");

    // Editing the original box leaves the clone untouched (true independence).
    if let Some(FeatureKind::Box { size }) = doc.history.get_mut(a).map(|f| &mut f.kind) {
        size.x = 99.0;
    }
    let unaffected = matches!(
        &doc.history.get(a2).unwrap().kind,
        FeatureKind::Box { size } if (size.x - 10.0).abs() < 1e-9
    );
    assert!(unaffected, "clone's box is independent of the original");

    // Both bodies build, and every reference still points backward.
    assert!(doc.history.validate().is_ok());
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok());
    assert_eq!(regen.visible(), &[b, b2], "two independent visible bodies");
}

#[test]
fn grouping_merges_lanes_and_ungroup_splits_them() {
    let mut doc = Document::new("grp");
    let a = doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
    let b = doc.add("B", FeatureKind::Box { size: DVec3::splat(10.0) });
    let g = doc.add("Group", FeatureKind::Group { members: vec![a, b] });

    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok());
    // Two lanes merge into the one group body (a compound, not a fuse).
    assert_eq!(regen.visible(), &[g], "group is the single visible body");
    assert_eq!(
        regen.body(g).unwrap(),
        "group(box(10,10,10),box(10,10,10))"
    );

    // Ungroup = delete the Group node; its members become visible again.
    doc.history.remove(g);
    assert!(doc.history.validate().is_ok());
    let regen = regenerate(&doc, &mut Recording);
    assert_eq!(regen.visible(), &[a, b], "members independently visible after ungroup");
}
