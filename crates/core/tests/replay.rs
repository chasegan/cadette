//! End-to-end tests for the parametric replay engine — with no kernel.
//!
//! A `Recording` backend renders each operation to a symbolic string, so we can
//! assert the exact expression the history reduces to. This is the payoff of
//! the data-first design: the entire parametric model is verifiable in plain
//! Rust, independent of OCCT.

use cdt_core::{
    regenerate, BooleanOp, Document, EdgeAnchor, FaceAnchor, FeatureId, FeatureKind, GeometryBackend, ImportFormat, LengthUnit,
    MissingReason, Profile, RegenError, SketchPlane, DVec3,
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
    fn import(&mut self, format: ImportFormat, data: &[u8], scale: f64) -> Result<String, String> {
        if data.starts_with(b"bad") {
            return Err(format!("not a {} file", format.label()));
        }
        Ok(format!("{}({}b,{scale})", format.label().to_lowercase(), data.len()))
    }
    fn sketch(&mut self, plane: SketchPlane, profile: Profile) -> Result<String, String> {
        Ok(format!("sketch({},{})", plane.label(), profile.type_name()))
    }
    fn extrude(&mut self, profile: &String, distance: f64) -> Result<String, String> {
        Ok(format!("extrude({profile},{distance:.0})"))
    }
    fn sweep(
        &mut self,
        profile: &String,
        _plane: SketchPlane,
        path: &[cdt_core::ProfileElem],
    ) -> Result<String, String> {
        Ok(format!("sweep({profile},{})", path.len()))
    }
    fn sketch_profile(
        &mut self,
        plane: cdt_core::SketchPlane,
        elements: &[cdt_core::ProfileElem],
    ) -> Result<String, String> {
        let curves = elements
            .iter()
            .filter(|e| matches!(e, cdt_core::ProfileElem::Bezier { .. }))
            .count();
        Ok(format!("loop({},{}seg,{}bez)", plane.label(), elements.len(), curves))
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
        // A radius beyond any body stands in for a kernel refusal (regen
        // itself already rejects non-positive radii before the backend).
        if radius > 1e6 {
            return Err(format!("infeasible fillet radius {radius}"));
        }
        Ok(format!("fillet({body},{radius:.0})"))
    }
    fn fillet_edges(
        &mut self,
        body: &String,
        anchors: &[cdt_core::EdgeAnchor],
        radius: f64,
    ) -> Result<String, String> {
        if radius > 1e6 {
            return Err(format!("infeasible fillet radius {radius}"));
        }
        Ok(format!("fillet_edges({body},{},{radius:.0})", anchors.len()))
    }
    fn chamfer_all(&mut self, body: &String, distance: f64) -> Result<String, String> {
        if distance <= 0.0 {
            return Err(format!("infeasible chamfer distance {distance}"));
        }
        Ok(format!("chamfer({body},{distance:.0})"))
    }
    fn chamfer_edges(
        &mut self,
        body: &String,
        anchors: &[cdt_core::EdgeAnchor],
        distance: f64,
    ) -> Result<String, String> {
        if distance <= 0.0 {
            return Err(format!("infeasible chamfer distance {distance}"));
        }
        Ok(format!("chamfer({body},{}e,{distance:.0})", anchors.len()))
    }
    fn shell(
        &mut self,
        body: &String,
        thickness: f64,
        open: &[cdt_core::FaceAnchor],
    ) -> Result<String, String> {
        Ok(format!("shell({body},{thickness:.0},{}open)", open.len()))
    }
    fn push_pull(
        &mut self,
        body: &String,
        _anchor: cdt_core::FaceAnchor,
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
    fn sweep_helix(&mut self, profile: &String, helix: cdt_core::Helix) -> Result<String, String> {
        Ok(format!("coil({profile},r={},p={},l={})", helix.radius, helix.pitch, helix.length))
    }
    fn helix_sweep(&mut self, profile: &[cdt_core::ProfileElem], helix: cdt_core::Helix) -> Result<String, String> {
        Ok(format!("helix({} segs,r={},p={},l={})", profile.len(), helix.radius, helix.pitch, helix.length))
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
        RegenError::MissingInput { feature, input, reason: MissingReason::Suppressed }
            if feature == cut && input == f
    ));
    // The fillet was skipped, so the box is no longer consumed; the failed cut
    // consumes nothing. Box and cylinder remain visible.
    assert_eq!(regen.visible(), &[b, c]);
}

#[test]
fn missing_input_reasons_tell_cascades_from_structural_errors() {
    // A cascade: the fillet fails (an absurd radius is rejected by Recording),
    // so the boolean that consumes it is missing its input because it *failed*.
    let (mut doc, [_b, f, _c, cut]) = sample();
    doc.history.get_mut(f).unwrap().kind = FeatureKind::FilletAll { source: _b, radius: 1e9 };
    let regen = regenerate(&doc, &mut Recording);
    assert!(matches!(
        regen.errors()[0],
        RegenError::Backend { feature, .. } if feature == f
    ));
    assert!(matches!(
        regen.errors()[1],
        RegenError::MissingInput { feature, input, reason: MissingReason::Failed }
            if feature == cut && input == f
    ));

    // A forward reference (only reachable by hand-editing a file): the cut
    // names a tool that comes after it.
    let (mut doc, [_b, _f, c, cut]) = sample();
    assert!(doc.history.reorder(c, 3).is_err(), "reorder refuses to break the invariant");
    // So build the broken order directly through the serde surface.
    let mut json = serde_json::to_value(&doc).unwrap();
    let features = json["history"]["features"].as_array_mut().unwrap();
    let tool = features.remove(2); // the cylinder
    features.push(tool); // ...now after the cut
    let doc: Document = serde_json::from_value(json).unwrap();
    let regen = regenerate(&doc, &mut Recording);
    assert!(matches!(
        regen.errors()[0],
        RegenError::MissingInput { feature, input, reason: MissingReason::Later }
            if feature == cut && input == c
    ));

    // A deleted input: the referenced id is gone from history entirely.
    let (doc, [_b, _f, c, cut]) = sample();
    let mut json = serde_json::to_value(&doc).unwrap();
    json["history"]["features"].as_array_mut().unwrap().remove(2);
    let doc: Document = serde_json::from_value(json).unwrap();
    let regen = regenerate(&doc, &mut Recording);
    assert!(matches!(
        regen.errors()[0],
        RegenError::MissingInput { feature, input, reason: MissingReason::Deleted }
            if feature == cut && input == c
    ));
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
    let bad = doc.add("Fillet", FeatureKind::FilletAll { source: b, radius: 1e9 });
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
fn chamfers_and_shells_replay_as_their_own_kinds() {
    let mut doc = Document::new("cs");
    let block = doc.add("Block", FeatureKind::Box { size: DVec3::new(30.0, 20.0, 10.0) });
    let bevel = doc.add("Bevel", FeatureKind::ChamferAll { source: block, distance: 1.0 });
    let edge = doc.add(
        "Edge",
        FeatureKind::Chamfer {
            source: bevel,
            edges: vec![EdgeAnchor { point: DVec3::new(15.0, 0.0, 10.0) }],
            distance: 2.0,
        },
    );
    let hollow = doc.add(
        "Hollow",
        FeatureKind::Shell {
            source: edge,
            thickness: 2.0,
            open: vec![FaceAnchor { point: DVec3::new(15.0, 10.0, 10.0), normal: DVec3::Z }],
        },
    );
    assert!(doc.history.validate().is_ok());
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(regen.visible(), &[hollow], "each step consumes the one before");
    assert_eq!(
        regen.body(hollow).unwrap(),
        "shell(chamfer(chamfer(box(30,20,10),1),1e,2),2,1open)"
    );

    // A non-positive distance or wall is refused with a reason, before the
    // backend sees it.
    doc.history.get_mut(bevel).unwrap().kind = FeatureKind::ChamferAll { source: block, distance: 0.0 };
    let regen = regenerate(&doc, &mut Recording);
    assert!(
        regen.errors().iter().any(|e| matches!(e, RegenError::Invalid { reason, .. } if reason.contains("positive distance"))),
        "{:?}",
        regen.errors()
    );
    doc.history.get_mut(bevel).unwrap().kind = FeatureKind::ChamferAll { source: block, distance: 1.0 };
    // Anchored blends also need at least one edge, fillets and chamfers alike.
    for (kind, want) in [
        (FeatureKind::Chamfer { source: bevel, edges: Vec::new(), distance: 1.0 }, "chamfer needs at least one edge"),
        (FeatureKind::Fillet { source: bevel, edges: Vec::new(), radius: 1.0 }, "fillet needs at least one edge"),
        (FeatureKind::Fillet { source: bevel, edges: vec![EdgeAnchor { point: DVec3::ZERO }], radius: 0.0 }, "fillet needs a positive radius"),
        (FeatureKind::Chamfer { source: bevel, edges: vec![EdgeAnchor { point: DVec3::ZERO }], distance: f64::NAN }, "chamfer needs a positive distance"),
    ] {
        doc.history.get_mut(edge).unwrap().kind = kind;
        let regen = regenerate(&doc, &mut Recording);
        assert!(
            regen.errors().iter().any(|e| matches!(e, RegenError::Invalid { reason, .. } if reason.contains(want))),
            "{want}: {:?}",
            regen.errors()
        );
    }
    doc.history.get_mut(edge).unwrap().kind = FeatureKind::Chamfer {
        source: bevel,
        edges: vec![EdgeAnchor { point: DVec3::new(15.0, 0.0, 10.0) }],
        distance: 2.0,
    };
    doc.history.get_mut(hollow).unwrap().kind =
        FeatureKind::Shell { source: edge, thickness: -2.0, open: Vec::new() };
    let regen = regenerate(&doc, &mut Recording);
    assert!(
        regen.errors().iter().any(|e| matches!(e, RegenError::Invalid { reason, .. } if reason.contains("wall thickness"))),
        "{:?}",
        regen.errors()
    );
}

#[test]
fn a_pattern_repeats_its_seed_and_splices_into_a_cut() {
    use cdt_core::PatternLayout;
    let mut doc = Document::new("pat");
    let block = doc.add("Block", FeatureKind::Box { size: DVec3::new(100.0, 20.0, 10.0) });
    let pin = doc.add("Pin", FeatureKind::Cylinder { radius: 2.0, height: 10.0 });
    let hole = doc.add(
        "Hole",
        FeatureKind::Boolean { op: BooleanOp::Subtract, target: block, tool: pin },
    );
    // Splice a linear pattern onto the pin: the Subtract's tool becomes the
    // pattern, so every instance is cut.
    let pat = doc.insert_transform(
        "Pattern",
        FeatureKind::Pattern {
            source: pin,
            layout: PatternLayout::Linear { step: DVec3::new(20.0, 0.0, 0.0) },
            count: 3,
            skip: Vec::new(),
        },
    );
    assert!(doc.history.validate().is_ok());

    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(regen.visible(), &[hole], "the pattern is consumed by the cut");
    assert_eq!(
        regen.body(pat).unwrap(),
        "group(cyl(2,10),xlat(cyl(2,10),20,0,0),xlat(cyl(2,10),40,0,0))",
        "instance 0 is the seed in place; the rest step along +X"
    );
    assert_eq!(*regen.body(hole).unwrap(), format!("sub(box(100,20,10),{})", regen.body(pat).unwrap()));

    // A circular layout rotates by i × angle; a count of zero is an error, not a hang.
    doc.history.get_mut(pat).unwrap().kind = FeatureKind::Pattern {
        source: pin,
        layout: PatternLayout::Circular { center: DVec3::ZERO, axis: DVec3::Z, angle: 1.0 },
        count: 2,
        skip: Vec::new(),
    };
    let regen = regenerate(&doc, &mut Recording);
    assert_eq!(regen.body(pat).unwrap(), "group(cyl(2,10),rotate(cyl(2,10),1.00))");
    // Helical: each instance is turned AND lifted along the axis; skipped
    // instances (here the seed and #2) are left out, keeping the others' places.
    doc.history.get_mut(pat).unwrap().kind = FeatureKind::Pattern {
        source: pin,
        layout: PatternLayout::Helical { center: DVec3::ZERO, axis: DVec3::Z, angle: 0.5, rise: 3.0 },
        count: 4,
        skip: vec![0, 2, 99],
    };
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(
        regen.body(pat).unwrap(),
        "group(xlat(rotate(cyl(2,10),0.50),0,0,3),xlat(rotate(cyl(2,10),1.50),0,0,9))"
    );
    assert_eq!(cdt_core::pattern_instances(4, &[0, 2, 99]), vec![1, 3]);
    doc.history.get_mut(pat).unwrap().kind = FeatureKind::Pattern {
        source: pin,
        layout: PatternLayout::Linear { step: DVec3::X },
        count: 0,
        skip: Vec::new(),
    };
    let regen = regenerate(&doc, &mut Recording);
    assert!(!regen.is_ok(), "an empty pattern is reported");
}

#[test]
fn a_thread_cuts_a_helical_groove_from_its_body() {
    use cdt_core::{thread_profile, Helix};
    let mut doc = Document::new("thr");
    let shaft = doc.add("Shaft", FeatureKind::Cylinder { radius: 5.0, height: 20.0 });
    let helix = Helix {
        center: DVec3::ZERO,
        axis: DVec3::Z,
        radius: 5.0,
        pitch: 1.5,
        length: 20.0,
        left_handed: false,
    };
    let thread = doc.add(
        "Thread",
        FeatureKind::Thread { source: shaft, helix, depth: Helix::iso_depth(1.5), internal: false, starts: 1 },
    );
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(regen.visible(), &[thread], "the thread consumes the shaft");
    assert_eq!(regen.body(thread).unwrap(), "sub(cyl(5,20),helix(4 segs,r=5,p=1.5,l=20))");

    // The groove: four straight segments closing on themselves, centred on
    // v=0, base just outside the surface (external) and root `depth` inside.
    let groove = thread_profile(1.5, Helix::iso_depth(1.5), 5.0, false);
    assert_eq!(groove.len(), 4);
    for w in groove.windows(2) {
        assert_eq!(w[0].end(), w[1].start(), "segments chain");
    }
    assert_eq!(groove[3].end(), groove[0].start(), "the loop closes");
    let us: Vec<f64> = groove.iter().map(|e| e.start()[0]).collect();
    assert!(us.iter().cloned().fold(f64::MIN, f64::max) > 5.0, "base is past the surface: {us:?}");
    assert!((us.iter().cloned().fold(f64::MAX, f64::min) - (5.0 - Helix::iso_depth(1.5))).abs() < 1e-9);
    let vs: Vec<f64> = groove.iter().map(|e| e.start()[1]).collect();
    assert!(vs.iter().all(|v| v.abs() < 0.75), "narrower than the pitch: {vs:?}");
    // Internal threads cut outward instead.
    let groove = thread_profile(1.5, 0.9, 5.0, true);
    assert!(groove.iter().map(|e| e.start()[0]).fold(f64::MIN, f64::max) > 5.5);

    // ISO defaults: M6 is 1.0 mm coarse; M10 is 1.5. A hole is seen at its
    // tap-drill size: 8.5 mm is M10, 5.0 mm is M6, 4.2 mm is M5.
    assert_eq!(Helix::iso_pitch(6.0), 1.0);
    assert_eq!(Helix::iso_pitch(10.2), 1.5);
    assert_eq!(Helix::iso_pitch_for_minor(8.5), 1.5);
    assert_eq!(Helix::iso_pitch_for_minor(5.0), 1.0);
    assert_eq!(Helix::iso_pitch_for_minor(4.2), 0.8);

    // An internal thread whose groove would cross the axis is refused.
    let mut hole_doc = Document::new("hole");
    let tiny = hole_doc.add("Tiny", FeatureKind::Cylinder { radius: 1.0, height: 5.0 });
    hole_doc.add(
        "Bad",
        FeatureKind::Thread {
            source: tiny,
            helix: Helix { radius: 1.0, pitch: 60.0, length: 5.0, ..helix },
            depth: 0.5,
            internal: true,
            starts: 1,
        },
    );
    let regen = regenerate(&hole_doc, &mut Recording);
    assert!(regen.errors().iter().any(|e| matches!(e, cdt_core::RegenError::Invalid { reason, .. } if reason.contains("too large for this hole"))), "{:?}", regen.errors());

    // Bad parameters are reported, not swept.
    doc.history.get_mut(thread).unwrap().kind = FeatureKind::Thread {
        source: shaft,
        helix: Helix { pitch: 0.0, ..helix },
        depth: 0.9,
        internal: false,
        starts: 1,
    };
    assert!(!regenerate(&doc, &mut Recording).is_ok(), "zero pitch is an error");

    // A multi-start thread turns the groove about the axis and cuts the lot
    // as one compound tool — a 3-start is the seed plus two rotated copies.
    doc.history.get_mut(thread).unwrap().kind = FeatureKind::Thread {
        source: shaft,
        helix,
        depth: 0.9,
        internal: false,
        starts: 3,
    };
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(
        regen.body(thread).unwrap(),
        "sub(cyl(5,20),group(helix(4 segs,r=5,p=1.5,l=20),rotate(helix(4 segs,r=5,p=1.5,l=20),2.09),rotate(helix(4 segs,r=5,p=1.5,l=20),4.19)))",
        "the groove itself is sized for pitch/starts, but the helix is the full lead"
    );

    // A coil: a sketched circle swept along a helix.
    let mut doc = Document::new("coil");
    let ring = doc.add("Ring", FeatureKind::Sketch { plane: SketchPlane::Xz, profile: Profile::Circle { radius: 1.0 } });
    let coil = doc.add("Coil", FeatureKind::Sweep { profile: ring, path: cdt_core::SweepPath::Helix { helix } });
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(regen.visible(), &[coil]);
    assert!(regen.body(coil).unwrap().starts_with("coil(sketch("), "{}", regen.body(coil).unwrap());
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

/// A solver that "fails" every sketch with a constraint and passes the rest.
struct FailConstrained;

impl cdt_core::SketchSolver for FailConstrained {
    fn solve(&mut self, sketch: &mut cdt_core::Sketch2d) -> cdt_core::SolveStatus {
        if sketch.constraints.is_empty() {
            cdt_core::SolveStatus::Converged
        } else {
            cdt_core::SolveStatus::Contradictory { residual_norm: 1.0 }
        }
    }
}

#[test]
fn an_unsolvable_sketch_is_an_error_and_is_not_built() {
    use cdt_core::{regenerate_solved, Constraint, Sketch2d, SolveStatus};
    let mut sketch = Sketch2d::new();
    let a = sketch.add_point(0.0, 0.0);
    let b = sketch.add_point(10.0, 0.0);
    let c = sketch.add_point(10.0, 10.0);
    sketch.add_line(a, b);
    sketch.add_line(b, c);
    sketch.add_line(c, a);
    let mut doc = Document::new("sk");
    let s = doc.add(
        "Sketch",
        FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: sketch.clone() },
    );
    let e = doc.add("Extrude", FeatureKind::Extrude { source: s, distance: 5.0 });

    // No constraints: the solver passes and the sketch builds as before.
    let regen = regenerate_solved(&mut doc, &mut Recording, &mut FailConstrained);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(regen.visible(), &[e]);

    // Add a constraint the (fake) solver can't satisfy: the sketch is reported
    // and skipped, so the extrude is missing its input because it *failed*.
    if let Some(FeatureKind::ConstraintSketch { sketch, .. }) =
        doc.history.get_mut(s).map(|f| &mut f.kind)
    {
        sketch.add_constraint(Constraint::Fixed(a));
    }
    let regen = regenerate_solved(&mut doc, &mut Recording, &mut FailConstrained);
    assert_eq!(regen.errors().len(), 2, "{:?}", regen.errors());
    assert!(matches!(
        regen.errors()[0],
        RegenError::UnsolvedSketch { feature, status: SolveStatus::Contradictory { .. } }
            if feature == s
    ));
    assert!(matches!(
        regen.errors()[1],
        RegenError::MissingInput { feature, input, reason: MissingReason::Failed }
            if feature == e && input == s
    ));
    assert!(regen.visible().is_empty(), "nothing was built from the unsolved coordinates");

    // The cached engine agrees.
    let mut cache = cdt_core::RegenCache::new();
    let regen = cache.regenerate_solved(&mut doc, &mut Recording, &mut FailConstrained);
    assert_eq!(regen.errors().len(), 2);
}

#[test]
fn an_import_replays_its_asset_scaled_into_the_document_unit() {
    let mut doc = Document::new("imp");
    let asset = doc.assets.insert("servo.step", vec![0u8; 120]);
    let servo = doc.add("servo", FeatureKind::Import { asset: asset.clone() });
    let rounded = doc.add("Round", FeatureKind::FilletAll { source: servo, radius: 1.0 });
    let regen = regenerate(&doc, &mut Recording);
    assert!(regen.is_ok(), "{:?}", regen.errors());
    assert_eq!(regen.visible(), &[rounded]);
    assert_eq!(regen.body(rounded).unwrap(), "fillet(step(120b,1),1)", "millimetre documents take the file as is");

    // An inch document scales the millimetre file down into its unit.
    doc.units = LengthUnit::Inch;
    let regen = regenerate(&doc, &mut Recording);
    assert_eq!(regen.body(servo).unwrap(), &format!("step(120b,{})", 1.0 / 25.4));

    // A feature whose file is gone from the project fails by name, and its
    // dependents report the missing input rather than crashing.
    doc.assets.remove(&asset);
    let regen = regenerate(&doc, &mut Recording);
    assert!(
        regen.errors().iter().any(|e| matches!(e, RegenError::Invalid { feature, reason } if *feature == servo && reason.contains("missing from the project"))),
        "{:?}",
        regen.errors()
    );
    assert!(regen.body(rounded).is_none());

    // A file the kernel rejects is a backend error on the import itself.
    let bad = doc.assets.insert("bad.step", b"bad bytes".to_vec());
    doc.history.get_mut(servo).unwrap().kind = FeatureKind::Import { asset: bad };
    let regen = regenerate(&doc, &mut Recording);
    assert!(
        regen.errors().iter().any(|e| matches!(e, RegenError::Backend { feature, source } if *feature == servo && source == "not a STEP file")),
        "{:?}",
        regen.errors()
    );
}
