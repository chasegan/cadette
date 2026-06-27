//! Incremental regeneration: the cache must reuse unchanged features and rebuild
//! exactly the edited feature plus everything downstream — never the unchanged
//! upstream, and never stale geometry.

use rmf_core::{
    DVec3, Document, FeatureKind, GeometryBackend, Profile, RegenCache, SketchPlane,
};

/// A backend that counts how many operations it performs, so a test can assert
/// which features were rebuilt. Bodies are strings encoding their derivation.
#[derive(Default)]
struct Counting {
    ops: usize,
}

impl GeometryBackend for Counting {
    type Body = String;
    type Error = String;

    fn make_box(&mut self, size: DVec3) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("box({},{},{})", size.x, size.y, size.z))
    }
    fn make_cylinder(&mut self, r: f64, h: f64) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("cyl({r},{h})"))
    }
    fn make_sphere(&mut self, r: f64) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("sphere({r})"))
    }
    fn sketch(&mut self, _p: SketchPlane, _profile: Profile) -> Result<String, String> {
        self.ops += 1;
        Ok("sketch".into())
    }
    fn sketch_loop(&mut self, _p: SketchPlane, pts: &[[f64; 2]]) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("loop({})", pts.len()))
    }
    fn extrude(&mut self, b: &String, d: f64) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("extrude({b},{d})"))
    }
    fn translate(&mut self, b: &String, o: DVec3) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("move({b},{},{},{})", o.x, o.y, o.z))
    }
    fn boolean(
        &mut self,
        _op: rmf_core::BooleanOp,
        a: &String,
        b: &String,
    ) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("bool({a},{b})"))
    }
    fn fillet_all(&mut self, b: &String, r: f64) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("fillet({b},{r})"))
    }
    fn fillet_edges(
        &mut self,
        b: &String,
        e: &[rmf_core::EdgeAnchor],
        r: f64,
    ) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("filletE({b},{},{r})", e.len()))
    }
    fn push_pull(
        &mut self,
        b: &String,
        _a: rmf_core::FaceAnchor,
        d: f64,
    ) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("pp({b},{d})"))
    }
    fn rotate(
        &mut self,
        b: &String,
        _c: DVec3,
        _ax: DVec3,
        angle: f64,
    ) -> Result<String, String> {
        self.ops += 1;
        Ok(format!("rot({b},{angle})"))
    }
}

/// A 3-step chain: box -> move -> fillet. Returns (doc, [box, move, fillet]).
fn chain() -> (Document, [rmf_core::FeatureId; 3]) {
    let mut doc = Document::new("chain");
    let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
    let m = doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::X * 5.0 });
    let f = doc.add("Fillet", FeatureKind::FilletAll { source: m, radius: 1.0 });
    (doc, [b, m, f])
}

#[test]
fn unchanged_features_are_reused() {
    let (doc, _) = chain();
    let mut cache = RegenCache::new();
    let mut backend = Counting::default();

    let r = cache.regenerate(&doc, &mut backend);
    assert!(r.is_ok());
    assert_eq!(backend.ops, 3, "cold regen builds all three");

    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 3, "a second regen with no change rebuilds nothing");
}

#[test]
fn editing_a_feature_rebuilds_it_and_only_downstream() {
    let (mut doc, [_b, m, f]) = chain();
    let mut cache = RegenCache::new();
    let mut backend = Counting::default();
    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 3);

    // Edit the MIDDLE feature (move): it + the fillet rebuild, the box is reused.
    if let FeatureKind::Translate { offset, .. } = &mut doc.history.get_mut(m).unwrap().kind {
        offset.x = 9.0;
    }
    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 5, "move + fillet rebuild (+2); box reused");

    // Edit the TIP (fillet radius): only the fillet rebuilds.
    if let FeatureKind::FilletAll { radius, .. } = &mut doc.history.get_mut(f).unwrap().kind {
        *radius = 2.0;
    }
    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 6, "only the fillet rebuilds (+1)");

    // And it produced the right geometry (not a stale cached body).
    let r = cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 6, "no further change");
    let tip = r.visible().last().copied().unwrap();
    assert_eq!(r.body(tip).unwrap(), "fillet(move(box(10,10,10),9,0,0),2)");
}

#[test]
fn deleting_a_feature_prunes_it_and_reuses_the_rest() {
    let (mut doc, [_b, _m, f]) = chain();
    let mut cache = RegenCache::new();
    let mut backend = Counting::default();
    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 3);

    // Delete the tip fillet; box + move are still reused (no new ops).
    doc.history.remove(f);
    let r = cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 3, "remaining features reused after a delete");
    assert_eq!(r.visible().len(), 1, "the move is the lone visible body");
}

#[test]
fn suppress_then_unsuppress_rebuilds_only_when_needed() {
    // box -> bool(box, cyl) so suppressing an input actually breaks the boolean.
    let mut doc = Document::new("sup");
    let a = doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
    let c = doc.add("C", FeatureKind::Cylinder { radius: 2.0, height: 20.0 });
    let u = doc.add(
        "U",
        FeatureKind::Boolean { op: rmf_core::BooleanOp::Union, target: a, tool: c },
    );
    let mut cache = RegenCache::new();
    let mut backend = Counting::default();
    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 3);

    // Suppress the cylinder: the union loses an input and errors.
    doc.history.get_mut(c).unwrap().suppressed = true;
    let r = cache.regenerate(&doc, &mut backend);
    assert!(!r.is_ok(), "union errors with a suppressed input");
    let _ = u;

    // Unsuppress: only the union rebuilds (+1) — its input had vanished and its
    // cache entry was dropped. The box and cylinder are reused: their own
    // definitions never changed, so re-running them would be wasted work.
    doc.history.get_mut(c).unwrap().suppressed = false;
    cache.regenerate(&doc, &mut backend);
    assert_eq!(backend.ops, 4, "only the union rebuilds; box and cylinder reused");
}
