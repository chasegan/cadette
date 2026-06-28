//! The replay engine: turn a [`Document`]'s active history into geometry.
//!
//! [`regenerate`] walks the active, non-suppressed features in order, evaluating
//! each against a [`GeometryBackend`] and caching the result by [`FeatureId`].
//! A feature's inputs are looked up from that cache, so the whole history
//! reduces to a single forward pass.
//!
//! Failures are collected, not fatal: a feature that errors (or whose input is
//! missing) is recorded and skipped, and regeneration continues. This is what
//! lets the history panel show a single broken step in red while everything
//! else still displays.
//!
//! **Visibility.** Each feature *consumes* its inputs: when feature `C = A ∪ B`
//! succeeds, `A` and `B` stop being independently visible and `C` takes their
//! place. The leftover set — bodies no later feature consumed — is what the
//! viewport should show.

use std::collections::{HashMap, HashSet};

use crate::backend::GeometryBackend;
use crate::document::Document;
use crate::features::{FeatureId, FeatureKind};

/// A single feature's failure during regeneration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegenError<E> {
    /// The backend rejected the operation (e.g. infeasible fillet).
    Backend { feature: FeatureId, source: E },
    /// A referenced input wasn't available (failed, suppressed, or rolled back).
    MissingInput { feature: FeatureId, input: FeatureId },
    /// The feature's own definition is invalid (e.g. a sketch with no closed
    /// loop to extrude).
    Invalid {
        feature: FeatureId,
        reason: &'static str,
    },
}

impl<E> RegenError<E> {
    /// The feature that failed.
    pub fn feature(&self) -> FeatureId {
        match self {
            RegenError::Backend { feature, .. }
            | RegenError::MissingInput { feature, .. }
            | RegenError::Invalid { feature, .. } => *feature,
        }
    }
}

/// The outcome of replaying a document: the bodies produced, which are visible,
/// and any per-feature failures.
pub struct Regeneration<B, E> {
    bodies: HashMap<FeatureId, B>,
    visible: Vec<FeatureId>,
    errors: Vec<RegenError<E>>,
}

impl<B, E> Regeneration<B, E> {
    /// The body produced by a specific feature, if it evaluated successfully.
    pub fn body(&self, id: FeatureId) -> Option<&B> {
        self.bodies.get(&id)
    }

    /// Feature ids whose bodies are visible (not consumed by a later feature),
    /// in history order.
    pub fn visible(&self) -> &[FeatureId] {
        &self.visible
    }

    /// The visible bodies themselves, in history order.
    pub fn visible_bodies(&self) -> impl Iterator<Item = &B> {
        self.visible.iter().filter_map(|id| self.bodies.get(id))
    }

    /// Take ownership of the visible bodies, in history order. Lets the host
    /// keep them after regeneration (e.g. to query a picked face's plane).
    pub fn into_visible_bodies(mut self) -> Vec<B> {
        self.visible
            .iter()
            .filter_map(|id| self.bodies.remove(id))
            .collect()
    }

    /// Per-feature failures collected during regeneration.
    pub fn errors(&self) -> &[RegenError<E>] {
        &self.errors
    }

    /// True if every active feature evaluated without error.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Replay the document's active history against `backend`.
pub fn regenerate<B>(document: &Document, backend: &mut B) -> Regeneration<B::Body, B::Error>
where
    B: GeometryBackend,
{
    let mut bodies: HashMap<FeatureId, B::Body> = HashMap::new();
    let mut visible: Vec<FeatureId> = Vec::new();
    let mut errors: Vec<RegenError<B::Error>> = Vec::new();

    for feature in document.active_features() {
        if feature.suppressed {
            continue;
        }

        match eval(backend, feature.id, &feature.kind, &bodies) {
            Ok(body) => {
                consume_inputs(&mut visible, &feature.kind.inputs());
                bodies.insert(feature.id, body);
                visible.push(feature.id);
            }
            Err(err) => errors.push(err),
        }
    }

    Regeneration {
        bodies,
        visible,
        errors,
    }
}

/// A feature consumes its inputs: drop them from the visible set so the feature
/// can take their place.
fn consume_inputs(visible: &mut Vec<FeatureId>, inputs: &[FeatureId]) {
    for input in inputs {
        if let Some(pos) = visible.iter().position(|v| v == input) {
            visible.remove(pos);
        }
    }
}

/// Persistent cache for **incremental** regeneration.
///
/// [`regenerate`] rebuilds the whole history every call; this remembers each
/// feature's built body, the parameters (`kind`) it was built from, and a
/// `version` that bumps on every rebuild. A feature is rebuilt only when its
/// `kind` changes or one of its inputs was rebuilt (detected by a version
/// mismatch) — so an edit, or a push/pull drag, recomputes only the changed
/// feature and everything downstream, never the unchanged upstream. The
/// `Body` must be cheap to clone (an OCCT shape is a shared handle).
pub struct RegenCache<B> {
    entries: HashMap<FeatureId, CacheEntry<B>>,
    next_version: u64,
}

struct CacheEntry<B> {
    kind: FeatureKind,
    /// The version of each input at the time this feature was built.
    input_versions: Vec<u64>,
    version: u64,
    body: B,
}

impl<B> Default for RegenCache<B> {
    fn default() -> Self {
        RegenCache {
            entries: HashMap::new(),
            next_version: 0,
        }
    }
}

impl<B: Clone> RegenCache<B> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replay the document's active history, reusing cached bodies whose `kind`
    /// and inputs are unchanged. Same result as [`regenerate`], only faster on
    /// repeat calls after a small edit.
    pub fn regenerate<BK>(
        &mut self,
        document: &Document,
        backend: &mut BK,
    ) -> Regeneration<B, BK::Error>
    where
        BK: GeometryBackend<Body = B>,
    {
        let mut bodies: HashMap<FeatureId, B> = HashMap::new();
        let mut visible: Vec<FeatureId> = Vec::new();
        let mut errors: Vec<RegenError<BK::Error>> = Vec::new();

        for feature in document.active_features() {
            if feature.suppressed {
                continue;
            }
            let inputs = feature.kind.inputs();

            // Inputs precede this feature in history order, so their entries are
            // already at this pass's version (0 = not built this pass).
            let input_versions: Vec<u64> = inputs
                .iter()
                .map(|id| self.entries.get(id).map_or(0, |e| e.version))
                .collect();

            let reusable = self.entries.get(&feature.id).is_some_and(|e| {
                e.kind == feature.kind
                    && e.input_versions == input_versions
                    && inputs.iter().all(|id| bodies.contains_key(id))
            });

            if reusable {
                let body = self.entries[&feature.id].body.clone();
                consume_inputs(&mut visible, &inputs);
                bodies.insert(feature.id, body);
                visible.push(feature.id);
                continue;
            }

            match eval(backend, feature.id, &feature.kind, &bodies) {
                Ok(body) => {
                    self.next_version += 1;
                    self.entries.insert(
                        feature.id,
                        CacheEntry {
                            kind: feature.kind.clone(),
                            input_versions,
                            version: self.next_version,
                            body: body.clone(),
                        },
                    );
                    consume_inputs(&mut visible, &inputs);
                    bodies.insert(feature.id, body);
                    visible.push(feature.id);
                }
                Err(err) => {
                    // Drop the stale entry so dependents see the change next pass.
                    self.entries.remove(&feature.id);
                    errors.push(err);
                }
            }
        }

        // Forget features removed from the document entirely (deleted). Rolled-
        // back-but-present features stay cached, ready to reuse when they return.
        let live: HashSet<FeatureId> =
            document.history.features().iter().map(|f| f.id).collect();
        self.entries.retain(|id, _| live.contains(id));

        Regeneration {
            bodies,
            visible,
            errors,
        }
    }
}

/// Evaluate one feature, resolving its inputs from already-built bodies.
fn eval<B>(
    backend: &mut B,
    feature: FeatureId,
    kind: &FeatureKind,
    bodies: &HashMap<FeatureId, B::Body>,
) -> Result<B::Body, RegenError<B::Error>>
where
    B: GeometryBackend,
{
    // Look up an input body or report precisely which reference was missing.
    let input = |id: FeatureId| -> Result<&B::Body, RegenError<B::Error>> {
        bodies
            .get(&id)
            .ok_or(RegenError::MissingInput { feature, input: id })
    };
    let backend_err = |source: B::Error| RegenError::Backend { feature, source };

    // Matched by reference: some variants (constraint sketches) are not Copy.
    match kind {
        FeatureKind::Box { size } => backend.make_box(*size).map_err(backend_err),
        FeatureKind::Cylinder { radius, height } => {
            backend.make_cylinder(*radius, *height).map_err(backend_err)
        }
        FeatureKind::Sphere { radius } => backend.make_sphere(*radius).map_err(backend_err),
        FeatureKind::Sketch { plane, profile } => {
            backend.sketch(*plane, *profile).map_err(backend_err)
        }
        FeatureKind::ConstraintSketch { plane, sketch } => match sketch.profile_loop() {
            Some(points) => backend.sketch_loop(*plane, &points).map_err(backend_err),
            None => Err(RegenError::Invalid {
                feature,
                reason: "sketch has no closed loop to build a profile",
            }),
        },
        FeatureKind::Extrude { source, distance } => {
            let body = input(*source)?;
            backend.extrude(body, *distance).map_err(backend_err)
        }
        FeatureKind::Translate { source, offset } => {
            let body = input(*source)?;
            backend.translate(body, *offset).map_err(backend_err)
        }
        FeatureKind::Boolean { op, target, tool } => {
            let target_body = input(*target)?;
            let tool_body = input(*tool)?;
            backend.boolean(*op, target_body, tool_body).map_err(backend_err)
        }
        FeatureKind::FilletAll { source, radius } => {
            let body = input(*source)?;
            backend.fillet_all(body, *radius).map_err(backend_err)
        }
        FeatureKind::Fillet {
            source,
            edges,
            radius,
        } => {
            let body = input(*source)?;
            backend.fillet_edges(body, edges, *radius).map_err(backend_err)
        }
        FeatureKind::PushPull {
            source,
            anchor,
            distance,
        } => {
            let body = input(*source)?;
            backend.push_pull(body, *anchor, *distance).map_err(backend_err)
        }
        FeatureKind::Rotate {
            source,
            axis,
            angle,
            center,
        } => {
            let body = input(*source)?;
            backend.rotate(body, *center, *axis, *angle).map_err(backend_err)
        }
        FeatureKind::Scale {
            source,
            factors,
            anchor,
        } => {
            let body = input(*source)?;
            backend.scale(body, *factors, *anchor).map_err(backend_err)
        }
        FeatureKind::Revolve {
            source,
            axis,
            angle,
        } => {
            let body = input(*source)?;
            backend.revolve(body, axis.point, *angle).map_err(backend_err)
        }
        FeatureKind::Mirror {
            source,
            origin,
            normal,
        } => {
            let body = input(*source)?;
            backend.mirror(body, *origin, *normal).map_err(backend_err)
        }
    }
}
