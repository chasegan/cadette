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

use std::collections::HashMap;

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
                // The feature consumes its inputs: drop them from the visible
                // set and present this body in their place.
                for input in feature.kind.inputs() {
                    if let Some(pos) = visible.iter().position(|v| *v == input) {
                        visible.remove(pos);
                    }
                }
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
            edge,
            radius,
        } => {
            let body = input(*source)?;
            backend.fillet_edge(body, *edge, *radius).map_err(backend_err)
        }
        FeatureKind::PushPull {
            source,
            anchor,
            distance,
        } => {
            let body = input(*source)?;
            backend.push_pull(body, *anchor, *distance).map_err(backend_err)
        }
    }
}
