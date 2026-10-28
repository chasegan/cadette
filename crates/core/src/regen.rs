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
use crate::features::{BooleanOp, FeatureId, FeatureKind, ImportFormat};
use crate::solver::{SketchSolver, SolveStatus};

/// A single feature's failure during regeneration.
#[derive(Clone, Debug, PartialEq)]
pub enum RegenError<E> {
    /// The backend rejected the operation (e.g. infeasible fillet).
    Backend { feature: FeatureId, source: E },
    /// A referenced input wasn't available; `reason` says why, so hosts can
    /// tell a cascade from a suppressed step or a structural error.
    MissingInput {
        feature: FeatureId,
        input: FeatureId,
        reason: MissingReason,
    },
    /// The feature's own definition is invalid (e.g. a sketch with no closed
    /// loop to extrude).
    Invalid {
        feature: FeatureId,
        reason: &'static str,
    },
    /// The feature's sketch constraints could not be solved, so its geometry
    /// was not built (see [`regenerate_solved`]).
    UnsolvedSketch {
        feature: FeatureId,
        status: SolveStatus,
    },
}

/// Why an input body was not available when a feature needed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingReason {
    /// The input is present and active but failed to build (a cascade: fix the
    /// input and this feature follows).
    Failed,
    /// The input is suppressed.
    Suppressed,
    /// The input sits behind the rollback bar.
    RolledBack,
    /// The input comes *later* in history than the feature that needs it — a
    /// forward reference, which no valid history contains.
    Later,
    /// No feature with that id exists in the history.
    Deleted,
}

impl MissingReason {
    fn of(document: &Document, feature: FeatureId, input: FeatureId) -> Self {
        let Some(input_feature) = document.history.get(input) else {
            return MissingReason::Deleted;
        };
        if input_feature.suppressed {
            return MissingReason::Suppressed;
        }
        let (Some(fi), Some(ii)) =
            (document.history.index_of(feature), document.history.index_of(input))
        else {
            return MissingReason::Deleted;
        };
        if ii >= fi {
            MissingReason::Later
        } else if ii >= document.rollback() {
            MissingReason::RolledBack
        } else {
            MissingReason::Failed
        }
    }
}

impl<E> RegenError<E> {
    /// The feature that failed.
    pub fn feature(&self) -> FeatureId {
        match self {
            RegenError::Backend { feature, .. }
            | RegenError::MissingInput { feature, .. }
            | RegenError::Invalid { feature, .. }
            | RegenError::UnsolvedSketch { feature, .. } => *feature,
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
    regenerate_inner(document, backend, &HashMap::new())
}

/// Solve the document's constraint sketches, then [`regenerate`]. A sketch
/// whose constraints can't be satisfied is reported as
/// [`RegenError::UnsolvedSketch`] and skipped (its dependents then miss their
/// input), instead of being built from whatever coordinates the failed solve
/// left behind. This is the entry point for anything that builds a document
/// without going through the interactive sketcher.
pub fn regenerate_solved<B>(
    document: &mut Document,
    backend: &mut B,
    solver: &mut impl SketchSolver,
) -> Regeneration<B::Body, B::Error>
where
    B: GeometryBackend,
{
    let unsolved = unsolved_sketches(document, solver);
    regenerate_inner(document, backend, &unsolved)
}

/// Solve every sketch and collect the ones that failed.
fn unsolved_sketches(
    document: &mut Document,
    solver: &mut impl SketchSolver,
) -> HashMap<FeatureId, SolveStatus> {
    document
        .solve_sketches(solver)
        .into_iter()
        .filter(|(_, status)| !status.is_converged())
        .collect()
}

fn regenerate_inner<B>(
    document: &Document,
    backend: &mut B,
    unsolved: &HashMap<FeatureId, SolveStatus>,
) -> Regeneration<B::Body, B::Error>
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
        if let Some(&status) = unsolved.get(&feature.id) {
            errors.push(RegenError::UnsolvedSketch { feature: feature.id, status });
            continue;
        }

        backend.begin_feature(feature.id);
        match eval(backend, document, feature.id, &feature.kind, &bodies) {
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
    /// What the feature consumed beyond its kind and inputs — an Import's
    /// file bytes and the document unit (see [`kind_fingerprint`]).
    fingerprint: u64,
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

    /// The cached body of `id`, if one has been built — read-only
    /// interrogation for previews (e.g. the Build Graph's hover ghost), no
    /// regeneration. May be stale for a feature that is currently rolled
    /// back or suppressed: it's whatever was last built for it.
    pub fn cached_body(&self, id: FeatureId) -> Option<&B> {
        self.entries.get(&id).map(|e| &e.body)
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
        self.regenerate_inner(document, backend, &HashMap::new())
    }

    /// Solve the document's sketches, then regenerate incrementally — the
    /// cached counterpart of [`regenerate_solved`].
    pub fn regenerate_solved<BK>(
        &mut self,
        document: &mut Document,
        backend: &mut BK,
        solver: &mut impl SketchSolver,
    ) -> Regeneration<B, BK::Error>
    where
        BK: GeometryBackend<Body = B>,
    {
        let unsolved = unsolved_sketches(document, solver);
        self.regenerate_inner(document, backend, &unsolved)
    }

    fn regenerate_inner<BK>(
        &mut self,
        document: &Document,
        backend: &mut BK,
        unsolved: &HashMap<FeatureId, SolveStatus>,
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
            if let Some(&status) = unsolved.get(&feature.id) {
                // Drop the stale entry so dependents see the change next pass.
                self.entries.remove(&feature.id);
                errors.push(RegenError::UnsolvedSketch { feature: feature.id, status });
                continue;
            }
            let inputs = feature.kind.inputs();

            // Inputs precede this feature in history order, so their entries are
            // already at this pass's version (0 = not built this pass).
            let input_versions: Vec<u64> = inputs
                .iter()
                .map(|id| self.entries.get(id).map_or(0, |e| e.version))
                .collect();

            let fingerprint = kind_fingerprint(document, &feature.kind);
            let reusable = self.entries.get(&feature.id).is_some_and(|e| {
                e.kind == feature.kind
                    && e.fingerprint == fingerprint
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

            backend.begin_feature(feature.id);
            match eval(backend, document, feature.id, &feature.kind, &bodies) {
                Ok(body) => {
                    self.next_version += 1;
                    self.entries.insert(
                        feature.id,
                        CacheEntry {
                            kind: feature.kind.clone(),
                            fingerprint,
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

/// Everything a feature's result depends on that its kind does not spell
/// out. An Import names its asset, but builds from the asset's bytes scaled
/// by the document unit; if either changes under the same name the cached
/// body is stale. Zero for every other kind.
fn kind_fingerprint(document: &Document, kind: &FeatureKind) -> u64 {
    match kind {
        FeatureKind::Import { asset } => {
            let bytes = document.assets.fingerprint(asset).unwrap_or(0);
            bytes ^ document.units.mm_per_unit().to_bits().rotate_left(17)
        }
        _ => 0,
    }
}

/// Evaluate one feature, resolving its inputs from already-built bodies.
fn eval<B>(
    backend: &mut B,
    document: &Document,
    feature: FeatureId,
    kind: &FeatureKind,
    bodies: &HashMap<FeatureId, B::Body>,
) -> Result<B::Body, RegenError<B::Error>>
where
    B: GeometryBackend,
{
    // Look up an input body or report precisely which reference was missing,
    // and why.
    let input = |id: FeatureId| -> Result<&B::Body, RegenError<B::Error>> {
        bodies.get(&id).ok_or_else(|| RegenError::MissingInput {
            feature,
            input: id,
            reason: MissingReason::of(document, feature, id),
        })
    };
    let backend_err = |source: B::Error| RegenError::Backend { feature, source };

    // Matched by reference: some variants (constraint sketches) are not Copy.
    match kind {
        FeatureKind::Box { size } => backend.make_box(*size).map_err(backend_err),
        FeatureKind::Cylinder { radius, height } => {
            backend.make_cylinder(*radius, *height).map_err(backend_err)
        }
        FeatureKind::Sphere { radius } => backend.make_sphere(*radius).map_err(backend_err),
        FeatureKind::Import { asset } => match (ImportFormat::of_name(asset), document.assets.get(asset)) {
            (Some(format), Some(data)) => {
                let scale = 1.0 / document.units.mm_per_unit();
                backend.import(format, data, scale).map_err(backend_err)
            }
            (None, _) => Err(RegenError::Invalid {
                feature,
                reason: "the imported file isn't a STEP or STL file (by its extension)",
            }),
            (_, None) => Err(RegenError::Invalid {
                feature,
                reason: "the imported file is missing from the project — import it again",
            }),
        },
        FeatureKind::Sketch { plane, profile } => {
            backend.sketch(*plane, *profile).map_err(backend_err)
        }
        FeatureKind::ConstraintSketch { plane, sketch } => match sketch.profile_elements() {
            Some(elements) => backend.sketch_profile(*plane, &elements).map_err(backend_err),
            None => Err(RegenError::Invalid {
                feature,
                reason: "sketch has no closed loop to build a profile",
            }),
        },
        FeatureKind::Extrude { source, distance } => {
            let body = input(*source)?;
            backend.extrude(body, *distance).map_err(backend_err)
        }
        FeatureKind::Sweep { profile, path } => {
            let face = input(*profile)?;
            match path {
                crate::SweepPath::Planar { plane, sketch } => match sketch.path_elements() {
                    Some(elements) => backend.sweep(face, *plane, &elements).map_err(backend_err),
                    None => Err(RegenError::Invalid {
                        feature,
                        reason: "sweep path is not a single open chain",
                    }),
                },
                crate::SweepPath::Helix { helix } => {
                    if let Some(reason) = helix_problem(helix) {
                        return Err(RegenError::Invalid { feature, reason });
                    }
                    backend.sweep_helix(face, *helix).map_err(backend_err)
                }
            }
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
            if !positive(*radius) {
                return Err(RegenError::Invalid { feature, reason: FILLET_RADIUS });
            }
            backend.fillet_all(body, *radius).map_err(backend_err)
        }
        FeatureKind::Fillet {
            source,
            edges,
            radius,
        } => {
            let body = input(*source)?;
            if let Some(reason) = blend_problem(edges.len(), *radius, FILLET_RADIUS, FILLET_EDGES) {
                return Err(RegenError::Invalid { feature, reason });
            }
            backend.fillet_edges(body, edges, *radius).map_err(backend_err)
        }
        FeatureKind::ChamferAll { source, distance } => {
            let body = input(*source)?;
            if !positive(*distance) {
                return Err(RegenError::Invalid { feature, reason: CHAMFER_DISTANCE });
            }
            backend.chamfer_all(body, *distance).map_err(backend_err)
        }
        FeatureKind::Chamfer {
            source,
            edges,
            distance,
        } => {
            let body = input(*source)?;
            if let Some(reason) =
                blend_problem(edges.len(), *distance, CHAMFER_DISTANCE, CHAMFER_EDGES)
            {
                return Err(RegenError::Invalid { feature, reason });
            }
            backend.chamfer_edges(body, edges, *distance).map_err(backend_err)
        }
        FeatureKind::Shell {
            source,
            thickness,
            open,
        } => {
            let body = input(*source)?;
            if !positive(*thickness) {
                return Err(RegenError::Invalid { feature, reason: "a shell needs a positive wall thickness" });
            }
            backend.shell(body, *thickness, open).map_err(backend_err)
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
        FeatureKind::Group { members } => {
            if members.is_empty() {
                return Err(RegenError::Invalid {
                    feature,
                    reason: "a group needs at least one member",
                });
            }
            let mut bodies = Vec::with_capacity(members.len());
            for m in members {
                bodies.push(input(*m)?);
            }
            backend.compound(&bodies).map_err(backend_err)
        }
        FeatureKind::Pattern {
            source,
            layout,
            count,
            skip,
        } => {
            let seed = input(*source)?;
            if *count > MAX_PATTERN_COUNT {
                return Err(RegenError::Invalid {
                    feature,
                    reason: PATTERN_COUNT_LIMIT,
                });
            }
            let instances = pattern_instances(*count, skip);
            if instances.is_empty() {
                return Err(RegenError::Invalid {
                    feature,
                    reason: "a pattern needs at least one instance that isn't skipped",
                });
            }
            // Instance 0 is the seed in place; instance i is the seed moved by
            // i steps. The kept instances are bundled into one compound (like
            // a group), in instance order.
            let mut moved = Vec::with_capacity(instances.len());
            for &i in &instances {
                if i == 0 {
                    continue;
                }
                let n = f64::from(i);
                let body = match *layout {
                    crate::PatternLayout::Linear { step } => backend.translate(seed, step * n),
                    crate::PatternLayout::Circular { center, axis, angle } => {
                        backend.rotate(seed, center, axis, angle * n)
                    }
                    crate::PatternLayout::Helical { center, axis, angle, rise } => {
                        let turned = backend.rotate(seed, center, axis, angle * n).map_err(backend_err)?;
                        backend.translate(&turned, axis.normalize_or_zero() * (rise * n))
                    }
                };
                moved.push(body.map_err(backend_err)?);
            }
            let mut members: Vec<&B::Body> = Vec::with_capacity(instances.len());
            if instances[0] == 0 {
                members.push(seed);
            }
            members.extend(moved.iter());
            backend.compound(&members).map_err(backend_err)
        }
        FeatureKind::Thread {
            source,
            helix,
            depth,
            internal,
            starts,
        } => {
            let body = input(*source)?;
            let reason = if let Some(r) = helix_problem(helix) {
                Some(r)
            } else if !positive(*depth) {
                Some("a thread needs a positive depth")
            } else if !*internal && *depth >= helix.radius {
                Some("a thread's depth must be less than its radius")
            } else if *internal && crate::features::thread_overlap(helix.pitch / f64::from((*starts).max(1))) >= helix.radius {
                Some("a thread's pitch is too large for this hole")
            } else if *starts == 0 || *starts > MAX_THREAD_STARTS {
                Some(THREAD_STARTS_LIMIT)
            } else {
                None
            };
            if let Some(reason) = reason {
                return Err(RegenError::Invalid { feature, reason });
            }
            // The groove, swept along the helix, is cut from the body: a shaft
            // is threaded inward from its major diameter, a hole outward from
            // its minor diameter. Extra starts are the same groove turned
            // about the axis, cut together as one compound tool. The helix
            // pitch is the LEAD (advance per turn); adjacent grooves of a
            // multi-start thread sit lead/starts apart, which sizes the groove.
            let spacing = helix.pitch / f64::from(*starts);
            let groove = crate::thread_profile(spacing, *depth, helix.radius, *internal);
            let tool = backend.helix_sweep(&groove, *helix).map_err(backend_err)?;
            let mut turned = Vec::with_capacity(*starts as usize - 1);
            for s in 1..*starts {
                let angle = std::f64::consts::TAU * f64::from(s) / f64::from(*starts);
                turned.push(backend.rotate(&tool, helix.center, helix.axis, angle).map_err(backend_err)?);
            }
            let cut_with = if turned.is_empty() {
                tool
            } else {
                let mut members: Vec<&B::Body> = Vec::with_capacity(*starts as usize);
                members.push(&tool);
                members.extend(turned.iter());
                backend.compound(&members).map_err(backend_err)?
            };
            backend
                .boolean(BooleanOp::Subtract, body, &cut_with)
                .map_err(backend_err)
        }
    }
}

/// The most turns one helix (a [`FeatureKind::Thread`] or a
/// [`crate::SweepPath::Helix`]) will sweep — a guard against a mistyped pitch
/// stalling regeneration.
pub const MAX_THREAD_TURNS: f64 = 500.0;
const THREAD_TURNS_LIMIT: &str = "a helix is limited to 500 turns";

/// The most starts one [`FeatureKind::Thread`] will cut.
pub const MAX_THREAD_STARTS: u32 = 200;
const THREAD_STARTS_LIMIT: &str = "a thread has between 1 and 200 starts";

const FILLET_RADIUS: &str = "a fillet needs a positive radius";
const FILLET_EDGES: &str = "a fillet needs at least one edge";
const CHAMFER_DISTANCE: &str = "a chamfer needs a positive distance";
const CHAMFER_EDGES: &str = "a chamfer needs at least one edge";

/// Why an anchored edge blend (fillet/chamfer) can't be attempted: no edges,
/// or a size that isn't a positive finite length.
fn blend_problem(
    edges: usize,
    size: f64,
    size_reason: &'static str,
    edges_reason: &'static str,
) -> Option<&'static str> {
    if edges == 0 {
        Some(edges_reason)
    } else if !positive(size) {
        Some(size_reason)
    } else {
        None
    }
}

/// A finite, strictly positive length or count.
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Why a helix can't be swept, if it can't.
fn helix_problem(helix: &crate::Helix) -> Option<&'static str> {
    if !positive(helix.pitch) {
        Some("a helix needs a positive pitch")
    } else if !positive(helix.length) {
        Some("a helix needs a positive length")
    } else if !positive(helix.radius) {
        Some("a helix needs a positive radius")
    } else if helix.axis.length() < 1e-9 {
        Some("a helix needs an axis direction")
    } else if helix.turns() > MAX_THREAD_TURNS {
        Some(THREAD_TURNS_LIMIT)
    } else {
        None
    }
}

/// The instances a [`FeatureKind::Pattern`] builds: `0..count` minus `skip`,
/// in order.
pub fn pattern_instances(count: u32, skip: &[u32]) -> Vec<u32> {
    (0..count).filter(|i| !skip.contains(i)).collect()
}

/// The most instances one [`FeatureKind::Pattern`] will build — a guard
/// against a mistyped count stalling regeneration. The message quotes the
/// number (reasons are `&'static str`); a test keeps them in step.
pub const MAX_PATTERN_COUNT: u32 = 1000;
const PATTERN_COUNT_LIMIT: &str = "a pattern is limited to 1000 instances";

#[cfg(test)]
mod limit_messages {
    use super::*;

    #[test]
    fn limit_messages_quote_the_limits() {
        assert!(PATTERN_COUNT_LIMIT.contains(&MAX_PATTERN_COUNT.to_string()));
        assert!(THREAD_STARTS_LIMIT.contains(&MAX_THREAD_STARTS.to_string()));
        assert!(THREAD_TURNS_LIMIT.contains(&(MAX_THREAD_TURNS as u32).to_string()));
    }
}
