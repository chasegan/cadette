//! The sketch-solver seam: the trait a constraint solver implements, mirroring
//! [`GeometryBackend`](crate::GeometryBackend) for geometry.
//!
//! Constraint sketches store their point coordinates; a solver moves those
//! coordinates to satisfy the constraints. Core owns the *contract* (solve in
//! place, report whether it worked) but no numerics, so the solver crate can
//! stay swappable (the hand-rolled Levenberg-Marquardt today, planegcs later)
//! and core stays FFI-free. Regeneration takes a solver so that a document
//! built or loaded by anything — the app, a test, a CLI — is solved before its
//! sketches become geometry, and a sketch that can't be solved is a
//! regeneration error rather than a silently wrong profile.

use crate::document::Document;
use crate::features::{FeatureId, FeatureKind, SweepPath};
use crate::sketch::Sketch2d;

/// How a sketch solve ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolveStatus {
    /// Every constraint is satisfied (to the solver's tolerance).
    Converged,
    /// The constraints contradict each other: some restate or fight others,
    /// and no arrangement satisfies them all. `residual_norm` is how far the
    /// worst constraint is from satisfied.
    Contradictory { residual_norm: f64 },
    /// The solver ran out of iterations without contradiction being proven —
    /// usually still an impossible sketch (a distance triangle that can't
    /// close), occasionally a starting guess too far from any solution.
    Stalled { residual_norm: f64 },
}

impl SolveStatus {
    pub fn is_converged(self) -> bool {
        matches!(self, SolveStatus::Converged)
    }
}

/// A constraint solver: solve a sketch in place and say how it went.
pub trait SketchSolver {
    fn solve(&mut self, sketch: &mut Sketch2d) -> SolveStatus;
}

/// A solver that does nothing and reports success — for documents whose
/// sketches carry no constraints, and for tests of the replay engine that
/// don't exercise solving.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSolver;

impl SketchSolver for NoSolver {
    fn solve(&mut self, _sketch: &mut Sketch2d) -> SolveStatus {
        SolveStatus::Converged
    }
}

impl Document {
    /// Solve every constraint sketch in the document in place — profile
    /// sketches and planar sweep paths, active or not — returning each sketch
    /// feature's outcome. Solving is idempotent for an already-solved sketch,
    /// so a converged sketch's coordinates (and any regeneration cache keyed
    /// on them) are left untouched.
    pub fn solve_sketches(
        &mut self,
        solver: &mut impl SketchSolver,
    ) -> Vec<(FeatureId, SolveStatus)> {
        let mut out = Vec::new();
        for feature in self.history.features_mut() {
            let sketch = match &mut feature.kind {
                FeatureKind::ConstraintSketch { sketch, .. } => sketch,
                FeatureKind::Sweep { path: SweepPath::Planar { sketch, .. }, .. } => sketch,
                _ => continue,
            };
            out.push((feature.id, solver.solve(sketch)));
        }
        out
    }
}
