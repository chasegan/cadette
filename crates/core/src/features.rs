//! Features as **data**.
//!
//! A feature is an immutable-ish description of one modeling operation — never
//! the resulting geometry. The [`crate::regen`] engine turns a list of these
//! into real solids by replaying them against a [`crate::GeometryBackend`].
//! Keeping features kernel-free is what makes the document serializable, undo-
//! able, and testable without OCCT.
//!
//! References between features use [`FeatureId`] ("the body produced by feature
//! X"). Topological references (a specific face or edge that survives across
//! edits) are the harder problem deferred to a later phase; for now selections
//! are coarse (e.g. *all* edges).

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Stable identity for a feature, independent of its position in history.
/// Survives reordering, so references never dangle when steps move.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct FeatureId(pub u64);

/// The three combination operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BooleanOp {
    /// Merge two bodies into one.
    Union,
    /// Remove the tool from the target.
    Subtract,
    /// Keep only the overlapping volume.
    Intersect,
}

/// The operation a feature performs, with its parameters and input references.
///
/// All lengths are in the document's internal unit (millimeters).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeatureKind {
    // --- Primitives (no inputs) ---
    /// Axis-aligned box from the origin to `size`.
    Box { size: DVec3 },
    /// Cylinder of `radius`/`height`, axis along +Z from the origin.
    Cylinder { radius: f64, height: f64 },
    /// Sphere of `radius`, centered at the origin.
    Sphere { radius: f64 },

    // --- Operations (reference earlier features) ---
    /// Translate `source` by `offset`.
    Translate { source: FeatureId, offset: DVec3 },
    /// Combine `target` and `tool` with a boolean `op`.
    Boolean {
        op: BooleanOp,
        target: FeatureId,
        tool: FeatureId,
    },
    /// Fillet every edge of `source` with a constant `radius`.
    FilletAll { source: FeatureId, radius: f64 },
}

impl FeatureKind {
    /// The feature ids this operation consumes as inputs, in dependency order.
    /// Primitives return an empty list.
    pub fn inputs(&self) -> Vec<FeatureId> {
        match self {
            FeatureKind::Box { .. }
            | FeatureKind::Cylinder { .. }
            | FeatureKind::Sphere { .. } => Vec::new(),
            FeatureKind::Translate { source, .. } => vec![*source],
            FeatureKind::FilletAll { source, .. } => vec![*source],
            FeatureKind::Boolean { target, tool, .. } => vec![*target, *tool],
        }
    }

    /// A short, stable type label for UI/history display.
    pub fn type_name(&self) -> &'static str {
        match self {
            FeatureKind::Box { .. } => "Box",
            FeatureKind::Cylinder { .. } => "Cylinder",
            FeatureKind::Sphere { .. } => "Sphere",
            FeatureKind::Translate { .. } => "Translate",
            FeatureKind::Boolean { .. } => "Boolean",
            FeatureKind::FilletAll { .. } => "Fillet",
        }
    }
}

/// One entry in the history tree: an identified, named, optionally suppressed
/// operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    pub id: FeatureId,
    /// User-facing label shown in the history panel.
    pub name: String,
    /// A suppressed feature is skipped during regeneration but kept in history.
    pub suppressed: bool,
    pub kind: FeatureKind,
}
