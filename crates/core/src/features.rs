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

use crate::sketch::{Profile, Sketch2d, SketchPlane};

/// Stable identity for a feature, independent of its position in history.
/// Survives reordering, so references never dangle when steps move.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct FeatureId(pub u64);

/// A durable reference to a planar face by its geometry: a point on the face's
/// plane and the plane normal. The kernel re-finds the matching face on each
/// rebuild — a pragmatic "robust reference" that survives upstream edits which
/// don't move or split the face. (Topological naming is the fuller solution.)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceAnchor {
    pub point: DVec3,
    pub normal: DVec3,
}

/// A durable reference to an edge by a point lying on it. The kernel re-finds
/// the nearest edge to this point on each rebuild — same pragmatic "robust
/// reference" approach as [`FaceAnchor`], adequate for edits that don't move or
/// split the edge.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeAnchor {
    pub point: DVec3,
}

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

    /// A 2D sketch: a closed `profile` on `plane`. Evaluates to a planar face,
    /// which can be extruded into a solid.
    Sketch { plane: SketchPlane, profile: Profile },

    /// A constraint-driven 2D sketch on `plane`. Its solved geometry forms a
    /// closed loop that evaluates to a planar face (see [`Sketch2d`]). The
    /// solver runs before regeneration, so the stored coordinates are solved.
    ConstraintSketch { plane: SketchPlane, sketch: Sketch2d },

    // --- Operations (reference earlier features) ---
    /// Extrude a sketch (`source`) perpendicular to its plane by `distance`.
    Extrude { source: FeatureId, distance: f64 },
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
    /// Fillet one or more edges of `source` (identified by `edges`) with a
    /// constant `radius`. A single-edge fillet is just a one-element list.
    Fillet {
        source: FeatureId,
        edges: Vec<EdgeAnchor>,
        radius: f64,
    },

    /// Push or pull a planar face of `source` along its normal by `distance`
    /// (positive adds material, negative removes). The face is identified by
    /// `anchor`, re-found each rebuild.
    PushPull {
        source: FeatureId,
        anchor: FaceAnchor,
        distance: f64,
    },

    /// Rotate `source` by `angle` radians about the line through `center` with
    /// direction `axis` (used by the gizmo's rotation rings).
    Rotate {
        source: FeatureId,
        axis: DVec3,
        angle: f64,
        center: DVec3,
    },

    /// Non-uniformly scale `source` by per-axis `factors` about the fixed point
    /// `anchor` (used by the resize grips on non-primitive bodies). A point `p`
    /// maps to `anchor + factors * (p - anchor)`.
    Scale {
        source: FeatureId,
        factors: DVec3,
        anchor: DVec3,
    },

    /// Revolve a planar profile `source` through `angle` radians about the
    /// straight edge identified by `axis` (one of the profile's own segments or
    /// an adjacent model edge). A full turn is `2π`.
    Revolve {
        source: FeatureId,
        axis: EdgeAnchor,
        angle: f64,
    },

    /// Reflect `source` across the plane through `origin` with unit `normal`
    /// (a face's plane, an origin plane, or a body-center plane). Mirror-copy
    /// is a [`FeatureKind::Mirror`] of a duplicated body.
    Mirror {
        source: FeatureId,
        origin: DVec3,
        normal: DVec3,
    },

    /// Bundle `members` into one body that moves/rotates/scales as a unit but
    /// keeps each member distinct (a compound, not a fuse). Ungrouping deletes
    /// this feature, and the members become independently visible again.
    Group { members: Vec<FeatureId> },

    /// Sweep the planar profile `profile` (a sketch face) along an open `path`,
    /// keeping the profile normal to the path.
    Sweep { profile: FeatureId, path: SweepPath },
}

/// The path a [`FeatureKind::Sweep`] follows. An enum so future path *sources* —
/// a free-space 3D curve, or references to existing model edges — slot in as new
/// variants without disturbing the planar case or breaking saved projects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SweepPath {
    /// An open chain drawn on `plane` (the current MVP). The kernel still sweeps
    /// along a generic spine, so a 3D variant needs no kernel change.
    Planar { plane: SketchPlane, sketch: Sketch2d },
}

impl FeatureKind {
    /// The feature ids this operation consumes as inputs, in dependency order.
    /// Primitives return an empty list.
    pub fn inputs(&self) -> Vec<FeatureId> {
        match self {
            FeatureKind::Box { .. }
            | FeatureKind::Cylinder { .. }
            | FeatureKind::Sphere { .. }
            | FeatureKind::Sketch { .. }
            | FeatureKind::ConstraintSketch { .. } => Vec::new(),
            FeatureKind::Translate { source, .. } => vec![*source],
            FeatureKind::FilletAll { source, .. } => vec![*source],
            FeatureKind::Fillet { source, .. } => vec![*source],
            FeatureKind::Extrude { source, .. } => vec![*source],
            FeatureKind::PushPull { source, .. } => vec![*source],
            FeatureKind::Rotate { source, .. } => vec![*source],
            FeatureKind::Scale { source, .. } => vec![*source],
            FeatureKind::Revolve { source, .. } => vec![*source],
            FeatureKind::Mirror { source, .. } => vec![*source],
            FeatureKind::Boolean { target, tool, .. } => vec![*target, *tool],
            FeatureKind::Group { members } => members.clone(),
            FeatureKind::Sweep { profile, .. } => vec![*profile],
        }
    }

    /// The input a dependent should rewire to if this feature is deleted — its
    /// primary upstream body. `None` for features with no usable input
    /// (primitives, sketches), whose dependents can't be healed.
    pub fn primary_input(&self) -> Option<FeatureId> {
        match self {
            FeatureKind::Translate { source, .. }
            | FeatureKind::FilletAll { source, .. }
            | FeatureKind::Fillet { source, .. }
            | FeatureKind::Extrude { source, .. }
            | FeatureKind::PushPull { source, .. }
            | FeatureKind::Rotate { source, .. }
            | FeatureKind::Scale { source, .. }
            | FeatureKind::Revolve { source, .. }
            | FeatureKind::Mirror { source, .. }
            | FeatureKind::Sweep { profile: source, .. } => Some(*source),
            // Heal to the kept body of a boolean.
            FeatureKind::Boolean { target, .. } => Some(*target),
            // Heal to the first member (ungroup proper bakes any group transform
            // into the members — see the app's ungroup).
            FeatureKind::Group { members } => members.first().copied(),
            _ => None,
        }
    }

    /// Replace every reference to feature `from` with `to`.
    pub fn remap_input(&mut self, from: FeatureId, to: FeatureId) {
        let swap = |id: &mut FeatureId| {
            if *id == from {
                *id = to;
            }
        };
        match self {
            FeatureKind::Translate { source, .. }
            | FeatureKind::FilletAll { source, .. }
            | FeatureKind::Fillet { source, .. }
            | FeatureKind::Extrude { source, .. }
            | FeatureKind::PushPull { source, .. }
            | FeatureKind::Rotate { source, .. }
            | FeatureKind::Scale { source, .. }
            | FeatureKind::Revolve { source, .. }
            | FeatureKind::Mirror { source, .. }
            | FeatureKind::Sweep { profile: source, .. } => swap(source),
            FeatureKind::Boolean { target, tool, .. } => {
                swap(target);
                swap(tool);
            }
            FeatureKind::Group { members } => members.iter_mut().for_each(swap),
            _ => {}
        }
    }

    /// A short, stable type label for UI/history display.
    pub fn type_name(&self) -> &'static str {
        match self {
            FeatureKind::Box { .. } => "Box",
            FeatureKind::Cylinder { .. } => "Cylinder",
            FeatureKind::Sphere { .. } => "Sphere",
            FeatureKind::Sketch { .. } => "Sketch",
            FeatureKind::ConstraintSketch { .. } => "Sketch",
            FeatureKind::Translate { .. } => "Translate",
            FeatureKind::Extrude { .. } => "Extrude",
            FeatureKind::Boolean { .. } => "Boolean",
            FeatureKind::FilletAll { .. } => "Fillet",
            FeatureKind::Fillet { .. } => "Fillet",
            FeatureKind::PushPull { .. } => "Push/Pull",
            FeatureKind::Rotate { .. } => "Rotate",
            FeatureKind::Scale { .. } => "Scale",
            FeatureKind::Revolve { .. } => "Revolve",
            FeatureKind::Mirror { .. } => "Mirror",
            FeatureKind::Group { .. } => "Group",
            FeatureKind::Sweep { .. } => "Sweep",
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
