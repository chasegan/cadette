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

use crate::sketch::{Profile, ProfileElem, Sketch2d, SketchPlane};

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
///
/// # Save-format compatibility — read before adding a field
///
/// This enum (and everything reachable from it: [`Sketch2d`], [`SketchPlane`],
/// anchors, …) is serialized verbatim into `.cdt` project files, and files in
/// the wild must keep loading. Adding a NEW variant is safe; adding a new
/// field to an EXISTING variant or serialized struct is not, unless it carries
/// `#[serde(default)]` (or ships with a versioned migration): a bare new field
/// makes every previously saved project fail to deserialize. See
/// `Sketch2d::beziers` for the pattern. Never rename or remove a serialized
/// field/variant without a migration.
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
    /// stored coordinates are what regeneration reads, so they must be solved
    /// first: [`regenerate_solved`](crate::regenerate_solved) does that, and
    /// reports a sketch that can't be solved instead of building it.
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
    /// Chamfer every edge of `source`: a symmetric bevel set back `distance`
    /// along both faces of each edge. Its own kind, not a fillet variant —
    /// a bevel and a rounding are different features of a part.
    ChamferAll { source: FeatureId, distance: f64 },
    /// Chamfer one or more edges of `source` (identified by `edges`) by a
    /// constant `distance`.
    Chamfer {
        source: FeatureId,
        edges: Vec<EdgeAnchor>,
        distance: f64,
    },
    /// Hollow `source` into a shell of uniform wall `thickness`, measured
    /// inward from its faces, removing the faces in `open` so the cavity is
    /// reachable (an empty `open` keeps a closed hollow body).
    Shell {
        source: FeatureId,
        thickness: f64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        open: Vec<FaceAnchor>,
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

    /// Sweep the planar profile `profile` (a sketch face) along `path`: an
    /// open planar chain (the profile kept normal to it) or a helix (the
    /// profile carried round in the axial plane — a coil or a spring).
    Sweep { profile: FeatureId, path: SweepPath },

    /// Repeat `source` `count` times on a `layout` (the seed stays in place as
    /// instance 0) and bundle the instances into one compound body, like a
    /// [`FeatureKind::Group`]. Splicing a pattern into a chain (see
    /// [`crate::History::insert_transform`]) repeats the body for everything
    /// downstream — pattern a boolean's tool and the cut repeats with it.
    Pattern {
        source: FeatureId,
        layout: PatternLayout,
        count: u32,
        /// Instance indices (0 = the seed) left out — a bolt circle with one
        /// bolt missing. Out-of-range entries are ignored.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        skip: Vec<u32>,
    },

    /// Cut a screw thread into `source`: a 60° V-groove (ISO basic profile:
    /// flat crest and root) swept along `helix` and subtracted. The helix
    /// radius is the surface being threaded — a shaft at its major diameter
    /// (`internal: false`, the groove cuts inward) or a hole at its minor
    /// diameter (`internal: true`, the groove cuts outward). `depth` is the
    /// groove depth; [`Helix::iso_depth`] gives the ISO value for a pitch.
    Thread {
        source: FeatureId,
        helix: Helix,
        depth: f64,
        internal: bool,
        /// Thread starts: the groove is repeated `starts` times around the axis
        /// (`2π / starts` apart), each on the same helix, so the helix pitch is
        /// the lead and adjacent grooves sit `pitch / starts` apart (which sizes
        /// the groove). 1 for an ordinary screw; many, with a steep pitch, cut
        /// in both hands, make a knurl.
        #[serde(default = "one_start")]
        starts: u32,
    },
}

fn one_start() -> u32 {
    1
}

/// A helix: the spine of a thread or a coil. It winds around the axis through
/// `center` with direction `axis` at `radius`, climbing `pitch` per turn for
/// `length` along the axis (so `length / pitch` turns), right-handed unless
/// `left_handed`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Helix {
    pub center: DVec3,
    pub axis: DVec3,
    pub radius: f64,
    pub pitch: f64,
    pub length: f64,
    pub left_handed: bool,
}

impl Helix {
    /// The ISO 261 coarse pitch for a nominal `diameter` (mm), by nearest
    /// standard size — the default a thread tool offers for a picked cylinder.
    pub fn iso_pitch(diameter: f64) -> f64 {
        const COARSE: [(f64, f64); 16] = [
            (1.6, 0.35), (2.0, 0.4), (2.5, 0.45), (3.0, 0.5), (4.0, 0.7), (5.0, 0.8),
            (6.0, 1.0), (8.0, 1.25), (10.0, 1.5), (12.0, 1.75), (16.0, 2.0), (20.0, 2.5),
            (24.0, 3.0), (30.0, 3.5), (36.0, 4.0), (42.0, 4.5),
        ];
        COARSE
            .iter()
            .min_by(|a, b| (a.0 - diameter).abs().total_cmp(&(b.0 - diameter).abs()))
            .map_or(1.0, |&(_, p)| p)
    }

    /// The ISO coarse pitch for a hole seen at its MINOR diameter (the
    /// tap-drill size): the nominal size is the minor diameter plus twice the
    /// thread depth, which itself depends on the pitch — so iterate to a
    /// fixed point. An 8.5 mm hole is M10 (1.5), not M8.
    pub fn iso_pitch_for_minor(minor_diameter: f64) -> f64 {
        let mut pitch = Self::iso_pitch(minor_diameter);
        for _ in 0..4 {
            pitch = Self::iso_pitch(minor_diameter + 2.0 * Self::iso_depth(pitch));
        }
        pitch
    }

    /// The ISO basic-profile thread depth for `pitch`: 5/8 of the fundamental
    /// triangle height (H = 0.866 p), between the crest and root flats.
    pub fn iso_depth(pitch: f64) -> f64 {
        0.625 * 0.866_025_4 * pitch
    }

    /// The number of turns.
    pub fn turns(&self) -> f64 {
        self.length / self.pitch
    }
}

/// How far past the surface (into air) a thread groove's base reaches, for a
/// groove `pitch` apart from its neighbours — what [`thread_profile`] uses,
/// and what must stay smaller than a hole's radius.
pub fn thread_overlap(pitch: f64) -> f64 {
    0.05 * pitch
}

/// The groove a [`FeatureKind::Thread`] cuts, as a closed loop in the helix's
/// axial plane: `u` is the distance from the axis, `v` the position along it
/// (0 at the helix point, so the groove is centred on the helix). ISO basic
/// profile: a crest flat of `p/8` is left standing at the surface and the root
/// flat is `p/4`, which with the ISO depth gives 60° flanks. The groove's base
/// is pushed a little past the surface (into air) so the cut is clean.
pub fn thread_profile(pitch: f64, depth: f64, radius: f64, internal: bool) -> Vec<ProfileElem> {
    let overlap = thread_overlap(pitch);
    // Half-widths: at the surface the groove is p - p/8 wide; the flank
    // continues through the overlap at the same slope.
    let flank = (0.4375 - 0.125) / Helix::iso_depth(1.0); // Δu per Δv at ISO depth
    let half_base = 0.4375 * pitch + overlap * flank;
    let half_root = 0.125 * pitch;
    // The groove points INTO the material: inward on a shaft, outward in a hole.
    let (u_base, u_root) = if internal {
        (radius - overlap, radius + depth)
    } else {
        (radius + overlap, radius - depth)
    };
    let pts = [
        [u_base, -half_base],
        [u_root, -half_root],
        [u_root, half_root],
        [u_base, half_base],
    ];
    (0..4)
        .map(|i| ProfileElem::Line { a: pts[i], b: pts[(i + 1) % 4] })
        .collect()
}

/// Where a [`FeatureKind::Pattern`] puts its instances. Instance `i` (from 0)
/// is the seed moved by `i` steps; the step is stored per instance so the
/// layout is independent of `count`. An enum so a path layout (along a sketch
/// curve or a helix) can be added as a variant without breaking saved projects.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PatternLayout {
    /// Instances along a straight line: instance `i` is the seed translated by
    /// `i * step`.
    Linear { step: DVec3 },
    /// Instances around an axis: instance `i` is the seed rotated by
    /// `i * angle` radians about the line through `center` with direction
    /// `axis`. Spread `count` instances over a full turn with
    /// `angle = 2π / count`.
    Circular {
        center: DVec3,
        axis: DVec3,
        angle: f64,
    },
    /// Instances along a helix (a screw motion): instance `i` is the seed
    /// rotated by `i * angle` about the axis AND moved `i * rise` along it.
    /// Bumps spiralling up a knob; with `rise = 0` it is a circular pattern.
    Helical {
        center: DVec3,
        axis: DVec3,
        angle: f64,
        rise: f64,
    },
}

impl PatternLayout {
    pub fn type_name(self) -> &'static str {
        match self {
            PatternLayout::Linear { .. } => "Linear",
            PatternLayout::Circular { .. } => "Circular",
            PatternLayout::Helical { .. } => "Helical",
        }
    }
}

/// The path a [`FeatureKind::Sweep`] follows. An enum so future path *sources* —
/// a free-space 3D curve, or references to existing model edges — slot in as new
/// variants without disturbing the planar case or breaking saved projects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SweepPath {
    /// An open chain drawn on `plane` (the current MVP). The kernel still sweeps
    /// along a generic spine, so a 3D variant needs no kernel change.
    Planar { plane: SketchPlane, sketch: Sketch2d },
    /// A helix. The profile is swept from wherever it was sketched with the
    /// helix axis as binormal, so a profile drawn in a plane containing the
    /// axis stays in that axial plane all the way round: sketch a circle at
    /// the coil radius on a plane through the axis and you get a spring.
    Helix { helix: Helix },
}

impl FeatureKind {
    /// The base of an auto-numbered name for a feature of this kind, matching
    /// what the app calls the ones it creates (`Box 1`, `Move 2`, `Subtract 3`).
    pub fn display_name(&self) -> &'static str {
        match self {
            FeatureKind::Box { .. } => "Box",
            FeatureKind::Cylinder { .. } => "Cylinder",
            FeatureKind::Sphere { .. } => "Sphere",
            FeatureKind::Sketch { .. } | FeatureKind::ConstraintSketch { .. } => "Sketch",
            FeatureKind::Extrude { .. } => "Extrude",
            FeatureKind::Translate { .. } => "Move",
            FeatureKind::Boolean { op: BooleanOp::Union, .. } => "Union",
            FeatureKind::Boolean { op: BooleanOp::Subtract, .. } => "Subtract",
            FeatureKind::Boolean { op: BooleanOp::Intersect, .. } => "Intersect",
            FeatureKind::FilletAll { .. } | FeatureKind::Fillet { .. } => "Fillet",
            FeatureKind::ChamferAll { .. } | FeatureKind::Chamfer { .. } => "Chamfer",
            FeatureKind::Shell { .. } => "Shell",
            FeatureKind::PushPull { .. } => "Push/Pull",
            FeatureKind::Rotate { .. } => "Rotate",
            FeatureKind::Scale { .. } => "Scale",
            FeatureKind::Revolve { .. } => "Revolve",
            FeatureKind::Mirror { .. } => "Mirror",
            FeatureKind::Group { .. } => "Group",
            FeatureKind::Sweep { .. } => "Sweep",
            FeatureKind::Pattern { .. } => "Pattern",
            FeatureKind::Thread { .. } => "Thread",
        }
    }

    /// Whether this is an in-place body transform — the kinds
    /// [`crate::History::insert_transform`] splices into a chain (and a
    /// group's chain bakes onto its members).
    pub fn is_in_place_transform(&self) -> bool {
        matches!(
            self,
            FeatureKind::Translate { .. }
                | FeatureKind::Rotate { .. }
                | FeatureKind::Scale { .. }
                | FeatureKind::Mirror { .. }
        )
    }

    /// Whether [`crate::History::insert_transform`] may splice this kind into
    /// its source's chain: the in-place transforms, plus a pattern (the same
    /// body, repeated — so a patterned tool repeats its cut downstream).
    pub fn can_splice(&self) -> bool {
        self.is_in_place_transform() || matches!(self, FeatureKind::Pattern { .. })
    }

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
            FeatureKind::ChamferAll { source, .. } => vec![*source],
            FeatureKind::Chamfer { source, .. } => vec![*source],
            FeatureKind::Shell { source, .. } => vec![*source],
            FeatureKind::Extrude { source, .. } => vec![*source],
            FeatureKind::PushPull { source, .. } => vec![*source],
            FeatureKind::Rotate { source, .. } => vec![*source],
            FeatureKind::Scale { source, .. } => vec![*source],
            FeatureKind::Revolve { source, .. } => vec![*source],
            FeatureKind::Mirror { source, .. } => vec![*source],
            FeatureKind::Boolean { target, tool, .. } => vec![*target, *tool],
            FeatureKind::Group { members } => members.clone(),
            FeatureKind::Sweep { profile, .. } => vec![*profile],
            FeatureKind::Pattern { source, .. } | FeatureKind::Thread { source, .. } => {
                vec![*source]
            }
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
            | FeatureKind::ChamferAll { source, .. }
            | FeatureKind::Chamfer { source, .. }
            | FeatureKind::Shell { source, .. }
            | FeatureKind::Extrude { source, .. }
            | FeatureKind::PushPull { source, .. }
            | FeatureKind::Rotate { source, .. }
            | FeatureKind::Scale { source, .. }
            | FeatureKind::Revolve { source, .. }
            | FeatureKind::Mirror { source, .. }
            | FeatureKind::Sweep { profile: source, .. }
            | FeatureKind::Pattern { source, .. }
            | FeatureKind::Thread { source, .. } => Some(*source),
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
            | FeatureKind::ChamferAll { source, .. }
            | FeatureKind::Chamfer { source, .. }
            | FeatureKind::Shell { source, .. }
            | FeatureKind::Extrude { source, .. }
            | FeatureKind::PushPull { source, .. }
            | FeatureKind::Rotate { source, .. }
            | FeatureKind::Scale { source, .. }
            | FeatureKind::Revolve { source, .. }
            | FeatureKind::Mirror { source, .. }
            | FeatureKind::Sweep { profile: source, .. }
            | FeatureKind::Pattern { source, .. }
            | FeatureKind::Thread { source, .. } => swap(source),
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
            FeatureKind::ChamferAll { .. } => "Chamfer",
            FeatureKind::Chamfer { .. } => "Chamfer",
            FeatureKind::Shell { .. } => "Shell",
            FeatureKind::PushPull { .. } => "Push/Pull",
            FeatureKind::Rotate { .. } => "Rotate",
            FeatureKind::Scale { .. } => "Scale",
            FeatureKind::Revolve { .. } => "Revolve",
            FeatureKind::Mirror { .. } => "Mirror",
            FeatureKind::Group { .. } => "Group",
            FeatureKind::Sweep { .. } => "Sweep",
            FeatureKind::Pattern { .. } => "Pattern",
            FeatureKind::Thread { .. } => "Thread",
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
    /// Optional user-chosen display colour (linear RGB). A body's colour is the
    /// nearest set colour walking down its `primary_input` chain; `None`
    /// everywhere means the viewport's theme default. `#[serde(default)]` keeps
    /// older project files (which lack the field) loading.
    #[serde(default)]
    pub color: Option<[f32; 3]>,
}
