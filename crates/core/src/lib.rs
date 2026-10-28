//! # cdt-core
//!
//! Pure domain logic with **no** dependency on the geometry kernel or GPU.
//! Features are described as *data* ([`FeatureKind`]); the [`regenerate`] engine
//! replays a [`Document`]'s history against any [`GeometryBackend`] to produce
//! geometry. Because the backend is abstract, this whole crate — including the
//! parametric replay — is testable without OCCT.
//!
//! Layering:
//! - [`features`] — operations as serializable data.
//! - [`edit`] — history mutations as serializable data ([`DocEdit`]).
//! - [`history`] — the ordered feature tree + dependency validation.
//! - [`document`] — a named history with units and a rollback bar.
//! - [`backend`] — the trait a kernel implements.
//! - [`solver`] — the trait a sketch constraint solver implements.
//! - [`regen`] — the replay engine tying them together.

pub mod assets;
pub mod backend;
pub mod document;
pub mod edit;
pub mod features;
pub mod build_graph;
pub mod history;
pub mod regen;
pub mod selection;
pub mod sketch;
pub mod solver;
pub mod units;

pub use assets::Assets;
pub use backend::GeometryBackend;
pub use document::Document;
pub use edit::{decode_import_data, import_feature_name, BatchError, DocEdit, EditError, EditOutcome};
pub use features::{
    thread_profile, BooleanOp, EdgeAnchor, FaceAnchor, Feature, FeatureId, FeatureKind, Helix,
    ImportFormat, PatternLayout, SweepPath,
};
pub use build_graph::{BuildGraph, BuildGraphRow, LANE_PALETTE};
pub use glam::DVec3;
pub use history::{DependencyError, History};
pub use regen::{
    regenerate, regenerate_solved, MissingReason, RegenCache, RegenError, Regeneration, pattern_instances, MAX_PATTERN_COUNT, MAX_THREAD_STARTS, MAX_THREAD_TURNS,
};
pub use selection::Pick;
pub use sketch::{
    CircleId, Constraint, LineId, PointId, Profile, ProfileElem, Sketch2d, SketchBezier,
    SketchCircle, SketchLine, SketchPlane, SketchPoint,
};
pub use solver::{NoSolver, SketchSolver, SolveStatus};
pub use units::LengthUnit;
