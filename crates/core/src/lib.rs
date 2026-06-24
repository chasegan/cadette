//! # rmf-core
//!
//! Pure domain logic with **no** dependency on the geometry kernel or GPU.
//! Features are described as *data* ([`FeatureKind`]); the [`regenerate`] engine
//! replays a [`Document`]'s history against any [`GeometryBackend`] to produce
//! geometry. Because the backend is abstract, this whole crate — including the
//! parametric replay — is testable without OCCT.
//!
//! Layering:
//! - [`features`] — operations as serializable data.
//! - [`history`] — the ordered feature tree + dependency validation.
//! - [`document`] — a named history with units and a rollback bar.
//! - [`backend`] — the trait a kernel implements.
//! - [`regen`] — the replay engine tying them together.

pub mod backend;
pub mod document;
pub mod features;
pub mod history;
pub mod regen;
pub mod selection;
pub mod sketch;
pub mod units;

pub use backend::GeometryBackend;
pub use document::Document;
pub use features::{BooleanOp, Feature, FeatureId, FeatureKind};
pub use glam::DVec3;
pub use history::{DependencyError, History};
pub use regen::{regenerate, RegenError, Regeneration};
pub use selection::Pick;
pub use sketch::{
    CircleId, Constraint, LineId, PointId, Profile, Sketch2d, SketchCircle, SketchLine,
    SketchPlane, SketchPoint,
};
pub use units::LengthUnit;
