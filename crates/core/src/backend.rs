//! The abstraction that lets pure `core` drive a real geometry kernel.
//!
//! `core` describes *what* to build (features as data); a `GeometryBackend`
//! knows *how* to build it. The kernel crate implements this trait over OCCT;
//! tests implement it over trivial in-memory values. This inversion is what
//! keeps `core` free of any C++/kernel dependency while still owning the replay
//! logic in [`crate::regen`].

use glam::DVec3;

use crate::features::BooleanOp;

/// A geometry engine capable of producing and combining solid bodies.
///
/// Operations take inputs by reference and return a new owned body, mirroring
/// OCCT's value semantics — implementations need not make `Body` cloneable.
/// `&mut self` is allowed so backends may carry caches or scratch state.
pub trait GeometryBackend {
    /// An opaque solid body produced by this backend.
    type Body;
    /// The backend's failure type (e.g. an infeasible fillet radius).
    type Error;

    fn make_box(&mut self, size: DVec3) -> Result<Self::Body, Self::Error>;
    fn make_cylinder(&mut self, radius: f64, height: f64) -> Result<Self::Body, Self::Error>;
    fn make_sphere(&mut self, radius: f64) -> Result<Self::Body, Self::Error>;

    fn translate(&mut self, body: &Self::Body, offset: DVec3) -> Result<Self::Body, Self::Error>;

    fn boolean(
        &mut self,
        op: BooleanOp,
        target: &Self::Body,
        tool: &Self::Body,
    ) -> Result<Self::Body, Self::Error>;

    fn fillet_all(&mut self, body: &Self::Body, radius: f64) -> Result<Self::Body, Self::Error>;
}
