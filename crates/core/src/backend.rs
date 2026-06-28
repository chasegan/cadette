//! The abstraction that lets pure `core` drive a real geometry kernel.
//!
//! `core` describes *what* to build (features as data); a `GeometryBackend`
//! knows *how* to build it. The kernel crate implements this trait over OCCT;
//! tests implement it over trivial in-memory values. This inversion is what
//! keeps `core` free of any C++/kernel dependency while still owning the replay
//! logic in [`crate::regen`].

use glam::DVec3;

use crate::features::{BooleanOp, EdgeAnchor, FaceAnchor};
use crate::sketch::{Profile, SketchPlane};

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

    /// Build a planar face from a closed profile on a base plane.
    fn sketch(&mut self, plane: SketchPlane, profile: Profile) -> Result<Self::Body, Self::Error>;

    /// Build a planar face from an ordered closed loop of 2D points (in plane
    /// coordinates) — the resolved geometry of a constraint sketch.
    fn sketch_loop(
        &mut self,
        plane: SketchPlane,
        points: &[[f64; 2]],
    ) -> Result<Self::Body, Self::Error>;

    /// Extrude a planar face (`profile`) along its normal by `distance`.
    fn extrude(&mut self, profile: &Self::Body, distance: f64) -> Result<Self::Body, Self::Error>;

    /// Revolve a planar `profile` by `angle` radians about the straight edge
    /// nearest `axis_point` (one of the profile's own segments or a model edge).
    fn revolve(
        &mut self,
        profile: &Self::Body,
        axis_point: DVec3,
        angle: f64,
    ) -> Result<Self::Body, Self::Error>;

    fn translate(&mut self, body: &Self::Body, offset: DVec3) -> Result<Self::Body, Self::Error>;

    fn boolean(
        &mut self,
        op: BooleanOp,
        target: &Self::Body,
        tool: &Self::Body,
    ) -> Result<Self::Body, Self::Error>;

    fn fillet_all(&mut self, body: &Self::Body, radius: f64) -> Result<Self::Body, Self::Error>;

    /// Fillet the edges of `body` nearest each anchor in `anchors` with
    /// `radius`, in a single operation.
    fn fillet_edges(
        &mut self,
        body: &Self::Body,
        anchors: &[EdgeAnchor],
        radius: f64,
    ) -> Result<Self::Body, Self::Error>;

    /// Push or pull the planar face of `body` identified by `anchor` along its
    /// normal by `distance` (positive adds material, negative removes).
    fn push_pull(
        &mut self,
        body: &Self::Body,
        anchor: FaceAnchor,
        distance: f64,
    ) -> Result<Self::Body, Self::Error>;

    /// Rotate `body` by `angle` radians about the line through `center` with
    /// direction `axis`.
    fn rotate(
        &mut self,
        body: &Self::Body,
        center: DVec3,
        axis: DVec3,
        angle: f64,
    ) -> Result<Self::Body, Self::Error>;

    /// Non-uniformly scale `body` by per-axis `factors` about the fixed point
    /// `anchor`: a point `p` maps to `anchor + factors * (p - anchor)`.
    fn scale(
        &mut self,
        body: &Self::Body,
        factors: DVec3,
        anchor: DVec3,
    ) -> Result<Self::Body, Self::Error>;
}
