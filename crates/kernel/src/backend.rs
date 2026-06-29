//! OCCT implementation of the pure-core [`GeometryBackend`] trait.
//!
//! This is the seam where `cdt-core`'s data-described features become real
//! B-rep solids. The replay engine in `cdt-core` calls these methods; we
//! forward each to the safe [`Solid`] API. Bodies are `Solid` and errors are
//! [`KernelError`], so a failed operation surfaces as a per-feature regen error
//! rather than aborting the whole rebuild.

use cdt_core::{
    BooleanOp, DVec3, EdgeAnchor, FaceAnchor, GeometryBackend, Profile, ProfileElem, SketchPlane,
};

use crate::{KernelError, Solid};

/// A [`GeometryBackend`] backed by OpenCASCADE.
#[derive(Default)]
pub struct KernelBackend;

impl GeometryBackend for KernelBackend {
    type Body = Solid;
    type Error = KernelError;

    fn make_box(&mut self, size: DVec3) -> Result<Solid, KernelError> {
        Solid::cuboid(size.x, size.y, size.z)
    }

    fn make_cylinder(&mut self, radius: f64, height: f64) -> Result<Solid, KernelError> {
        Solid::cylinder(radius, height)
    }

    fn make_sphere(&mut self, radius: f64) -> Result<Solid, KernelError> {
        Solid::sphere(radius)
    }

    fn sketch(&mut self, plane: SketchPlane, profile: Profile) -> Result<Solid, KernelError> {
        let origin = plane.origin().to_array();
        match profile {
            Profile::Rectangle { width, height } => Solid::rectangle_face(
                origin,
                plane.x_dir().to_array(),
                plane.y_dir().to_array(),
                width,
                height,
            ),
            Profile::Circle { radius } => {
                Solid::circle_face(origin, plane.normal().to_array(), radius)
            }
        }
    }

    fn sketch_profile(
        &mut self,
        plane: SketchPlane,
        elements: &[ProfileElem],
    ) -> Result<Solid, KernelError> {
        // Flatten to the FFI shape: the loop vertices (each element's start) and
        // 5 doubles per segment ([is_bezier, c1x, c1y, c2x, c2y]).
        let mut points: Vec<f64> = Vec::with_capacity(elements.len() * 2);
        let mut segs: Vec<f64> = Vec::with_capacity(elements.len() * 5);
        for e in elements {
            let [px, py] = e.start();
            points.push(px);
            points.push(py);
            match *e {
                ProfileElem::Line { .. } => segs.extend([0.0, 0.0, 0.0, 0.0, 0.0]),
                ProfileElem::Bezier { c1, c2, .. } => {
                    segs.extend([1.0, c1[0], c1[1], c2[0], c2[1]])
                }
            }
        }
        Solid::profile_face(
            plane.origin().to_array(),
            plane.x_dir().to_array(),
            plane.y_dir().to_array(),
            &points,
            &segs,
        )
    }

    fn extrude(&mut self, profile: &Solid, distance: f64) -> Result<Solid, KernelError> {
        profile.extrude(distance)
    }

    fn translate(&mut self, body: &Solid, offset: DVec3) -> Result<Solid, KernelError> {
        body.translate(offset.x, offset.y, offset.z)
    }

    fn boolean(
        &mut self,
        op: BooleanOp,
        target: &Solid,
        tool: &Solid,
    ) -> Result<Solid, KernelError> {
        match op {
            BooleanOp::Union => target.fuse(tool),
            BooleanOp::Subtract => target.cut(tool),
            BooleanOp::Intersect => target.common(tool),
        }
    }

    fn fillet_all(&mut self, body: &Solid, radius: f64) -> Result<Solid, KernelError> {
        body.fillet_all_edges(radius)
    }

    fn fillet_edges(
        &mut self,
        body: &Solid,
        anchors: &[EdgeAnchor],
        radius: f64,
    ) -> Result<Solid, KernelError> {
        let points: Vec<[f64; 3]> = anchors.iter().map(|a| a.point.to_array()).collect();
        body.fillet_edges(&points, radius)
    }

    fn push_pull(
        &mut self,
        body: &Solid,
        anchor: FaceAnchor,
        distance: f64,
    ) -> Result<Solid, KernelError> {
        body.push_pull(anchor.point.to_array(), anchor.normal.to_array(), distance)
    }

    fn rotate(
        &mut self,
        body: &Solid,
        center: DVec3,
        axis: DVec3,
        angle: f64,
    ) -> Result<Solid, KernelError> {
        body.rotate(center.to_array(), axis.to_array(), angle)
    }

    fn scale(
        &mut self,
        body: &Solid,
        factors: DVec3,
        anchor: DVec3,
    ) -> Result<Solid, KernelError> {
        body.scale(factors.to_array(), anchor.to_array())
    }

    fn revolve(
        &mut self,
        profile: &Solid,
        axis_point: DVec3,
        angle: f64,
    ) -> Result<Solid, KernelError> {
        profile.revolve(axis_point.to_array(), angle)
    }

    fn mirror(
        &mut self,
        body: &Solid,
        origin: DVec3,
        normal: DVec3,
    ) -> Result<Solid, KernelError> {
        body.mirror(origin.to_array(), normal.to_array())
    }

    fn compound(&mut self, members: &[&Solid]) -> Result<Solid, KernelError> {
        // Fold the members into one compound (regen guarantees ≥1).
        let (first, rest) = members.split_first().expect("group has ≥1 member");
        let mut acc = (*first).clone();
        for m in rest {
            acc = acc.compound(m)?;
        }
        Ok(acc)
    }
}
