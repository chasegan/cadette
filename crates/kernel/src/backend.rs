//! OCCT implementation of the pure-core [`GeometryBackend`] trait.
//!
//! This is the seam where `rmf-core`'s data-described features become real
//! B-rep solids. The replay engine in `rmf-core` calls these methods; we
//! forward each to the safe [`Solid`] API. Bodies are `Solid` and errors are
//! [`KernelError`], so a failed operation surfaces as a per-feature regen error
//! rather than aborting the whole rebuild.

use rmf_core::{BooleanOp, DVec3, EdgeAnchor, FaceAnchor, GeometryBackend, Profile, SketchPlane};

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

    fn sketch_loop(
        &mut self,
        plane: SketchPlane,
        points: &[[f64; 2]],
    ) -> Result<Solid, KernelError> {
        let flat: Vec<f64> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
        Solid::polygon_face(
            plane.origin().to_array(),
            plane.x_dir().to_array(),
            plane.y_dir().to_array(),
            &flat,
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
}
