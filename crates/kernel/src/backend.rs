//! OCCT implementation of the pure-core [`GeometryBackend`] trait.
//!
//! This is the seam where `cdt-core`'s data-described features become real
//! B-rep solids. The replay engine in `cdt-core` calls these methods; we
//! forward each to the safe [`Solid`] API. Bodies are [`Body`] — the solid
//! plus its element map — and errors are [`KernelError`], so a failed
//! operation surfaces as a per-feature regen error rather than aborting the
//! whole rebuild.

use std::sync::Arc;

use cdt_core::{
    BooleanOp, DVec3, EdgeAnchor, FaceAnchor, FeatureId, GeometryBackend, Profile, ProfileElem,
    SketchPlane,
};

use crate::elements::ElementMap;
use crate::{KernelError, Solid};

/// A kernel body: the OCCT solid plus the element map naming its faces and
/// edges (persistent identity — see [`crate::elements`]). Cheap to clone —
/// the solid is a shared handle and the map is behind an `Arc` — which the
/// regen cache relies on.
#[derive(Clone)]
pub struct Body {
    pub solid: Solid,
    /// Empty for bodies from operations that don't propagate identity yet
    /// (booleans, fillets, extrude, … — arriving slice by slice); lookups on
    /// an empty map miss, and callers fall back to geometric anchors.
    pub elements: Arc<ElementMap>,
}

/// The element map rides along wherever the body goes; everything else —
/// tessellation, measurement, picking queries, export — reads it as a
/// [`Solid`].
impl std::ops::Deref for Body {
    type Target = Solid;
    fn deref(&self) -> &Solid {
        &self.solid
    }
}

impl Body {
    /// A body whose elements are not (yet) tracked — the honest state for
    /// operations whose name propagation lands in a later slice.
    fn untracked(solid: Solid) -> Self {
        Body {
            solid,
            elements: Arc::new(ElementMap::default()),
        }
    }

    /// The same solid moved/reflected/scaled: a transform changes WHERE the
    /// elements are, never WHICH elements exist, so the map carries over
    /// (sub-shape order is preserved — locked by tests/element_map.rs).
    fn transformed(&self, solid: Solid) -> Self {
        Body {
            solid,
            elements: self.elements.clone(),
        }
    }
}

/// A [`GeometryBackend`] backed by OpenCASCADE.
#[derive(Default)]
pub struct KernelBackend {
    /// The feature currently being evaluated — the scope ("origin") under
    /// which a newly born body's element names are minted. Set by regen via
    /// [`GeometryBackend::begin_feature`]; 0 outside a regen pass (e.g. a
    /// body built directly in a test), which is harmless: no real feature has
    /// id 0.
    current_feature: u64,
}

impl KernelBackend {
    /// Wrap a newly BORN body (primitive or sketch face): every face and edge
    /// is named `Born` under the feature being evaluated.
    fn born(&self, solid: Solid) -> Result<Body, KernelError> {
        let faces = solid.face_count()?;
        let edges = solid.edge_count()?;
        Ok(Body {
            elements: Arc::new(ElementMap::born(self.current_feature, faces, edges)),
            solid,
        })
    }
}

impl GeometryBackend for KernelBackend {
    type Body = Body;
    type Error = KernelError;

    fn begin_feature(&mut self, id: FeatureId) {
        self.current_feature = id.0;
    }

    fn make_box(&mut self, size: DVec3) -> Result<Body, KernelError> {
        self.born(Solid::cuboid(size.x, size.y, size.z)?)
    }

    fn make_cylinder(&mut self, radius: f64, height: f64) -> Result<Body, KernelError> {
        self.born(Solid::cylinder(radius, height)?)
    }

    fn make_sphere(&mut self, radius: f64) -> Result<Body, KernelError> {
        self.born(Solid::sphere(radius)?)
    }

    fn sketch(&mut self, plane: SketchPlane, profile: Profile) -> Result<Body, KernelError> {
        let origin = plane.origin().to_array();
        let solid = match profile {
            Profile::Rectangle { width, height } => Solid::rectangle_face(
                origin,
                plane.x_dir().to_array(),
                plane.y_dir().to_array(),
                width,
                height,
            )?,
            Profile::Circle { radius } => {
                Solid::circle_face(origin, plane.normal().to_array(), radius)?
            }
        };
        self.born(solid)
    }

    fn sketch_profile(
        &mut self,
        plane: SketchPlane,
        elements: &[ProfileElem],
    ) -> Result<Body, KernelError> {
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
        let solid = Solid::profile_face(
            plane.origin().to_array(),
            plane.x_dir().to_array(),
            plane.y_dir().to_array(),
            &points,
            &segs,
        )?;
        self.born(solid)
    }

    fn extrude(&mut self, profile: &Body, distance: f64) -> Result<Body, KernelError> {
        Ok(Body::untracked(profile.solid.extrude(distance)?))
    }

    fn sweep(
        &mut self,
        profile: &Body,
        path_plane: SketchPlane,
        path: &[ProfileElem],
    ) -> Result<Body, KernelError> {
        // Flatten the open chain to path_wire's shape: the chain vertices (each
        // element's start, plus the final element's end) and 5 doubles per
        // segment ([is_bezier, c1x, c1y, c2x, c2y]).
        let mut points: Vec<f64> = Vec::with_capacity((path.len() + 1) * 2);
        let mut segs: Vec<f64> = Vec::with_capacity(path.len() * 5);
        for e in path {
            let [px, py] = e.start();
            points.push(px);
            points.push(py);
            match *e {
                ProfileElem::Line { .. } => segs.extend([0.0, 0.0, 0.0, 0.0, 0.0]),
                ProfileElem::Bezier { c1, c2, .. } => segs.extend([1.0, c1[0], c1[1], c2[0], c2[1]]),
            }
        }
        if let Some(last) = path.last() {
            let [ex, ey] = last.end();
            points.push(ex);
            points.push(ey);
        }
        let spine = Solid::path_wire(
            path_plane.origin().to_array(),
            path_plane.x_dir().to_array(),
            path_plane.y_dir().to_array(),
            &points,
            &segs,
        )?;
        // Lock the section's roll to the path plane's normal so an asymmetric
        // profile keeps a predictable orientation through bends.
        Ok(Body::untracked(
            profile.solid.sweep(&spine, path_plane.normal().to_array())?,
        ))
    }

    fn translate(&mut self, body: &Body, offset: DVec3) -> Result<Body, KernelError> {
        Ok(body.transformed(body.solid.translate(offset.x, offset.y, offset.z)?))
    }

    fn boolean(
        &mut self,
        op: BooleanOp,
        target: &Body,
        tool: &Body,
    ) -> Result<Body, KernelError> {
        let solid = match op {
            BooleanOp::Union => target.solid.fuse(&tool.solid),
            BooleanOp::Subtract => target.solid.cut(&tool.solid),
            BooleanOp::Intersect => target.solid.common(&tool.solid),
        }?;
        Ok(Body::untracked(solid))
    }

    fn fillet_all(&mut self, body: &Body, radius: f64) -> Result<Body, KernelError> {
        Ok(Body::untracked(body.solid.fillet_all_edges(radius)?))
    }

    fn fillet_edges(
        &mut self,
        body: &Body,
        anchors: &[EdgeAnchor],
        radius: f64,
    ) -> Result<Body, KernelError> {
        let points: Vec<[f64; 3]> = anchors.iter().map(|a| a.point.to_array()).collect();
        Ok(Body::untracked(body.solid.fillet_edges(&points, radius)?))
    }

    fn push_pull(
        &mut self,
        body: &Body,
        anchor: FaceAnchor,
        distance: f64,
    ) -> Result<Body, KernelError> {
        Ok(Body::untracked(body.solid.push_pull(
            anchor.point.to_array(),
            anchor.normal.to_array(),
            distance,
        )?))
    }

    fn rotate(
        &mut self,
        body: &Body,
        center: DVec3,
        axis: DVec3,
        angle: f64,
    ) -> Result<Body, KernelError> {
        Ok(body.transformed(body.solid.rotate(center.to_array(), axis.to_array(), angle)?))
    }

    fn scale(
        &mut self,
        body: &Body,
        factors: DVec3,
        anchor: DVec3,
    ) -> Result<Body, KernelError> {
        Ok(body.transformed(body.solid.scale(factors.to_array(), anchor.to_array())?))
    }

    fn revolve(
        &mut self,
        profile: &Body,
        axis_point: DVec3,
        angle: f64,
    ) -> Result<Body, KernelError> {
        Ok(Body::untracked(
            profile.solid.revolve(axis_point.to_array(), angle)?,
        ))
    }

    fn mirror(
        &mut self,
        body: &Body,
        origin: DVec3,
        normal: DVec3,
    ) -> Result<Body, KernelError> {
        Ok(body.transformed(body.solid.mirror(origin.to_array(), normal.to_array())?))
    }

    fn compound(&mut self, members: &[&Body]) -> Result<Body, KernelError> {
        // Fold the members into one compound (regen guarantees ≥1).
        let (first, rest) = members.split_first().expect("group has ≥1 member");
        let mut acc = first.solid.clone();
        for m in rest {
            acc = acc.compound(&m.solid)?;
        }
        Ok(Body::untracked(acc))
    }
}
