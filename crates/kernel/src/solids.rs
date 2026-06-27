//! Safe, owned wrappers over the OCCT FFI.
//!
//! [`Solid`] owns a B-rep `TopoDS_Shape` behind a `UniquePtr` and exposes the
//! modeling operations as ordinary, fallible Rust methods. This is the surface
//! the rest of the application builds on — nothing above this layer should ever
//! reach into [`crate::ffi`].

use cxx::UniquePtr;

use crate::{ffi, Result};

/// Re-export the tessellated mesh produced by [`Solid::tessellate`].
pub use crate::ffi::Mesh;

/// An owned B-rep solid (or, generally, any OCCT topology).
pub struct Solid(UniquePtr<ffi::Shape>);

/// A `TopoDS_Shape` is a handle to shared, ref-counted geometry, so cloning is
/// cheap (a handle copy, not a deep geometry copy). The regen cache relies on
/// this to hand out cached bodies each frame without re-running OCCT.
impl Clone for Solid {
    fn clone(&self) -> Self {
        Solid::wrap(ffi::copy_shape(&self.0))
    }
}

impl Solid {
    fn wrap(inner: UniquePtr<ffi::Shape>) -> Self {
        Solid(inner)
    }

    // --- Primitives ---------------------------------------------------------

    /// An axis-aligned box with one corner at the origin and the opposite at
    /// `(dx, dy, dz)`.
    pub fn cuboid(dx: f64, dy: f64, dz: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::make_box(dx, dy, dz)?))
    }

    /// A sphere of the given `radius`, centered at the origin.
    pub fn sphere(radius: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::make_sphere(radius)?))
    }

    /// A cylinder of `radius` and `height`, axis along +Z from the origin.
    pub fn cylinder(radius: f64, height: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::make_cylinder(radius, height)?))
    }

    // --- Sketch profiles ----------------------------------------------------

    /// A rectangular planar face centered at `origin`, spanning `width` along
    /// the unit `x_dir` and `height` along the unit `y_dir`.
    pub fn rectangle_face(
        origin: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        width: f64,
        height: f64,
    ) -> Result<Self> {
        Ok(Self::wrap(ffi::make_rectangle_face(
            origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], y_dir[0], y_dir[1],
            y_dir[2], width, height,
        )?))
    }

    /// A circular planar face centered at `origin` on the plane with unit
    /// `normal`.
    pub fn circle_face(origin: [f64; 3], normal: [f64; 3], radius: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::make_circle_face(
            origin[0], origin[1], origin[2], normal[0], normal[1], normal[2], radius,
        )?))
    }

    /// A polygonal planar face from a flat `[x0, y0, x1, y1, ...]` loop mapped
    /// onto the plane `origin + x*x_dir + y*y_dir`.
    pub fn polygon_face(
        origin: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        points: &[f64],
    ) -> Result<Self> {
        Ok(Self::wrap(ffi::make_polygon_face(
            origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], y_dir[0], y_dir[1],
            y_dir[2], points,
        )?))
    }

    // --- Transforms ---------------------------------------------------------

    /// A copy of this solid translated by `(dx, dy, dz)`.
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::translate(&self.0, dx, dy, dz)?))
    }

    /// A copy of this solid rotated by `angle` radians about the line through
    /// `center` with direction `axis`.
    pub fn rotate(&self, center: [f64; 3], axis: [f64; 3], angle: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::rotate(
            &self.0, center[0], center[1], center[2], axis[0], axis[1], axis[2], angle,
        )?))
    }

    /// Extrude this planar face along its normal by `distance` into a solid.
    pub fn extrude(&self, distance: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::extrude(&self.0, distance)?))
    }

    /// Push/pull the planar face anchored at `point` with `normal` along that
    /// normal by `distance` (positive fuses material, negative cuts).
    pub fn push_pull(&self, point: [f64; 3], normal: [f64; 3], distance: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::push_pull(
            &self.0, point[0], point[1], point[2], normal[0], normal[1], normal[2], distance,
        )?))
    }

    // --- Booleans -----------------------------------------------------------

    /// Union: merge `self` and `other` into one body.
    pub fn fuse(&self, other: &Solid) -> Result<Self> {
        Ok(Self::wrap(ffi::fuse(&self.0, &other.0)?))
    }

    /// Difference: remove `other` from `self`.
    pub fn cut(&self, other: &Solid) -> Result<Self> {
        Ok(Self::wrap(ffi::cut(&self.0, &other.0)?))
    }

    /// Intersection: keep only the volume shared by `self` and `other`.
    pub fn common(&self, other: &Solid) -> Result<Self> {
        Ok(Self::wrap(ffi::common(&self.0, &other.0)?))
    }

    /// Bundle `self` and `other` into one compound shape (no boolean) — for
    /// exporting several visible bodies as a single STL.
    pub fn compound(&self, other: &Solid) -> Result<Self> {
        Ok(Self::wrap(ffi::compound(&self.0, &other.0)?))
    }

    // --- Edge treatments ----------------------------------------------------

    /// Fillet every edge with a constant `radius`. Returns an error if the
    /// radius is infeasible for some edge (OCCT rejects the whole operation).
    pub fn fillet_all_edges(&self, radius: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::fillet_all_edges(&self.0, radius)?))
    }

    /// Fillet the edges nearest each `point` with `radius`, in one operation.
    pub fn fillet_edges(&self, points: &[[f64; 3]], radius: f64) -> Result<Self> {
        let coords: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
        Ok(Self::wrap(ffi::fillet_edges(&self.0, &coords, radius)?))
    }

    /// Fillet the single edge nearest `point` with `radius`.
    pub fn fillet_edge(&self, point: [f64; 3], radius: f64) -> Result<Self> {
        self.fillet_edges(&[point], radius)
    }

    // --- Display / export ---------------------------------------------------

    /// Tessellate into a render-ready triangle [`Mesh`]. `deflection` is the
    /// maximum chord deviation in model units (smaller = smoother, heavier).
    pub fn tessellate(&self, deflection: f64) -> Result<Mesh> {
        Ok(ffi::tessellate(&self.0, deflection)?)
    }

    /// Write a binary STL to `path` at the given mesh `deflection`.
    pub fn write_stl(&self, path: &str, deflection: f64) -> Result<()> {
        ffi::write_stl(&self.0, path, deflection)?;
        Ok(())
    }

    /// The plane of the face at `index` (TopExp order) as `(origin, x_dir,
    /// y_dir)`, or `None` if the index is out of range or the face isn't planar.
    pub fn face_plane(&self, index: u32) -> Result<Option<([f64; 3], [f64; 3], [f64; 3])>> {
        let pf = ffi::face_plane(&self.0, index)?;
        Ok(pf.ok.then_some((
            [pf.ox, pf.oy, pf.oz],
            [pf.xx, pf.xy, pf.xz],
            [pf.yx, pf.yy, pf.yz],
        )))
    }
}
