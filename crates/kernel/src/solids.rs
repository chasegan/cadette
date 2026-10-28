//! Safe, owned wrappers over the OCCT FFI.
//!
//! [`Solid`] owns a B-rep `TopoDS_Shape` behind a `UniquePtr` and exposes the
//! modeling operations as ordinary, fallible Rust methods. This is the surface
//! the rest of the application builds on — nothing above this layer should ever
//! reach into [`crate::ffi`].

use cxx::UniquePtr;

use crate::elements::{ElemKind, GenRecord, Lineage};
use crate::{ffi, Result};

/// Parse the kernel's flat `[tag, in, out]` generative-lineage triples.
fn gen_records(records: Vec<i32>) -> Vec<GenRecord> {
    records
        .chunks_exact(3)
        .filter_map(|r| GenRecord::parse(r[0], r[1] as u32, r[2] as u32))
        .collect()
}

/// Parse the kernel's flat `[which, in, out|-1]` survival triples.
fn lineage(records: Vec<i32>) -> Vec<Lineage> {
    records
        .chunks_exact(3)
        .map(|r| Lineage {
            input: r[0] as u8,
            index: r[1] as u32,
            output: (r[2] >= 0).then_some(r[2] as u32),
        })
        .collect()
}

/// Flatten `(point, normal)` face anchors into the FFI's sextuple layout.
fn shell_anchors(open: &[([f64; 3], [f64; 3])]) -> Vec<f64> {
    open.iter()
        .flat_map(|(p, n)| p.iter().chain(n.iter()).copied())
        .collect()
}

/// Parse the kernel's flat `[in, out]` generated-face pairs.
fn pairs(records: Vec<i32>) -> Vec<(u32, u32)> {
    records
        .chunks_exact(2)
        .map(|r| (r[0] as u32, r[1] as u32))
        .collect()
}

/// Re-export the tessellated mesh produced by [`Solid::tessellate`].
pub use crate::ffi::{Mesh, Topology};

/// A planar face's frame as `(origin, x_dir, y_dir)` — see [`Solid::face_plane`].
pub type PlaneBasis = ([f64; 3], [f64; 3], [f64; 3]);

/// An owned B-rep solid (or, generally, any OCCT topology).
pub struct Solid(UniquePtr<ffi::Shape>);

/// Which boolean a [`Solid::boolean_tracked`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackedBoolOp {
    Fuse = 0,
    Cut = 1,
    Common = 2,
}

/// A boolean result with its evolution history: the lineage of both operands'
/// faces and edges, plus the NEW section edges generated where faces
/// intersected — everything element-map composition needs.
pub struct TrackedBoolean {
    pub result: Solid,
    pub faces: Vec<Lineage>,
    pub edges: Vec<Lineage>,
    /// `(which_input, input_face_index, result_edge_index)`: the boolean
    /// generated this new edge where that face crossed the other body.
    pub sections: Vec<(u8, u32, u32)>,
}

/// A modifying op's result (fillet, chamfer, shell) with its evolution
/// history. Lineage `input` is always 0 (one source body); `generated` pairs
/// each source element with the result face it generated — the fillet or
/// bevel face of an edge, the inner wall of a shelled face. All vecs are
/// EMPTY when the result needed healing.
pub struct TrackedModify {
    pub result: Solid,
    pub faces: Vec<Lineage>,
    pub edges: Vec<Lineage>,
    /// `(source_element_index, result_face_index)` per generated face.
    pub generated: Vec<(u32, u32)>,
}

/// A push/pull result with its evolution history: boolean-style lineage where
/// operand 0 is the body and operand 1 the offset prism.
pub struct TrackedPushPull {
    pub result: Solid,
    pub tracking: PushPullTracking,
    pub faces: Vec<Lineage>,
    pub edges: Vec<Lineage>,
    pub sections: Vec<(u8, u32, u32)>,
}

/// How a push/pull result relates to its input's element map.
pub enum PushPullTracking {
    /// The zero-distance no-op: the result IS the input — keep its map.
    Identity,
    /// The result was healed (rebuilt without history) or unmappable: good
    /// geometry, no lineage — the map must be empty, never stale.
    Untracked,
    /// Full lineage, ready for composition.
    Tracked(PushPullInfo),
}

/// Where the pushed face sits — everything needed to seed the prism operand's
/// pseudo-map (its far cap IS the pushed face, moved).
#[derive(Clone, Copy, Debug)]
pub struct PushPullInfo {
    /// Face index of the pushed face in the INPUT body.
    pub pushed_face: u32,
    /// Face index of the prism's far cap in the PRISM.
    pub far_cap: u32,
    pub prism_faces: usize,
    pub prism_edges: usize,
}

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

    /// A planar face whose closed boundary mixes line and cubic-bezier segments.
    /// `points` is the flat 2D loop (in plane coords); `segs` carries 5 doubles
    /// per segment `i` (vertex `i`→`i+1`): `[is_bezier, c1x, c1y, c2x, c2y]`.
    pub fn profile_face(
        origin: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        points: &[f64],
        segs: &[f64],
    ) -> Result<Self> {
        Ok(Self::wrap(ffi::profile_face(
            origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], y_dir[0], y_dir[1],
            y_dir[2], points, segs,
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

    /// A copy of this solid non-uniformly scaled by per-axis `factors` about the
    /// fixed point `anchor`.
    pub fn scale(&self, factors: [f64; 3], anchor: [f64; 3]) -> Result<Self> {
        Ok(Self::wrap(ffi::scale(
            &self.0, factors[0], factors[1], factors[2], anchor[0], anchor[1], anchor[2],
        )?))
    }

    /// A copy of this solid reflected across the plane through `origin` with
    /// unit `normal`.
    pub fn mirror(&self, origin: [f64; 3], normal: [f64; 3]) -> Result<Self> {
        Ok(Self::wrap(ffi::mirror(
            &self.0, origin[0], origin[1], origin[2], normal[0], normal[1], normal[2],
        )?))
    }

    /// Extrude this planar face along its normal by `distance` into a solid.
    pub fn extrude(&self, distance: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::extrude(&self.0, distance)?))
    }

    /// [`Self::extrude`] with its lineage report — which result face/edge
    /// each profile element became (element-map composition input).
    pub fn extrude_tracked(&self, distance: f64) -> Result<(Self, Vec<GenRecord>)> {
        let mut records: Vec<i32> = Vec::new();
        let shape = ffi::extrude_tracked(&self.0, distance, &mut records)?;
        Ok((Self::wrap(shape), gen_records(records)))
    }

    /// An OPEN planar wire (a sweep path) on the frame `origin + u*x_dir +
    /// v*y_dir`. `points` is the flat 2D polyline, `segs` 5 doubles per segment.
    pub fn path_wire(
        origin: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        points: &[f64],
        segs: &[f64],
    ) -> Result<Self> {
        Ok(Self::wrap(ffi::path_wire(
            origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], y_dir[0], y_dir[1],
            y_dir[2], points, segs,
        )?))
    }

    /// A helix wire on the cylinder of `radius` about the axis through
    /// `center` with direction `axis`, starting at `center + radius * x_dir`
    /// (`x_dir` ⊥ `axis`) and climbing `pitch` per turn for `length` along the
    /// axis. The natural spine for a thread or a coil: sweep a profile drawn on
    /// the frame `(center, x_dir, axis)` along it with the axis as binormal and
    /// the profile stays in the axial plane all the way round.
    #[allow(clippy::too_many_arguments)]
    pub fn helix_wire(
        center: [f64; 3],
        axis: [f64; 3],
        x_dir: [f64; 3],
        radius: f64,
        pitch: f64,
        length: f64,
        left_handed: bool,
    ) -> Result<Self> {
        Ok(Self::wrap(ffi::helix_wire(
            center[0], center[1], center[2], axis[0], axis[1], axis[2], x_dir[0], x_dir[1],
            x_dir[2], radius, pitch, length, left_handed,
        )?))
    }

    /// Sweep this planar face along `spine` into a solid, keeping it normal to
    /// the path with its roll locked to `binormal` (the path plane's normal).
    pub fn sweep(&self, spine: &Solid, binormal: [f64; 3]) -> Result<Self> {
        Ok(Self::wrap(ffi::sweep(
            &self.0, &spine.0, binormal[0], binormal[1], binormal[2],
        )?))
    }

    /// [`Self::sweep`] with its lineage report. The records are EMPTY when
    /// the swept solid needed healing (ShapeFix discards history).
    pub fn sweep_tracked(
        &self,
        spine: &Solid,
        binormal: [f64; 3],
    ) -> Result<(Self, Vec<GenRecord>)> {
        let mut records: Vec<i32> = Vec::new();
        let shape = ffi::sweep_tracked(
            &self.0, &spine.0, binormal[0], binormal[1], binormal[2], &mut records,
        )?;
        Ok((Self::wrap(shape), gen_records(records)))
    }

    /// Revolve this planar profile by `angle` radians about the straight edge
    /// nearest `axis_point` (a full turn is `2π`). The axis edge is one of the
    /// profile's own segments or an adjacent model edge.
    pub fn revolve(&self, axis_point: [f64; 3], angle: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::revolve(
            &self.0, axis_point[0], axis_point[1], axis_point[2], angle,
        )?))
    }

    /// [`Self::revolve`] with its lineage report.
    pub fn revolve_tracked(
        &self,
        axis_point: [f64; 3],
        angle: f64,
    ) -> Result<(Self, Vec<GenRecord>)> {
        let mut records: Vec<i32> = Vec::new();
        let shape = ffi::revolve_tracked(
            &self.0, axis_point[0], axis_point[1], axis_point[2], angle, &mut records,
        )?;
        Ok((Self::wrap(shape), gen_records(records)))
    }

    /// Which faces bound each edge, as `(edge_index, face_index)` pairs
    /// (TopExp map order) — names a generative op's lateral edges.
    pub fn edge_face_adjacency(&self) -> Result<Vec<(u32, u32)>> {
        let flat = ffi::edge_face_adjacency(&self.0)?;
        Ok(flat.chunks_exact(2).map(|p| (p[0], p[1])).collect())
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

    /// Run a boolean through the SAME pipeline as [`Self::fuse`]/
    /// [`Self::cut`]/[`Self::common`] but keep the composed evolution
    /// history: which face/edge of each operand became which face/edge of the
    /// result, and which new section edges appeared — the raw material for
    /// element-map composition.
    pub fn boolean_tracked(&self, other: &Solid, op: TrackedBoolOp) -> Result<TrackedBoolean> {
        let mut faces: Vec<i32> = Vec::new();
        let mut edges: Vec<i32> = Vec::new();
        let mut sections: Vec<i32> = Vec::new();
        let shape = ffi::boolean_tracked(
            &self.0,
            &other.0,
            op as i32,
            &mut faces,
            &mut edges,
            &mut sections,
        )?;
        Ok(TrackedBoolean {
            result: Self::wrap(shape),
            faces: lineage(faces),
            edges: lineage(edges),
            sections: sections
                .chunks_exact(3)
                .map(|r| (r[0] as u8, r[1] as u32, r[2] as u32))
                .collect(),
        })
    }

    /// Centre of mass of the face/edge at `index` (TopExp map order) — the
    /// deterministic key ranking a split element's pieces.
    pub fn elem_centroid(&self, kind: ElemKind, index: u32) -> Result<[f64; 3]> {
        let v = ffi::elem_centroid(&self.0, kind as i32, index)?;
        Ok([v[0], v[1], v[2]])
    }

    /// Bundle `self` and `other` into one compound shape (no boolean) — for
    /// exporting several visible bodies as a single STL.
    pub fn compound(&self, other: &Solid) -> Result<Self> {
        Ok(Self::wrap(ffi::compound(&self.0, &other.0)?))
    }

    /// Merge same-domain faces and edges — remove the artificial seam edges left
    /// between coplanar neighbours (e.g. by a boolean). Booleans already apply
    /// this; exposed for an explicit "refine".
    pub fn unify(&self) -> Result<Self> {
        Ok(Self::wrap(ffi::unify(&self.0)?))
    }

    /// One-line structural summary (root type, sub-shape counts, closedness,
    /// validity) — for diagnostics and tests.
    pub fn describe(&self) -> Result<String> {
        Ok(ffi::describe(&self.0)?)
    }

    /// The number of faces in this solid (TopExp order).
    pub fn face_count(&self) -> Result<usize> {
        Ok(ffi::count_faces(&self.0)?)
    }

    /// The number of unique edges in this solid (TopExp map order, matching
    /// tessellation edge ids).
    pub fn edge_count(&self) -> Result<usize> {
        Ok(ffi::count_edges(&self.0)?)
    }

    /// The enclosed volume of this solid (model units³).
    pub fn volume(&self) -> Result<f64> {
        Ok(ffi::volume(&self.0)?)
    }

    /// Exact area (model units²) of the face at `index` (TopExp order,
    /// matching tessellation face ids).
    pub fn face_area(&self, index: u32) -> Result<f64> {
        Ok(ffi::face_area(&self.0, index)?)
    }

    /// Exact length (model units) of the edge at `index` (unique-edge order,
    /// matching tessellation edge ids).
    pub fn edge_length(&self, index: u32) -> Result<f64> {
        Ok(ffi::edge_length(&self.0, index)?)
    }

    // --- Interrogation ------------------------------------------------------

    /// The tight axis-aligned bounding box `(min, max)` from the exact
    /// geometry (not a tessellation).
    pub fn bbox(&self) -> Result<([f64; 3], [f64; 3])> {
        let b = ffi::bbox(&self.0)?;
        Ok(([b[0], b[1], b[2]], [b[3], b[4], b[5]]))
    }

    /// Volume, surface area, centre of mass, inertia and bounding box.
    pub fn mass_properties(&self) -> Result<MassProperties> {
        let p = ffi::body_props(&self.0)?;
        Ok(MassProperties {
            volume: p.volume,
            surface_area: p.surface_area,
            centroid: [p.cx, p.cy, p.cz],
            inertia: [[p.ixx, p.ixy, p.ixz], [p.iyx, p.iyy, p.iyz], [p.izx, p.izy, p.izz]],
            bbox: ([p.min_x, p.min_y, p.min_z], [p.max_x, p.max_y, p.max_z]),
        })
    }

    /// Sub-shape counts, closedness and validity — the typed form of
    /// [`Self::describe`].
    pub fn topology(&self) -> Result<Topology> {
        Ok(ffi::topology(&self.0)?)
    }

    /// Whether OCCT's checker accepts the shape (a closed, non-self-
    /// intersecting solid prints and booleans cleanly).
    pub fn is_valid(&self) -> Result<bool> {
        Ok(ffi::topology(&self.0)?.is_valid)
    }

    /// Every face, indexed the way tessellation and [`Self::face_area`] number
    /// them, with its surface type, area, centre, outward normal and edges.
    pub fn faces(&self) -> Result<Vec<FaceInfo>> {
        Ok(ffi::faces(&self.0)?.into_iter().map(FaceInfo::from).collect())
    }

    /// Every unique edge, indexed the way tessellation and
    /// [`Self::edge_length`] number them, with its curve type, endpoints, a
    /// point on the curve (an anchor), length and adjacent faces.
    pub fn edges(&self) -> Result<Vec<EdgeInfo>> {
        Ok(ffi::edges(&self.0)?.into_iter().map(EdgeInfo::from).collect())
    }

    /// Minimum distance to `other`: `0.0` when the bodies touch or overlap.
    pub fn distance_to(&self, other: &Solid) -> Result<f64> {
        Ok(ffi::distance(&self.0, &other.0)?)
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

    /// [`Self::fillet_all_edges`] with its lineage report.
    pub fn fillet_all_tracked(&self, radius: f64) -> Result<TrackedModify> {
        let (mut f, mut e, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let shape = ffi::fillet_all_tracked(&self.0, radius, &mut f, &mut e, &mut r)?;
        Ok(TrackedModify {
            result: Self::wrap(shape),
            faces: lineage(f),
            edges: lineage(e),
            generated: pairs(r),
        })
    }

    /// [`Self::fillet_edges`] with its lineage report (empty when healed).
    pub fn fillet_edges_tracked(
        &self,
        points: &[[f64; 3]],
        radius: f64,
    ) -> Result<TrackedModify> {
        let coords: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let (mut f, mut e, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let shape =
            ffi::fillet_edges_tracked(&self.0, &coords, radius, &mut f, &mut e, &mut r)?;
        Ok(TrackedModify {
            result: Self::wrap(shape),
            faces: lineage(f),
            edges: lineage(e),
            generated: pairs(r),
        })
    }

    /// Chamfer every edge: a symmetric bevel set back `distance` along both
    /// faces of each edge. Errors like [`Self::fillet_all_edges`].
    pub fn chamfer_all_edges(&self, distance: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::chamfer_all_edges(&self.0, distance)?))
    }

    /// Chamfer the edges nearest each `point` by `distance`, in one operation.
    pub fn chamfer_edges(&self, points: &[[f64; 3]], distance: f64) -> Result<Self> {
        let coords: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
        Ok(Self::wrap(ffi::chamfer_edges(&self.0, &coords, distance)?))
    }

    /// [`Self::chamfer_all_edges`] with its lineage report.
    pub fn chamfer_all_tracked(&self, distance: f64) -> Result<TrackedModify> {
        let (mut f, mut e, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let shape = ffi::chamfer_all_tracked(&self.0, distance, &mut f, &mut e, &mut r)?;
        Ok(TrackedModify {
            result: Self::wrap(shape),
            faces: lineage(f),
            edges: lineage(e),
            generated: pairs(r),
        })
    }

    /// [`Self::chamfer_edges`] with its lineage report (empty when healed).
    pub fn chamfer_edges_tracked(
        &self,
        points: &[[f64; 3]],
        distance: f64,
    ) -> Result<TrackedModify> {
        let coords: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let (mut f, mut e, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let shape =
            ffi::chamfer_edges_tracked(&self.0, &coords, distance, &mut f, &mut e, &mut r)?;
        Ok(TrackedModify {
            result: Self::wrap(shape),
            faces: lineage(f),
            edges: lineage(e),
            generated: pairs(r),
        })
    }

    // --- Shell --------------------------------------------------------------

    /// Hollow the solid to a uniform wall of `thickness`, measured inward,
    /// removing each face anchored by a `(point, normal)` in `open` so the
    /// cavity is reachable (no open faces → a closed hollow body).
    pub fn shell(&self, open: &[([f64; 3], [f64; 3])], thickness: f64) -> Result<Self> {
        Ok(Self::wrap(ffi::shell(&self.0, &shell_anchors(open), thickness)?))
    }

    /// [`Self::shell`] with its lineage report: `generated` pairs each kept
    /// outer face with its inner wall (empty when healed).
    pub fn shell_tracked(
        &self,
        open: &[([f64; 3], [f64; 3])],
        thickness: f64,
    ) -> Result<TrackedModify> {
        let (mut f, mut e, mut o) = (Vec::new(), Vec::new(), Vec::new());
        let shape =
            ffi::shell_tracked(&self.0, &shell_anchors(open), thickness, &mut f, &mut e, &mut o)?;
        Ok(TrackedModify {
            result: Self::wrap(shape),
            faces: lineage(f),
            edges: lineage(e),
            generated: pairs(o),
        })
    }

    /// [`Self::push_pull`] with its lineage report.
    pub fn push_pull_tracked(
        &self,
        point: [f64; 3],
        normal: [f64; 3],
        distance: f64,
    ) -> Result<TrackedPushPull> {
        let mut meta: Vec<i32> = Vec::new();
        let (mut f, mut e, mut s) = (Vec::new(), Vec::new(), Vec::new());
        let shape = ffi::push_pull_tracked(
            &self.0, point[0], point[1], point[2], normal[0], normal[1], normal[2], distance,
            &mut meta, &mut f, &mut e, &mut s,
        )?;
        let tracking = match meta.len() {
            0 => PushPullTracking::Identity,
            4 => PushPullTracking::Tracked(PushPullInfo {
                pushed_face: meta[0] as u32,
                far_cap: meta[1] as u32,
                prism_faces: meta[2] as usize,
                prism_edges: meta[3] as usize,
            }),
            _ => PushPullTracking::Untracked,
        };
        Ok(TrackedPushPull {
            result: Self::wrap(shape),
            tracking,
            faces: lineage(f),
            edges: lineage(e),
            sections: s
                .chunks_exact(3)
                .map(|r| (r[0] as u8, r[1] as u32, r[2] as u32))
                .collect(),
        })
    }

    // --- Data exchange ------------------------------------------------------

    /// Read a STEP file (its bytes) into one solid — a compound of them when
    /// the file holds several — in millimetres. Closed free shells become
    /// solids; a file with no solid at all is an error, as is one the reader
    /// can't parse.
    pub fn from_step(data: &[u8]) -> Result<Self> {
        Ok(Self::wrap(ffi::read_step(data)?))
    }

    /// Write this solid to `path` as a STEP file (AP214), in model units.
    pub fn write_step(&self, path: &str) -> Result<()> {
        ffi::write_step(&self.0, path)?;
        Ok(())
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
    pub fn face_plane(&self, index: u32) -> Result<Option<PlaneBasis>> {
        let pf = ffi::face_plane(&self.0, index)?;
        Ok(pf.ok.then_some((
            [pf.ox, pf.oy, pf.oz],
            [pf.xx, pf.xy, pf.xz],
            [pf.yx, pf.yy, pf.yz],
        )))
    }

    /// The cylinder behind the face at `index` (TopExp order), or `None` if
    /// the index is out of range or the face isn't cylindrical.
    pub fn face_cylinder(&self, index: u32) -> Result<Option<CylinderFace>> {
        let cf = ffi::face_cylinder(&self.0, index)?;
        Ok(cf.ok.then_some(CylinderFace {
            location: [cf.lx, cf.ly, cf.lz],
            axis: [cf.ax, cf.ay, cf.az],
            radius: cf.radius,
            v0: cf.v0,
            v1: cf.v1,
            convex: cf.convex,
        }))
    }

    /// Midpoints of the edges bounding face `index` (TopExp order) — a point on
    /// each edge, for filleting a whole face's edges.
    pub fn face_edge_midpoints(&self, index: u32) -> Result<Vec<[f64; 3]>> {
        let flat = ffi::face_edge_midpoints(&self.0, index)?;
        Ok(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
    }
}

/// Whole-body measurements from the exact geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Enclosed volume (model units³).
    pub volume: f64,
    /// Total surface area (model units²).
    pub surface_area: f64,
    /// Centre of mass (uniform density).
    pub centroid: [f64; 3],
    /// Inertia matrix about the centre of mass, row-major (unit density).
    pub inertia: [[f64; 3]; 3],
    /// Tight axis-aligned bounding box `(min, max)`.
    pub bbox: ([f64; 3], [f64; 3]),
}

/// The underlying surface type of a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    Plane,
    Cylinder,
    Cone,
    Sphere,
    Torus,
    Bezier,
    BSpline,
    Other,
}

impl SurfaceKind {
    fn from_code(code: u8) -> Self {
        match code {
            0 => SurfaceKind::Plane,
            1 => SurfaceKind::Cylinder,
            2 => SurfaceKind::Cone,
            3 => SurfaceKind::Sphere,
            4 => SurfaceKind::Torus,
            5 => SurfaceKind::Bezier,
            6 => SurfaceKind::BSpline,
            _ => SurfaceKind::Other,
        }
    }
}

/// The underlying curve type of an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveKind {
    Line,
    Circle,
    Ellipse,
    Bezier,
    BSpline,
    Other,
}

impl CurveKind {
    fn from_code(code: u8) -> Self {
        match code {
            0 => CurveKind::Line,
            1 => CurveKind::Circle,
            2 => CurveKind::Ellipse,
            3 => CurveKind::Bezier,
            4 => CurveKind::BSpline,
            _ => CurveKind::Other,
        }
    }
}

/// A cylindrical face's geometry (see [`Solid::face_cylinder`]): the axis is
/// `location + v * axis` and the face spans `v0..v1` along it. `convex` means
/// the material is inside the cylinder (a shaft) rather than outside (a hole).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderFace {
    pub location: [f64; 3],
    pub axis: [f64; 3],
    pub radius: f64,
    pub v0: f64,
    pub v1: f64,
    pub convex: bool,
}

/// One face of a solid (see [`Solid::faces`]).
#[derive(Clone, Debug, PartialEq)]
pub struct FaceInfo {
    /// Face index — the id tessellation, picking and `face_area` use.
    pub index: u32,
    pub surface: SurfaceKind,
    pub area: f64,
    /// Centre of mass of the face (on the face for convex faces).
    pub centroid: [f64; 3],
    /// Outward unit normal sampled at the face's parametric middle (constant
    /// for planes); zero if undefined.
    pub normal: [f64; 3],
    /// Radius for cylinders, spheres and cones (reference radius) and the
    /// major radius for tori; `None` for other surfaces.
    pub radius: Option<f64>,
    /// The surface axis for cylinders, cones and tori, as `(point on it,
    /// unit direction)`; `None` for other surfaces.
    pub axis: Option<([f64; 3], [f64; 3])>,
    /// Bounding edges, as unique-edge indices (see [`Solid::edges`]).
    pub edges: Vec<u32>,
}

impl From<ffi::FaceInfo> for FaceInfo {
    fn from(f: ffi::FaceInfo) -> Self {
        let surface = SurfaceKind::from_code(f.surface);
        let has_radius = matches!(
            surface,
            SurfaceKind::Cylinder | SurfaceKind::Sphere | SurfaceKind::Cone | SurfaceKind::Torus
        );
        FaceInfo {
            index: f.index,
            surface,
            area: f.area,
            centroid: [f.cx, f.cy, f.cz],
            normal: [f.nx, f.ny, f.nz],
            radius: has_radius.then_some(f.radius),
            axis: matches!(surface, SurfaceKind::Cylinder | SurfaceKind::Cone | SurfaceKind::Torus)
                .then_some(([f.lx, f.ly, f.lz], [f.ax, f.ay, f.az])),
            edges: f.edges,
        }
    }
}

/// One unique edge of a solid (see [`Solid::edges`]).
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeInfo {
    /// Edge index — the id tessellation, picking and `edge_length` use.
    pub index: u32,
    pub curve: CurveKind,
    pub length: f64,
    /// Endpoints in the edge's own orientation.
    pub start: [f64; 3],
    pub end: [f64; 3],
    /// A point genuinely on the curve at its parametric middle — what an
    /// [`cdt_core::EdgeAnchor`] wants (the chord midpoint of a curved edge is
    /// NOT on the edge).
    pub midpoint: [f64; 3],
    /// Radius for circular edges.
    pub radius: Option<f64>,
    /// Adjacent faces, as face indices.
    pub faces: Vec<u32>,
}

impl From<ffi::EdgeInfo> for EdgeInfo {
    fn from(e: ffi::EdgeInfo) -> Self {
        let curve = CurveKind::from_code(e.curve);
        EdgeInfo {
            index: e.index,
            curve,
            length: e.length,
            start: [e.sx, e.sy, e.sz],
            end: [e.ex, e.ey, e.ez],
            midpoint: [e.mx, e.my, e.mz],
            radius: (curve == CurveKind::Circle).then_some(e.radius),
            faces: e.faces,
        }
    }
}
