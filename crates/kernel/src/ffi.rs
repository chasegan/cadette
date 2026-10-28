//! The raw OCCT FFI boundary.
//!
//! This is the **only** place in the codebase that crosses into C++. Everything
//! declared here is unsafe-by-nature; the safe, ergonomic surface lives in
//! [`crate::solids`]. Keep this module small, mechanical, and well-mirrored by
//! the C++ in `src/ffi/bridge.{hpp,cpp}`.
//!
//! Conventions:
//! - A B-rep solid is an opaque C++ `Shape` handed back as `UniquePtr<Shape>`.
//! - Any OCCT failure is translated to a thrown `std::runtime_error` on the C++
//!   side, which cxx surfaces to Rust as `Err` because every fallible function
//!   returns `Result`.

#[cxx::bridge(namespace = "cdt")]
mod bridge {
    /// A tessellated, render-ready triangle mesh produced from a B-rep `Shape`.
    ///
    /// All arrays are flat. `positions` and `normals` hold `3 * vertex_count`
    /// floats (xyz); `indices` holds `3 * triangle_count` vertex indices wound
    /// counter-clockwise (front-facing) in a right-handed, Z-up frame.
    #[derive(Clone, Debug, Default)]
    struct Mesh {
        positions: Vec<f32>,
        normals: Vec<f32>,
        indices: Vec<u32>,
        /// Per-vertex source face index (one entry per `positions` triple).
        /// Faces are numbered in OCCT exploration order; used for GPU picking.
        face_ids: Vec<u32>,
        /// Crisp feature edges: flat xyz polyline points, line-segment index
        /// pairs into them, and a per-point source edge id (for edge picking).
        edge_positions: Vec<f32>,
        edge_indices: Vec<u32>,
        edge_ids: Vec<u32>,
    }

    /// A planar face's local frame: origin plus in-plane x/y axes. `ok` is false
    /// if the queried face index is out of range or the face isn't planar.
    #[derive(Clone, Copy, Debug, Default)]
    struct PlaneFrame {
        ok: bool,
        ox: f64,
        oy: f64,
        oz: f64,
        xx: f64,
        xy: f64,
        xz: f64,
        yx: f64,
        yy: f64,
        yz: f64,
    }

    /// The cylinder behind a face, for threading. `ok` is false if the face
    /// index is out of range or the face isn't cylindrical. The axis is
    /// `(lx,ly,lz) + v * (ax,ay,az)`; the face spans `v0..v1` along it.
    /// `convex` = the material is inside the cylinder (a shaft, not a hole).
    #[derive(Clone, Copy, Debug, Default)]
    struct CylinderFrame {
        ok: bool,
        lx: f64,
        ly: f64,
        lz: f64,
        ax: f64,
        ay: f64,
        az: f64,
        radius: f64,
        v0: f64,
        v1: f64,
        convex: bool,
    }

    /// Exact whole-body measurements (see `Solid::mass_properties` /
    /// `Solid::bbox`). Inertia is the 3×3 matrix about the centre of mass,
    /// row-major.
    #[derive(Clone, Copy, Debug, Default)]
    struct BodyProps {
        volume: f64,
        surface_area: f64,
        cx: f64,
        cy: f64,
        cz: f64,
        ixx: f64,
        ixy: f64,
        ixz: f64,
        iyx: f64,
        iyy: f64,
        iyz: f64,
        izx: f64,
        izy: f64,
        izz: f64,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    }

    /// Structural summary of a shape — the typed form of `describe()`.
    #[derive(Clone, Debug, Default)]
    struct Topology {
        kind: String,
        solids: u32,
        shells: u32,
        closed_shells: u32,
        faces: u32,
        edges: u32,
        vertices: u32,
        is_valid: bool,
    }

    /// One face of a shape (index in TopExp exploration order, matching
    /// tessellation face ids and `face_area`/`face_plane`). `surface` is a
    /// `SurfaceKind` code; `radius` is set for cylinders, spheres, cones and
    /// tori. The normal points OUT of the solid, sampled at the face's
    /// parametric middle; `(cx, cy, cz)` is the face's centre of mass.
    #[derive(Clone, Debug, Default)]
    struct FaceInfo {
        index: u32,
        surface: u8,
        area: f64,
        cx: f64,
        cy: f64,
        cz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
        radius: f64,
        /// The surface axis for cylinders, cones and tori: a point on it
        /// `(lx,ly,lz)` and its direction `(ax,ay,az)`; all zero otherwise.
        lx: f64,
        ly: f64,
        lz: f64,
        ax: f64,
        ay: f64,
        az: f64,
        /// Bounding edges, as unique-edge indices.
        edges: Vec<u32>,
    }

    /// One edge of a shape (unique-edge index, matching tessellation edge ids
    /// and `edge_length`). `curve` is a `CurveKind` code; `radius` is set for
    /// circles. `start`/`end` follow the edge's orientation; `mid` is a point
    /// genuinely on the curve (usable as a fillet anchor).
    #[derive(Clone, Debug, Default)]
    struct EdgeInfo {
        index: u32,
        curve: u8,
        length: f64,
        sx: f64,
        sy: f64,
        sz: f64,
        ex: f64,
        ey: f64,
        ez: f64,
        mx: f64,
        my: f64,
        mz: f64,
        radius: f64,
        /// Adjacent faces, as face indices.
        faces: Vec<u32>,
    }

    unsafe extern "C++" {
        include!("cdt-kernel/src/ffi/bridge.hpp");

        /// Opaque handle to an OCCT `TopoDS_Shape`.
        type Shape;

        // --- Primitives ---------------------------------------------------
        fn make_box(dx: f64, dy: f64, dz: f64) -> Result<UniquePtr<Shape>>;
        fn make_sphere(radius: f64) -> Result<UniquePtr<Shape>>;
        fn make_cylinder(radius: f64, height: f64) -> Result<UniquePtr<Shape>>;

        // --- Sketch profiles (planar faces) -------------------------------
        /// A rectangular face centered at `origin`, spanning `width` along the
        /// unit `x_dir` and `height` along the unit `y_dir`.
        #[allow(clippy::too_many_arguments)]
        fn make_rectangle_face(
            ox: f64, oy: f64, oz: f64,
            xx: f64, xy: f64, xz: f64,
            yx: f64, yy: f64, yz: f64,
            width: f64, height: f64,
        ) -> Result<UniquePtr<Shape>>;
        /// A circular face centered at `origin` on the plane with unit `normal`.
        fn make_circle_face(
            ox: f64, oy: f64, oz: f64,
            nx: f64, ny: f64, nz: f64,
            radius: f64,
        ) -> Result<UniquePtr<Shape>>;
        /// A polygonal face from a closed loop of 2D points (`points` is a flat
        /// `[x0, y0, x1, y1, ...]`) mapped onto the plane `origin + x*xdir + y*ydir`.
        #[allow(clippy::too_many_arguments)]
        fn make_polygon_face(
            ox: f64, oy: f64, oz: f64,
            xx: f64, xy: f64, xz: f64,
            yx: f64, yy: f64, yz: f64,
            points: &[f64],
        ) -> Result<UniquePtr<Shape>>;
        /// A planar face whose boundary mixes line and cubic-bezier segments.
        /// `points` is the flat 2D loop; `segs` has 5 doubles per segment:
        /// `[is_bezier, c1x, c1y, c2x, c2y]`.
        #[allow(clippy::too_many_arguments)]
        fn profile_face(
            ox: f64, oy: f64, oz: f64,
            xx: f64, xy: f64, xz: f64,
            yx: f64, yy: f64, yz: f64,
            points: &[f64],
            segs: &[f64],
        ) -> Result<UniquePtr<Shape>>;

        // --- Extrude ------------------------------------------------------
        /// Extrude a planar face along its normal by `distance` into a solid.
        fn extrude(shape: &Shape, distance: f64) -> Result<UniquePtr<Shape>>;
        /// Extrude with lineage: `records` receives flat `[tag, in, out]`
        /// triples — 0 = the profile FACE's image at the sweep start (the
        /// profile-side cap), 1 = its image at the end (the far cap), 2 =
        /// side face swept from profile edge `in`, 3/4 = profile edge `in`'s
        /// start/end rim image. Indices are TopExp map order.
        fn extrude_tracked(
            shape: &Shape,
            distance: f64,
            records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        // --- Sweep --------------------------------------------------------
        /// An OPEN planar wire (a sweep path) on the frame `origin + u*xdir +
        /// v*ydir`. `points` is the flat 2D polyline; `segs` has 5 doubles per
        /// segment (`n-1` of them): `[is_bezier, c1u, c1v, c2u, c2v]`.
        #[allow(clippy::too_many_arguments)]
        fn path_wire(
            ox: f64, oy: f64, oz: f64,
            xx: f64, xy: f64, xz: f64,
            yx: f64, yy: f64, yz: f64,
            points: &[f64],
            segs: &[f64],
        ) -> Result<UniquePtr<Shape>>;
        /// A helix wire on the cylinder of `radius` about the axis through
        /// `(cx,cy,cz)` with direction `(ax,ay,az)`, starting at
        /// `center + radius * (xx,xy,xz)` and climbing `pitch` per turn for
        /// `length` along the axis. `left_handed` winds the other way.
        #[allow(clippy::too_many_arguments)]
        fn helix_wire(
            cx: f64, cy: f64, cz: f64,
            ax: f64, ay: f64, az: f64,
            xx: f64, xy: f64, xz: f64,
            radius: f64,
            pitch: f64,
            length: f64,
            left_handed: bool,
        ) -> Result<UniquePtr<Shape>>;
        /// Sweep the planar `profile` face along the `spine` wire into a solid,
        /// keeping it normal to the path with its roll locked to the constant
        /// `binormal` (the path plane's normal), so orientation is preserved.
        fn sweep(
            profile: &Shape,
            spine: &Shape,
            bnx: f64,
            bny: f64,
            bnz: f64,
        ) -> Result<UniquePtr<Shape>>;
        /// Sweep with lineage (same record scheme as `extrude_tracked`).
        /// `records` stays EMPTY when the swept solid needed healing —
        /// ShapeFix rebuilds sub-shapes without history.
        fn sweep_tracked(
            profile: &Shape,
            spine: &Shape,
            bnx: f64,
            bny: f64,
            bnz: f64,
            records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        // --- Revolve ------------------------------------------------------
        /// Revolve the planar profile `shape` by `angle` radians about the
        /// straight edge nearest `(ax,ay,az)` (a full turn is `2π`).
        fn revolve(
            shape: &Shape,
            ax: f64, ay: f64, az: f64,
            angle: f64,
        ) -> Result<UniquePtr<Shape>>;
        /// Revolve with lineage (same record scheme as `extrude_tracked`).
        fn revolve_tracked(
            shape: &Shape,
            ax: f64, ay: f64, az: f64,
            angle: f64,
            records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        /// Which faces bound each edge: flat `(edge_index, face_index)` pairs
        /// in TopExp map order — names a generative op's lateral edges.
        fn edge_face_adjacency(s: &Shape) -> Result<Vec<u32>>;

        // --- Push/pull -----------------------------------------------------
        /// Offset the planar face of `shape` anchored at `(px,py,pz)` with
        /// normal `(nx,ny,nz)` along that normal by `distance` (fuse if
        /// positive, cut if negative).
        #[allow(clippy::too_many_arguments)]
        fn push_pull(
            shape: &Shape,
            px: f64, py: f64, pz: f64,
            nx: f64, ny: f64, nz: f64,
            distance: f64,
        ) -> Result<UniquePtr<Shape>>;

        /// Push/pull with lineage. `meta` stays EMPTY for the zero-distance
        /// no-op; otherwise it receives `[pushed_face_index (in shape),
        /// prism_far_cap_face_index (in the prism), prism_face_count,
        /// prism_edge_count]` and the record vecs receive boolean_tracked-style
        /// lineage with operand 0 = shape, operand 1 = the offset prism.
        #[allow(clippy::too_many_arguments)]
        fn push_pull_tracked(
            shape: &Shape,
            px: f64, py: f64, pz: f64,
            nx: f64, ny: f64, nz: f64,
            distance: f64,
            meta: &mut Vec<i32>,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            section_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        /// A cheap shallow copy (shares the underlying ref-counted geometry).
        fn copy_shape(shape: &Shape) -> UniquePtr<Shape>;

        /// Bundle two shapes into one compound (for exporting several bodies).
        fn compound(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;

        // --- Transforms ---------------------------------------------------
        fn translate(shape: &Shape, dx: f64, dy: f64, dz: f64) -> Result<UniquePtr<Shape>>;

        /// Rotate `shape` by `angle` radians about the axis through `(cx,cy,cz)`
        /// with direction `(ax,ay,az)`.
        #[allow(clippy::too_many_arguments)]
        fn rotate(
            shape: &Shape,
            cx: f64, cy: f64, cz: f64,
            ax: f64, ay: f64, az: f64,
            angle: f64,
        ) -> Result<UniquePtr<Shape>>;

        /// Non-uniformly scale `shape` by `(sx,sy,sz)` about the fixed point
        /// `(ax,ay,az)`.
        #[allow(clippy::too_many_arguments)]
        fn scale(
            shape: &Shape,
            sx: f64, sy: f64, sz: f64,
            ax: f64, ay: f64, az: f64,
        ) -> Result<UniquePtr<Shape>>;

        /// Reflect `shape` across the plane through `(ox,oy,oz)` with normal
        /// `(nx,ny,nz)`.
        #[allow(clippy::too_many_arguments)]
        fn mirror(
            shape: &Shape,
            ox: f64, oy: f64, oz: f64,
            nx: f64, ny: f64, nz: f64,
        ) -> Result<UniquePtr<Shape>>;

        // --- Booleans -----------------------------------------------------
        fn fuse(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;
        fn cut(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;
        fn common(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;

        /// The same pipeline as fuse/cut/common — `BRepAlgoAPI_*` then
        /// unify-same-domain — but with both stages' evolution histories
        /// composed and reported (the raw material for element maps). `op`:
        /// 0=fuse, 1=cut, 2=common. `face_records`/`edge_records` receive flat
        /// triples `[which_input (0=a, 1=b), input_index, output_index]` — one
        /// per (input element, surviving image) pair, `output_index` -1 when
        /// the element was consumed. `section_records` receives
        /// `[which_input, input_FACE_index, output_EDGE_index]` for the NEW
        /// edges generated where faces intersect. Indices are TopExp map order
        /// per shape.
        fn boolean_tracked(
            a: &Shape,
            b: &Shape,
            op: i32,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            section_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        /// Centre of mass (x, y, z) of the face (`kind` 0) or edge (`kind` 1)
        /// at `index` — ranks the pieces of a split element deterministically.
        fn elem_centroid(s: &Shape, kind: i32, index: u32) -> Result<Vec<f64>>;

        /// Merge same-domain faces/edges (remove seam edges between coplanar
        /// neighbours) — an explicit "refine". Booleans already do this.
        fn unify(s: &Shape) -> Result<UniquePtr<Shape>>;

        /// One-line structural summary of a shape (root type, sub-shape
        /// counts, closedness, validity) — for diagnostics and tests.
        fn describe(s: &Shape) -> Result<String>;

        /// Count the faces of a shape (for topology assertions/debugging).
        fn count_faces(s: &Shape) -> Result<usize>;

        /// Count the UNIQUE edges of a shape (TopExp map order, matching
        /// tessellation edge ids) — sizes element maps.
        fn count_edges(s: &Shape) -> Result<usize>;

        /// Enclosed volume of a solid (model units³), for validating builds.
        fn volume(s: &Shape) -> Result<f64>;

        /// Exact area of the face at `index` (TopExp order, matching
        /// tessellation face ids), for measurement.
        fn face_area(s: &Shape, index: u32) -> Result<f64>;

        /// Exact length of the edge at `index` (unique-edge order, matching
        /// tessellation edge ids), for measurement.
        fn edge_length(s: &Shape, index: u32) -> Result<f64>;

        // --- Interrogation ------------------------------------------------
        /// The tight axis-aligned bounding box alone (`[min xyz, max xyz]`) —
        /// cheap enough to ask every frame.
        fn bbox(s: &Shape) -> Result<Vec<f64>>;

        /// Volume, surface area, centre of mass, inertia and the tight
        /// axis-aligned bounding box, in one pass.
        fn body_props(s: &Shape) -> Result<BodyProps>;

        /// Sub-shape counts, closedness and BRepCheck validity, typed.
        fn topology(s: &Shape) -> Result<Topology>;

        /// Every face, in tessellation/face_area order.
        fn faces(s: &Shape) -> Result<Vec<FaceInfo>>;

        /// Every unique edge, in tessellation/edge_length order.
        fn edges(s: &Shape) -> Result<Vec<EdgeInfo>>;

        /// Minimum distance between two shapes (0 when they touch or overlap).
        fn distance(a: &Shape, b: &Shape) -> Result<f64>;

        // --- Edge treatments ----------------------------------------------
        /// Fillet every edge of `shape` with a constant `radius`.
        fn fillet_all_edges(shape: &Shape, radius: f64) -> Result<UniquePtr<Shape>>;

        /// Fillet the edges nearest each xyz triple in `coords` with `radius`.
        fn fillet_edges(
            shape: &Shape,
            coords: &[f64],
            radius: f64,
        ) -> Result<UniquePtr<Shape>>;

        /// Fillet-every-edge with lineage: `face_records`/`edge_records` carry
        /// `[0, input_index, output_index|-1]` survival triples;
        /// `rounded_records` carries `[input_edge, output_face]` pairs — the
        /// fillet face each rounded edge GENERATED.
        fn fillet_all_tracked(
            shape: &Shape,
            radius: f64,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            rounded_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        /// Fillet-by-anchors with lineage (same record scheme as
        /// `fillet_all_tracked`). Records stay EMPTY when the result needed
        /// healing — ShapeFix discards history.
        fn fillet_edges_tracked(
            shape: &Shape,
            coords: &[f64],
            radius: f64,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            rounded_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        /// Chamfer every edge of `shape`: a symmetric bevel set back
        /// `distance` along both faces of each edge.
        fn chamfer_all_edges(shape: &Shape, distance: f64) -> Result<UniquePtr<Shape>>;

        /// Chamfer the edges nearest each xyz triple in `coords` by `distance`.
        fn chamfer_edges(
            shape: &Shape,
            coords: &[f64],
            distance: f64,
        ) -> Result<UniquePtr<Shape>>;

        /// Chamfer-every-edge with lineage (the `fillet_all_tracked` record
        /// scheme; `bevel_records` pairs each chamfered edge with the bevel
        /// face it GENERATED).
        fn chamfer_all_tracked(
            shape: &Shape,
            distance: f64,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            bevel_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        /// Chamfer-by-anchors with lineage. Records stay EMPTY when the
        /// result needed healing.
        fn chamfer_edges_tracked(
            shape: &Shape,
            coords: &[f64],
            distance: f64,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            bevel_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        // --- Shell ----------------------------------------------------------
        /// Hollow `shape` to a uniform inward wall `thickness`, removing the
        /// faces anchored by each `[px,py,pz, nx,ny,nz]` sextuple in
        /// `anchors` (none → a closed hollow).
        fn shell(shape: &Shape, anchors: &[f64], thickness: f64) -> Result<UniquePtr<Shape>>;

        /// `shell` with lineage: survival triples for the outer faces/edges
        /// and `offset_records` `[input_face, output_face]` pairs — the inner
        /// wall each kept face GENERATED. EMPTY when healed.
        fn shell_tracked(
            shape: &Shape,
            anchors: &[f64],
            thickness: f64,
            face_records: &mut Vec<i32>,
            edge_records: &mut Vec<i32>,
            offset_records: &mut Vec<i32>,
        ) -> Result<UniquePtr<Shape>>;

        // --- Data exchange ------------------------------------------------
        /// Read a STEP file's bytes: every solid in it (closed free shells
        /// are promoted to solids), as one solid or a compound of them, in
        /// the file's units converted to millimetres. A file the parser
        /// can't fully read (truncated, syntax errors) is refused, not
        /// patched.
        fn read_step(data: &[u8]) -> Result<UniquePtr<Shape>>;

        /// Read an STL file's bytes (ASCII or binary) as a faceted solid:
        /// one planar face per triangle, sharing vertices and edges, with
        /// face orientation repaired and inward shells attached to the
        /// solid they hollow. Each connected closed mesh is one solid, in
        /// millimetres. A mesh with open or non-manifold edges is refused.
        fn read_stl(data: &[u8]) -> Result<UniquePtr<Shape>>;

        /// Write the shape as an AP214 STEP file.
        fn write_step(shape: &Shape, path: &str) -> Result<()>;

        // --- Display / export ---------------------------------------------
        /// Tessellate to a triangle mesh. `deflection` is the max chord
        /// deviation (model units); smaller = smoother and heavier.
        fn tessellate(shape: &Shape, deflection: f64) -> Result<Mesh>;

        /// Write a binary STL. `deflection` controls mesh resolution.
        fn write_stl(shape: &Shape, path: &str, deflection: f64) -> Result<()>;

        /// The plane of the face at `index` (TopExp order), for sketch-on-face.
        fn face_plane(shape: &Shape, index: u32) -> Result<PlaneFrame>;

        /// The cylinder of the face at `index` (TopExp order), for threading.
        fn face_cylinder(shape: &Shape, index: u32) -> Result<CylinderFrame>;

        /// Midpoints (flat x,y,z) of the edges bounding face `index`, for
        /// filleting a whole face's edges.
        fn face_edge_midpoints(shape: &Shape, index: u32) -> Result<Vec<f64>>;
    }
}

// Re-export the bridge contents at module root so crate-internal consumers
// write `crate::ffi::Shape` (the module is `pub(crate)` — `solids` is the only
// intended user).
pub use bridge::*;
