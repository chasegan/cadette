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
pub mod ffi {
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
        /// Sweep the planar `profile` face along the `spine` wire into a solid,
        /// keeping the profile normal to the path (corrected-Frenet frame).
        fn sweep(profile: &Shape, spine: &Shape) -> Result<UniquePtr<Shape>>;

        // --- Revolve ------------------------------------------------------
        /// Revolve the planar profile `shape` by `angle` radians about the
        /// straight edge nearest `(ax,ay,az)` (a full turn is `2π`).
        fn revolve(
            shape: &Shape,
            ax: f64, ay: f64, az: f64,
            angle: f64,
        ) -> Result<UniquePtr<Shape>>;

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

        /// Merge same-domain faces/edges (remove seam edges between coplanar
        /// neighbours) — an explicit "refine". Booleans already do this.
        fn unify(s: &Shape) -> Result<UniquePtr<Shape>>;

        /// Count the faces of a shape (for topology assertions/debugging).
        fn count_faces(s: &Shape) -> usize;

        /// Enclosed volume of a solid (model units³), for validating builds.
        fn volume(s: &Shape) -> f64;

        // --- Edge treatments ----------------------------------------------
        /// Fillet every edge of `shape` with a constant `radius`.
        fn fillet_all_edges(shape: &Shape, radius: f64) -> Result<UniquePtr<Shape>>;

        /// Fillet the edges nearest each xyz triple in `coords` with `radius`.
        fn fillet_edges(
            shape: &Shape,
            coords: &[f64],
            radius: f64,
        ) -> Result<UniquePtr<Shape>>;

        // --- Display / export ---------------------------------------------
        /// Tessellate to a triangle mesh. `deflection` is the max chord
        /// deviation (model units); smaller = smoother and heavier.
        fn tessellate(shape: &Shape, deflection: f64) -> Result<Mesh>;

        /// Write a binary STL. `deflection` controls mesh resolution.
        fn write_stl(shape: &Shape, path: &str, deflection: f64) -> Result<()>;

        /// The plane of the face at `index` (TopExp order), for sketch-on-face.
        fn face_plane(shape: &Shape, index: u32) -> Result<PlaneFrame>;

        /// Midpoints (flat x,y,z) of the edges bounding face `index`, for
        /// filleting a whole face's edges.
        fn face_edge_midpoints(shape: &Shape, index: u32) -> Result<Vec<f64>>;
    }
}

// Re-export the bridge contents at module root so consumers write
// `kernel::ffi::Shape` rather than `kernel::ffi::ffi::Shape`.
pub use ffi::*;
