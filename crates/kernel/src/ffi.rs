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

#[cxx::bridge(namespace = "rmf")]
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
        include!("rmf-kernel/src/ffi/bridge.hpp");

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

        // --- Extrude ------------------------------------------------------
        /// Extrude a planar face along its normal by `distance` into a solid.
        fn extrude(shape: &Shape, distance: f64) -> Result<UniquePtr<Shape>>;

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

        // --- Booleans -----------------------------------------------------
        fn fuse(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;
        fn cut(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;
        fn common(a: &Shape, b: &Shape) -> Result<UniquePtr<Shape>>;

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
    }
}

// Re-export the bridge contents at module root so consumers write
// `kernel::ffi::Shape` rather than `kernel::ffi::ffi::Shape`.
pub use ffi::*;
