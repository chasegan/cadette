#pragma once

// cxx runtime types (rust::Str, rust::Vec, ...).
#include "rust/cxx.h"

#include <memory>
#include <TopoDS_Shape.hxx>

namespace cdt {

// Forward declarations of the cxx-generated shared structs. Their full
// definitions live in the generated "cdt-kernel/src/ffi.rs.h" header, which
// bridge.cpp includes. A forward declaration is sufficient for these by-value
// declarations; only the definitions (in bridge.cpp) need the full type.
struct Mesh;
struct PlaneFrame;

// Opaque-to-Rust handle around an OCCT B-rep shape. We store by value so the
// UniquePtr fully owns the topology and cxx can hand ownership to Rust.
class Shape {
public:
  TopoDS_Shape shape;

  Shape() = default;
  explicit Shape(const TopoDS_Shape& s) : shape(s) {}
};

// --- Primitives -------------------------------------------------------------
std::unique_ptr<Shape> make_box(double dx, double dy, double dz);
std::unique_ptr<Shape> make_sphere(double radius);
std::unique_ptr<Shape> make_cylinder(double radius, double height);

// --- Sketch profiles (planar faces) -----------------------------------------
std::unique_ptr<Shape> make_rectangle_face(double ox, double oy, double oz,
                                           double xx, double xy, double xz,
                                           double yx, double yy, double yz,
                                           double width, double height);
std::unique_ptr<Shape> make_circle_face(double ox, double oy, double oz,
                                        double nx, double ny, double nz,
                                        double radius);
std::unique_ptr<Shape> make_polygon_face(double ox, double oy, double oz,
                                         double xx, double xy, double xz,
                                         double yx, double yy, double yz,
                                         rust::Slice<const double> points);
std::unique_ptr<Shape> profile_face(double ox, double oy, double oz,
                                    double xx, double xy, double xz,
                                    double yx, double yy, double yz,
                                    rust::Slice<const double> points,
                                    rust::Slice<const double> segs);

// --- Extrude ----------------------------------------------------------------
std::unique_ptr<Shape> extrude(const Shape& s, double distance);

// --- Revolve ----------------------------------------------------------------
std::unique_ptr<Shape> revolve(const Shape& s, double ax, double ay, double az,
                               double angle);

// --- Push/pull --------------------------------------------------------------
std::unique_ptr<Shape> push_pull(const Shape& s, double px, double py, double pz,
                                 double nx, double ny, double nz, double distance);

// --- Transforms -------------------------------------------------------------
std::unique_ptr<Shape> translate(const Shape& s, double dx, double dy, double dz);

// --- Booleans ---------------------------------------------------------------
std::unique_ptr<Shape> fuse(const Shape& a, const Shape& b);
std::unique_ptr<Shape> cut(const Shape& a, const Shape& b);
std::unique_ptr<Shape> common(const Shape& a, const Shape& b);
std::unique_ptr<Shape> unify(const Shape& s);
std::size_t count_faces(const Shape& s);

// --- Edge treatments --------------------------------------------------------
std::unique_ptr<Shape> rotate(const Shape& s, double cx, double cy, double cz,
                              double ax, double ay, double az, double angle);

std::unique_ptr<Shape> scale(const Shape& s, double sx, double sy, double sz,
                             double ax, double ay, double az);

std::unique_ptr<Shape> mirror(const Shape& s, double ox, double oy, double oz,
                              double nx, double ny, double nz);

std::unique_ptr<Shape> copy_shape(const Shape& s);
std::unique_ptr<Shape> compound(const Shape& a, const Shape& b);

std::unique_ptr<Shape> fillet_all_edges(const Shape& s, double radius);
std::unique_ptr<Shape> fillet_edges(const Shape& s,
                                    rust::Slice<const double> coords,
                                    double radius);

// --- Display / export -------------------------------------------------------
Mesh tessellate(const Shape& s, double deflection);
void write_stl(const Shape& s, rust::Str path, double deflection);
PlaneFrame face_plane(const Shape& s, uint32_t index);
rust::Vec<double> face_edge_midpoints(const Shape& s, uint32_t index);

}  // namespace cdt
