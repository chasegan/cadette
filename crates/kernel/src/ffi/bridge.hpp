#pragma once

// cxx runtime types (rust::Str, rust::Vec, ...).
#include "rust/cxx.h"

#include <memory>
#include <TopoDS_Shape.hxx>

namespace rmf {

// Forward declaration of the cxx-generated shared struct. Its full definition
// lives in the generated "rmf-kernel/src/ffi.rs.h" header, which bridge.cpp
// includes. A forward declaration is sufficient for these by-value
// declarations; only the definitions (in bridge.cpp) need the full type.
struct Mesh;

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

// --- Extrude ----------------------------------------------------------------
std::unique_ptr<Shape> extrude(const Shape& s, double distance);

// --- Transforms -------------------------------------------------------------
std::unique_ptr<Shape> translate(const Shape& s, double dx, double dy, double dz);

// --- Booleans ---------------------------------------------------------------
std::unique_ptr<Shape> fuse(const Shape& a, const Shape& b);
std::unique_ptr<Shape> cut(const Shape& a, const Shape& b);
std::unique_ptr<Shape> common(const Shape& a, const Shape& b);

// --- Edge treatments --------------------------------------------------------
std::unique_ptr<Shape> fillet_all_edges(const Shape& s, double radius);

// --- Display / export -------------------------------------------------------
Mesh tessellate(const Shape& s, double deflection);
void write_stl(const Shape& s, rust::Str path, double deflection);

}  // namespace rmf
