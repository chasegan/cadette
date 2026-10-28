#pragma once

// cxx runtime types (rust::Str, rust::Vec, ...).
#include "rust/cxx.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <TopoDS_Shape.hxx>

namespace cdt {

// Forward declarations of the cxx-generated shared structs. Their full
// definitions live in the generated "cdt-kernel/src/ffi.rs.h" header, which
// bridge.cpp includes. A forward declaration is sufficient for these by-value
// declarations; only the definitions (in bridge.cpp) need the full type.
struct Mesh;
struct PlaneFrame;
struct CylinderFrame;
struct BodyProps;
struct Topology;
struct FaceInfo;
struct EdgeInfo;

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
std::unique_ptr<Shape> extrude_tracked(const Shape& s, double distance,
                                       rust::Vec<int32_t>& records);

// --- Sweep ------------------------------------------------------------------
std::unique_ptr<Shape> path_wire(double ox, double oy, double oz,
                                 double xx, double xy, double xz,
                                 double yx, double yy, double yz,
                                 rust::Slice<const double> points,
                                 rust::Slice<const double> segs);
std::unique_ptr<Shape> helix_wire(double cx, double cy, double cz,
                                  double ax, double ay, double az,
                                  double xx, double xy, double xz,
                                  double radius, double pitch, double length,
                                  bool left_handed);
std::unique_ptr<Shape> sweep(const Shape& profile, const Shape& spine,
                             double bnx, double bny, double bnz);
std::unique_ptr<Shape> sweep_tracked(const Shape& profile, const Shape& spine,
                                     double bnx, double bny, double bnz,
                                     rust::Vec<int32_t>& records);

// --- Revolve ----------------------------------------------------------------
std::unique_ptr<Shape> revolve(const Shape& s, double ax, double ay, double az,
                               double angle);
std::unique_ptr<Shape> revolve_tracked(const Shape& s, double ax, double ay,
                                       double az, double angle,
                                       rust::Vec<int32_t>& records);
rust::Vec<uint32_t> edge_face_adjacency(const Shape& s);

// --- Push/pull --------------------------------------------------------------
std::unique_ptr<Shape> push_pull(const Shape& s, double px, double py, double pz,
                                 double nx, double ny, double nz, double distance);
std::unique_ptr<Shape> push_pull_tracked(const Shape& s, double px, double py,
                                         double pz, double nx, double ny,
                                         double nz, double distance,
                                         rust::Vec<int32_t>& meta,
                                         rust::Vec<int32_t>& face_records,
                                         rust::Vec<int32_t>& edge_records,
                                         rust::Vec<int32_t>& section_records);

// --- Transforms -------------------------------------------------------------
std::unique_ptr<Shape> translate(const Shape& s, double dx, double dy, double dz);

// --- Booleans ---------------------------------------------------------------
std::unique_ptr<Shape> fuse(const Shape& a, const Shape& b);
std::unique_ptr<Shape> cut(const Shape& a, const Shape& b);
std::unique_ptr<Shape> common(const Shape& a, const Shape& b);
std::unique_ptr<Shape> boolean_tracked(const Shape& a, const Shape& b, int32_t op,
                                       rust::Vec<int32_t>& face_records,
                                       rust::Vec<int32_t>& edge_records,
                                       rust::Vec<int32_t>& section_records);
rust::Vec<double> elem_centroid(const Shape& s, int32_t kind, uint32_t index);
std::unique_ptr<Shape> unify(const Shape& s);
rust::String describe(const Shape& s);
std::size_t count_faces(const Shape& s);
std::size_t count_edges(const Shape& s);
double volume(const Shape& s);
double face_area(const Shape& s, uint32_t index);
double edge_length(const Shape& s, uint32_t index);

// --- Interrogation ----------------------------------------------------------
rust::Vec<double> bbox(const Shape& s);
BodyProps body_props(const Shape& s);
Topology topology(const Shape& s);
rust::Vec<FaceInfo> faces(const Shape& s);
rust::Vec<EdgeInfo> edges(const Shape& s);
double distance(const Shape& a, const Shape& b);

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
std::unique_ptr<Shape> fillet_all_tracked(const Shape& s, double radius,
                                          rust::Vec<int32_t>& face_records,
                                          rust::Vec<int32_t>& edge_records,
                                          rust::Vec<int32_t>& rounded_records);
std::unique_ptr<Shape> fillet_edges_tracked(const Shape& s,
                                            rust::Slice<const double> coords,
                                            double radius,
                                            rust::Vec<int32_t>& face_records,
                                            rust::Vec<int32_t>& edge_records,
                                            rust::Vec<int32_t>& rounded_records);
std::unique_ptr<Shape> chamfer_all_edges(const Shape& s, double distance);
std::unique_ptr<Shape> chamfer_edges(const Shape& s,
                                     rust::Slice<const double> coords,
                                     double distance);
std::unique_ptr<Shape> chamfer_all_tracked(const Shape& s, double distance,
                                           rust::Vec<int32_t>& face_records,
                                           rust::Vec<int32_t>& edge_records,
                                           rust::Vec<int32_t>& bevel_records);
std::unique_ptr<Shape> chamfer_edges_tracked(const Shape& s,
                                             rust::Slice<const double> coords,
                                             double distance,
                                             rust::Vec<int32_t>& face_records,
                                             rust::Vec<int32_t>& edge_records,
                                             rust::Vec<int32_t>& bevel_records);

// --- Shell ------------------------------------------------------------------
std::unique_ptr<Shape> shell(const Shape& s, rust::Slice<const double> anchors,
                             double thickness);
std::unique_ptr<Shape> shell_tracked(const Shape& s,
                                     rust::Slice<const double> anchors,
                                     double thickness,
                                     rust::Vec<int32_t>& face_records,
                                     rust::Vec<int32_t>& edge_records,
                                     rust::Vec<int32_t>& offset_records);

// --- Data exchange ----------------------------------------------------------
std::unique_ptr<Shape> read_step(rust::Slice<const uint8_t> data);
void write_step(const Shape& s, rust::Str path);

// --- Display / export -------------------------------------------------------
Mesh tessellate(const Shape& s, double deflection);
void write_stl(const Shape& s, rust::Str path, double deflection);
PlaneFrame face_plane(const Shape& s, uint32_t index);
CylinderFrame face_cylinder(const Shape& s, uint32_t index);
rust::Vec<double> face_edge_midpoints(const Shape& s, uint32_t index);

}  // namespace cdt
