#include "cdt-kernel/src/ffi/bridge.hpp"
// cxx-generated header: defines cdt::Mesh and the extern shims.
#include "cdt-kernel/src/ffi.rs.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <Geom_Plane.hxx>
#include <GeomLib_IsPlanarSurface.hxx>
#include <Geom_BezierCurve.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <GProp_GProps.hxx>
#include <BRepGProp.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Vertex.hxx>
#include <Geom_Curve.hxx>
#include <Poly_Triangulation.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <StlAPI_Writer.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <Standard_Failure.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Mat.hxx>
#include <gp_XYZ.hxx>
#include <gp_Vec.hxx>

namespace cdt {
namespace {

// OCCT signals errors by throwing Standard_Failure, which does not reliably
// derive from std::exception across versions. Re-throw as std::runtime_error so
// cxx can translate it into a Rust `Err`. Calling Rust-bound code across an
// un-caught C++ exception is undefined behavior, so every entry point funnels
// through this guard.
template <typename F>
auto guard(const char* op, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const Standard_Failure& e) {
    const char* msg = e.GetMessageString();
    throw std::runtime_error(std::string(op) + ": " +
                             (msg ? msg : "OCCT failure"));
  } catch (const std::exception&) {
    throw;  // already a std::exception; let cxx handle it
  } catch (...) {
    throw std::runtime_error(std::string(op) + ": unknown C++ exception");
  }
}

// Merge faces that lie on the same underlying surface (and edges on the same
// curve), removing the artificial seam edges that boolean operations leave
// between coplanar neighbours — OCCT's equivalent of FreeCAD's "Refine shape".
// Only same-domain geometry is unified, so intentional features are untouched.
TopoDS_Shape unify_same_domain(const TopoDS_Shape& s) {
  ShapeUpgrade_UnifySameDomain tool(s, /*unifyEdges=*/true, /*unifyFaces=*/true,
                                    /*concatBSplines=*/false);
  tool.Build();
  return tool.Shape();
}

}  // namespace

// --- Primitives -------------------------------------------------------------

std::unique_ptr<Shape> make_box(double dx, double dy, double dz) {
  return guard("make_box", [&] {
    return std::make_unique<Shape>(BRepPrimAPI_MakeBox(dx, dy, dz).Shape());
  });
}

std::unique_ptr<Shape> make_sphere(double radius) {
  return guard("make_sphere", [&] {
    return std::make_unique<Shape>(BRepPrimAPI_MakeSphere(radius).Shape());
  });
}

std::unique_ptr<Shape> make_cylinder(double radius, double height) {
  return guard("make_cylinder", [&] {
    return std::make_unique<Shape>(
        BRepPrimAPI_MakeCylinder(radius, height).Shape());
  });
}

// --- Sketch profiles (planar faces) -----------------------------------------

std::unique_ptr<Shape> make_rectangle_face(double ox, double oy, double oz,
                                           double xx, double xy, double xz,
                                           double yx, double yy, double yz,
                                           double width, double height) {
  return guard("make_rectangle_face", [&] {
    const gp_Vec o(ox, oy, oz);
    const gp_Vec x(xx, xy, xz);
    const gp_Vec y(yx, yy, yz);
    const gp_Vec hx = x * (width * 0.5);
    const gp_Vec hy = y * (height * 0.5);

    // Corners wound counter-clockwise in the plane so the face normal agrees
    // with x_dir x y_dir.
    auto corner = [&](const gp_Vec& a, const gp_Vec& b) {
      const gp_Vec p = o + a + b;
      return gp_Pnt(p.X(), p.Y(), p.Z());
    };
    BRepBuilderAPI_MakePolygon poly(corner(-hx, -hy), corner(hx, -hy),
                                    corner(hx, hy), corner(-hx, hy),
                                    /*close=*/Standard_True);
    BRepBuilderAPI_MakeFace face(poly.Wire());
    return std::make_unique<Shape>(face.Shape());
  });
}

std::unique_ptr<Shape> make_circle_face(double ox, double oy, double oz,
                                        double nx, double ny, double nz,
                                        double radius) {
  return guard("make_circle_face", [&] {
    gp_Ax2 axis(gp_Pnt(ox, oy, oz), gp_Dir(nx, ny, nz));
    gp_Circ circle(axis, radius);
    BRepBuilderAPI_MakeEdge edge(circle);
    BRepBuilderAPI_MakeWire wire(edge.Edge());
    BRepBuilderAPI_MakeFace face(wire.Wire());
    return std::make_unique<Shape>(face.Shape());
  });
}

std::unique_ptr<Shape> make_polygon_face(double ox, double oy, double oz,
                                         double xx, double xy, double xz,
                                         double yx, double yy, double yz,
                                         rust::Slice<const double> points) {
  return guard("make_polygon_face", [&] {
    const std::size_t count = points.size() / 2;
    if (count < 3) {
      throw std::runtime_error("make_polygon_face: need at least 3 points");
    }
    const gp_Vec o(ox, oy, oz);
    const gp_Vec x(xx, xy, xz);
    const gp_Vec y(yx, yy, yz);

    BRepBuilderAPI_MakePolygon poly;
    for (std::size_t i = 0; i < count; ++i) {
      const double u = points[2 * i];
      const double v = points[2 * i + 1];
      const gp_Vec p = o + x * u + y * v;
      poly.Add(gp_Pnt(p.X(), p.Y(), p.Z()));
    }
    poly.Close();
    BRepBuilderAPI_MakeFace face(poly.Wire());
    return std::make_unique<Shape>(face.Shape());
  });
}

// A planar face whose closed boundary mixes straight and cubic-bezier segments.
// `points` is the flat 2D loop vertices (in plane coords); `segs` has 5 doubles
// per segment i (from vertex i to vertex i+1): [is_bezier, c1x, c1y, c2x, c2y]
// where the control points are used only when is_bezier != 0.
std::unique_ptr<Shape> profile_face(double ox, double oy, double oz,
                                    double xx, double xy, double xz,
                                    double yx, double yy, double yz,
                                    rust::Slice<const double> points,
                                    rust::Slice<const double> segs) {
  return guard("profile_face", [&] {
    const std::size_t n = points.size() / 2;
    if (n < 3 || segs.size() != n * 5) {
      throw std::runtime_error("profile_face: bad point/segment counts");
    }
    const gp_Vec o(ox, oy, oz), x(xx, xy, xz), y(yx, yy, yz);
    auto to3d = [&](double u, double v) {
      const gp_Vec p = o + x * u + y * v;
      return gp_Pnt(p.X(), p.Y(), p.Z());
    };

    BRepBuilderAPI_MakeWire wire;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j = (i + 1) % n;
      const gp_Pnt a = to3d(points[2 * i], points[2 * i + 1]);
      const gp_Pnt b = to3d(points[2 * j], points[2 * j + 1]);
      if (segs[5 * i] != 0.0) {
        const gp_Pnt c1 = to3d(segs[5 * i + 1], segs[5 * i + 2]);
        const gp_Pnt c2 = to3d(segs[5 * i + 3], segs[5 * i + 4]);
        TColgp_Array1OfPnt poles(1, 4);
        poles.SetValue(1, a);
        poles.SetValue(2, c1);
        poles.SetValue(3, c2);
        poles.SetValue(4, b);
        Handle(Geom_BezierCurve) curve = new Geom_BezierCurve(poles);
        wire.Add(BRepBuilderAPI_MakeEdge(curve).Edge());
      } else {
        wire.Add(BRepBuilderAPI_MakeEdge(a, b).Edge());
      }
    }
    BRepBuilderAPI_MakeFace face(wire.Wire(), /*OnlyPlane=*/Standard_True);
    return std::make_unique<Shape>(face.Shape());
  });
}

// --- Extrude ----------------------------------------------------------------

std::unique_ptr<Shape> extrude(const Shape& s, double distance) {
  return guard("extrude", [&] {
    // Find the planar face and extrude along its normal.
    TopExp_Explorer ex(s.shape, TopAbs_FACE);
    if (!ex.More()) {
      throw std::runtime_error("extrude: shape has no face to extrude");
    }
    TopoDS_Face face = TopoDS::Face(ex.Current());
    Handle(Geom_Surface) surface = BRep_Tool::Surface(face);
    Handle(Geom_Plane) plane = Handle(Geom_Plane)::DownCast(surface);
    if (plane.IsNull()) {
      throw std::runtime_error("extrude: profile face is not planar");
    }
    gp_Dir normal = plane->Pln().Axis().Direction();
    gp_Vec direction(normal);
    direction *= distance;
    BRepPrimAPI_MakePrism prism(s.shape, direction);
    return std::make_unique<Shape>(prism.Shape());
  });
}

// --- Revolve ----------------------------------------------------------------

// Revolve the profile `s` (a planar face — from a sketch or a model face)
// through `angle` radians about the straight edge nearest the anchor
// `(ax,ay,az)`. The axis edge is one of the profile's own segments (a sketch
// centerline) or an adjacent model edge; either way it resolves the same way as
// a fillet anchor. A full revolution is `angle = 2π`.
std::unique_ptr<Shape> revolve(const Shape& s, double ax, double ay, double az,
                               double angle) {
  return guard("revolve", [&] {
    const gp_Pnt anchor(ax, ay, az);
    const TopoDS_Vertex probe_vertex = BRepBuilderAPI_MakeVertex(anchor).Vertex();

    // Find the edge nearest the anchor — the durable axis reference (same scheme
    // as fillet_edges: the picked point lies on the chosen edge).
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);
    TopoDS_Edge target;
    double best = 1e30;
    for (int i = 1; i <= edges.Extent(); ++i) {
      const TopoDS_Edge edge = TopoDS::Edge(edges(i));
      BRepExtrema_DistShapeShape probe(probe_vertex, edge);
      if (!probe.IsDone()) continue;
      if (probe.Value() < best) {
        best = probe.Value();
        target = edge;
      }
    }
    if (target.IsNull() || best > 1.0) {
      throw std::runtime_error("revolve: no matching axis edge");
    }

    // The axis is the edge's straight line, taken from its two endpoints.
    TopoDS_Vertex v1, v2;
    TopExp::Vertices(target, v1, v2);
    const gp_Pnt p1 = BRep_Tool::Pnt(v1);
    const gp_Pnt p2 = BRep_Tool::Pnt(v2);
    const gp_Vec dir(p1, p2);
    if (dir.Magnitude() < 1e-9) {
      throw std::runtime_error("revolve: degenerate axis edge");
    }
    const gp_Ax1 axis(p1, gp_Dir(dir));
    const TopoDS_Shape solid = BRepPrimAPI_MakeRevol(s.shape, axis, angle).Shape();
    return std::make_unique<Shape>(solid);
  });
}

// --- Push/pull --------------------------------------------------------------

// A face's plane, accepting both true Geom_Plane surfaces and surfaces that are
// only *geometrically* planar — notably the b-splines a non-uniform GTransform
// (our resize/Scale) produces from planes. Without this, a resized body's flat
// faces stop reading as planar, so push/pull and sketch-on-face fail on them.
// The Geom_Plane fast path keeps unscaled bodies free of any sampling cost.
static bool face_pln(const TopoDS_Face& face, gp_Pln& out) {
  Handle(Geom_Surface) surface = BRep_Tool::Surface(face);
  Handle(Geom_Plane) plane = Handle(Geom_Plane)::DownCast(surface);
  if (!plane.IsNull()) {
    out = plane->Pln();
    return true;
  }
  GeomLib_IsPlanarSurface checker(surface, 1e-6);
  if (checker.IsPlanar()) {
    out = checker.Plan();
    return true;
  }
  return false;
}

std::unique_ptr<Shape> push_pull(const Shape& s, double px, double py, double pz,
                                 double nx, double ny, double nz,
                                 double distance) {
  return guard("push_pull", [&] {
    const gp_Pnt anchor(px, py, pz);
    const gp_Dir wanted(nx, ny, nz);

    // Find the planar face whose plane passes through the anchor with a
    // matching normal (the durable face reference).
    TopoDS_Face target;
    bool found = false;
    for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next()) {
      const TopoDS_Face face = TopoDS::Face(ex.Current());
      gp_Pln pln;
      if (!face_pln(face, pln)) continue;
      if (std::abs(pln.Axis().Direction().Dot(wanted)) < 0.99) continue;
      // The anchor must lie on THIS face, not merely on its (infinite) plane —
      // otherwise a coplanar neighbour could be picked instead. The tolerance
      // absorbs the small error in a depth-reconstructed clicked point while
      // still being far tighter than any real gap to a separate face.
      BRepExtrema_DistShapeShape probe(BRepBuilderAPI_MakeVertex(anchor).Vertex(), face);
      if (!probe.IsDone() || probe.Value() > 0.05) continue;
      target = face;
      found = true;
      break;
    }
    if (!found) {
      throw std::runtime_error("push_pull: no matching planar face");
    }

    gp_Vec direction(wanted);
    direction *= distance;
    if (direction.Magnitude() < 1e-9) {
      return std::make_unique<Shape>(s.shape);  // no-op
    }
    const TopoDS_Shape prism = BRepPrimAPI_MakePrism(target, direction).Shape();
    const TopoDS_Shape result =
        distance >= 0.0 ? BRepAlgoAPI_Fuse(s.shape, prism).Shape()
                        : BRepAlgoAPI_Cut(s.shape, prism).Shape();
    // Pulling a face flush with a neighbour leaves a coplanar seam — merge it.
    return std::make_unique<Shape>(unify_same_domain(result));
  });
}

// --- Transforms -------------------------------------------------------------

std::unique_ptr<Shape> translate(const Shape& s, double dx, double dy,
                                 double dz) {
  return guard("translate", [&] {
    gp_Trsf t;
    t.SetTranslation(gp_Vec(dx, dy, dz));
    BRepBuilderAPI_Transform xf(s.shape, t, /*copy=*/true);
    return std::make_unique<Shape>(xf.Shape());
  });
}

std::unique_ptr<Shape> rotate(const Shape& s, double cx, double cy, double cz,
                              double ax, double ay, double az, double angle) {
  return guard("rotate", [&] {
    gp_Trsf t;
    t.SetRotation(gp_Ax1(gp_Pnt(cx, cy, cz), gp_Dir(ax, ay, az)), angle);
    BRepBuilderAPI_Transform xf(s.shape, t, /*copy=*/true);
    return std::make_unique<Shape>(xf.Shape());
  });
}

// Non-uniform scale about an anchor: p' = anchor + diag(sx,sy,sz)*(p - anchor).
// A general (anisotropic) scale needs gp_GTrsf + BRepBuilderAPI_GTransform; the
// uniform gp_Trsf::SetScale can't do per-axis factors. GTransform may convert
// surfaces to b-splines, so it's the right tool for resizing arbitrary bodies.
std::unique_ptr<Shape> scale(const Shape& s, double sx, double sy, double sz,
                             double ax, double ay, double az) {
  return guard("scale", [&] {
    gp_GTrsf t;
    t.SetVectorialPart(gp_Mat(sx, 0, 0, 0, sy, 0, 0, 0, sz));
    t.SetTranslationPart(gp_XYZ(ax * (1.0 - sx), ay * (1.0 - sy), az * (1.0 - sz)));
    BRepBuilderAPI_GTransform xf(s.shape, t, /*copy=*/true);
    return std::make_unique<Shape>(xf.Shape());
  });
}

// Reflect across the plane through `(ox,oy,oz)` with normal `(nx,ny,nz)`. A
// mirror gp_Trsf has negative determinant; BRepBuilderAPI_Transform reverses the
// shape's faces so the result stays a properly-oriented (outward-normal) solid.
std::unique_ptr<Shape> mirror(const Shape& s, double ox, double oy, double oz,
                              double nx, double ny, double nz) {
  return guard("mirror", [&] {
    gp_Trsf t;
    t.SetMirror(gp_Ax2(gp_Pnt(ox, oy, oz), gp_Dir(nx, ny, nz)));
    BRepBuilderAPI_Transform xf(s.shape, t, /*copy=*/true);
    return std::make_unique<Shape>(xf.Shape());
  });
}

// A shallow copy: TopoDS_Shape is a handle to a shared, ref-counted TShape, so
// this shares the underlying geometry rather than deep-copying it — cheap, which
// is what lets the regen cache hand out cloned bodies per frame.
std::unique_ptr<Shape> copy_shape(const Shape& s) {
  return std::make_unique<Shape>(s.shape);
}

// Bundle two shapes into one compound (no boolean) — used to export several
// visible bodies as a single STL. Nesting compounds is fine; the mesher and STL
// writer walk all sub-shapes.
std::unique_ptr<Shape> compound(const Shape& a, const Shape& b) {
  return guard("compound", [&] {
    TopoDS_Compound comp;
    BRep_Builder builder;
    builder.MakeCompound(comp);
    builder.Add(comp, a.shape);
    builder.Add(comp, b.shape);
    return std::make_unique<Shape>(comp);
  });
}

// --- Booleans ---------------------------------------------------------------

// Each boolean unifies its result so coplanar faces of the two operands merge
// into one instead of staying split by a seam edge.
std::unique_ptr<Shape> fuse(const Shape& a, const Shape& b) {
  return guard("fuse", [&] {
    return std::make_unique<Shape>(
        unify_same_domain(BRepAlgoAPI_Fuse(a.shape, b.shape).Shape()));
  });
}

std::unique_ptr<Shape> cut(const Shape& a, const Shape& b) {
  return guard("cut", [&] {
    return std::make_unique<Shape>(
        unify_same_domain(BRepAlgoAPI_Cut(a.shape, b.shape).Shape()));
  });
}

std::unique_ptr<Shape> common(const Shape& a, const Shape& b) {
  return guard("common", [&] {
    return std::make_unique<Shape>(
        unify_same_domain(BRepAlgoAPI_Common(a.shape, b.shape).Shape()));
  });
}

// Standalone refine — merge same-domain faces/edges of any shape on demand.
std::unique_ptr<Shape> unify(const Shape& s) {
  return guard("unify", [&] {
    return std::make_unique<Shape>(unify_same_domain(s.shape));
  });
}

// Number of faces in the shape (TopExp order) — for asserting topology.
std::size_t count_faces(const Shape& s) {
  std::size_t n = 0;
  for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next()) n++;
  return n;
}

// --- Edge treatments --------------------------------------------------------

std::unique_ptr<Shape> fillet_all_edges(const Shape& s, double radius) {
  return guard("fillet_all_edges", [&] {
    BRepFilletAPI_MakeFillet mk(s.shape);
    // Use a uniqued edge map: a raw explorer visits shared edges once per
    // adjacent face, which would add each edge to the fillet twice.
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);
    for (int i = 1; i <= edges.Extent(); ++i) {
      mk.Add(radius, TopoDS::Edge(edges(i)));
    }
    return std::make_unique<Shape>(mk.Shape());
  });
}

std::unique_ptr<Shape> fillet_edges(const Shape& s, rust::Slice<const double> coords,
                                    double radius) {
  return guard("fillet_edges", [&] {
    if (coords.size() % 3 != 0) {
      throw std::runtime_error("fillet_edges: coords must be xyz triples");
    }

    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);

    BRepFilletAPI_MakeFillet mk(s.shape);
    // A given edge must be added only once, even if two anchors resolve to it.
    TopTools_MapOfShape added;
    for (size_t base = 0; base < coords.size(); base += 3) {
      const gp_Pnt anchor(coords[base], coords[base + 1], coords[base + 2]);
      const TopoDS_Vertex probe_vertex =
          BRepBuilderAPI_MakeVertex(anchor).Vertex();

      // Find the edge nearest this anchor point (the durable edge reference).
      TopoDS_Edge target;
      double best = 1e30;
      for (int i = 1; i <= edges.Extent(); ++i) {
        const TopoDS_Edge edge = TopoDS::Edge(edges(i));
        BRepExtrema_DistShapeShape probe(probe_vertex, edge);
        if (!probe.IsDone()) continue;
        if (probe.Value() < best) {
          best = probe.Value();
          target = edge;
        }
      }
      // The anchor lies on the picked edge, so a close match is expected; a
      // large gap means the edge moved or vanished after an upstream edit.
      if (target.IsNull() || best > 1.0) {
        throw std::runtime_error("fillet_edges: no matching edge");
      }
      if (added.Add(target)) {
        mk.Add(radius, target);
      }
    }
    return std::make_unique<Shape>(mk.Shape());
  });
}

// --- Display / export -------------------------------------------------------

Mesh tessellate(const Shape& s, double deflection) {
  return guard("tessellate", [&] {
    Mesh m;

    // Mesh a COPY, never the caller's shape: BRepMesh stores its triangulation
    // inside the TShape, and the regen cache hands out shallow Solid clones that
    // share one TShape. Meshing the original would bake a stale triangulation
    // into the shared shape, which then corrupts downstream transforms (e.g. a
    // resize's GTransform reuses it → a "floating" top face).
    const TopoDS_Shape shape = BRepBuilderAPI_Copy(s.shape).Shape();
    BRepMesh_IncrementalMesh mesher(shape, deflection, /*isRelative=*/false,
                                    /*angDeflection=*/0.5,
                                    /*isInParallel=*/true);
    mesher.Perform();

    uint32_t base = 0;        // running vertex offset across faces
    // Face id = TopExp face index, incremented for every face so it matches
    // face_plane()'s indexing even if some face lacks a triangulation.
    uint32_t face_index = 0;
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More();
         ex.Next(), ++face_index) {
      const TopoDS_Face face = TopoDS::Face(ex.Current());
      TopLoc_Location loc;
      Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
      if (tri.IsNull()) continue;

      const gp_Trsf trsf = loc.Transformation();
      const bool reversed = (face.Orientation() == TopAbs_REVERSED);
      const int nb_nodes = tri->NbNodes();

      // Positions (transformed to world space); normals zero-initialized and
      // accumulated from adjacent triangles below for smooth shading. Every
      // vertex of this face carries the same face id for picking.
      for (int i = 1; i <= nb_nodes; ++i) {
        gp_Pnt p = tri->Node(i).Transformed(trsf);
        m.positions.push_back(static_cast<float>(p.X()));
        m.positions.push_back(static_cast<float>(p.Y()));
        m.positions.push_back(static_cast<float>(p.Z()));
        m.normals.push_back(0.0f);
        m.normals.push_back(0.0f);
        m.normals.push_back(0.0f);
        m.face_ids.push_back(face_index);
      }

      const int nb_tris = tri->NbTriangles();
      for (int i = 1; i <= nb_tris; ++i) {
        int a, b, c;
        tri->Triangle(i).Get(a, b, c);  // 1-based, local to this face
        if (reversed) std::swap(b, c);  // keep CCW winding for front faces

        const uint32_t ia = base + (a - 1);
        const uint32_t ib = base + (b - 1);
        const uint32_t ic = base + (c - 1);
        m.indices.push_back(ia);
        m.indices.push_back(ib);
        m.indices.push_back(ic);

        // Area-weighted face normal accumulated into each vertex.
        const float* pa = &m.positions[3 * ia];
        const float* pb = &m.positions[3 * ib];
        const float* pc = &m.positions[3 * ic];
        const float ux = pb[0] - pa[0], uy = pb[1] - pa[1], uz = pb[2] - pa[2];
        const float vx = pc[0] - pa[0], vy = pc[1] - pa[1], vz = pc[2] - pa[2];
        const float nx = uy * vz - uz * vy;
        const float ny = uz * vx - ux * vz;
        const float nz = ux * vy - uy * vx;
        for (uint32_t idx : {ia, ib, ic}) {
          m.normals[3 * idx + 0] += nx;
          m.normals[3 * idx + 1] += ny;
          m.normals[3 * idx + 2] += nz;
        }
      }

      base += static_cast<uint32_t>(nb_nodes);
    }

    // Normalize accumulated vertex normals.
    for (size_t i = 0; i + 2 < m.normals.size(); i += 3) {
      float nx = m.normals[i], ny = m.normals[i + 1], nz = m.normals[i + 2];
      float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (len > 1e-12f) {
        m.normals[i] = nx / len;
        m.normals[i + 1] = ny / len;
        m.normals[i + 2] = nz / len;
      }
    }

    // Crisp feature edges: one 3D polyline per unique edge, taken from the
    // edge's tessellation on an adjacent face. Edge ids match exploration order.
    TopTools_IndexedMapOfShape edge_map;
    TopExp::MapShapes(shape, TopAbs_EDGE, edge_map);
    std::vector<bool> emitted(edge_map.Extent() + 1, false);
    for (TopExp_Explorer fex(shape, TopAbs_FACE); fex.More(); fex.Next()) {
      const TopoDS_Face face = TopoDS::Face(fex.Current());
      TopLoc_Location loc;
      Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
      if (tri.IsNull()) continue;
      const gp_Trsf trsf = loc.Transformation();

      for (TopExp_Explorer eex(face, TopAbs_EDGE); eex.More(); eex.Next()) {
        const TopoDS_Edge edge = TopoDS::Edge(eex.Current());
        const int eid = edge_map.FindIndex(edge);
        if (eid == 0 || emitted[eid]) continue;
        Handle(Poly_PolygonOnTriangulation) poly =
            BRep_Tool::PolygonOnTriangulation(edge, tri, loc);
        if (poly.IsNull()) continue;

        const TColStd_Array1OfInteger& nodes = poly->Nodes();
        const uint32_t first = static_cast<uint32_t>(m.edge_positions.size() / 3);
        for (int i = nodes.Lower(); i <= nodes.Upper(); ++i) {
          gp_Pnt p = tri->Node(nodes(i)).Transformed(trsf);
          m.edge_positions.push_back(static_cast<float>(p.X()));
          m.edge_positions.push_back(static_cast<float>(p.Y()));
          m.edge_positions.push_back(static_cast<float>(p.Z()));
          m.edge_ids.push_back(static_cast<uint32_t>(eid - 1));
        }
        const uint32_t count =
            static_cast<uint32_t>(nodes.Upper() - nodes.Lower() + 1);
        for (uint32_t i = 0; i + 1 < count; ++i) {
          m.edge_indices.push_back(first + i);
          m.edge_indices.push_back(first + i + 1);
        }
        emitted[eid] = true;
      }
    }

    return m;
  });
}

PlaneFrame face_plane(const Shape& s, uint32_t index) {
  return guard("face_plane", [&] {
    PlaneFrame pf{};
    pf.ok = false;
    uint32_t i = 0;
    for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next(), ++i) {
      if (i != index) continue;
      const TopoDS_Face face = TopoDS::Face(ex.Current());
      gp_Pln pln;
      if (!face_pln(face, pln)) break;  // not planar — can't sketch on it
      const gp_Ax3 axis = pln.Position();
      // Origin = face centroid: a point that actually lies on the face (the
      // plane's reference point may not), so a push/pull anchor placed here
      // matches this face rather than a coplanar neighbour.
      GProp_GProps props;
      BRepGProp::SurfaceProperties(face, props);
      const gp_Pnt o = props.CentreOfMass();
      const gp_Dir x = axis.XDirection();
      const gp_Dir y = axis.YDirection();
      pf.ok = true;
      pf.ox = o.X(); pf.oy = o.Y(); pf.oz = o.Z();
      pf.xx = x.X(); pf.xy = x.Y(); pf.xz = x.Z();
      pf.yx = y.X(); pf.yy = y.Y(); pf.yz = y.Z();
      break;
    }
    return pf;
  });
}

// Midpoints (flat x,y,z triples) of the edges bounding face `index` (TopExp
// order, matching face_plane / picking). Used to fillet a whole face's edges:
// the host turns each point into an edge anchor for fillet_edges.
rust::Vec<double> face_edge_midpoints(const Shape& s, uint32_t index) {
  return guard("face_edge_midpoints", [&] {
    rust::Vec<double> out;
    uint32_t i = 0;
    for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next(), ++i) {
      if (i != index) continue;
      const TopoDS_Face face = TopoDS::Face(ex.Current());
      TopTools_IndexedMapOfShape edges;
      TopExp::MapShapes(face, TopAbs_EDGE, edges);
      for (int k = 1; k <= edges.Extent(); ++k) {
        const TopoDS_Edge edge = TopoDS::Edge(edges(k));
        gp_Pnt mid;
        Standard_Real f, l;
        Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, f, l);
        if (!curve.IsNull()) {
          mid = curve->Value(0.5 * (f + l)); // a point genuinely on the edge
        } else {
          TopoDS_Vertex v1, v2;
          TopExp::Vertices(edge, v1, v2);
          const gp_Pnt p1 = BRep_Tool::Pnt(v1), p2 = BRep_Tool::Pnt(v2);
          mid = gp_Pnt((p1.X() + p2.X()) / 2, (p1.Y() + p2.Y()) / 2,
                       (p1.Z() + p2.Z()) / 2);
        }
        out.push_back(mid.X());
        out.push_back(mid.Y());
        out.push_back(mid.Z());
      }
      break;
    }
    return out;
  });
}

void write_stl(const Shape& s, rust::Str path, double deflection) {
  guard("write_stl", [&] {
    // Mesh a copy (like tessellate): the export compound shares its sub-shapes'
    // TShapes with the cached bodies, so meshing in place would bake a stale
    // triangulation into them and corrupt a later resize.
    const TopoDS_Shape shape = BRepBuilderAPI_Copy(s.shape).Shape();
    BRepMesh_IncrementalMesh mesher(shape, deflection, false, 0.5, true);
    mesher.Perform();
    StlAPI_Writer writer;
    writer.ASCIIMode() = false;  // compact binary STL
    std::string p(path);
    if (!writer.Write(shape, p.c_str())) {
      throw std::runtime_error("write_stl: writer reported failure for " + p);
    }
    return 0;  // guard<F> needs a value; ignored by the void wrapper
  });
}

}  // namespace cdt
