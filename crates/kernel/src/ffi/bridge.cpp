#include "cdt-kernel/src/ffi/bridge.hpp"
// cxx-generated header: defines cdt::Mesh and the extern shims.
#include "cdt-kernel/src/ffi.rs.h"

#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepBuilderAPI_TransitionMode.hxx>
#include <BRepTools.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_BooleanOperation.hxx>
#include <BRepTools_History.hxx>
#include <TopTools_ListOfShape.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <BRepBuilderAPI_MakeShape.hxx>
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
#include <BRepCheck_Analyzer.hxx>
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
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
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

namespace {

// Push one shape through the unify pass's history: its merged images, itself
// if untouched, nothing if removed.
void through_unify(const Handle(BRepTools_History)& uhist, const TopoDS_Shape& m,
                   TopTools_ListOfShape& fin) {
  int mapped = 0;
  if (!uhist.IsNull()) {
    const TopTools_ListOfShape& umods = uhist->Modified(m);
    for (TopTools_ListOfShape::Iterator jt(umods); jt.More(); jt.Next()) {
      fin.Append(jt.Value());
      ++mapped;
    }
  }
  if (mapped == 0 && (uhist.IsNull() || !uhist->IsRemoved(m))) {
    fin.Append(m);
  }
}

// Track one element kind (faces or edges) of each input through a modifying
// operation (+ optional unify), appending flat
// [which_input, input_index, output_index|-1] triples. Two stages, two
// histories: the maker reports splits via Modified() and consumption via
// IsDeleted(); the seam-removing unify pass (when run) reports its merges via
// a BRepTools_History. An input element's final images are its maker images
// (itself, if untouched) pushed through the unify history. Coverage of this
// composition is locked by tests/history_spike.rs.
void track_elements(BRepBuilderAPI_MakeShape& bop,
                    const Handle(BRepTools_History)& uhist,
                    std::initializer_list<const TopoDS_Shape*> inputs,
                    const TopoDS_Shape& result, TopAbs_ShapeEnum elem,
                    rust::Vec<int32_t>& records) {
  TopTools_IndexedMapOfShape out_map;
  TopExp::MapShapes(result, elem, out_map);

  int which = -1;
  for (const TopoDS_Shape* input_ptr : inputs) {
    ++which;
    const TopoDS_Shape& input = *input_ptr;
    TopTools_IndexedMapOfShape in_map;
    TopExp::MapShapes(input, elem, in_map);

    for (int i = 1; i <= in_map.Extent(); ++i) {
      const TopoDS_Shape& e = in_map(i);

      // Stage 1 — through the boolean: splits, or the element itself.
      TopTools_ListOfShape mid;
      const TopTools_ListOfShape& mods = bop.Modified(e);
      if (!mods.IsEmpty()) {
        mid = mods;
      } else if (!bop.IsDeleted(e)) {
        mid.Append(e);
      }

      // Stage 2 — each image through the unify pass.
      TopTools_ListOfShape fin;
      for (TopTools_ListOfShape::Iterator it(mid); it.More(); it.Next()) {
        through_unify(uhist, it.Value(), fin);
      }

      // Emit records, deduping images (several splits can unify back into
      // one output face; both operands can map to the same fused face).
      std::set<int32_t> outs;
      for (TopTools_ListOfShape::Iterator it(fin); it.More(); it.Next()) {
        const int oi = out_map.FindIndex(it.Value());
        if (oi > 0) outs.insert(static_cast<int32_t>(oi - 1));
      }
      if (outs.empty()) {
        records.push_back(static_cast<int32_t>(which));
        records.push_back(static_cast<int32_t>(i - 1));
        records.push_back(-1);
      } else {
        for (const int32_t oi : outs) {
          records.push_back(static_cast<int32_t>(which));
          records.push_back(static_cast<int32_t>(i - 1));
          records.push_back(oi);
        }
      }
    }
  }
}

// The NEW edges a boolean creates where faces intersect (e.g. the rim circles
// where a bore meets a box top): for each input FACE, the edges the boolean
// GENERATED from it, pushed through unify. Appends flat
// [which_input, input_face_index, output_edge_index] triples — a section edge
// typically appears once per parent face (one from each operand).
void track_sections(BRepAlgoAPI_BooleanOperation& bop,
                    const Handle(BRepTools_History)& uhist,
                    const TopoDS_Shape& a, const TopoDS_Shape& b,
                    const TopoDS_Shape& result,
                    rust::Vec<int32_t>& records) {
  TopTools_IndexedMapOfShape out_edges;
  TopExp::MapShapes(result, TopAbs_EDGE, out_edges);

  for (int which = 0; which < 2; ++which) {
    const TopoDS_Shape& input = which == 0 ? a : b;
    TopTools_IndexedMapOfShape in_faces;
    TopExp::MapShapes(input, TopAbs_FACE, in_faces);

    for (int i = 1; i <= in_faces.Extent(); ++i) {
      const TopTools_ListOfShape& gen = bop.Generated(in_faces(i));
      TopTools_ListOfShape fin;
      for (TopTools_ListOfShape::Iterator it(gen); it.More(); it.Next()) {
        if (it.Value().ShapeType() != TopAbs_EDGE) continue;
        through_unify(uhist, it.Value(), fin);
      }
      std::set<int32_t> outs;
      for (TopTools_ListOfShape::Iterator it(fin); it.More(); it.Next()) {
        const int oi = out_edges.FindIndex(it.Value());
        if (oi > 0) outs.insert(static_cast<int32_t>(oi - 1));
      }
      for (const int32_t oi : outs) {
        records.push_back(static_cast<int32_t>(which));
        records.push_back(static_cast<int32_t>(i - 1));
        records.push_back(oi);
      }
    }
  }
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

// An OPEN planar wire (a sweep path) from a polyline that may include cubic
// beziers — the same plane mapping and segment encoding as profile_face, but the
// loop is NOT closed and no face is built. `n` points yield `n-1` segments;
// `segs` has 5 doubles per segment `[is_bezier, c1u, c1v, c2u, c2v]`.
std::unique_ptr<Shape> path_wire(double ox, double oy, double oz,
                                 double xx, double xy, double xz,
                                 double yx, double yy, double yz,
                                 rust::Slice<const double> points,
                                 rust::Slice<const double> segs) {
  return guard("path_wire", [&] {
    const std::size_t n = points.size() / 2;
    if (n < 2 || segs.size() != (n - 1) * 5) {
      throw std::runtime_error("path_wire: bad point/segment counts");
    }
    const gp_Vec o(ox, oy, oz), x(xx, xy, xz), y(yx, yy, yz);
    auto to3d = [&](double u, double v) {
      const gp_Vec p = o + x * u + y * v;
      return gp_Pnt(p.X(), p.Y(), p.Z());
    };

    BRepBuilderAPI_MakeWire wire;
    for (std::size_t i = 0; i + 1 < n; ++i) {
      const gp_Pnt a = to3d(points[2 * i], points[2 * i + 1]);
      const gp_Pnt b = to3d(points[2 * (i + 1)], points[2 * (i + 1) + 1]);
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
    return std::make_unique<Shape>(wire.Wire());
  });
}

// --- Extrude ----------------------------------------------------------------

namespace {

// Report a generative op's lineage (extrude/revolve — anything exposing
// per-subshape FirstShape/LastShape and Generated). Flat [tag, in, out]
// triples: 0 = profile FACE's start image (the profile-side cap), 1 = its end
// image (the far cap), 2 = side FACE swept from profile edge `in`, 3/4 =
// profile edge `in`'s start/end rim image. Lateral edges (swept from
// vertices, which element maps don't track) are named later by face
// adjacency; anything the maker doesn't report simply stays unnamed.
template <typename Maker>
void track_generated(Maker& mk, const TopoDS_Shape& profile,
                     const TopoDS_Shape& result, rust::Vec<int32_t>& records) {
  TopTools_IndexedMapOfShape out_faces, out_edges;
  TopExp::MapShapes(result, TopAbs_FACE, out_faces);
  TopExp::MapShapes(result, TopAbs_EDGE, out_edges);
  auto push = [&](int32_t tag, int32_t in_idx, const TopoDS_Shape& out,
                  const TopTools_IndexedMapOfShape& map) {
    if (out.IsNull()) return;
    const int oi = map.FindIndex(out);
    if (oi > 0) {
      records.push_back(tag);
      records.push_back(in_idx);
      records.push_back(static_cast<int32_t>(oi - 1));
    }
  };

  TopExp_Explorer fx(profile, TopAbs_FACE);
  if (fx.More()) {
    push(0, 0, mk.FirstShape(fx.Current()), out_faces);
    push(1, 0, mk.LastShape(fx.Current()), out_faces);
  }
  TopTools_IndexedMapOfShape in_edges;
  TopExp::MapShapes(profile, TopAbs_EDGE, in_edges);
  for (int i = 1; i <= in_edges.Extent(); ++i) {
    const TopoDS_Shape& e = in_edges(i);
    const TopTools_ListOfShape& gen = mk.Generated(e);
    for (TopTools_ListOfShape::Iterator it(gen); it.More(); it.Next()) {
      if (it.Value().ShapeType() == TopAbs_FACE) {
        push(2, i - 1, it.Value(), out_faces);
      }
    }
    push(3, i - 1, mk.FirstShape(e), out_edges);
    push(4, i - 1, mk.LastShape(e), out_edges);
  }
}

}  // namespace

std::unique_ptr<Shape> extrude_tracked(const Shape& s, double distance,
                                       rust::Vec<int32_t>& records) {
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
    const TopoDS_Shape result = prism.Shape();
    track_generated(prism, s.shape, result, records);
    return std::make_unique<Shape>(result);
  });
}

std::unique_ptr<Shape> extrude(const Shape& s, double distance) {
  rust::Vec<int32_t> ignored;
  return extrude_tracked(s, distance, ignored);
}

// --- Revolve ----------------------------------------------------------------

// Revolve the profile `s` (a planar face — from a sketch or a model face)
// through `angle` radians about the straight edge nearest the anchor
// `(ax,ay,az)`. The axis edge is one of the profile's own segments (a sketch
// centerline) or an adjacent model edge; either way it resolves the same way as
// a fillet anchor. A full revolution is `angle = 2π`.
std::unique_ptr<Shape> revolve_tracked(const Shape& s, double ax, double ay,
                                       double az, double angle,
                                       rust::Vec<int32_t>& records) {
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
    BRepPrimAPI_MakeRevol revol(s.shape, axis, angle);
    const TopoDS_Shape solid = revol.Shape();
    track_generated(revol, s.shape, solid, records);
    return std::make_unique<Shape>(solid);
  });
}

std::unique_ptr<Shape> revolve(const Shape& s, double ax, double ay, double az,
                               double angle) {
  rust::Vec<int32_t> ignored;
  return revolve_tracked(s, ax, ay, az, angle, ignored);
}

// --- Sweep ------------------------------------------------------------------

// Sweep the planar `profile` face along the `spine` wire into a solid, with the
// section's roll locked to the constant `binormal` (the path plane's normal).
std::unique_ptr<Shape> sweep_tracked(const Shape& profile, const Shape& spine,
                                     double bnx, double bny, double bnz,
                                     rust::Vec<int32_t>& records) {
  return guard("sweep", [&] {
    TopoDS_Wire spine_wire;
    if (spine.shape.ShapeType() == TopAbs_WIRE) {
      spine_wire = TopoDS::Wire(spine.shape);
    } else {
      // Assemble any loose edges of the spine shape into one wire.
      BRepBuilderAPI_MakeWire mw;
      for (TopExp_Explorer ex(spine.shape, TopAbs_EDGE); ex.More(); ex.Next()) {
        mw.Add(TopoDS::Edge(ex.Current()));
      }
      if (!mw.IsDone()) {
        throw std::runtime_error("sweep: spine is not a usable wire");
      }
      spine_wire = mw.Wire();
    }
    // The profile to sweep is the face's outer boundary wire.
    TopoDS_Wire profile_wire;
    if (profile.shape.ShapeType() == TopAbs_FACE) {
      profile_wire = BRepTools::OuterWire(TopoDS::Face(profile.shape));
    } else if (profile.shape.ShapeType() == TopAbs_WIRE) {
      profile_wire = TopoDS::Wire(profile.shape);
    } else {
      throw std::runtime_error("sweep: profile is not a face or wire");
    }

    // MakePipeShell with a CONSTANT-BINORMAL trihedron (SetMode(dir)): the
    // section stays normal to the path AND its roll is locked to a fixed
    // direction — the path plane's normal. A Frenet/discrete frame instead picks
    // the roll from curvature, which twists or flips the section unpredictably
    // (an asymmetric profile lands on the wrong side of a bend). With the
    // binormal fixed, one section axis always points out of the path plane and
    // the other stays radial, so the profile's orientation is preserved through
    // every bend. For a planar path the plane normal is always perpendicular to
    // the tangent, so it's a valid binormal everywhere.
    BRepOffsetAPI_MakePipeShell shell(spine_wire);
    shell.SetMode(gp_Dir(bnx, bny, bnz));
    // Sharp polyline corners need an explicit transition (the default transforms
    // the section straight through). RoundCorner reorients onto each next segment
    // and, unlike RightCorner, also works on smooth (bezier) spines.
    shell.SetTransitionMode(BRepBuilderAPI_RoundCorner);
    shell.Add(profile_wire, /*WithContact=*/Standard_False,
              /*WithCorrection=*/Standard_False);
    shell.Build();
    if (!shell.IsDone()) {
      throw std::runtime_error("sweep: MakePipeShell failed");
    }
    if (!shell.MakeSolid()) {
      throw std::runtime_error("sweep: could not cap the swept shell into a solid");
    }
    TopoDS_Shape result = shell.Shape();
    // PipeShell output — especially RoundCorner transitions over bezier
    // profiles — is routinely slightly invalid (tolerance gaps, bad pcurves).
    // It renders and even measures fine, but booleans silently classify an
    // invalid solid as nothing (empty cut, fuse that drops it). Heal it here
    // so downstream ops see a genuinely valid solid.
    bool healed = false;
    if (!BRepCheck_Analyzer(result).IsValid()) {
      ShapeFix_Shape fixer(result);
      fixer.Perform();
      result = fixer.Shape();
      healed = true;
      // A sweep that stays invalid after healing (typically a bend tighter
      // than the profile is wide, self-intersecting at the elbow) must be an
      // error, not a silently broken body: it renders fine but corrupts every
      // downstream boolean and STL.
      if (!BRepCheck_Analyzer(result).IsValid()) {
        throw std::runtime_error(
            "sweep: the swept solid self-intersects (the bend is too tight "
            "for this profile size) — widen the bend or shrink the profile");
      }
    }
    // Lineage — only when the shape was NOT healed: ShapeFix rebuilds
    // sub-shapes without history, so records against a healed result would
    // point at stale topology. An unhealed sweep reports side faces via
    // Generated(profile edge); PipeShell has no per-subshape FirstShape/
    // LastShape, so caps get lineage only if the no-arg accessors surface a
    // result face (rim edges stay unnamed — honest gaps).
    if (!healed) {
      TopTools_IndexedMapOfShape out_faces;
      TopExp::MapShapes(result, TopAbs_FACE, out_faces);
      auto push_face = [&](int32_t tag, int32_t in_idx, const TopoDS_Shape& out) {
        if (out.IsNull()) return;
        const int oi = out_faces.FindIndex(out);
        if (oi > 0) {
          records.push_back(tag);
          records.push_back(in_idx);
          records.push_back(static_cast<int32_t>(oi - 1));
        }
      };
      push_face(0, 0, shell.FirstShape());
      push_face(1, 0, shell.LastShape());
      TopTools_IndexedMapOfShape in_edges;
      TopExp::MapShapes(profile.shape, TopAbs_EDGE, in_edges);
      for (int i = 1; i <= in_edges.Extent(); ++i) {
        const TopTools_ListOfShape& gen = shell.Generated(in_edges(i));
        for (TopTools_ListOfShape::Iterator it(gen); it.More(); it.Next()) {
          if (it.Value().ShapeType() == TopAbs_FACE) {
            push_face(2, i - 1, it.Value());
          }
        }
      }
    }
    return std::make_unique<Shape>(result);
  });
}

std::unique_ptr<Shape> sweep(const Shape& profile, const Shape& spine,
                             double bnx, double bny, double bnz) {
  rust::Vec<int32_t> ignored;
  return sweep_tracked(profile, spine, bnx, bny, bnz, ignored);
}

// Which faces bound each edge — flat (edge_index, face_index) pairs, TopExp
// map order. Names the lateral edges of a generative op (an edge swept from
// an untracked vertex is identified by the two named faces it separates).
rust::Vec<uint32_t> edge_face_adjacency(const Shape& s) {
  return guard("edge_face_adjacency", [&] {
    TopTools_IndexedMapOfShape edges, faces;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);
    TopExp::MapShapes(s.shape, TopAbs_FACE, faces);
    TopTools_IndexedDataMapOfShapeListOfShape ancestors;
    TopExp::MapShapesAndAncestors(s.shape, TopAbs_EDGE, TopAbs_FACE, ancestors);

    rust::Vec<uint32_t> out;
    for (int i = 1; i <= edges.Extent(); ++i) {
      const TopTools_ListOfShape* bound = ancestors.Seek(edges(i));
      if (!bound) continue;
      std::set<uint32_t> fs;
      for (TopTools_ListOfShape::Iterator it(*bound); it.More(); it.Next()) {
        const int fi = faces.FindIndex(it.Value());
        if (fi > 0) fs.insert(static_cast<uint32_t>(fi - 1));
      }
      for (const uint32_t f : fs) {
        out.push_back(static_cast<uint32_t>(i - 1));
        out.push_back(f);
      }
    }
    return out;
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

// Push/pull with lineage. `meta` stays EMPTY for the zero-distance no-op
// (identity — the caller keeps the input's map); otherwise it receives
// [pushed_face_index (in `s`), prism_far_cap_face_index (in the prism),
// prism_face_count, prism_edge_count], and the record vecs receive the same
// boolean + unify lineage as boolean_tracked, with operand 0 = `s` and
// operand 1 = the offset prism. The far cap matters because it becomes the
// MOVED face: the caller names it verbatim as the pushed face, so the face's
// identity survives the push.
std::unique_ptr<Shape> push_pull_tracked(const Shape& s, double px, double py,
                                         double pz, double nx, double ny,
                                         double nz, double distance,
                                         rust::Vec<int32_t>& meta,
                                         rust::Vec<int32_t>& face_records,
                                         rust::Vec<int32_t>& edge_records,
                                         rust::Vec<int32_t>& section_records) {
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
      return std::make_unique<Shape>(s.shape);  // no-op; meta stays empty
    }
    BRepPrimAPI_MakePrism prism_mk(target, direction);
    const TopoDS_Shape prism = prism_mk.Shape();

    std::unique_ptr<BRepAlgoAPI_BooleanOperation> bop;
    if (distance >= 0.0) {
      bop = std::make_unique<BRepAlgoAPI_Fuse>(s.shape, prism);
    } else {
      bop = std::make_unique<BRepAlgoAPI_Cut>(s.shape, prism);
    }
    if (!bop->IsDone()) {
      throw std::runtime_error("push_pull: the boolean failed");
    }
    // Pulling a face flush with a neighbour leaves a coplanar seam — merge it.
    ShapeUpgrade_UnifySameDomain usd(bop->Shape(), /*unifyEdges=*/true,
                                     /*unifyFaces=*/true,
                                     /*concatBSplines=*/false);
    usd.Build();
    TopoDS_Shape result = usd.Shape();
    const Handle(BRepTools_History) uhist = usd.History();

    // A pull whose prism runs tangent along existing walls (e.g. raising a
    // pocket floor beside its own fillets) can make the fuse emit a
    // self-intersecting body: IsDone, renders, even measures — but corrupts
    // every downstream boolean and STL. Same heal-or-error policy as the
    // swept-solid and fillet paths: recover tolerance-level defects, REQUIRE
    // a valid closed solid, fail loudly otherwise.
    bool healed = false;
    if (!BRepCheck_Analyzer(result).IsValid()) {
      ShapeFix_Shape fixer(result);
      fixer.Perform();
      result = fixer.Shape();
      healed = true;
      bool has_solid = false;
      for (TopExp_Explorer ex(result, TopAbs_SOLID); ex.More(); ex.Next()) {
        has_solid = true;
        break;
      }
      if (!has_solid || !BRepCheck_Analyzer(result).IsValid()) {
        throw std::runtime_error(
            "push_pull: the result self-intersects (the moved face collides "
            "with nearby walls or fillets) — reduce the distance");
      }
    }

    // Lineage: where the pushed face sits in `s`, where its offset image (the
    // prism's far cap) sits in the prism, and how everything flowed through
    // the boolean + unify.
    TopTools_IndexedMapOfShape s_faces, prism_faces, prism_edges;
    TopExp::MapShapes(s.shape, TopAbs_FACE, s_faces);
    TopExp::MapShapes(prism, TopAbs_FACE, prism_faces);
    TopExp::MapShapes(prism, TopAbs_EDGE, prism_edges);
    const int pushed = s_faces.FindIndex(target);
    const int far_cap = prism_faces.FindIndex(prism_mk.LastShape());
    if (!healed && pushed > 0 && far_cap > 0) {
      meta.push_back(static_cast<int32_t>(pushed - 1));
      meta.push_back(static_cast<int32_t>(far_cap - 1));
      meta.push_back(static_cast<int32_t>(prism_faces.Extent()));
      meta.push_back(static_cast<int32_t>(prism_edges.Extent()));
      track_elements(*bop, uhist, {&s.shape, &prism}, result, TopAbs_FACE,
                     face_records);
      track_elements(*bop, uhist, {&s.shape, &prism}, result, TopAbs_EDGE,
                     edge_records);
      track_sections(*bop, uhist, s.shape, prism, result, section_records);
    } else {
      // A healed (rebuilt-without-history) or unmappable result: geometry is
      // good but lineage is not — signal "untracked", distinct from the
      // empty-meta zero-distance no-op (whose result IS the input).
      meta.push_back(-1);
    }
    return std::make_unique<Shape>(result);
  });
}

std::unique_ptr<Shape> push_pull(const Shape& s, double px, double py, double pz,
                                 double nx, double ny, double nz,
                                 double distance) {
  rust::Vec<int32_t> meta, f, e, sec;
  return push_pull_tracked(s, px, py, pz, nx, ny, nz, distance, meta, f, e, sec);
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

// The production boolean pipeline (BRepAlgoAPI_* then unify-same-domain, the
// same result as fuse/cut/common) with its evolution history kept instead of
// discarded — the raw material element maps are composed from. `op`: 0=fuse,
// 1=cut, 2=common.
std::unique_ptr<Shape> boolean_tracked(const Shape& a, const Shape& b, int32_t op,
                                       rust::Vec<int32_t>& face_records,
                                       rust::Vec<int32_t>& edge_records,
                                       rust::Vec<int32_t>& section_records) {
  return guard("boolean_tracked", [&] {
    std::unique_ptr<BRepAlgoAPI_BooleanOperation> bop;
    switch (op) {
      case 0: bop = std::make_unique<BRepAlgoAPI_Fuse>(a.shape, b.shape); break;
      case 1: bop = std::make_unique<BRepAlgoAPI_Cut>(a.shape, b.shape); break;
      case 2: bop = std::make_unique<BRepAlgoAPI_Common>(a.shape, b.shape); break;
      default: throw std::runtime_error("boolean_tracked: bad op code");
    }
    if (!bop->IsDone()) {
      throw std::runtime_error("boolean_tracked: boolean failed");
    }

    ShapeUpgrade_UnifySameDomain usd(bop->Shape(), /*unifyEdges=*/true,
                                     /*unifyFaces=*/true,
                                     /*concatBSplines=*/false);
    usd.Build();
    const TopoDS_Shape result = usd.Shape();
    const Handle(BRepTools_History) uhist = usd.History();

    track_elements(*bop, uhist, {&a.shape, &b.shape}, result, TopAbs_FACE, face_records);
    track_elements(*bop, uhist, {&a.shape, &b.shape}, result, TopAbs_EDGE, edge_records);
    track_sections(*bop, uhist, a.shape, b.shape, result, section_records);

    return std::make_unique<Shape>(result);
  });
}

// Centre of mass of the face/edge at `index` (TopExp map order) — the
// deterministic geometric key that ranks the pieces of a SPLIT element.
rust::Vec<double> elem_centroid(const Shape& s, int32_t kind, uint32_t index) {
  return guard("elem_centroid", [&] {
    TopTools_IndexedMapOfShape map;
    TopExp::MapShapes(s.shape, kind == 0 ? TopAbs_FACE : TopAbs_EDGE, map);
    const int i = static_cast<int>(index) + 1;
    if (i < 1 || i > map.Extent()) {
      throw std::runtime_error("elem_centroid: no element at index");
    }
    GProp_GProps props;
    if (kind == 0) {
      BRepGProp::SurfaceProperties(TopoDS::Face(map(i)), props);
    } else {
      BRepGProp::LinearProperties(TopoDS::Edge(map(i)), props);
    }
    const gp_Pnt c = props.CentreOfMass();
    rust::Vec<double> out;
    out.push_back(c.X());
    out.push_back(c.Y());
    out.push_back(c.Z());
    return out;
  });
}

std::unique_ptr<Shape> common(const Shape& a, const Shape& b) {
  return guard("common", [&] {
    return std::make_unique<Shape>(
        unify_same_domain(BRepAlgoAPI_Common(a.shape, b.shape).Shape()));
  });
}

// Standalone refine — merge same-domain faces/edges of any shape on demand.
// One-line structural summary — root type, sub-shape counts, closedness,
// validity — the first thing to look at when a boolean misbehaves: booleans
// only classify volumes for true closed SOLIDs, so a shape that renders and
// even measures fine (a closed SHELL) still contributes nothing to a cut/fuse.
rust::String describe(const Shape& s) {
  return guard("describe", [&]() -> rust::String {
    static const char* kinds[] = {"COMPOUND", "COMPSOLID", "SOLID",  "SHELL",
                                  "FACE",     "WIRE",      "EDGE",   "VERTEX",
                                  "SHAPE"};
    std::string out = kinds[s.shape.ShapeType()];
    int solids = 0, shells = 0, closed_shells = 0, faces = 0;
    for (TopExp_Explorer ex(s.shape, TopAbs_SOLID); ex.More(); ex.Next()) solids++;
    for (TopExp_Explorer ex(s.shape, TopAbs_SHELL); ex.More(); ex.Next()) {
      shells++;
      if (BRep_Tool::IsClosed(ex.Current())) closed_shells++;
    }
    for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next()) faces++;
    BRepCheck_Analyzer check(s.shape);
    out += ": solids=" + std::to_string(solids) +
           " shells=" + std::to_string(shells) +
           " (closed " + std::to_string(closed_shells) + ")" +
           " faces=" + std::to_string(faces) +
           (check.IsValid() ? " valid" : " INVALID");
    return rust::String(out);
  });
}

std::unique_ptr<Shape> unify(const Shape& s) {
  return guard("unify", [&] {
    return std::make_unique<Shape>(unify_same_domain(s.shape));
  });
}

// Number of faces in the shape (TopExp order) — for asserting topology.
std::size_t count_faces(const Shape& s) {
  return guard("count_faces", [&] {
    std::size_t n = 0;
    for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next()) n++;
    return n;
  });
}

// Number of UNIQUE edges (TopExp map order, matching tessellate's edge ids and
// edge_length) — a raw explorer would visit shared edges once per adjacent face.
std::size_t count_edges(const Shape& s) {
  return guard("count_edges", [&] {
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);
    return static_cast<std::size_t>(edges.Extent());
  });
}

// Enclosed volume of the solid (model units³) — for validating swept/built bodies.
double volume(const Shape& s) {
  return guard("volume", [&] {
    GProp_GProps props;
    BRepGProp::VolumeProperties(s.shape, props);
    return props.Mass();
  });
}

// Exact area (model units²) of the face at `index` (TopExp order, matching
// tessellate's face ids / face_plane) — for the measurement read-out.
double face_area(const Shape& s, uint32_t index) {
  return guard("face_area", [&] {
    uint32_t i = 0;
    for (TopExp_Explorer ex(s.shape, TopAbs_FACE); ex.More(); ex.Next(), ++i) {
      if (i != index) continue;
      GProp_GProps props;
      BRepGProp::SurfaceProperties(TopoDS::Face(ex.Current()), props);
      return props.Mass();
    }
    throw std::runtime_error("face_area: no face at index");
  });
}

// Exact length (model units) of the edge at `index` (unique-edge map order,
// matching tessellate's edge ids) — for the measurement read-out.
double edge_length(const Shape& s, uint32_t index) {
  return guard("edge_length", [&] {
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);
    // A plain int index: the map's 1-based operator() takes Standard_Integer,
    // and MSVC finds the call ambiguous for an unsigned argument (C3066).
    const int i = static_cast<int>(index) + 1;
    if (i < 1 || i > edges.Extent()) {
      throw std::runtime_error("edge_length: no edge at index");
    }
    GProp_GProps props;
    BRepGProp::LinearProperties(TopoDS::Edge(edges(i)), props);
    return props.Mass();
  });
}

// --- Edge treatments --------------------------------------------------------

namespace {

// The lineage of a fillet: how each input face/edge survived, plus which
// result faces each rounded edge GENERATED ([input_edge, output_face] pairs
// in `rounded_records`). No unify pass runs in the fillet pipeline, so the
// history is single-stage (a null unify handle).
void track_fillet(BRepFilletAPI_MakeFillet& mk, const TopoDS_Shape& input,
                  const TopoDS_Shape& result,
                  rust::Vec<int32_t>& face_records,
                  rust::Vec<int32_t>& edge_records,
                  rust::Vec<int32_t>& rounded_records) {
  const Handle(BRepTools_History) no_unify;
  track_elements(mk, no_unify, {&input}, result, TopAbs_FACE, face_records);
  track_elements(mk, no_unify, {&input}, result, TopAbs_EDGE, edge_records);

  TopTools_IndexedMapOfShape out_faces, in_edges;
  TopExp::MapShapes(result, TopAbs_FACE, out_faces);
  TopExp::MapShapes(input, TopAbs_EDGE, in_edges);
  for (int i = 1; i <= in_edges.Extent(); ++i) {
    const TopTools_ListOfShape& gen = mk.Generated(in_edges(i));
    for (TopTools_ListOfShape::Iterator it(gen); it.More(); it.Next()) {
      if (it.Value().ShapeType() != TopAbs_FACE) continue;
      const int oi = out_faces.FindIndex(it.Value());
      if (oi > 0) {
        rounded_records.push_back(static_cast<int32_t>(i - 1));
        rounded_records.push_back(static_cast<int32_t>(oi - 1));
      }
    }
  }
}

}  // namespace

std::unique_ptr<Shape> fillet_all_tracked(const Shape& s, double radius,
                                          rust::Vec<int32_t>& face_records,
                                          rust::Vec<int32_t>& edge_records,
                                          rust::Vec<int32_t>& rounded_records) {
  return guard("fillet_all_edges", [&] {
    BRepFilletAPI_MakeFillet mk(s.shape);
    // Use a uniqued edge map: a raw explorer visits shared edges once per
    // adjacent face, which would add each edge to the fillet twice.
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(s.shape, TopAbs_EDGE, edges);
    for (int i = 1; i <= edges.Extent(); ++i) {
      mk.Add(radius, TopoDS::Edge(edges(i)));
    }
    const TopoDS_Shape result = mk.Shape();
    track_fillet(mk, s.shape, result, face_records, edge_records, rounded_records);
    return std::make_unique<Shape>(result);
  });
}

std::unique_ptr<Shape> fillet_all_edges(const Shape& s, double radius) {
  rust::Vec<int32_t> f, e, r;
  return fillet_all_tracked(s, radius, f, e, r);
}

std::unique_ptr<Shape> fillet_edges_tracked(const Shape& s,
                                            rust::Slice<const double> coords,
                                            double radius,
                                            rust::Vec<int32_t>& face_records,
                                            rust::Vec<int32_t>& edge_records,
                                            rust::Vec<int32_t>& rounded_records) {
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
    mk.Build();
    if (!mk.IsDone()) {
      throw std::runtime_error("fillet: the rounding failed to build");
    }
    TopoDS_Shape result = mk.Shape();
    // When a fillet radius is too large for the face between two rounded edges,
    // the rolling surfaces overlap and OCCT leaves a self-intersecting sliver:
    // MakeFillet reports IsDone, but the solid is INVALID. It renders yet
    // silently corrupts every downstream boolean and STL. Try to heal it (the
    // same policy as the swept-solid path) — minor tolerance/pcurve defects
    // recover into a clean solid.
    bool healed = false;
    if (!BRepCheck_Analyzer(result).IsValid()) {
      ShapeFix_Shape fixer(result);
      fixer.Perform();
      result = fixer.Shape();
      healed = true;
    }
    // A genuine radius-too-large overlap can't be made watertight: ShapeFix
    // strips the tangled faces and leaves an OPEN shell (valid per BRepCheck,
    // but no solid and not closed). Require a real closed solid, else fail
    // loudly with an actionable message rather than emit an unprintable body.
    bool has_solid = false;
    for (TopExp_Explorer ex(result, TopAbs_SOLID); ex.More(); ex.Next()) {
      has_solid = true;
      break;
    }
    if (!has_solid || !BRepCheck_Analyzer(result).IsValid()) {
      throw std::runtime_error(
          "fillet: the rounded edges overlap (the radius is too large for the "
          "face between them) — reduce the fillet radius");
    }
    // Lineage — only when the shape was NOT healed: ShapeFix rebuilds
    // sub-shapes without history, so records against a healed result would
    // point at stale topology (its elements stay unnamed, anchor fallback).
    if (!healed) {
      track_fillet(mk, s.shape, result, face_records, edge_records, rounded_records);
    }
    return std::make_unique<Shape>(result);
  });
}

std::unique_ptr<Shape> fillet_edges(const Shape& s, rust::Slice<const double> coords,
                                    double radius) {
  rust::Vec<int32_t> f, e, r;
  return fillet_edges_tracked(s, coords, radius, f, e, r);
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
