//! The query surface an agent (or a CLI) verifies a model with, no pixels
//! involved: exact bounds and mass properties, a typed topology summary, face
//! and edge listings with types and normals, and body-to-body distance.

use cdt_kernel::{CurveKind, Solid, SurfaceKind};
use std::f64::consts::PI;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn bbox_and_mass_properties_of_a_box_are_exact() {
    let b = Solid::cuboid(10.0, 20.0, 30.0).unwrap();
    let (min, max) = b.bbox().unwrap();
    assert!(min.iter().all(|v| close(*v, 0.0)), "{min:?}");
    assert!(close(max[0], 10.0) && close(max[1], 20.0) && close(max[2], 30.0), "{max:?}");
    let m = b.mass_properties().unwrap();
    assert!(close(m.volume, 6000.0));
    assert!(close(m.surface_area, 2.0 * (200.0 + 300.0 + 600.0)));
    assert!(close(m.centroid[0], 5.0) && close(m.centroid[1], 10.0) && close(m.centroid[2], 15.0));
    // Inertia about the centre of mass: Ixx = V·(b²+c²)/12 for a box, and no
    // products of inertia for an axis-aligned one.
    assert!(close(m.inertia[0][0], 6000.0 * (400.0 + 900.0) / 12.0), "{:?}", m.inertia);
    assert!(close(m.inertia[0][1], 0.0) && close(m.inertia[1][2], 0.0));
    assert_eq!(m.bbox, (min, max));
}

#[test]
fn a_curved_body_gets_tight_bounds_not_padded_ones() {
    let c = Solid::cylinder(10.0, 20.0).unwrap();
    let (min, max) = c.bbox().unwrap();
    assert!(close(min[0], -10.0) && close(max[0], 10.0), "x {min:?}..{max:?}");
    assert!(close(min[2], 0.0) && close(max[2], 20.0));
    let s = Solid::sphere(5.0).unwrap();
    let (min, max) = s.bbox().unwrap();
    assert!(close(min[1], -5.0) && close(max[1], 5.0), "{min:?}..{max:?}");
}

#[test]
fn topology_is_typed_and_describe_is_its_display() {
    let b = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let t = b.topology().unwrap();
    assert_eq!(t.kind, "SOLID");
    assert_eq!((t.solids, t.shells, t.closed_shells), (1, 1, 1));
    assert_eq!((t.faces, t.edges, t.vertices), (6, 12, 8));
    assert!(t.is_valid);
    assert!(b.is_valid().unwrap());
    let desc = b.describe().unwrap();
    assert_eq!(desc, "SOLID: solids=1 shells=1 (closed 1) faces=6 valid");
}

#[test]
fn faces_carry_type_area_normal_and_edges() {
    let b = Solid::cuboid(10.0, 20.0, 30.0).unwrap();
    let faces = b.faces().unwrap();
    assert_eq!(faces.len(), 6);
    for (i, f) in faces.iter().enumerate() {
        assert_eq!(f.index as usize, i, "indexed in tessellation order");
        assert_eq!(f.surface, SurfaceKind::Plane);
        assert!(close(f.area, b.face_area(f.index).unwrap()), "matches face_area");
        assert_eq!(f.edges.len(), 4);
        assert!(f.radius.is_none());
        // The normal points away from the box centre.
        let out = [f.centroid[0] - 5.0, f.centroid[1] - 10.0, f.centroid[2] - 15.0];
        let dot = out[0] * f.normal[0] + out[1] * f.normal[1] + out[2] * f.normal[2];
        assert!(dot > 0.0, "face {i}: normal {:?} at {:?} points inward", f.normal, f.centroid);
        assert!(close(f.normal.iter().map(|v| v * v).sum::<f64>(), 1.0));
    }
    // The top face: +Z normal, centre at z=30.
    let top = faces.iter().find(|f| f.normal[2] > 0.99).expect("a +Z face");
    assert!(close(top.centroid[2], 30.0) && close(top.area, 200.0));

    let c = Solid::cylinder(10.0, 20.0).unwrap();
    let faces = c.faces().unwrap();
    let lateral = faces.iter().find(|f| f.surface == SurfaceKind::Cylinder).expect("lateral");
    assert_eq!(lateral.radius, Some(10.0));
    assert!(close(lateral.area, 2.0 * PI * 10.0 * 20.0));
    assert_eq!(faces.iter().filter(|f| f.surface == SurfaceKind::Plane).count(), 2);
}

#[test]
fn edges_carry_type_endpoints_anchor_and_faces() {
    let b = Solid::cuboid(10.0, 20.0, 30.0).unwrap();
    let edges = b.edges().unwrap();
    assert_eq!(edges.len(), 12);
    for e in &edges {
        assert_eq!(e.curve, CurveKind::Line);
        assert!(close(e.length, b.edge_length(e.index).unwrap()), "matches edge_length");
        assert_eq!(e.faces.len(), 2, "every box edge joins two faces");
        let chord: f64 = (0..3).map(|k| (e.end[k] - e.start[k]).powi(2)).sum::<f64>().sqrt();
        assert!(close(chord, e.length));
        for k in 0..3 {
            assert!(close(e.midpoint[k], 0.5 * (e.start[k] + e.end[k])));
        }
        assert!(e.radius.is_none());
    }
    // Face/edge adjacency is consistent both ways.
    let faces = b.faces().unwrap();
    for e in &edges {
        for &fi in &e.faces {
            assert!(faces[fi as usize].edges.contains(&e.index), "face {fi} lists edge {}", e.index);
        }
    }

    let c = Solid::cylinder(10.0, 20.0).unwrap();
    let edges = c.edges().unwrap();
    let rims: Vec<_> = edges.iter().filter(|e| e.curve == CurveKind::Circle).collect();
    assert_eq!(rims.len(), 2, "{edges:?}");
    for rim in rims {
        assert_eq!(rim.radius, Some(10.0));
        assert!(close(rim.length, 2.0 * PI * 10.0));
        // The anchor lies ON the circle, not at the chord midpoint (the seam
        // edge's start and end coincide, so a chord midpoint would be a
        // vertex — or for a half-circle, the axis).
        let r = rim.midpoint[0].hypot(rim.midpoint[1]);
        assert!(close(r, 10.0), "anchor {:?} is not on the rim", rim.midpoint);
    }
}

#[test]
fn distance_between_bodies() {
    let a = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let far = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(25.0, 0.0, 0.0).unwrap();
    assert!(close(a.distance_to(&far).unwrap(), 15.0));
    let touching = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(10.0, 0.0, 0.0).unwrap();
    assert!(close(a.distance_to(&touching).unwrap(), 0.0));
    let overlapping = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(5.0, 5.0, 5.0).unwrap();
    assert!(close(a.distance_to(&overlapping).unwrap(), 0.0));
}

#[test]
fn face_numbering_is_the_same_everywhere() {
    // Explorer-order faces (tessellation, face_area, faces()) and the
    // indexed-map order (element maps, elem_centroid) are assumed equal; pin
    // it, for a body with a shared seam edge and a compound of two bodies.
    let c = Solid::cylinder(10.0, 20.0).unwrap();
    let two = Solid::cuboid(5.0, 5.0, 5.0).unwrap().compound(&c).unwrap();
    for body in [&c, &two] {
        let faces = body.faces().unwrap();
        let mesh = body.tessellate(0.1).unwrap();
        let max_face_id = mesh.face_ids.iter().copied().max().unwrap() as usize;
        assert_eq!(max_face_id + 1, faces.len());
        for f in &faces {
            let c = body.elem_centroid(cdt_kernel::ElemKind::Face, f.index).unwrap();
            for k in 0..3 {
                assert!(close(c[k], f.centroid[k]), "face {} centroid differs by order", f.index);
            }
        }
    }
}
