//! STEP in and out: what a vendor part becomes once it is read, and that
//! Cadette's own bodies survive a round trip.

use cdt_core::{DVec3, FeatureId, GeometryBackend, ImportFormat};
use cdt_kernel::{ElemKind, ElementName, KernelBackend, Solid};

fn step_bytes(solid: &Solid) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("cdt-step-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{:p}.step", solid));
    solid.write_step(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    bytes
}

fn bored_block() -> Solid {
    let block = Solid::cuboid(40.0, 30.0, 20.0).unwrap();
    let bore = Solid::cylinder(5.0, 20.0).unwrap().translate(20.0, 15.0, 0.0).unwrap();
    block.cut(&bore).unwrap()
}

#[test]
fn a_body_round_trips_through_step_with_its_exact_geometry() {
    let block = bored_block();
    let bytes = step_bytes(&block);
    assert!(bytes.starts_with(b"ISO-10303-21;"), "a STEP file starts with its header");

    let back = Solid::from_step(&bytes).unwrap();
    let want = block.mass_properties().unwrap();
    let got = back.mass_properties().unwrap();
    assert!((got.volume - want.volume).abs() < 1e-6, "volume {} vs {}", got.volume, want.volume);
    assert!((got.surface_area - want.surface_area).abs() < 1e-6);
    // The bore is still a cylinder, not a mesh: six planar faces plus one
    // cylindrical wall, exactly as built.
    assert_eq!(back.face_count().unwrap(), 7);
    assert_eq!(back.topology().unwrap().solids, 1);
    assert!(back.is_valid().unwrap());
}

#[test]
fn several_solids_arrive_as_one_compound_body() {
    let a = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let b = Solid::sphere(4.0).unwrap().translate(30.0, 0.0, 0.0).unwrap();
    let bytes = step_bytes(&a.compound(&b).unwrap());
    let back = Solid::from_step(&bytes).unwrap();
    let topo = back.topology().unwrap();
    assert_eq!(topo.solids, 2);
    let got = back.mass_properties().unwrap().volume;
    let want = 1000.0 + 4.0 / 3.0 * std::f64::consts::PI * 64.0;
    assert!((got - want).abs() < 1e-3, "{got} vs {want}");
}

#[test]
fn an_imported_body_gets_born_names_and_takes_a_fillet() {
    let bytes = step_bytes(&bored_block());
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(7));
    let body = backend.import(ImportFormat::Step, &bytes, 1.0).unwrap();
    assert_eq!(body.face_count().unwrap(), 7);
    assert!(matches!(
        body.elements.face(6),
        Some(ElementName::Born { origin: 7, kind: ElemKind::Face, index: 6 })
    ));
    // A fillet on an imported edge works like one on a box edge: anchor the
    // bore's top rim.
    let rounded = backend.fillet_all(&body, 1.0).unwrap();
    assert!(rounded.mass_properties().unwrap().volume < body.mass_properties().unwrap().volume);
}

#[test]
fn import_scales_into_the_document_unit() {
    let bytes = step_bytes(&Solid::cuboid(25.4, 25.4, 25.4).unwrap());
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(1));
    let inch = backend.import(ImportFormat::Step, &bytes, 1.0 / 25.4).unwrap();
    let props = inch.mass_properties().unwrap();
    assert!((props.volume - 1.0).abs() < 1e-9, "a 1-inch cube in an inch document is 1 unit³");
    let _ = DVec3::ZERO;
}

#[test]
fn garbage_empty_truncated_and_damaged_files_are_refused_by_name() {
    let err = Solid::from_step(b"hello, not a step file").err().unwrap();
    assert!(err.0.contains("not a STEP file, or truncated"), "{}", err.0);
    let err = Solid::from_step(b"").err().unwrap();
    assert!(err.0.contains("empty"), "{}", err.0);

    // A file cut in half: the parser's own diagnosis comes through.
    let bytes = step_bytes(&bored_block());
    let err = Solid::from_step(&bytes[..bytes.len() / 2]).err().unwrap();
    assert!(err.0.contains("unexpected end of file"), "{}", err.0);

    // One entity made unparsable: OCCT would drop it and hand back a body
    // with a hole; the import refuses instead.
    let text = String::from_utf8(bytes.clone()).unwrap();
    let at = text.find("CARTESIAN_POINT(").unwrap();
    let damaged = format!("{}CARTESIAN_POINT(!!{}", &text[..at], &text[at + "CARTESIAN_POINT(".len()..]);
    let err = Solid::from_step(damaged.as_bytes()).err().unwrap();
    assert!(err.0.contains("syntax error"), "{}", err.0);
}

#[test]
fn a_uniform_scale_keeps_imported_surfaces_analytic() {
    let bytes = step_bytes(&bored_block());
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(1));
    let inch = backend.import(ImportFormat::Step, &bytes, 1.0 / 25.4).unwrap();
    let kinds: Vec<_> = inch.faces().unwrap().into_iter().map(|f| f.surface).collect();
    assert!(
        kinds.iter().any(|k| matches!(k, cdt_kernel::SurfaceKind::Cylinder)),
        "the bore stays a cylinder after scaling, not a B-spline: {kinds:?}"
    );
    assert!(!kinds.iter().any(|k| matches!(k, cdt_kernel::SurfaceKind::BSpline)), "{kinds:?}");
}

// --- STL -------------------------------------------------------------------

fn stl_bytes(solid: &Solid, deflection: f64) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("cdt-stl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{:p}.stl", solid));
    solid.write_stl(path.to_str().unwrap(), deflection).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    bytes
}

#[test]
fn a_binary_stl_becomes_a_faceted_solid_with_shared_edges() {
    let block = Solid::cuboid(40.0, 30.0, 20.0).unwrap();
    let bytes = stl_bytes(&block, 0.1);
    let back = Solid::from_stl(&bytes).unwrap();
    let props = back.mass_properties().unwrap();
    assert!((props.volume - 24000.0).abs() < 1e-6, "volume {}", props.volume);
    assert!((props.surface_area - 5200.0).abs() < 1e-6);
    let topo = back.topology().unwrap();
    assert_eq!(topo.solids, 1);
    assert_eq!(topo.faces, 12, "two triangles per side");
    // Edges are shared, not duplicated per triangle: a cube's 12 edges plus
    // one diagonal per side, as Euler demands for 8 vertices and 12 faces.
    assert_eq!(topo.edges, 18);
    assert!(back.is_valid().unwrap());
    // Drawn like a box, not a wireframe: the diagonals across each flat
    // side are not feature edges, the twelve creases are.
    let mesh = back.tessellate(0.1).unwrap();
    let drawn: std::collections::BTreeSet<u32> = mesh.edge_ids.iter().copied().collect();
    assert_eq!(drawn.len(), 12, "{drawn:?}");
    // Faces point outward: a cut by a smaller box removes exactly its volume.
    let bite = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let cut = back.cut(&bite).unwrap();
    assert!((cut.mass_properties().unwrap().volume - 23000.0).abs() < 1e-6);
}

#[test]
fn an_ascii_stl_with_inconsistent_winding_is_repaired() {
    // A tetrahedron, one face wound the wrong way, inconsistent normals.
    let stl = "solid tet
facet normal 0 0 -1
 outer loop
  vertex 0 0 0
  vertex 0 10 0
  vertex 10 0 0
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 0 0 0
  vertex 10 0 0
  vertex 0 0 10
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex 0 0 0
  vertex 0 10 0
  vertex 0 0 10
 endloop
endfacet
facet normal 1 1 1
 outer loop
  vertex 10 0 0
  vertex 0 10 0
  vertex 0 0 10
 endloop
endfacet
endsolid tet
";
    let tet = Solid::from_stl(stl.as_bytes()).unwrap();
    let props = tet.mass_properties().unwrap();
    assert!((props.volume - 1000.0 / 6.0).abs() < 1e-6, "volume {}", props.volume);
    assert_eq!(tet.topology().unwrap().faces, 4);
    assert!(tet.is_valid().unwrap());
}

#[test]
fn an_stl_keeps_cavities_and_separate_bodies() {
    // A hollow box: the inner shell comes back as a cavity, not a second body.
    let outer = Solid::cuboid(30.0, 30.0, 30.0).unwrap();
    let inner = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(10.0, 10.0, 10.0).unwrap();
    let hollow = outer.cut(&inner).unwrap();
    let back = Solid::from_stl(&stl_bytes(&hollow, 0.1)).unwrap();
    assert_eq!(back.topology().unwrap().solids, 1);
    assert!((back.mass_properties().unwrap().volume - 26000.0).abs() < 1e-6);

    // Two separate boxes: two solids in one body, as a STEP assembly.
    let a = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let b = Solid::cuboid(5.0, 5.0, 5.0).unwrap().translate(50.0, 0.0, 0.0).unwrap();
    let both = a.fuse(&b).unwrap();
    let back = Solid::from_stl(&stl_bytes(&both, 0.1)).unwrap();
    assert_eq!(back.topology().unwrap().solids, 2);
    assert!((back.mass_properties().unwrap().volume - 1125.0).abs() < 1e-6);

    // An inside-out mesh (every normal flipped) is turned right side out.
    let flipped = stl_bytes(&a, 0.1);
    let mut inside_out = flipped.clone();
    // Binary STL: 80-byte header, u32 count, then 50-byte records of
    // normal, v0, v1, v2, attr. Swap v1 and v2 of every record.
    for rec in inside_out[84..].chunks_exact_mut(50) {
        let (v1, v2) = rec[24..48].split_at_mut(12);
        v1.swap_with_slice(v2);
    }
    let back = Solid::from_stl(&inside_out).unwrap();
    assert!((back.mass_properties().unwrap().volume - 1000.0).abs() < 1e-6);
}

#[test]
fn concatenated_ascii_solids_and_nesting_are_read_by_geometry() {
    // Two `solid … endsolid` blocks: the second is not dropped.
    let a = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let b = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(40.0, 0.0, 0.0).unwrap();
    let ascii = |s: &Solid| {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cdt-ascii-{}-{:p}.stl", std::process::id(), s));
        // OCCT writes binary; make ASCII from the binary records.
        s.write_stl(path.to_str().unwrap(), 0.1).unwrap();
        let bin = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let mut text = String::from("solid part\r\n");
        for rec in bin[84..].chunks_exact(50) {
            let f = |at: usize| f32::from_le_bytes(rec[at..at + 4].try_into().unwrap());
            text += "facet normal 0 0 0\r\n outer loop\r\n";
            for v in 0..3 {
                text += &format!("  vertex {} {} {}\r\n", f(12 + v * 12), f(16 + v * 12), f(20 + v * 12));
            }
            text += " endloop\r\nendfacet\r\n";
        }
        text + "endsolid part\r\n"
    };
    let two = format!("{}\n\n{}", ascii(&a), ascii(&b));
    let back = Solid::from_stl(two.as_bytes()).unwrap();
    assert_eq!(back.topology().unwrap().solids, 2);
    assert!((back.mass_properties().unwrap().volume - 9000.0).abs() < 1e-6);

    // One body split over two blocks shares its vertices by position.
    let text = ascii(&a);
    let (head, tail) = text.split_at(text.len() / 2);
    let cut = head.rfind("facet normal").unwrap();
    let split = format!("{}endsolid x\nsolid y\n{}{}", &head[..cut], &head[cut..], tail);
    let back = Solid::from_stl(split.as_bytes()).unwrap();
    assert_eq!(back.topology().unwrap().solids, 1);
    assert!((back.mass_properties().unwrap().volume - 8000.0).abs() < 1e-6);

    // A hollow box exported mirrored (every shell inward) is still one
    // solid with a cavity, not a solid and a second body.
    let outer = Solid::cuboid(30.0, 30.0, 30.0).unwrap();
    let inner = Solid::cuboid(10.0, 10.0, 10.0).unwrap().translate(10.0, 10.0, 10.0).unwrap();
    let mut mirrored = stl_bytes(&outer.cut(&inner).unwrap(), 0.1);
    for rec in mirrored[84..].chunks_exact_mut(50) {
        let (v1, v2) = rec[24..48].split_at_mut(12);
        v1.swap_with_slice(v2);
    }
    let back = Solid::from_stl(&mirrored).unwrap();
    assert_eq!(back.topology().unwrap().solids, 1);
    assert!((back.mass_properties().unwrap().volume - 26000.0).abs() < 1e-6);
    assert!(back.is_valid().unwrap());

    // A hollow block sitting in the notch of an L-bracket whose bounding
    // box surrounds it: the cavity belongs to the block, not the bracket.
    let bracket = Solid::cuboid(100.0, 100.0, 20.0).unwrap()
        .cut(&Solid::cuboid(80.0, 80.0, 20.0).unwrap().translate(20.0, 20.0, 0.0).unwrap())
        .unwrap();
    let block = Solid::cuboid(40.0, 40.0, 20.0).unwrap()
        .cut(&Solid::cuboid(20.0, 20.0, 10.0).unwrap().translate(10.0, 10.0, 5.0).unwrap())
        .unwrap()
        .translate(50.0, 50.0, 0.0)
        .unwrap();
    let scene = stl_bytes(&bracket.fuse(&block).unwrap(), 0.1);
    let back = Solid::from_stl(&scene).unwrap();
    assert_eq!(back.topology().unwrap().solids, 2);
    assert!(back.is_valid().unwrap());
    let want = 100.0 * 100.0 * 20.0 - 80.0 * 80.0 * 20.0 + 40.0 * 40.0 * 20.0 - 20.0 * 20.0 * 10.0;
    assert!((back.mass_properties().unwrap().volume - want).abs() < 1e-6);
}

#[test]
fn a_small_binary_stl_of_round_numbers_is_still_binary() {
    // Zero normals and whole-number coordinates: every byte printable.
    let tet = [[0.0f32, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let mut bytes = vec![b' '; 80];
    bytes.extend_from_slice(&4u32.to_le_bytes());
    for f in faces {
        bytes.extend_from_slice(&[0u8; 12]);
        for v in f {
            for c in tet[v] {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&[0u8; 2]);
    }
    assert!(bytes.iter().all(|&b| b < 0x7f));
    let back = Solid::from_stl(&bytes).unwrap();
    assert!((back.mass_properties().unwrap().volume - 1000.0 / 6.0).abs() < 1e-6);
}

#[test]
fn a_mesh_cut_by_a_box_still_renders_shaded_and_a_pyramid_keeps_its_edges() {
    // The boolean merges coplanar triangles into polygons (each side facet
    // of the cylinder becomes a quad); the seams between them stay hidden.
    let cyl = Solid::cylinder(10.0, 20.0).unwrap();
    let mesh = Solid::from_stl(&stl_bytes(&cyl, 0.002)).unwrap();
    let bitten = mesh.cut(&Solid::cuboid(20.0, 20.0, 20.0).unwrap()).unwrap();
    let topo = bitten.topology().unwrap();
    assert!(topo.faces >= 100, "{} faces", topo.faces);
    let drawn: std::collections::BTreeSet<u32> = bitten.tessellate(0.1).unwrap().edge_ids.iter().copied().collect();
    // Of a side facet's four edges the two rims are creases and the two
    // seams to its neighbours are not: well under three quarters drawn.
    assert!(drawn.len() * 4 < topo.edges as usize * 3, "{} of {} edges drawn", drawn.len(), topo.edges);

    // Six faces, all triangles, meeting at 16°: a design, every edge drawn.
    let pyramid = "solid p
facet normal 0 0 0
outer loop
vertex 0 0 0
vertex 0 100 0
vertex 100 100 0
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 0 0 0
vertex 100 100 0
vertex 100 0 0
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 0 0 0
vertex 100 0 0
vertex 50 50 10
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 100 0 0
vertex 100 100 0
vertex 50 50 10
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 100 100 0
vertex 0 100 0
vertex 50 50 10
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 0 100 0
vertex 0 0 0
vertex 50 50 10
endloop
endfacet
endsolid p
";
    let p = Solid::from_stl(pyramid.as_bytes()).unwrap();
    assert!((p.mass_properties().unwrap().volume - 100.0 * 100.0 * 10.0 / 3.0).abs() < 1e-6);
    let drawn: std::collections::BTreeSet<u32> = p.tessellate(0.1).unwrap().edge_ids.iter().copied().collect();
    assert_eq!(drawn.len(), 9, "four base edges, four slopes, one base diagonal");
}

#[test]
fn a_curved_body_through_stl_is_a_mesh_and_can_be_cut_and_exported() {
    let cyl = Solid::cylinder(10.0, 20.0).unwrap();
    let back = Solid::from_stl(&stl_bytes(&cyl, 0.05)).unwrap();
    let faces = back.faces().unwrap();
    assert!(faces.len() > 50, "{} faces", faces.len());
    assert!(faces.iter().all(|f| f.surface == cdt_kernel::SurfaceKind::Plane));
    let v = back.mass_properties().unwrap().volume;
    let exact = std::f64::consts::PI * 100.0 * 20.0;
    assert!(v < exact && v > exact * 0.99, "volume {v} vs {exact}");
    // Booleans see it as a solid: the box takes about a quadrant, and what
    // it takes plus what is left is the whole.
    let bite = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let cut = back.cut(&bite).unwrap().mass_properties().unwrap().volume;
    let common = back.common(&bite).unwrap().mass_properties().unwrap().volume;
    assert!((cut + common - v).abs() < 1e-6, "{cut} + {common} vs {v}");
    assert!((common - v / 4.0).abs() < v * 0.01, "{common} vs a quadrant of {v}");
    // And out again, as a mesh of the same size.
    let again = stl_bytes(&back.cut(&bite).unwrap(), 0.1);
    let third = Solid::from_stl(&again).unwrap();
    assert!((third.mass_properties().unwrap().volume - cut).abs() < 1e-6);

    // The backend scales it into the document unit like a STEP.
    let mut backend = KernelBackend::default();
    backend.begin_feature(FeatureId(1));
    let inch = backend.import(ImportFormat::Stl, &stl_bytes(&cyl, 0.05), 1.0 / 25.4).unwrap();
    assert!((inch.mass_properties().unwrap().volume - v / 25.4f64.powi(3)).abs() < 1e-9);
}

#[test]
fn broken_meshes_are_refused_by_name() {
    let err = |bytes: &[u8]| Solid::from_stl(bytes).err().unwrap().0;
    assert!(err(b"").contains("empty"));
    assert!(err(b"not an stl").contains("could not parse"), "{}", err(b"not an stl"));
    // One triangle: three open edges.
    let open = "solid one\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid one\n";
    let e = err(open.as_bytes());
    assert!(e.contains("isn't watertight") && e.contains("3 open edge"), "{e}");
    // A Klein bottle (the figure-8 immersion) is closed and every edge is
    // shared by two triangles, but it has no inside: refused as unorientable.
    let klein = {
        let (n, m) = (12usize, 12usize);
        let p = |i: usize, j: usize| {
            let u = i as f64 / n as f64 * std::f64::consts::TAU;
            // Half a step off the grid: the figure-8 crosses itself at v = 0
            // and π, and a vertex there would be shared by four triangles.
            let v = (j as f64 + 0.5) / m as f64 * std::f64::consts::TAU;
            let r = 3.0 + (u / 2.0).cos() * v.sin() - (u / 2.0).sin() * (2.0 * v).sin();
            [r * u.cos(), r * u.sin(), (u / 2.0).sin() * v.sin() + (u / 2.0).cos() * (2.0 * v).sin()]
        };
        // Going once round in u lands on the same curve traversed backwards.
        let at = |i: usize, j: usize| if i < n { p(i, j % m) } else { p(0, m - 1 - j % m) };
        let mut s = String::from("solid k\n");
        let mut tri = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
            s += "facet normal 0 0 0\nouter loop\n";
            for v in [a, b, c] {
                s += &format!("vertex {} {} {}\n", v[0], v[1], v[2]);
            }
            s += "endloop\nendfacet\n";
        };
        for i in 0..n {
            for j in 0..m {
                tri(at(i, j), at(i + 1, j), at(i + 1, j + 1));
                tri(at(i, j), at(i + 1, j + 1), at(i, j + 1));
            }
        }
        s + "endsolid k\n"
    };
    let e = err(klein.as_bytes());
    assert!(e.contains("no inside and outside"), "{e}");
    // A truncated binary file.
    let whole = stl_bytes(&Solid::cuboid(1.0, 1.0, 1.0).unwrap(), 0.1);
    let e = err(&whole[..whole.len() - 20]);
    assert!(e.contains("could not parse"), "{e}");
    // The wrong format under the wrong name goes through `import` by format.
    assert!(Solid::import(ImportFormat::Step, &whole).is_err());
    assert!(Solid::import(ImportFormat::Stl, &step_bytes(&Solid::cuboid(1.0, 1.0, 1.0).unwrap())).is_err());
}
