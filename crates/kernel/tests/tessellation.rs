//! Tessellation tagging: every vertex carries its source face id, used for
//! GPU picking.

use std::collections::BTreeSet;

use cdt_kernel::Solid;

#[test]
fn box_tessellation_tags_six_faces() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let mesh = cube.tessellate(0.5).unwrap();

    // One face id per vertex (per positions triple).
    assert_eq!(mesh.face_ids.len(), mesh.positions.len() / 3);
    assert!(!mesh.face_ids.is_empty());

    // A box has exactly six planar faces, numbered 0..6.
    let distinct: BTreeSet<u32> = mesh.face_ids.iter().copied().collect();
    assert_eq!(distinct.len(), 6, "a box has six faces, got {distinct:?}");
    assert_eq!(*distinct.iter().max().unwrap(), 5);

    // Every triangle's three vertices belong to one and the same face.
    for tri in mesh.indices.chunks_exact(3) {
        let f = mesh.face_ids[tri[0] as usize];
        assert!(tri.iter().all(|&v| mesh.face_ids[v as usize] == f));
    }

    // A box has twelve edges; each emits a polyline tagged with its edge id.
    assert_eq!(mesh.edge_ids.len(), mesh.edge_positions.len() / 3);
    assert!(!mesh.edge_indices.is_empty());
    let edges: BTreeSet<u32> = mesh.edge_ids.iter().copied().collect();
    assert_eq!(edges.len(), 12, "a box has twelve edges, got {edges:?}");
}

#[test]
fn box_faces_report_planar_frames() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();

    // Every one of the six faces is planar with an orthonormal frame.
    for index in 0..6 {
        let (_o, x, y) = cube
            .face_plane(index)
            .unwrap()
            .unwrap_or_else(|| panic!("face {index} should be planar"));
        let dot = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        assert!(dot.abs() < 1e-9, "axes not perpendicular on face {index}");
        let xlen = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        assert!((xlen - 1.0).abs() < 1e-9, "x axis not unit on face {index}");
    }
    // Out of range -> None, not an error.
    assert!(cube.face_plane(99).unwrap().is_none());
}

fn max_z(solid: &Solid) -> f32 {
    let mesh = solid.tessellate(0.5).unwrap();
    mesh.positions
        .chunks_exact(3)
        .map(|p| p[2])
        .fold(f32::MIN, f32::max)
}

#[test]
fn push_pull_offsets_the_anchored_face() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    assert!((max_z(&cube) - 10.0).abs() < 0.5);

    // Top face at z=10, normal +Z. Pushing out by 5 fuses material (z -> 15).
    let taller = cube
        .push_pull([5.0, 5.0, 10.0], [0.0, 0.0, 1.0], 5.0)
        .unwrap();
    assert!((max_z(&taller) - 15.0).abs() < 0.5, "pushed max z {}", max_z(&taller));

    // Pulling in by 3 cuts material (z -> 7).
    let shorter = cube
        .push_pull([5.0, 5.0, 10.0], [0.0, 0.0, 1.0], -3.0)
        .unwrap();
    assert!((max_z(&shorter) - 7.0).abs() < 0.5, "pulled max z {}", max_z(&shorter));
}

#[test]
fn push_pull_disambiguates_coplanar_faces() {
    // Two separate boxes with coplanar top faces (both at z=10, normal +Z).
    let a = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let b = Solid::cuboid(10.0, 10.0, 10.0)
        .unwrap()
        .translate(20.0, 0.0, 0.0)
        .unwrap();
    let both = a.fuse(&b).unwrap();

    // Push only B's top (its centroid is (25,5,10)). A must stay put — the old
    // plane-only match could grab A instead since both tops share the plane.
    let pushed = both
        .push_pull([25.0, 5.0, 10.0], [0.0, 0.0, 1.0], 5.0)
        .unwrap();
    let mesh = pushed.tessellate(0.5).unwrap();
    let region_max_z = |keep: fn(f32) -> bool| {
        mesh.positions
            .chunks_exact(3)
            .filter(|p| keep(p[0]))
            .map(|p| p[2])
            .fold(f32::MIN, f32::max)
    };
    assert!((region_max_z(|x| x < 15.0) - 10.0).abs() < 0.5, "A moved");
    assert!((region_max_z(|x| x > 15.0) - 15.0).abs() < 0.5, "B not pushed");
}

#[test]
fn fillet_edge_rounds_only_the_nearest_edge() {
    // A 10mm cube; fillet just the top edge that runs along +Y at x=0, z=10.
    // Anchor on that edge's midpoint (0, 5, 10).
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let rounded = cube.fillet_edge([0.0, 5.0, 10.0], 2.0).unwrap();

    // Filleting one edge adds a rounded face: the box gains faces (6 -> 7).
    let mesh = rounded.tessellate(0.3).unwrap();
    let faces: BTreeSet<u32> = mesh.face_ids.iter().copied().collect();
    assert_eq!(faces.len(), 7, "one fillet adds exactly one face, got {faces:?}");

    // The sharp corner at (0,0,10)/(0,10,10) is gone: no vertex sits within the
    // 2mm fillet radius of the original edge line (x≈0, z≈10).
    let near_old_edge = mesh
        .positions
        .chunks_exact(3)
        .any(|p| p[0].abs() < 0.2 && (p[2] - 10.0).abs() < 0.2);
    assert!(!near_old_edge, "the sharp edge should have been rounded away");
}

#[test]
fn rotate_about_z_swaps_the_footprint() {
    // A 20x10x10 box at the origin, rotated 90° about Z through its center
    // (10,5,5): its X-span (20) and Y-span (10) should swap.
    let bar = Solid::cuboid(20.0, 10.0, 10.0).unwrap();
    let turned = bar
        .rotate([10.0, 5.0, 5.0], [0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)
        .unwrap();
    let mesh = turned.tessellate(0.5).unwrap();
    let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in mesh.positions.chunks_exact(3) {
        minx = minx.min(p[0]);
        maxx = maxx.max(p[0]);
        miny = miny.min(p[1]);
        maxy = maxy.max(p[1]);
    }
    // After a 90° turn the footprint is 10 wide in X and 20 deep in Y.
    assert!((maxx - minx - 10.0).abs() < 0.5, "x span {}", maxx - minx);
    assert!((maxy - miny - 20.0).abs() < 0.5, "y span {}", maxy - miny);
}

#[test]
fn fillet_edges_rounds_several_at_once() {
    // Round all four vertical edges of a 10mm cube in one operation. Anchors at
    // each vertical edge's midpoint (z=5).
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    let rounded = cube
        .fillet_edges(
            &[
                [0.0, 0.0, 5.0],
                [10.0, 0.0, 5.0],
                [10.0, 10.0, 5.0],
                [0.0, 10.0, 5.0],
            ],
            2.0,
        )
        .unwrap();

    // 6 original faces + 4 rounded fillet faces = 10.
    let mesh = rounded.tessellate(0.3).unwrap();
    let faces: BTreeSet<u32> = mesh.face_ids.iter().copied().collect();
    assert_eq!(faces.len(), 10, "four fillets add four faces, got {faces:?}");

    // Duplicate anchors landing on the same edge must not double-add it.
    let twice = cube
        .fillet_edges(&[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]], 2.0)
        .unwrap();
    let f = twice.tessellate(0.3).unwrap();
    let n: BTreeSet<u32> = f.face_ids.iter().copied().collect();
    assert_eq!(n.len(), 7, "a deduped single fillet adds one face, got {n:?}");
}

#[test]
fn profile_face_builds_a_curved_boundary() {
    // A 20×20 square whose BOTTOM edge is a cubic bezier bulging down to ~y=-7.5
    // (controls at y=-10). The other three sides stay straight.
    let points = [0.0, 0.0, 20.0, 0.0, 20.0, 20.0, 0.0, 20.0];
    #[rustfmt::skip]
    let segs = [
        1.0, 5.0, -10.0, 15.0, -10.0, // seg 0: bezier
        0.0, 0.0, 0.0, 0.0, 0.0,       // seg 1: line
        0.0, 0.0, 0.0, 0.0, 0.0,       // seg 2: line
        0.0, 0.0, 0.0, 0.0, 0.0,       // seg 3: line
    ];
    let face =
        Solid::profile_face([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], &points, &segs)
            .unwrap();
    let prism = face.extrude(10.0).unwrap();
    let mesh = prism.tessellate(0.1).unwrap();
    assert!(!mesh.indices.is_empty(), "curved profile extrudes to a solid");

    let (mut ymin, mut ymax) = (f32::MAX, f32::MIN);
    for p in mesh.positions.chunks_exact(3) {
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    // The straight top stays at y=20; the bezier bottom dips well below 0.
    assert!((ymax - 20.0).abs() < 0.2, "top straight at y=20, got {ymax}");
    assert!(ymin < -5.0, "bottom edge bulges down past y=−5, got {ymin}");
}

#[test]
fn mirror_reflects_a_box_into_a_valid_oriented_solid() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap(); // [0,10]^3
    // Reflect across the x=10 plane.
    let mirrored = cube.mirror([10.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();

    let mesh = mirrored.tessellate(0.5).unwrap();
    assert!(!mesh.indices.is_empty());
    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in mesh.positions.chunks_exact(3) {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    // The reflected box occupies [10,20] in X, unchanged in Y/Z.
    assert!((min[0] - 10.0).abs() < 0.1 && (max[0] - 20.0).abs() < 0.1, "x {}..{}", min[0], max[0]);
    assert!(min[1].abs() < 0.1 && (max[1] - 10.0).abs() < 0.1, "y unchanged");

    // The mirror is properly oriented (outward normals): fusing it with the
    // original yields one connected 20-wide solid — an inside-out reflection
    // would fail or carve a void here. (OCCT leaves the coplanar seam faces
    // unmerged, so we check the bounds, not the face count.)
    let joined = cube.fuse(&mirrored).unwrap();
    let jmesh = joined.tessellate(0.5).unwrap();
    let (mut jmin, mut jmax) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in jmesh.positions.chunks_exact(3) {
        for k in 0..3 {
            jmin[k] = jmin[k].min(p[k]);
            jmax[k] = jmax[k].max(p[k]);
        }
    }
    assert!((jmin[0]).abs() < 0.1 && (jmax[0] - 20.0).abs() < 0.1, "fused x {}..{}", jmin[0], jmax[0]);
    assert!((jmax[1] - 10.0).abs() < 0.1 && (jmax[2] - 10.0).abs() < 0.1, "fused y/z unchanged");
}

#[test]
fn revolve_a_rectangle_about_its_edge_makes_a_cylinder() {
    use std::f64::consts::TAU;
    // A 10(radius) x 20(height) rectangle in the XZ plane (y=0), spanning
    // x in [0,10], z in [0,20] — so its left edge (x=0) lies on the Z axis.
    let rect = Solid::rectangle_face(
        [5.0, 0.0, 10.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        20.0,
    )
    .unwrap();

    // A full turn about that left edge (anchor: its midpoint) → a solid cylinder
    // of radius 10, height 20: x,y in [-10,10], z in [0,20].
    let solid = rect.revolve([0.0, 0.0, 10.0], TAU).unwrap();
    let mesh = solid.tessellate(0.2).unwrap();
    assert!(!mesh.indices.is_empty(), "revolve produced a meshable solid");

    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in mesh.positions.chunks_exact(3) {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    assert!((min[0] + 10.0).abs() < 0.3 && (max[0] - 10.0).abs() < 0.3, "x {}..{}", min[0], max[0]);
    assert!((min[1] + 10.0).abs() < 0.3 && (max[1] - 10.0).abs() < 0.3, "y {}..{}", min[1], max[1]);
    assert!(min[2].abs() < 0.3 && (max[2] - 20.0).abs() < 0.3, "z {}..{}", min[2], max[2]);

    // A partial turn (90°) sweeps only one quadrant: the profile (at +X) reaches
    // ±Y on one side, so the Y extent is a single 10mm quadrant touching y=0.
    // (Which side depends on the axis edge's orientation — not fixed here.)
    let quarter = rect.revolve([0.0, 0.0, 10.0], TAU / 4.0).unwrap();
    let qmesh = quarter.tessellate(0.2).unwrap();
    let (mut qymin, mut qymax) = (f32::MAX, f32::MIN);
    for p in qmesh.positions.chunks_exact(3) {
        qymin = qymin.min(p[1]);
        qymax = qymax.max(p[1]);
    }
    assert!((qymax - qymin - 10.0).abs() < 0.3, "quarter y span {}", qymax - qymin);
    assert!(qymin.abs() < 0.3 || qymax.abs() < 0.3, "quadrant touches y=0: {qymin}..{qymax}");
}
