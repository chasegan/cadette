//! Chamfer and shell: the two modifying ops that share the fillet pipeline.
//!
//! Chamfer: a symmetric bevel set back `distance` along both faces of each
//! edge, so one edge of an L-long block removes exactly d²/2·L. Names follow
//! the fillet rules — survivors verbatim, the bevel is `Chamfered { edge }`.
//! Shell: an INWARD wall of uniform thickness; opened faces vanish, kept
//! faces survive verbatim, and each kept face's inner wall is `Offset {
//! face }`.

use cdt_core::{DVec3, EdgeAnchor, FaceAnchor, FeatureId, GeometryBackend};
use cdt_kernel::{Body, ElemKind, ElementName, KernelBackend, Solid};

fn tracked_box(backend: &mut KernelBackend, id: u64) -> Body {
    backend.begin_feature(FeatureId(id));
    backend.make_box(DVec3::splat(20.0)).unwrap()
}

/// The face of `body` whose plane passes the predicate (planar faces only).
fn planar_face_index(body: &Body, pred: impl Fn([f64; 3], [f64; 3]) -> bool) -> u32 {
    let n = body.face_count().unwrap();
    for i in 0..n as u32 {
        if let Some((origin, x_dir, y_dir)) = body.face_plane(i).unwrap() {
            let normal = [
                x_dir[1] * y_dir[2] - x_dir[2] * y_dir[1],
                x_dir[2] * y_dir[0] - x_dir[0] * y_dir[2],
                x_dir[0] * y_dir[1] - x_dir[1] * y_dir[0],
            ];
            if pred(origin, normal) {
                return i;
            }
        }
    }
    panic!("no face matched the predicate");
}

fn edge_index_near(body: &Body, point: [f64; 3]) -> u32 {
    let n = body.edge_count().unwrap();
    let mut best = (0u32, f64::INFINITY);
    for i in 0..n as u32 {
        let c = body.solid.elem_centroid(ElemKind::Edge, i).unwrap();
        let d2 = (0..3).map(|k| (c[k] - point[k]).powi(2)).sum::<f64>();
        if d2 < best.1 {
            best = (i, d2);
        }
    }
    best.0
}

#[test]
fn chamfering_one_edge_removes_a_triangular_prism() {
    // The top-front edge of a 20 cube, d = 3: a 3×3 right triangle times the
    // 20 mm edge → 90 mm³ gone, one new PLANAR face.
    let cube = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let chamfered = cube.chamfer_edges(&[[10.0, 0.0, 20.0]], 3.0).unwrap();
    let v = chamfered.volume().unwrap();
    assert!((v - (8000.0 - 90.0)).abs() < 1e-6, "volume {v}");
    assert_eq!(chamfered.face_count().unwrap(), 7);
    let desc = chamfered.describe().unwrap();
    assert!(desc.contains("closed 1") && !desc.contains("INVALID"), "{desc}");
}

#[test]
fn chamfering_every_edge_bevels_all_twelve_and_the_corners() {
    let cube = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let chamfered = cube.chamfer_all_edges(2.0).unwrap();
    // 6 trimmed faces + 12 bevels + 8 corner triangles.
    assert_eq!(chamfered.face_count().unwrap(), 26);
    let v = chamfered.volume().unwrap();
    // Twelve 2×2 prisms minus the corner overlaps: strictly between the two
    // simple bounds.
    assert!(v < 8000.0 - 12.0 * 40.0 + 8.0 * 8.0 && v > 8000.0 - 12.0 * 40.0, "volume {v}");
}

#[test]
fn an_oversized_chamfer_is_a_clear_error() {
    // Chamfers of 12 on opposite edges of a 20 face would cross.
    let cube = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let err = cube
        .chamfer_edges(&[[10.0, 0.0, 20.0], [10.0, 20.0, 20.0]], 12.0)
        .err()
        .expect("crossing bevels must fail, not emit a broken body");
    let msg = err.to_string();
    assert!(msg.contains("chamfer"), "the error names the op: {msg}");
}

#[test]
fn a_chamfer_names_the_bevel_and_keeps_survivors_verbatim() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let top_name = a.elements.face(top).unwrap().clone();
    let edge = edge_index_near(&a, [10.0, 0.0, 20.0]);
    let bevel_name = ElementName::Chamfered {
        parent: Box::new(a.elements.edge(edge).unwrap().clone()),
    };

    for distance in [1.0, 3.0] {
        backend.begin_feature(FeatureId(2));
        let chamfered = backend
            .chamfer_edges(&a, &[EdgeAnchor { point: DVec3::new(10.0, 0.0, 20.0) }], distance)
            .unwrap();
        let ti = chamfered.elements.find_face(&top_name).expect("trimmed top keeps its name");
        let area = chamfered.face_area(ti).unwrap();
        assert!(
            (area - (400.0 - 20.0 * distance)).abs() < 0.1,
            "d={distance}: the top is trimmed by the distance, got {area:.2}"
        );
        let bi = chamfered.elements.find_face(&bevel_name).expect("the bevel is Chamfered{edge}");
        let (origin, x_dir, y_dir) = chamfered.face_plane(bi).unwrap().expect("a bevel is planar");
        let normal = [
            x_dir[1] * y_dir[2] - x_dir[2] * y_dir[1],
            x_dir[2] * y_dir[0] - x_dir[0] * y_dir[2],
            x_dir[0] * y_dir[1] - x_dir[1] * y_dir[0],
        ];
        // A symmetric chamfer between the -Y and +Z faces faces (0, -1, 1)/√2
        // (either sign of the basis is fine).
        assert!(
            normal[0].abs() < 1e-6 && (normal[1].abs() - normal[2].abs()).abs() < 1e-6,
            "d={distance}: a 45° bevel, got normal {normal:?} at {origin:?}"
        );
        let area = chamfered.face_area(bi).unwrap();
        assert!(
            (area - 20.0 * distance * 2f64.sqrt()).abs() < 1e-6,
            "d={distance}: the bevel spans d√2 × 20, got {area:.3}"
        );
    }
}

#[test]
fn chamfer_all_names_every_bevel() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    backend.begin_feature(FeatureId(2));
    let chamfered = backend.chamfer_all(&a, 2.0).unwrap();
    let bevels = chamfered
        .elements
        .faces
        .iter()
        .flatten()
        .filter(|n| matches!(n, ElementName::Chamfered { .. }))
        .count();
    assert_eq!(bevels, 12, "one Chamfered face per cube edge");
    for i in 0..6u32 {
        assert!(chamfered.elements.find_face(a.elements.face(i).unwrap()).is_some(), "face {i} survives");
    }
}

#[test]
fn shelling_from_a_face_hollows_the_body_inward() {
    // A 20 cube opened at the top with 2 mm walls: a 16×16×18 cavity.
    let cube = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let top = ([10.0, 10.0, 20.0], [0.0, 0.0, 1.0]);
    let shelled = cube.shell(&[top], 2.0).unwrap();
    let v = shelled.volume().unwrap();
    assert!((v - (8000.0 - 16.0 * 16.0 * 18.0)).abs() < 1e-6, "volume {v}");
    let desc = shelled.describe().unwrap();
    assert!(desc.contains("solids=1") && desc.contains("closed 1") && !desc.contains("INVALID"), "{desc}");
    // The outer box is untouched: the wall grew inward.
    let (lo, hi) = shelled.bbox().unwrap();
    assert!(lo.iter().all(|c| c.abs() < 1e-6) && hi.iter().all(|c| (c - 20.0).abs() < 1e-6), "bbox {lo:?}..{hi:?}");
    // Nothing to open → a closed hollow (a 16 cube cavity).
    let closed = Solid::cuboid(20.0, 20.0, 20.0).unwrap().shell(&[], 2.0).unwrap();
    let vc = closed.volume().unwrap();
    assert!((vc - (8000.0 - 16.0 * 16.0 * 16.0)).abs() < 1e-6, "closed hollow volume {vc}");
}

#[test]
fn a_shell_that_cannot_fit_is_a_clear_error() {
    let cube = Solid::cuboid(20.0, 20.0, 20.0).unwrap();
    let err = cube
        .shell(&[([10.0, 10.0, 20.0], [0.0, 0.0, 1.0])], 11.0)
        .err()
        .expect("walls thicker than half the body must fail");
    assert!(err.to_string().contains("thickness"), "the error points at the thickness: {err}");
    // A face anchor that matches nothing is an error, not a silent closed hollow.
    let err = cube.shell(&[([50.0, 50.0, 50.0], [0.0, 0.0, 1.0])], 2.0).err().expect("no such face");
    assert!(err.to_string().contains("no matching face"), "{err}");
}

#[test]
fn a_shell_names_the_inner_walls_and_keeps_the_outside_verbatim() {
    let mut backend = KernelBackend::default();
    let a = tracked_box(&mut backend, 1);
    let top = planar_face_index(&a, |o, n| o[2] > 19.9 && n[2].abs() > 0.99);
    let bottom = planar_face_index(&a, |o, n| o[2] < 0.1 && n[2].abs() > 0.99);
    let bottom_name = a.elements.face(bottom).unwrap().clone();
    let floor_name = ElementName::Offset { parent: Box::new(bottom_name.clone()) };

    for thickness in [1.0, 3.0] {
        backend.begin_feature(FeatureId(2));
        let shelled = backend
            .shell(
                &a,
                thickness,
                &[FaceAnchor { point: DVec3::new(10.0, 10.0, 20.0), normal: DVec3::Z }],
            )
            .unwrap();
        // The bottom survives verbatim at full size; the opened top survives
        // as the wall's RIM (the top trimmed by the cavity) — so anything
        // referencing the top face follows it onto the rim.
        let bi = shelled.elements.find_face(&bottom_name).expect("the bottom survives");
        assert!((shelled.face_area(bi).unwrap() - 400.0).abs() < 1e-6);
        let ri = shelled.elements.find_face(a.elements.face(top).unwrap()).expect("the top became the rim");
        let inner = 20.0 - 2.0 * thickness;
        assert!(
            (shelled.face_area(ri).unwrap() - (400.0 - inner * inner)).abs() < 1e-6,
            "t={thickness}: the rim is the top minus the cavity mouth"
        );
        // Its inner wall is Offset{bottom}: the cavity floor, `thickness` up.
        let fi = shelled.elements.find_face(&floor_name).expect("the floor is Offset{bottom}");
        let (origin, _, _) = shelled.face_plane(fi).unwrap().expect("the floor is planar");
        assert!((origin[2] - thickness).abs() < 1e-6, "t={thickness}: floor at z={}", origin[2]);
        let area = shelled.face_area(fi).unwrap();
        assert!((area - inner * inner).abs() < 1e-6, "t={thickness}: floor area {area}");
    }

    // A closed hollow names every inner wall too.
    backend.begin_feature(FeatureId(3));
    let closed = backend.shell(&a, 2.0, &[]).unwrap();
    let walls = closed
        .elements
        .faces
        .iter()
        .flatten()
        .filter(|n| matches!(n, ElementName::Offset { .. }))
        .count();
    assert_eq!(walls, 6, "one Offset face per outer face of a closed hollow");
    let fi = closed.elements.find_face(&floor_name).expect("the floor is Offset{bottom}");
    assert!((closed.face_area(fi).unwrap() - 256.0).abs() < 1e-6, "a 16 mm square floor");
}

#[test]
fn a_shell_passes_under_chamfers_and_fillets() {
    // The usual order is edge treatment first, then hollow. OCCT's default
    // arc-only offset join silently returns a chamfered body unchanged; the
    // intersection join handles bevels, and a wall thinner than a fillet
    // radius keeps the rounded inner corners.
    let block = Solid::cuboid(40.0, 30.0, 20.0).unwrap();
    let top = ([20.0, 15.0, 20.0], [0.0, 0.0, 1.0]);

    let chamfered = block.chamfer_all_edges(2.0).unwrap();
    for t in [1.0, 3.0] {
        let shelled = chamfered.shell(&[top], t).unwrap();
        let v = shelled.volume().unwrap();
        assert!(v > 0.0 && v < chamfered.volume().unwrap() * 0.5, "t={t}: hollowed, got {v}");
        let desc = shelled.describe().unwrap();
        assert!(desc.contains("closed 1") && !desc.contains("INVALID"), "t={t}: {desc}");
    }

    let filleted = block.fillet_all_edges(2.0).unwrap();
    let shelled = filleted.shell(&[top], 1.0).unwrap();
    let v = shelled.volume().unwrap();
    assert!(v > 0.0 && v < filleted.volume().unwrap() * 0.5, "a 1 mm wall under a 2 mm fillet: {v}");
    // A wall as thick as the fillet radius has nowhere to put the inner
    // rounding: refused, not silently ignored.
    let err = filleted.shell(&[top], 2.0).err().expect("wall ≥ fillet radius must fail");
    assert!(err.to_string().contains("thickness"), "{err}");

    // Closed hollows too: on a blended body OCCT hands the inner offset back
    // as a bare shell, which the kernel must promote to a solid before the
    // cut (a 3 mm wall under 2 mm blends loses the inner rounding — fine).
    for (label, body) in [("chamfered", &chamfered), ("filleted", &filleted)] {
        for t in [1.0, 3.0] {
            let hollow = body.shell(&[], t).unwrap();
            let v = hollow.volume().unwrap();
            assert!(v > 0.0 && v < body.volume().unwrap() * 0.6, "{label} t={t}: hollowed, got {v}");
            // One solid bounded by two closed shells: the outside and the cavity.
            let desc = hollow.describe().unwrap();
            assert!(desc.contains("solids=1") && desc.contains("closed 2") && !desc.contains("INVALID"), "{label} t={t}: {desc}");
        }
    }
}
