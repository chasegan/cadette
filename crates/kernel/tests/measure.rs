//! Interrogation for the measurement read-out: exact face areas and edge
//! lengths, indexed the same way tessellation ids the faces/edges it emits.

use cdt_kernel::Solid;

#[test]
fn every_face_of_a_box_has_its_exact_area() {
    let b = Solid::cuboid(10.0, 20.0, 30.0).unwrap();
    let n = b.face_count().unwrap();
    assert_eq!(n, 6);
    // A 10×20×30 box has face-area pairs 200, 300, 600 (two of each).
    let mut areas: Vec<f64> = (0..n as u32)
        .map(|i| b.face_area(i).unwrap())
        .collect();
    areas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expect = [200.0, 200.0, 300.0, 300.0, 600.0, 600.0];
    for (got, want) in areas.iter().zip(expect) {
        assert!((got - want).abs() < 1e-6, "area {got} != {want} ({areas:?})");
    }
}

#[test]
fn cylinder_faces_measure_lateral_and_cap_areas() {
    let c = Solid::cylinder(10.0, 20.0).unwrap();
    let n = c.face_count().unwrap() as u32;
    let mut areas: Vec<f64> = (0..n).map(|i| c.face_area(i).unwrap()).collect();
    areas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    use std::f64::consts::PI;
    // Two r=10 caps (100π) and the 2πrh lateral surface (400π).
    let expect = [100.0 * PI, 100.0 * PI, 400.0 * PI];
    assert_eq!(areas.len(), expect.len());
    for (got, want) in areas.iter().zip(expect) {
        assert!((got - want).abs() < 1e-6, "area {got} != {want}");
    }
}

#[test]
fn box_edges_measure_their_exact_lengths() {
    let b = Solid::cuboid(10.0, 20.0, 30.0).unwrap();
    // 12 edges: four of each length.
    let mut lens: Vec<f64> = (0..12).map(|i| b.edge_length(i).unwrap()).collect();
    lens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (k, want) in [(0..4, 10.0), (4..8, 20.0), (8..12, 30.0)] {
        for i in k {
            assert!(
                (lens[i] - want) < 1e-6 && (want - lens[i]) < 1e-6,
                "edge {i}: {} != {want} ({lens:?})",
                lens[i]
            );
        }
    }
}

#[test]
fn out_of_range_indices_are_errors_not_garbage() {
    let b = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    assert!(b.face_area(6).is_err(), "face index past the end errors");
    assert!(b.edge_length(12).is_err(), "edge index past the end errors");
}
