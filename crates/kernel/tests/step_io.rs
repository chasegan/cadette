//! STEP in and out: what a vendor part becomes once it is read, and that
//! Cadette's own bodies survive a round trip.

use cdt_core::{DVec3, FeatureId, GeometryBackend};
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
    let body = backend.import_step(&bytes, 1.0).unwrap();
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
    let inch = backend.import_step(&bytes, 1.0 / 25.4).unwrap();
    let props = inch.mass_properties().unwrap();
    assert!((props.volume - 1.0).abs() < 1e-9, "a 1-inch cube in an inch document is 1 unit³");
    let _ = DVec3::ZERO;
}

#[test]
fn garbage_and_empty_files_are_refused_by_name() {
    let err = Solid::from_step(b"hello, not a step file").err().unwrap();
    assert!(err.0.contains("not a STEP file"), "{}", err.0);
    let err = Solid::from_step(b"").err().unwrap();
    assert!(err.0.contains("empty"), "{}", err.0);
}
