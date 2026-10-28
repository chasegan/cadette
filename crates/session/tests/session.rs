//! The headless loop an agent runs: open, read the report, edit, verify,
//! undo, export — no window anywhere.

use cdt_core::{DocEdit, Document, FeatureId, FeatureKind, DVec3};
use cdt_session::{FeatureStatus, Session, PREVIEW_DEFLECTION_MM};
use std::path::PathBuf;

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/robot-gauge-bracket.cdt")
}

#[test]
fn the_example_opens_and_reports_one_valid_body() {
    let mut s = Session::open(example()).unwrap();
    let r = s.report();
    assert!(r.ok, "{:?}", r.errors);
    // The bracket and the gauge ring it carries.
    let names: Vec<&str> = r.bodies.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, ["Subtract 25", "Fillet 12"], "{names:?}");
    assert!(r.bodies.iter().all(|b| b.valid));
    let b = r.bodies.iter().find(|b| b.name == "Fillet 12").unwrap();
    assert!((b.volume - 5957.0).abs() < 5.0, "volume {}", b.volume);
    let w = b.bbox[1][0] - b.bbox[0][0];
    assert!((w - 50.15).abs() < 0.1, "width {w}");
    assert!(r.features.iter().all(|f| f.status == FeatureStatus::Built));
    assert_eq!(r.features.iter().filter(|f| f.visible).count(), 2);
    assert!(s.extras().contains_key("view.json"), "the app's view state is preserved");

    let detail = s.body_detail(b.feature).unwrap();
    assert_eq!(detail.faces.len(), b.faces);
    assert!(detail.faces.iter().any(|f| f.surface == "cylinder"));
    assert!(detail.edges.iter().any(|e| e.curve == "circle" && e.radius.is_some()));
}

#[test]
fn edit_verify_undo_redo() {
    let mut s = Session::new(Document::new("t"));
    let out = s
        .apply(&[DocEdit::Add { name: None, kind: FeatureKind::Box { size: DVec3::new(10.0, 20.0, 30.0) } }])
        .unwrap();
    let b = out[0].created().unwrap();
    let r = s.report();
    assert_eq!(r.bodies.len(), 1);
    assert!((r.bodies[0].volume - 6000.0).abs() < 1e-6);

    // Fillet it: the box is consumed, the fillet is the visible body.
    s.apply(&[DocEdit::Add { name: None, kind: FeatureKind::FilletAll { source: b, radius: 2.0 } }]).unwrap();
    let r = s.report();
    assert_eq!(r.bodies.len(), 1);
    assert_ne!(r.bodies[0].feature, b);
    assert!(r.bodies[0].volume < 6000.0, "rounding removes material");
    assert_eq!(r.features[0].status, FeatureStatus::Built);
    assert!(!r.features[0].visible, "consumed by the fillet");

    // A bad edit is refused and changes nothing.
    assert!(s.apply(&[DocEdit::Remove { id: FeatureId(999) }]).is_err());
    assert_eq!(s.report().features.len(), 2);

    // A failing feature is reported, not fatal.
    s.apply(&[DocEdit::Replace { id: r.bodies[0].feature, kind: FeatureKind::FilletAll { source: b, radius: 50.0 } }])
        .unwrap();
    let r = s.report();
    assert!(!r.ok);
    assert_eq!(r.features[1].status, FeatureStatus::Failed);
    assert!(r.features[1].error.is_some());
    assert_eq!(r.bodies[0].feature, b, "the box shows again because the fillet failed");

    assert!(s.undo());
    assert!(s.report().ok);
    assert!(s.undo() && s.undo(), "back to the empty document");
    assert!(s.report().bodies.is_empty());
    assert!(s.redo());
    assert_eq!(s.report().bodies.len(), 1);
}

#[test]
fn export_and_mesh_and_save_round_trip() {
    let mut s = Session::open(example()).unwrap();
    let dir = std::env::temp_dir();
    let stl = dir.join("cdt-session-test.stl");
    s.export_stl(&stl, 0.2).unwrap();
    assert!(std::fs::metadata(&stl).unwrap().len() > 1000);
    let _ = std::fs::remove_file(&stl);

    let scene = s.scene_mesh(PREVIEW_DEFLECTION_MM);
    assert_eq!(scene.bodies.len(), 2);
    assert!(scene.failures.is_empty());
    assert!(!scene.mesh.vertices.is_empty() && !scene.mesh.edge_vertices.is_empty());
    assert_eq!(scene.bodies[1].face_start, scene.bodies[0].face_end, "ids are contiguous per body");
    assert!(scene.mesh.vertices.iter().all(|v| v.face_id < scene.bodies[1].face_end));

    s.apply(&[DocEdit::Add { name: Some("Extra".into()), kind: FeatureKind::Sphere { radius: 3.0 } }]).unwrap();
    let cdt = dir.join("cdt-session-test.cdt");
    s.save_as(&cdt).unwrap();
    let mut back = Session::open(&cdt).unwrap();
    assert!(back.extras().contains_key("view.json"), "extras survive a headless save");
    assert_eq!(back.report().bodies.len(), 3);
    let _ = std::fs::remove_file(&cdt);

    let mut empty = Session::new(Document::new("e"));
    assert!(empty.export_stl(dir.join("never.stl"), 0.1).is_err(), "nothing to export");
}

#[test]
fn a_step_import_is_checked_before_it_lands_and_survives_a_save() {
    let bytes = {
        let path = std::env::temp_dir().join(format!("cdt-session-import-{}.step", std::process::id()));
        cdt_kernel::Solid::cuboid(10.0, 20.0, 30.0).unwrap().write_step(path.to_str().unwrap()).unwrap();
        let b = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        b
    };
    let mut s = Session::new(Document::new("imp"));

    // Junk is refused as a batch error — nothing lands, no undo entry.
    let err = s.apply(&[DocEdit::import_step("junk.step", b"hello", None)]).unwrap_err();
    assert_eq!(err.index, 0);
    assert!(err.error.to_string().contains("junk.step: could not parse"), "{err}");
    assert!(s.document().history.is_empty());
    assert!(s.document().assets.is_empty());
    assert!(!s.can_undo());

    // A real file lands, builds, and is in the saved archive.
    let out = s.apply(&[DocEdit::import_step("block.step", &bytes, None)]).unwrap();
    let id = out[0].created().unwrap();
    let r = s.report();
    assert!(r.ok, "{:?}", r.errors);
    assert_eq!(r.bodies.len(), 1);
    assert!((r.bodies[0].volume - 6000.0).abs() < 1e-6);
    assert_eq!(r.bodies[0].solids, 1);
    assert_eq!(r.features[0].summary, "Import block.step");

    let path = std::env::temp_dir().join(format!("cdt-session-import-{}.cdt", std::process::id()));
    s.save_as(&path).unwrap();
    let mut again = Session::open(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let r = again.report();
    assert!(r.ok, "{:?}", r.errors);
    assert!((r.bodies[0].volume - 6000.0).abs() < 1e-6);
    assert!(again.body(id).is_some());
}
