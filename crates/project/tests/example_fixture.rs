//! The checked-in example project is the format's compatibility fixture: it
//! must keep loading as the format evolves (add a copy per version bump).

use std::path::PathBuf;

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/robot-gauge-bracket.cdt")
}

#[test]
fn the_example_project_loads_and_validates() {
    let project = cdt_project::read_path(example()).expect("example loads");
    let doc = &project.document;
    assert!(doc.history.len() >= 60, "the bracket has a long history: {}", doc.history.len());
    assert_eq!(doc.rollback(), doc.history.len(), "saved at the tip");
    doc.history.validate().expect("references hold");
    assert!(project.extras.contains_key("view.json"), "the app's viewport state rides along");
    assert!(doc.history.features().iter().any(|f| f.name.starts_with("Cylinder")));
}

#[test]
fn the_example_project_survives_a_rewrite() {
    let project = cdt_project::read_path(example()).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    cdt_project::write(&mut buf, &project).unwrap();
    buf.set_position(0);
    let back = cdt_project::read(buf).unwrap();
    assert_eq!(back.document.history.features(), project.document.history.features());
    assert_eq!(back.extras, project.extras);
}
