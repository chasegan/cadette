//! Riemanifold — Phase 1 interactive shell.
//!
//! A live modeler window: the **history panel** (egui) on the left shows the
//! parametric feature tree with selection, suppression, reordering, a rollback
//! bar, and an inline parameter editor; the **viewport** (wgpu) on the right
//! shows the regenerated solid. Editing any step rebuilds the model live by
//! replaying the document through the OCCT backend.
//!
//! `--screenshot` renders one composited frame (panel + model) to a PNG for
//! headless verification.

use rmf_core::{regenerate, BooleanOp, Document, FeatureKind, RegenError, DVec3};
use rmf_kernel::KernelBackend;
use rmf_render::{Controller, MeshData};
use rmf_ui::{history_panel, HistoryState};

const DEFLECTION_MM: f64 = 0.1;

/// The default starting part: a 40 mm cube, edges filleted at 4 mm, with a
/// 12 mm hole bored through it — every step editable in the history panel.
fn build_document() -> Document {
    let mut doc = Document::new("spike-part");

    let cube = doc.add(
        "Box",
        FeatureKind::Box {
            size: DVec3::new(40.0, 40.0, 40.0),
        },
    );
    let rounded = doc.add(
        "Fillet edges",
        FeatureKind::FilletAll {
            source: cube,
            radius: 4.0,
        },
    );
    let drill = doc.add(
        "Drill",
        FeatureKind::Cylinder {
            radius: 6.0,
            height: 60.0,
        },
    );
    let drill = doc.add(
        "Position drill",
        FeatureKind::Translate {
            source: drill,
            offset: DVec3::new(20.0, 20.0, -10.0),
        },
    );
    doc.add(
        "Bore hole",
        FeatureKind::Boolean {
            op: BooleanOp::Subtract,
            target: rounded,
            tool: drill,
        },
    );

    doc
}

/// Ties the document, the OCCT backend, and the history panel together as a
/// [`Controller`] the viewport drives.
struct Modeler {
    doc: Document,
    backend: KernelBackend,
    ui: HistoryState,
}

impl Modeler {
    fn new() -> Self {
        Self {
            doc: build_document(),
            backend: KernelBackend::default(),
            ui: HistoryState::default(),
        }
    }
}

impl Controller for Modeler {
    fn ui(&mut self, ctx: &rmf_render::egui::Context) -> bool {
        history_panel(ctx, &mut self.doc, &mut self.ui)
    }

    /// Regenerate the document through OCCT and tessellate every visible body
    /// into one mesh. Records per-feature errors for the panel.
    fn mesh(&mut self) -> MeshData {
        let regen = regenerate(&self.doc, &mut self.backend);
        self.ui.errors = regen
            .errors()
            .iter()
            .map(|e| (e.feature(), error_message(e)))
            .collect();
        self.ui.visible = regen.visible().to_vec();

        let mut mesh = MeshData::default();
        for body in regen.visible_bodies() {
            match body.tessellate(DEFLECTION_MM) {
                Ok(part) => {
                    let base = mesh.vertices.len() as u32;
                    mesh.vertices
                        .extend(rmf_render::interleave(&part.positions, &part.normals));
                    mesh.indices.extend(part.indices.iter().map(|i| i + base));
                }
                Err(e) => self.ui.errors.push((rmf_core::FeatureId(0), e.to_string())),
            }
        }
        mesh
    }
}

fn error_message(err: &RegenError<rmf_kernel::KernelError>) -> String {
    match err {
        RegenError::Backend { source, .. } => source.to_string(),
        RegenError::MissingInput { input, .. } => format!("missing input {input:?}"),
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut modeler = Modeler::new();

    if std::env::args().any(|a| a == "--screenshot") {
        // Pre-select the fillet so the screenshot also shows the parameter editor.
        modeler.ui.selected = modeler.doc.history.features().get(1).map(|f| f.id);
        let path = "out/modeler.png";
        std::fs::create_dir_all("out")?;
        rmf_render::screenshot(modeler, 1280, 820, path)?;
        println!("wrote {path}");
        return Ok(());
    }

    println!("Riemanifold — interactive modeler");
    println!("  history panel: edit/suppress/reorder features, drag the rollback bar");
    println!("  viewport: left-drag orbit, right/middle-drag pan, scroll zoom\n");
    rmf_render::run(modeler)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! These exercise the Controller::mesh path through real OCCT, simulating
    //! the document mutations the history panel performs.
    use super::*;

    /// Axis-aligned bounds of a mesh's vertices: (min, max) per axis.
    fn bounds(mesh: &MeshData) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &mesh.vertices {
            for a in 0..3 {
                min[a] = min[a].min(v.position[a]);
                max[a] = max[a].max(v.position[a]);
            }
        }
        (min, max)
    }

    #[test]
    fn full_part_regenerates_without_errors() {
        let mut m = Modeler::new();
        let mesh = m.mesh();
        assert!(!mesh.indices.is_empty());
        assert!(m.ui.errors.is_empty());
        // The default part is a 40 mm cube.
        let (_min, max) = bounds(&mesh);
        assert!((max[0] - 40.0).abs() < 0.5);
    }

    #[test]
    fn rollback_shows_the_untranslated_cylinder() {
        let mut m = Modeler::new();
        let _ = m.mesh();
        // Roll back before "Position drill": box, fillet, and the cylinder at
        // the origin (height 60 along +Z) are active and both bodies visible.
        m.doc.set_rollback(3);
        let rolled = m.mesh();
        assert!(m.ui.errors.is_empty());
        let (_min, max) = bounds(&rolled);
        assert!(max[2] > 50.0, "the 60 mm cylinder should extend past z=50");
    }

    #[test]
    fn editing_the_box_rescales_the_model() {
        let mut m = Modeler::new();
        let before = bounds(&m.mesh());
        let box_id = m.doc.history.features()[0].id;
        m.doc.history.get_mut(box_id).unwrap().kind = FeatureKind::Box {
            size: DVec3::new(80.0, 80.0, 80.0),
        };
        let after = bounds(&m.mesh());
        assert!(before.1[0] < 50.0 && after.1[0] > 70.0, "box should grow");
    }

    #[test]
    fn adding_a_primitive_adds_a_visible_body() {
        let mut m = Modeler::new();
        let _ = m.mesh();
        let before = m.ui.visible.len();
        m.doc.add("Sphere", FeatureKind::Sphere { radius: 10.0 });
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), before + 1);
        assert!(m.ui.errors.is_empty());
    }

    #[test]
    fn suppressing_a_referenced_feature_surfaces_an_error() {
        let mut m = Modeler::new();
        let fillet_id = m.doc.history.features()[1].id;
        m.doc.history.set_suppressed(fillet_id, true);
        let _ = m.mesh();
        // "Bore hole" references the suppressed fillet -> a regen error.
        assert!(!m.ui.errors.is_empty());
    }
}
