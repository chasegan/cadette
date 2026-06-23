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

use rmf_core::{
    regenerate, BooleanOp, Constraint, Document, FeatureKind, Profile, RegenError, Sketch2d,
    SketchPlane, DVec3,
};
use rmf_kernel::KernelBackend;
use rmf_render::{Controller, MeshData};
use rmf_ui::{history_panel, HistoryState};

const DEFLECTION_MM: f64 = 0.1;

/// A centered, axis-aligned rectangle defined by constraints. The corner points
/// start deliberately rough; the solver snaps them to an exact `width x height`
/// rectangle. Anchored at its lower-left corner.
fn constraint_rectangle(width: f64, height: f64) -> Sketch2d {
    let (hw, hh) = (width / 2.0, height / 2.0);
    let mut s = Sketch2d::new();
    let p0 = s.add_point(-hw, -hh); // anchor (exact)
    let p1 = s.add_point(hw * 0.9, -hh * 0.8);
    let p2 = s.add_point(hw * 1.1, hh * 0.9);
    let p3 = s.add_point(-hw * 0.8, hh * 1.1);

    let bottom = s.add_line(p0, p1);
    let right = s.add_line(p1, p2);
    let top = s.add_line(p2, p3);
    let left = s.add_line(p3, p0);

    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Horizontal(bottom));
    s.add_constraint(Constraint::Vertical(right));
    s.add_constraint(Constraint::Horizontal(top));
    s.add_constraint(Constraint::Vertical(left));
    s.add_constraint(Constraint::Distance(p0, p1, width));
    s.add_constraint(Constraint::Distance(p1, p2, height));
    s
}

/// Solve every constraint sketch in the document in place, so regeneration
/// reads solved coordinates. (`core` can't solve — that would cycle with the
/// solver crate — so the app drives it here.)
fn solve_sketches(doc: &mut Document) {
    for feature in doc.history.features_mut() {
        if let FeatureKind::ConstraintSketch { sketch, .. } = &mut feature.kind {
            rmf_solver::solve_sketch(sketch);
        }
    }
}

/// The default starting part, built end to end from sketches: a sketched
/// rectangle extruded into a block, edges filleted, then a sketched circle
/// extruded into a pin and bored through it. Every step is editable in the
/// history panel.
fn build_document() -> Document {
    let mut doc = Document::new("spike-part");

    // Block: a constraint-driven 40 x 40 rectangle on XY, extruded 40 up,
    // edges rounded.
    let base = doc.add(
        "Base profile",
        FeatureKind::ConstraintSketch {
            plane: SketchPlane::Xy,
            sketch: constraint_rectangle(40.0, 40.0),
        },
    );
    let block = doc.add("Extrude base", FeatureKind::Extrude { source: base, distance: 40.0 });
    let rounded = doc.add(
        "Fillet edges",
        FeatureKind::FilletAll {
            source: block,
            radius: 4.0,
        },
    );

    // Drill: a 12 mm circle extruded into a pin, positioned to pass through.
    let hole = doc.add(
        "Hole profile",
        FeatureKind::Sketch {
            plane: SketchPlane::Xy,
            profile: Profile::Circle { radius: 6.0 },
        },
    );
    let pin = doc.add("Extrude hole", FeatureKind::Extrude { source: hole, distance: 60.0 });
    let pin = doc.add(
        "Position hole",
        FeatureKind::Translate {
            source: pin,
            offset: DVec3::new(0.0, 0.0, -10.0),
        },
    );
    doc.add(
        "Bore hole",
        FeatureKind::Boolean {
            op: BooleanOp::Subtract,
            target: rounded,
            tool: pin,
        },
    );

    doc
}

/// Maximum number of undo snapshots retained.
const UNDO_LIMIT: usize = 200;

/// Ties the document, the OCCT backend, and the history panel together as a
/// [`Controller`] the viewport drives.
struct Modeler {
    doc: Document,
    backend: KernelBackend,
    ui: HistoryState,
    /// Snapshot stacks for undo/redo. The whole document is cloned per edit;
    /// cheap for these models and trivially correct.
    undo: Vec<Document>,
    redo: Vec<Document>,
    /// True while a continuous edit (e.g. dragging a value) is in progress, so
    /// the whole drag collapses into a single undo entry.
    editing: bool,
}

impl Modeler {
    fn new() -> Self {
        Self {
            doc: build_document(),
            backend: KernelBackend::default(),
            ui: HistoryState::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            editing: false,
        }
    }

    /// Record the pre-edit document on the undo stack and drop the redo stack.
    fn record_undo(&mut self, before: Document) {
        if self.undo.len() >= UNDO_LIMIT {
            self.undo.remove(0);
        }
        self.undo.push(before);
        self.redo.clear();
    }

    /// Restore the previous document state. Returns whether anything changed.
    fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.doc, prev));
                true
            }
            None => false,
        }
    }

    /// Reapply an undone change. Returns whether anything changed.
    fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.doc, next));
                true
            }
            None => false,
        }
    }
}

impl Controller for Modeler {
    fn ui(&mut self, ctx: &rmf_render::egui::Context) -> bool {
        use rmf_render::egui::Key;

        self.ui.can_undo = !self.undo.is_empty();
        self.ui.can_redo = !self.redo.is_empty();

        // Snapshot the pre-edit state only when not already mid-edit, so a drag
        // produces one undo entry rather than one per frame.
        let snapshot = (!self.editing).then(|| self.doc.clone());
        let resp = history_panel(ctx, &mut self.doc, &mut self.ui);
        let mut changed = resp.changed;

        if resp.changed {
            if let Some(before) = snapshot {
                self.record_undo(before);
            }
            self.editing = true;
        } else {
            self.editing = false;
        }

        let (undo_key, redo_key) = ctx.input(|i| {
            let cmd = i.modifiers.command;
            let undo = cmd && !i.modifiers.shift && i.key_pressed(Key::Z);
            let redo = cmd
                && ((i.modifiers.shift && i.key_pressed(Key::Z)) || i.key_pressed(Key::Y));
            (undo, redo)
        });

        if resp.undo || undo_key {
            changed |= self.undo();
        }
        if resp.redo || redo_key {
            changed |= self.redo();
        }

        changed
    }

    /// Regenerate the document through OCCT and tessellate every visible body
    /// into one mesh. Records per-feature errors for the panel.
    fn mesh(&mut self) -> MeshData {
        solve_sketches(&mut self.doc);
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
        RegenError::Invalid { reason, .. } => reason.to_string(),
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut modeler = Modeler::new();

    if std::env::args().any(|a| a == "--screenshot") {
        // Pre-select the constraint sketch so the screenshot shows its editor.
        modeler.ui.selected = modeler.doc.history.features().first().map(|f| f.id);
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

    /// Span of a mesh along an axis (max - min).
    fn span(mesh: &MeshData, axis: usize) -> f64 {
        let (min, max) = bounds(mesh);
        (max[axis] - min[axis]) as f64
    }

    #[test]
    fn full_part_regenerates_without_errors() {
        let mut m = Modeler::new();
        let mesh = m.mesh();
        assert!(!mesh.indices.is_empty());
        assert!(m.ui.errors.is_empty());
        // The default block is a 40 mm cube (the sketched rectangle extruded).
        assert!((span(&mesh, 0) - 40.0).abs() < 0.5);
        assert!((span(&mesh, 2) - 40.0).abs() < 0.5);
    }

    #[test]
    fn rollback_shows_the_untranslated_pin() {
        let mut m = Modeler::new();
        let _ = m.mesh();
        // Roll back before "Position hole": the filleted block and the
        // origin-anchored extruded pin (60 mm along +Z) are both visible.
        m.doc.set_rollback(5);
        let rolled = m.mesh();
        assert!(m.ui.errors.is_empty());
        assert!(span(&rolled, 2) > 50.0, "the 60 mm pin should reach past z=50");
    }

    #[test]
    fn editing_the_sketch_rescales_the_model() {
        let mut m = Modeler::new();
        let before = span(&m.mesh(), 0);
        // Edit the base rectangle profile; the change flows through extrude,
        // fillet, and the boolean.
        let sketch_id = m.doc.history.features()[0].id;
        m.doc.history.get_mut(sketch_id).unwrap().kind = FeatureKind::Sketch {
            plane: SketchPlane::Xy,
            profile: Profile::Rectangle {
                width: 80.0,
                height: 80.0,
            },
        };
        let after = span(&m.mesh(), 0);
        assert!((before - 40.0).abs() < 0.5 && (after - 80.0).abs() < 0.5);
    }

    #[test]
    fn constraint_sketch_solves_and_extrudes_to_a_prism() {
        // A rough constraint rectangle must be solved, turned into a polygon
        // face, and extruded into a correctly-sized prism — the full Phase 2
        // geometry path.
        let mut doc = Document::new("c-prism");
        let base = doc.add(
            "Sketch",
            FeatureKind::ConstraintSketch {
                plane: SketchPlane::Xy,
                sketch: constraint_rectangle(30.0, 20.0),
            },
        );
        doc.add(
            "Extrude",
            FeatureKind::Extrude {
                source: base,
                distance: 15.0,
            },
        );
        let mut m = Modeler {
            doc,
            backend: KernelBackend::default(),
            ui: HistoryState::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            editing: false,
        };
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty());
        assert!((span(&mesh, 0) - 30.0).abs() < 0.5, "x span {}", span(&mesh, 0));
        assert!((span(&mesh, 1) - 20.0).abs() < 0.5, "y span {}", span(&mesh, 1));
        assert!((span(&mesh, 2) - 15.0).abs() < 0.5, "z span {}", span(&mesh, 2));
    }

    #[test]
    fn undo_and_redo_restore_document_state() {
        let mut m = Modeler::new();
        let n0 = m.doc.history.len();

        // Simulate an edit: record the pre-edit state, then mutate.
        m.record_undo(m.doc.clone());
        m.doc.add("Sphere", FeatureKind::Sphere { radius: 10.0 });
        assert_eq!(m.doc.history.len(), n0 + 1);

        assert!(m.undo());
        assert_eq!(m.doc.history.len(), n0, "undo removes the added feature");

        assert!(m.redo());
        assert_eq!(m.doc.history.len(), n0 + 1, "redo restores it");

        assert!(!m.redo(), "nothing left to redo");
    }

    #[test]
    fn a_new_edit_clears_the_redo_stack() {
        let mut m = Modeler::new();

        m.record_undo(m.doc.clone());
        m.doc.add("Sphere", FeatureKind::Sphere { radius: 10.0 });
        assert!(m.undo());
        assert!(!m.redo.is_empty(), "an undone change is redoable");

        // A fresh edit must invalidate the redo history.
        m.record_undo(m.doc.clone());
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(5.0) });
        assert!(m.redo.is_empty(), "a new edit clears redo");
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
    fn sketch_extrude_makes_a_prism_of_the_right_size() {
        use rmf_core::{Profile, SketchPlane};
        let mut doc = Document::new("prism");
        let sketch = doc.add(
            "Sketch",
            FeatureKind::Sketch {
                plane: SketchPlane::Xy,
                profile: Profile::Rectangle {
                    width: 30.0,
                    height: 20.0,
                },
            },
        );
        doc.add(
            "Extrude",
            FeatureKind::Extrude {
                source: sketch,
                distance: 15.0,
            },
        );

        let mut m = Modeler {
            doc,
            backend: KernelBackend::default(),
            ui: HistoryState::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            editing: false,
        };
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty());
        let (min, max) = bounds(&mesh);
        let span = |a: usize| (max[a] - min[a]) as f64;
        // Rectangle 30x20 centered, extruded 15 along +Z.
        assert!((span(0) - 30.0).abs() < 0.5, "x span {}", span(0));
        assert!((span(1) - 20.0).abs() < 0.5, "y span {}", span(1));
        assert!((span(2) - 15.0).abs() < 0.5, "z span {}", span(2));
    }

    #[test]
    fn suppressing_a_referenced_feature_surfaces_an_error() {
        let mut m = Modeler::new();
        // Index 2 is "Fillet edges", which "Bore hole" subtracts from.
        let fillet_id = m.doc.history.features()[2].id;
        m.doc.history.set_suppressed(fillet_id, true);
        let _ = m.mesh();
        assert!(!m.ui.errors.is_empty());
    }
}
