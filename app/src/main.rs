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
    regenerate, BooleanOp, Constraint, Document, FeatureKind, LineId, PointId, Profile, RegenError,
    Sketch2d, SketchPlane, DVec3,
};
use rmf_kernel::KernelBackend;
use rmf_render::egui;
use rmf_render::{Controller, MeshData, ViewContext};
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

/// What clicking in the viewport does during a sketch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SketchTool {
    /// Place points, chaining line segments.
    Line,
    /// Pick points and lines to constrain.
    Select,
}

/// An in-progress interactive sketch. Kept out of the document until finished,
/// so the half-drawn (open) profile never triggers regeneration errors.
struct SketchSession {
    plane: SketchPlane,
    sketch: Sketch2d,
    tool: SketchTool,
    /// First point placed (the loop closes back to it).
    start: Option<PointId>,
    /// Most recent point (the next segment starts here).
    last: Option<PointId>,
    /// True once the loop has been closed.
    closed: bool,
    /// Selected entities (for applying constraints).
    selected_points: Vec<PointId>,
    selected_lines: Vec<LineId>,
    /// Remaining degrees of freedom from the last solve (for UI feedback).
    dof: usize,
}

/// A deferred edit to the sketch session, collected during egui drawing and
/// applied afterward to keep borrows simple.
enum SketchAction {
    SetTool(SketchTool),
    AddPoint([f64; 2]),
    CloseLoop,
    PickAt(egui::Pos2),
    Apply(Constraint),
    Finish,
    Cancel,
}

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
    /// Active interactive sketch, if drawing.
    sketch_session: Option<SketchSession>,
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
            sketch_session: None,
        }
    }

    /// Begin drawing a new sketch on `plane`.
    fn start_sketch(&mut self, plane: SketchPlane) {
        self.sketch_session = Some(SketchSession {
            plane,
            sketch: Sketch2d::new(),
            tool: SketchTool::Line,
            start: None,
            last: None,
            closed: false,
            selected_points: Vec::new(),
            selected_lines: Vec::new(),
            dof: 0,
        });
    }

    /// Re-solve the session sketch and record its remaining DOF.
    fn resolve_session(&mut self) {
        if let Some(session) = self.sketch_session.as_mut() {
            let solution = rmf_solver::solve_sketch(&mut session.sketch);
            session.dof = solution.degrees_of_freedom;
        }
    }

    /// Commit the current sketch as a `ConstraintSketch` feature if it forms a
    /// closed loop. Returns whether the document changed.
    fn finish_sketch(&mut self) -> bool {
        let Some(session) = self.sketch_session.take() else {
            return false;
        };
        if session.sketch.profile_loop().is_none() {
            // Not a closed loop — discard rather than commit something invalid.
            return false;
        }
        let before = self.doc.clone();
        let id = self.doc.add(
            "Sketch",
            FeatureKind::ConstraintSketch {
                plane: session.plane,
                sketch: session.sketch,
            },
        );
        self.record_undo(before);
        self.ui.selected = Some(id);
        true
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

    /// The sketch toolbar (top panel): tool toggle, constraint buttons gated by
    /// the current selection, and status. Reads state; pushes actions.
    fn sketch_top_bar(&self, ctx: &egui::Context, actions: &mut Vec<SketchAction>) {
        let Some(session) = self.sketch_session.as_ref() else {
            return;
        };
        let np = session.selected_points.len();
        let nl = session.selected_lines.len();

        #[allow(deprecated)]
        egui::Panel::top("sketch_toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("✓ Finish").clicked() {
                    actions.push(SketchAction::Finish);
                }
                if ui.button("✗ Cancel").clicked() {
                    actions.push(SketchAction::Cancel);
                }
                ui.separator();
                let line = session.tool == SketchTool::Line;
                if ui.selectable_label(line, "✏ Line").clicked() {
                    actions.push(SketchAction::SetTool(SketchTool::Line));
                }
                if ui.selectable_label(!line, "◉ Select").clicked() {
                    actions.push(SketchAction::SetTool(SketchTool::Select));
                }
                ui.separator();
                let dof = session.dof;
                let status = if dof == 0 {
                    egui::RichText::new("fully constrained").color(egui::Color32::from_rgb(120, 200, 120))
                } else {
                    egui::RichText::new(format!("{dof} dof")).weak()
                };
                ui.label(status);
            });

            ui.horizontal_wrapped(|ui| {
                let mut button = |ui: &mut egui::Ui, label, on, c: Option<Constraint>| {
                    if ui.add_enabled(on, egui::Button::new(label)).clicked() {
                        if let Some(c) = c {
                            actions.push(SketchAction::Apply(c));
                        }
                    }
                };
                let lines2 = two_lines(session);
                let points2 = two_points(session);
                button(ui, "Horizontal", nl == 1, session.selected_lines.first().map(|l| Constraint::Horizontal(*l)));
                button(ui, "Vertical", nl == 1, session.selected_lines.first().map(|l| Constraint::Vertical(*l)));
                button(ui, "Perpendicular", nl == 2, lines2.map(|(a, b)| Constraint::Perpendicular(a, b)));
                button(ui, "Parallel", nl == 2, lines2.map(|(a, b)| Constraint::Parallel(a, b)));
                button(ui, "Equal", nl == 2, lines2.map(|(a, b)| Constraint::EqualLength(a, b)));
                button(ui, "Coincident", np == 2, points2.map(|(a, b)| Constraint::Coincident(a, b)));
                button(ui, "Fixed", np == 1, session.selected_points.first().map(|p| Constraint::Fixed(*p)));
                let dist = distance_constraint(session);
                button(ui, "Distance", dist.is_some(), dist);
            });

            ui.label(
                egui::RichText::new(match session.tool {
                    SketchTool::Line => {
                        "Line: click the plane to add points; click the first point to close."
                    }
                    SketchTool::Select => "Select: click points/lines, then apply a constraint.",
                })
                .small()
                .weak(),
            );
        });
    }

    /// The sketch canvas overlay: draws entities (selection highlighted) and
    /// turns clicks into actions.
    fn sketch_overlay(
        &self,
        ctx: &egui::Context,
        view: &ViewContext,
        actions: &mut Vec<SketchAction>,
    ) {
        let Some(session) = self.sketch_session.as_ref() else {
            return;
        };
        let plane = session.plane;
        let o = plane.origin().to_array();
        let xd = plane.x_dir().to_array();
        let yd = plane.y_dir().to_array();
        let nd = plane.normal().to_array();
        let proj = |x: f64, y: f64| view.project_plane_point(o, xd, yd, [x, y]);
        let accent = egui::Color32::from_rgb(120, 180, 255);
        let selected = egui::Color32::from_rgb(255, 180, 80);

        #[allow(deprecated)]
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let response =
                    ui.interact(rect, ui.id().with("sketch_canvas"), egui::Sense::click());
                let painter = ui.painter_at(rect);

                for (i, line) in session.sketch.lines.iter().enumerate() {
                    let a = session.sketch.point(line.a);
                    let b = session.sketch.point(line.b);
                    if let (Some(pa), Some(pb)) = (proj(a.x, a.y), proj(b.x, b.y)) {
                        let sel = session.selected_lines.contains(&LineId(i));
                        let color = if sel { selected } else { accent };
                        painter.line_segment([pa, pb], egui::Stroke::new(if sel { 2.5 } else { 1.5 }, color));
                    }
                }
                for (i, p) in session.sketch.points.iter().enumerate() {
                    if let Some(sp) = proj(p.x, p.y) {
                        let sel = session.selected_points.contains(&PointId(i));
                        painter.circle_filled(sp, if sel { 4.5 } else { 3.0 }, if sel { selected } else { accent });
                    }
                }

                if session.tool == SketchTool::Line {
                    if let (Some(last), Some(cursor)) = (session.last, response.hover_pos()) {
                        let lp = session.sketch.point(last);
                        if let Some(a) = proj(lp.x, lp.y) {
                            painter.line_segment([a, cursor], egui::Stroke::new(1.0, egui::Color32::from_gray(150)));
                        }
                    }
                    if let Some(start) = session.start {
                        let s = session.sketch.point(start);
                        if let Some(sp) = proj(s.x, s.y) {
                            painter.circle_stroke(sp, 6.0, egui::Stroke::new(1.5, accent));
                        }
                    }
                }

                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        match session.tool {
                            SketchTool::Line => {
                                let near_start = !session.closed
                                    && session
                                        .start
                                        .map(|id| session.sketch.point(id))
                                        .and_then(|p| proj(p.x, p.y))
                                        .is_some_and(|sp| sp.distance(pos) < 10.0);
                                if near_start && session.sketch.lines.len() >= 2 {
                                    actions.push(SketchAction::CloseLoop);
                                } else if let Some(uv) = view.cursor_on_plane(pos, o, xd, yd, nd) {
                                    actions.push(SketchAction::AddPoint(uv));
                                }
                            }
                            SketchTool::Select => actions.push(SketchAction::PickAt(pos)),
                        }
                    }
                }
            });
    }

    /// Apply collected sketch actions. Returns whether the document changed.
    fn apply_sketch_actions(&mut self, actions: Vec<SketchAction>, view: &ViewContext) -> bool {
        let mut changed = false;
        for action in actions {
            match action {
                SketchAction::Cancel => {
                    self.sketch_session = None;
                    return changed;
                }
                SketchAction::Finish => {
                    changed |= self.finish_sketch();
                    return changed;
                }
                SketchAction::SetTool(tool) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.tool = tool;
                        s.selected_points.clear();
                        s.selected_lines.clear();
                    }
                }
                SketchAction::AddPoint([u, v]) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        let p = s.sketch.add_point(u, v);
                        if let Some(last) = s.last {
                            s.sketch.add_line(last, p);
                        }
                        if s.start.is_none() {
                            s.start = Some(p);
                        }
                        s.last = Some(p);
                    }
                }
                SketchAction::CloseLoop => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let (Some(last), Some(start)) = (s.last, s.start) {
                            s.sketch.add_line(last, start);
                        }
                        s.closed = true;
                        s.tool = SketchTool::Select;
                    }
                    self.resolve_session();
                }
                SketchAction::PickAt(pos) => self.pick_entity(pos, view),
                SketchAction::Apply(constraint) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.sketch.add_constraint(constraint);
                        s.selected_points.clear();
                        s.selected_lines.clear();
                    }
                    self.resolve_session();
                }
            }
        }
        changed
    }

    /// Pick the nearest point (or line) under `pos` and toggle its selection.
    fn pick_entity(&mut self, pos: egui::Pos2, view: &ViewContext) {
        let Some(s) = self.sketch_session.as_mut() else {
            return;
        };
        let o = s.plane.origin().to_array();
        let xd = s.plane.x_dir().to_array();
        let yd = s.plane.y_dir().to_array();
        let proj = |x: f64, y: f64| view.project_plane_point(o, xd, yd, [x, y]);

        let mut best_point: (f32, Option<PointId>) = (10.0, None);
        for (i, p) in s.sketch.points.iter().enumerate() {
            if let Some(sp) = proj(p.x, p.y) {
                let d = sp.distance(pos);
                if d < best_point.0 {
                    best_point = (d, Some(PointId(i)));
                }
            }
        }
        if let Some(id) = best_point.1 {
            toggle(&mut s.selected_points, id);
            return;
        }

        let mut best_line: (f32, Option<LineId>) = (8.0, None);
        for (i, line) in s.sketch.lines.iter().enumerate() {
            let a = s.sketch.point(line.a);
            let b = s.sketch.point(line.b);
            if let (Some(pa), Some(pb)) = (proj(a.x, a.y), proj(b.x, b.y)) {
                let d = dist_to_segment(pos, pa, pb);
                if d < best_line.0 {
                    best_line = (d, Some(LineId(i)));
                }
            }
        }
        if let Some(id) = best_line.1 {
            toggle(&mut s.selected_lines, id);
            return;
        }

        s.selected_points.clear();
        s.selected_lines.clear();
    }
}

fn two_lines(s: &SketchSession) -> Option<(LineId, LineId)> {
    (s.selected_lines.len() == 2).then(|| (s.selected_lines[0], s.selected_lines[1]))
}

fn two_points(s: &SketchSession) -> Option<(PointId, PointId)> {
    (s.selected_points.len() == 2).then(|| (s.selected_points[0], s.selected_points[1]))
}

/// A Distance constraint for the current selection (2 points or 1 line), using
/// the present length, rounded to 0.1 mm, as its initial value.
fn distance_constraint(s: &SketchSession) -> Option<Constraint> {
    let (a, b) = if let Some((a, b)) = two_points(s) {
        (a, b)
    } else if let Some(&l) = s.selected_lines.first().filter(|_| s.selected_lines.len() == 1) {
        let line = s.sketch.line(l);
        (line.a, line.b)
    } else {
        return None;
    };
    let (pa, pb) = (s.sketch.point(a), s.sketch.point(b));
    let length = (pa.x - pb.x).hypot(pa.y - pb.y);
    Some(Constraint::Distance(a, b, (length * 10.0).round() / 10.0))
}

fn toggle<T: PartialEq>(v: &mut Vec<T>, item: T) {
    if let Some(i) = v.iter().position(|x| *x == item) {
        v.remove(i);
    } else {
        v.push(item);
    }
}

fn dist_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    let t = if len_sq > 0.0 {
        ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
}

impl Controller for Modeler {
    fn ui(&mut self, ctx: &egui::Context, view: &ViewContext) -> bool {
        use egui::Key;
        let mut changed = false;
        let mut sketch_actions: Vec<SketchAction> = Vec::new();

        // --- Sketch top bar (before the side panel, so it spans full width) ---
        if self.sketch_session.is_some() {
            self.sketch_top_bar(ctx, &mut sketch_actions);
        }

        // --- History panel + undo/redo ---
        self.ui.can_undo = !self.undo.is_empty();
        self.ui.can_redo = !self.redo.is_empty();
        // Snapshot the pre-edit state only when not already mid-edit, so a drag
        // produces one undo entry rather than one per frame.
        let snapshot = (!self.editing).then(|| self.doc.clone());
        let resp = history_panel(ctx, &mut self.doc, &mut self.ui);
        if resp.changed {
            if let Some(before) = snapshot {
                self.record_undo(before);
            }
            self.editing = true;
            changed = true;
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

        // The "Sketch" tool in the Add panel enters interactive draw mode.
        if resp.start_sketch && self.sketch_session.is_none() {
            self.start_sketch(SketchPlane::Xy);
        }

        // --- Sketch canvas overlay (after the side panel) ---
        if self.sketch_session.is_some() {
            self.sketch_overlay(ctx, view, &mut sketch_actions);
        }

        // --- Apply collected sketch edits ---
        changed |= self.apply_sketch_actions(sketch_actions, view);

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

    // Verification aid: draw a closed triangle and finish it, mirroring the
    // interactive flow, so we can confirm the Extrude button enables afterward.
    if std::env::args().any(|a| a == "--sketch-demo") {
        let mut empty = Document::new("draw");
        std::mem::swap(&mut modeler.doc, &mut empty); // start from a clean doc
        modeler.start_sketch(SketchPlane::Xy);
        if let Some(s) = modeler.sketch_session.as_mut() {
            let p0 = s.sketch.add_point(-15.0, -12.0);
            let p1 = s.sketch.add_point(15.0, -12.0);
            let p2 = s.sketch.add_point(0.0, 14.0);
            s.sketch.add_line(p0, p1);
            s.sketch.add_line(p1, p2);
            s.sketch.add_line(p2, p0); // close
            s.start = Some(p0);
            s.last = Some(p2);
        }
        modeler.finish_sketch(); // commits + selects the sketch
        let path = "out/sketch-demo.png";
        std::fs::create_dir_all("out")?;
        rmf_render::screenshot(modeler, 1280, 820, path)?;
        println!("wrote {path}");
        return Ok(());
    }

    // Verification aid: a skewed quad with Horizontal/Vertical constraints
    // applied should solve to an axis-aligned rectangle.
    if std::env::args().any(|a| a == "--constrain-demo") {
        // Keep the default part so the camera frames the scene and the sketch
        // overlay (drawn at the part's scale) is visible.
        modeler.start_sketch(SketchPlane::Xy);
        if let Some(s) = modeler.sketch_session.as_mut() {
            let p0 = s.sketch.add_point(-18.0, -15.0);
            let p1 = s.sketch.add_point(16.0, -12.0);
            let p2 = s.sketch.add_point(20.0, 13.0);
            let p3 = s.sketch.add_point(-15.0, 17.0);
            let bottom = s.sketch.add_line(p0, p1);
            let right = s.sketch.add_line(p1, p2);
            let top = s.sketch.add_line(p2, p3);
            let left = s.sketch.add_line(p3, p0);
            s.closed = true;
            s.tool = SketchTool::Select;
            for c in [
                Constraint::Fixed(p0),
                Constraint::Horizontal(bottom),
                Constraint::Vertical(right),
                Constraint::Horizontal(top),
                Constraint::Vertical(left),
            ] {
                s.sketch.add_constraint(c);
            }
            s.selected_lines = vec![bottom, top]; // show selection + enable Equal/Parallel
        }
        modeler.resolve_session();
        let path = "out/constrain-demo.png";
        std::fs::create_dir_all("out")?;
        rmf_render::screenshot(modeler, 1280, 820, path)?;
        println!("wrote {path}");
        return Ok(());
    }

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
            sketch_session: None,
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
            sketch_session: None,
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
