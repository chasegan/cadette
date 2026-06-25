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

use std::collections::HashMap;

use rmf_core::{
    regenerate, BooleanOp, Constraint, Document, EdgeAnchor, FaceAnchor, FeatureId, FeatureKind, LineId,
    PointId, Profile, RegenError, Sketch2d, SketchPlane, DVec3,
};
use rmf_interaction::Selection;
use rmf_kernel::{KernelBackend, Solid};
use rmf_render::egui;
use rmf_render::{
    Controller, Gizmo, GizmoHandle, Highlights, MeshData, Pick, TransformDelta, ViewContext,
};
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

/// An in-progress push/pull drag: the body being modified, the face anchor, and
/// the feature once it's been created (on the first non-trivial drag).
#[derive(Clone, Copy)]
struct Manipulation {
    source: FeatureId,
    anchor: FaceAnchor,
    feature: Option<FeatureId>,
}

/// An in-progress gizmo transform: the body being moved/rotated, the feature
/// once the first drag created it, and (for rotation) the fixed pivot — the
/// body's AABB center drifts as it turns, so the gizmo and rotation axis stay
/// pinned at the center captured when the drag began.
#[derive(Clone, Copy)]
struct TransformDrag {
    source: FeatureId,
    feature: Option<FeatureId>,
    pivot: Option<[f64; 3]>,
}

/// Bounding-box center of a flat `[x,y,z,...]` position array.
fn positions_center(positions: &[f32]) -> [f64; 3] {
    if positions.is_empty() {
        return [0.0; 3];
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions.chunks_exact(3) {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    [
        ((min[0] + max[0]) * 0.5) as f64,
        ((min[1] + max[1]) * 0.5) as f64,
        ((min[2] + max[2]) * 0.5) as f64,
    ]
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
    /// Viewport selection (clicked + hovered entity, and a selected face's
    /// plane). Transient — ids are per-regeneration.
    selection: Selection,
    /// Visible bodies from the last regeneration, kept so a picked face's plane
    /// can be queried. `face_ranges[i]` / `edge_ranges[i]` are the global face /
    /// edge id where body `i` starts.
    bodies: Vec<Solid>,
    face_ranges: Vec<u32>,
    edge_ranges: Vec<u32>,
    /// Bounding-box center of each visible body (aligned with `ui.visible`), the
    /// gizmo origin for that body.
    body_centers: Vec<[f64; 3]>,
    /// World-space anchor point for each selected edge (keyed by its global pick
    /// id), captured at click time so each edge in a multi-selection keeps its
    /// own durable fillet anchor.
    edge_anchors: HashMap<u32, DVec3>,
    /// Active push/pull drag, if any.
    manipulation: Option<Manipulation>,
    /// Active gizmo move drag, if any.
    transform: Option<TransformDrag>,
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
            selection: Selection::default(),
            bodies: Vec::new(),
            face_ranges: Vec::new(),
            edge_ranges: Vec::new(),
            body_centers: Vec::new(),
            edge_anchors: HashMap::new(),
            manipulation: None,
            transform: None,
        }
    }

    /// Body index + local face index for a global picked face id.
    fn locate_face(&self, global: u32) -> Option<(usize, u32)> {
        let i = self.face_ranges.iter().rposition(|&start| start <= global)?;
        Some((i, global - self.face_ranges[i]))
    }

    /// Body index + local edge index for a global picked edge id.
    fn locate_edge(&self, global: u32) -> Option<(usize, u32)> {
        let i = self.edge_ranges.iter().rposition(|&start| start <= global)?;
        Some((i, global - self.edge_ranges[i]))
    }

    /// The body feature id behind the current viewport selection, if any.
    fn selected_body(&self) -> Option<FeatureId> {
        let body_index = match self.selection.primary()? {
            Pick::Face(g) => self.locate_face(g)?.0,
            Pick::Edge(g) => self.locate_edge(g)?.0,
        };
        self.ui.visible.get(body_index).copied()
    }

    /// A short readout for an in-progress gizmo drag (angle / distance), shown
    /// floating next to the gizmo so the manipulation has a numeric value.
    fn transform_readout(&self) -> Option<String> {
        let id = self.transform?.feature?;
        match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::Rotate { angle, .. }) => Some(format!("{:.0}°", angle.to_degrees())),
            Some(FeatureKind::Translate { offset, .. }) => {
                Some(format!("{:.1} mm", offset.length()))
            }
            _ => None,
        }
    }

    /// The gizmo origin — the bounding-box center of a single selected face's
    /// body (the only selection that shows the gizmo).
    fn gizmo_origin(&self) -> Option<[f64; 3]> {
        match self.selection.selected() {
            [Pick::Face(_)] => {
                let body = self.selected_body()?;
                let idx = self.ui.visible.iter().position(|&f| f == body)?;
                self.body_centers.get(idx).copied()
            }
            _ => None,
        }
    }

    /// The plane of a picked face, if it is planar.
    fn face_plane(&self, global: u32) -> Option<SketchPlane> {
        let (i, local) = self.locate_face(global)?;
        let (origin, x, y) = self.bodies.get(i)?.face_plane(local).ok().flatten()?;
        Some(SketchPlane::from_frame(origin.into(), x.into(), y.into()))
    }

    /// Start a sketch on the currently selected face's plane.
    fn sketch_on_selected_face(&mut self) {
        if let Some(plane) = self.selection.face_plane {
            self.start_sketch(plane);
            self.selection.clear();
        }
    }

    /// Fillet the currently selected edges (each anchored at its clicked point),
    /// as one feature. Returns true if a feature was added.
    fn fillet_selected_edges(&mut self) -> bool {
        let Some(source) = self.selected_body() else {
            return false;
        };
        // Gather an anchor per selected edge, in click order.
        let edges: Vec<EdgeAnchor> = self
            .selection
            .selected()
            .iter()
            .filter_map(|pick| match pick {
                Pick::Edge(g) => self.edge_anchors.get(g).copied(),
                _ => None,
            })
            .map(|point| EdgeAnchor { point })
            .collect();
        if edges.is_empty() {
            return false;
        }

        self.record_undo(self.doc.clone());
        let name = match edges.len() {
            1 => "Fillet edge".to_string(),
            n => format!("Fillet {n} edges"),
        };
        let id = self.doc.add(name, FeatureKind::Fillet { source, edges, radius: 2.0 });
        self.ui.selected = Some(id);
        self.selection.clear();
        self.edge_anchors.clear();
        true
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

        // --- Selection action bar: contextual actions for the picked entity ---
        if self.sketch_session.is_none() && self.selection.can_sketch_on_face() {
            let mut start = false;
            #[allow(deprecated)]
            egui::Panel::top("selection_bar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Planar face selected");
                    ui.separator();
                    start = ui.button("✏ Sketch on this face").clicked();
                });
            });
            if start {
                self.sketch_on_selected_face();
            }
        } else if self.sketch_session.is_none() && self.selection.is_edges() {
            let n = self.selection.selected().len();
            let (label, action) = if n == 1 {
                ("1 edge selected".to_string(), "⌒ Fillet this edge".to_string())
            } else {
                (format!("{n} edges selected"), format!("⌒ Fillet these {n} edges"))
            };
            let mut fillet = false;
            #[allow(deprecated)]
            egui::Panel::top("selection_bar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(label);
                    ui.separator();
                    fillet = ui.button(action).clicked();
                    ui.separator();
                    ui.weak("⇧/⌘-click to add edges");
                });
            });
            if fillet {
                changed |= self.fillet_selected_edges();
            }
        }

        // --- Gizmo readout: the live angle / distance next to the gizmo ---
        if let (Some(text), Some(origin)) =
            (self.transform_readout(), self.gizmo().map(|g| g.origin))
        {
            if let Some(pos) = view.project(origin) {
                egui::Area::new(egui::Id::new("gizmo_readout"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(pos + egui::vec2(16.0, -16.0))
                    .show(ctx, |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.label(egui::RichText::new(text).strong().size(15.0));
                        });
                    });
            }
        }

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
            .map(|e| (e.feature(), error_message(e, &self.doc)))
            .collect();
        self.ui.visible = regen.visible().to_vec();
        let bodies = regen.into_visible_bodies();

        let mut mesh = MeshData::default();
        let mut face_offset = 0u32; // keep face/edge ids unique across bodies
        let mut edge_offset = 0u32;
        self.face_ranges.clear();
        self.edge_ranges.clear();
        self.body_centers.clear();
        for body in &bodies {
            self.face_ranges.push(face_offset);
            self.edge_ranges.push(edge_offset);
            match body.tessellate(DEFLECTION_MM) {
                Ok(part) => {
                    self.body_centers.push(positions_center(&part.positions));
                    let base = mesh.vertices.len() as u32;
                    let face_ids: Vec<u32> =
                        part.face_ids.iter().map(|f| f + face_offset).collect();
                    mesh.vertices.extend(rmf_render::interleave(
                        &part.positions,
                        &part.normals,
                        &face_ids,
                    ));
                    mesh.indices.extend(part.indices.iter().map(|i| i + base));
                    face_offset += part.face_ids.iter().copied().max().unwrap_or(0) + 1;

                    // Crisp edges (edge ids offset to stay unique per body).
                    let edge_base = mesh.edge_vertices.len() as u32;
                    let edge_ids: Vec<u32> =
                        part.edge_ids.iter().map(|e| e + edge_offset).collect();
                    mesh.edge_vertices.extend(rmf_render::interleave_edges(
                        &part.edge_positions,
                        &edge_ids,
                    ));
                    mesh.edge_indices
                        .extend(part.edge_indices.iter().map(|i| i + edge_base));
                    edge_offset += part.edge_ids.iter().copied().max().unwrap_or(0) + 1;
                }
                Err(e) => {
                    self.body_centers.push([0.0; 3]); // keep aligned with visible
                    self.ui.errors.push((rmf_core::FeatureId(0), e.to_string()));
                }
            }
        }
        self.bodies = bodies;
        mesh
    }

    fn highlights(&self) -> Highlights {
        Highlights {
            selected: self.selection.selected().to_vec(),
            hovered: self.selection.hovered,
        }
    }

    fn on_pick(&mut self, pick: Option<Pick>, point: Option<[f64; 3]>, additive: bool) {
        let face_plane = match pick {
            Some(Pick::Face(global)) => self.face_plane(global),
            _ => None,
        };
        self.selection.select(pick, additive, face_plane);
        // Remember the clicked point on a picked edge as its durable fillet
        // anchor, then drop anchors for edges no longer in the selection.
        if let (Some(Pick::Edge(g)), Some(p)) = (pick, point) {
            self.edge_anchors.insert(g, DVec3::new(p[0], p[1], p[2]));
        }
        let live: std::collections::HashSet<u32> = self
            .selection
            .selected()
            .iter()
            .filter_map(|pick| match pick {
                Pick::Edge(g) => Some(*g),
                _ => None,
            })
            .collect();
        self.edge_anchors.retain(|g, _| live.contains(g));
        // Clicking a body in the viewport makes it the operation target: select
        // its feature in the history (which is what the toolbar acts on).
        self.ui.selected = self.selected_body();
    }

    fn on_hover(&mut self, pick: Option<Pick>) {
        self.selection.hover(pick);
    }

    fn wants_picking(&self) -> bool {
        // In sketch mode, viewport clicks draw/select 2D entities instead.
        self.sketch_session.is_none()
    }

    fn start_manipulation(
        &mut self,
        pick: Pick,
        point: [f64; 3],
        eye: [f64; 3],
    ) -> Option<([f64; 3], [f64; 3])> {
        let Pick::Face(global) = pick else {
            return None; // only planar faces push/pull (for now)
        };
        // Anchor at the clicked point (guaranteed on the face, even with holes);
        // take the normal from the face plane.
        let point = DVec3::new(point[0], point[1], point[2]);
        let mut normal = self.face_plane(global)?.normal().normalize_or_zero();
        // Orient the normal outward (toward the camera) so dragging away from
        // the solid is a positive push.
        let eye = DVec3::new(eye[0], eye[1], eye[2]);
        if normal.dot(eye - point) < 0.0 {
            normal = -normal;
        }
        let (body_index, _) = self.locate_face(global)?;
        let source = *self.ui.visible.get(body_index)?;

        self.manipulation = Some(Manipulation {
            source,
            anchor: FaceAnchor { point, normal },
            feature: None,
        });
        Some((point.to_array(), normal.to_array()))
    }

    fn update_manipulation(&mut self, distance: f64) -> bool {
        let Some(mut m) = self.manipulation else {
            return false;
        };
        match m.feature {
            None => {
                // First real drag: record one undo entry, then add the feature.
                self.record_undo(self.doc.clone());
                let id = self.doc.add(
                    "Push/Pull",
                    FeatureKind::PushPull {
                        source: m.source,
                        anchor: m.anchor,
                        distance,
                    },
                );
                m.feature = Some(id);
            }
            Some(id) => {
                if let Some(FeatureKind::PushPull { distance: d, .. }) =
                    self.doc.history.get_mut(id).map(|f| &mut f.kind)
                {
                    *d = distance;
                }
            }
        }
        self.manipulation = Some(m);
        self.selection.clear();
        true
    }

    fn finish_manipulation(&mut self, commit: bool) {
        let Some(m) = self.manipulation.take() else {
            return;
        };
        let Some(id) = m.feature else {
            return; // never dragged far enough to create a feature
        };
        let distance = match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::PushPull { distance, .. }) => *distance,
            _ => 0.0,
        };
        // Discard a cancelled or no-op push/pull (and the undo it recorded).
        if !commit || distance.abs() < 1e-3 {
            self.doc.history.remove(id);
            self.undo.pop();
        }
    }

    fn gizmo(&self) -> Option<Gizmo> {
        if self.sketch_session.is_some() {
            return None;
        }
        // During a rotation the gizmo stays pinned at the captured pivot (the
        // body's AABB center drifts as it turns).
        if let Some(t) = self.transform {
            if let Some(pivot) = t.pivot {
                return Some(Gizmo { origin: pivot });
            }
        }
        self.gizmo_origin().map(|origin| Gizmo { origin })
    }

    fn start_transform(&mut self, handle: GizmoHandle) {
        let Some(source) = self.selected_body() else {
            return;
        };
        // Rotation pins the gizmo/axis at the body center captured now.
        let pivot = match handle {
            GizmoHandle::RotateAxis(_) => self.gizmo_origin(),
            _ => None,
        };
        self.transform = Some(TransformDrag { source, feature: None, pivot });
    }

    fn update_transform(&mut self, delta: TransformDelta) -> bool {
        let Some(mut t) = self.transform else {
            return false;
        };
        let (name, kind) = match delta {
            TransformDelta::Translate(offset) => (
                "Move",
                FeatureKind::Translate {
                    source: t.source,
                    offset: DVec3::from_array(offset),
                },
            ),
            TransformDelta::Rotate { axis, angle } => (
                "Rotate",
                FeatureKind::Rotate {
                    source: t.source,
                    axis: DVec3::from_array(axis),
                    angle,
                    center: DVec3::from_array(t.pivot.unwrap_or([0.0; 3])),
                },
            ),
        };
        match t.feature {
            None => {
                // First real drag: one undo entry, then add the feature.
                self.record_undo(self.doc.clone());
                let id = self.doc.add(name, kind);
                t.feature = Some(id);
                self.ui.selected = Some(id);
            }
            Some(id) => {
                if let Some(k) = self.doc.history.get_mut(id).map(|f| &mut f.kind) {
                    *k = kind;
                }
            }
        }
        self.transform = Some(t);
        true
    }

    fn finish_transform(&mut self, commit: bool) {
        let Some(t) = self.transform.take() else {
            return;
        };
        let Some(id) = t.feature else {
            return; // never dragged far enough to create a feature
        };
        // Discard a cancelled or no-op transform (and the undo it recorded).
        let nonzero = match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::Translate { offset, .. }) => offset.length() >= 1e-3,
            Some(FeatureKind::Rotate { angle, .. }) => angle.abs() >= 1e-4,
            _ => true,
        };
        if !commit || !nonzero {
            self.doc.history.remove(id);
            self.undo.pop();
        }
    }
}

fn error_message(err: &RegenError<rmf_kernel::KernelError>, doc: &Document) -> String {
    match err {
        RegenError::Backend { source, .. } => source.to_string(),
        RegenError::MissingInput { input, .. } => match doc.history.get(*input) {
            Some(f) => format!("needs '{}' (suppressed)", f.name),
            None => "needs a step that was deleted".to_string(),
        },
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

    // Verification aid: highlight a face by id to confirm the shader tint path.
    if std::env::args().any(|a| a == "--highlight-demo") {
        modeler.selection.select(Some(Pick::Face(0)), false, None); // strong face
        modeler.selection.hover(Some(Pick::Edge(0))); // hovered edge
        let path = "out/highlight-demo.png";
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
            selection: Selection::default(),
            bodies: Vec::new(),
            face_ranges: Vec::new(),
            edge_ranges: Vec::new(),
            body_centers: Vec::new(),
            edge_anchors: HashMap::new(),
            manipulation: None,
            transform: None,
        };
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty());
        assert!((span(&mesh, 0) - 30.0).abs() < 0.5, "x span {}", span(&mesh, 0));
        assert!((span(&mesh, 1) - 20.0).abs() < 0.5, "y span {}", span(&mesh, 1));
        assert!((span(&mesh, 2) - 15.0).abs() < 0.5, "z span {}", span(&mesh, 2));
    }

    #[test]
    fn sketch_on_a_custom_plane_extrudes_there() {
        // A rectangle on a plane lifted to z = 20, extruded +10, should produce
        // a solid spanning z in [20, 30] — the sketch-on-face geometry path.
        let plane = SketchPlane::from_frame(
            DVec3::new(0.0, 0.0, 20.0),
            DVec3::X,
            DVec3::Y,
        );
        let mut doc = Document::new("on-face");
        let base = doc.add(
            "Sketch",
            FeatureKind::ConstraintSketch {
                plane,
                sketch: constraint_rectangle(20.0, 20.0),
            },
        );
        doc.add(
            "Extrude",
            FeatureKind::Extrude {
                source: base,
                distance: 10.0,
            },
        );
        let mut m = Modeler::new();
        m.doc = doc;
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty());
        let (min, max) = bounds(&mesh);
        assert!((min[2] - 20.0).abs() < 0.5, "min z {}", min[2]);
        assert!((max[2] - 30.0).abs() < 0.5, "max z {}", max[2]);
    }

    #[test]
    fn push_pull_feature_extends_a_box() {
        use rmf_core::FaceAnchor;
        let mut doc = Document::new("pp");
        let b = doc.add(
            "Box",
            FeatureKind::Box {
                size: DVec3::new(10.0, 10.0, 10.0),
            },
        );
        doc.add(
            "Push/Pull",
            FeatureKind::PushPull {
                source: b,
                anchor: FaceAnchor {
                    point: DVec3::new(5.0, 5.0, 10.0),
                    normal: DVec3::Z,
                },
                distance: 5.0,
            },
        );
        let mut m = Modeler::new();
        m.doc = doc;
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (_min, max) = bounds(&mesh);
        assert!((max[2] - 15.0).abs() < 0.5, "max z {}", max[2]);
    }

    #[test]
    fn gizmo_shows_on_face_selection_and_drag_moves_the_body() {
        use rmf_render::{Axis3, GizmoHandle, TransformDelta};
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();

        // No selection → no gizmo. Selecting a face shows it at the body center.
        assert!(m.gizmo().is_none());
        m.on_pick(Some(Pick::Face(0)), None, false);
        let g = m.gizmo().expect("gizmo on face selection");
        assert!((g.origin[0] - 5.0).abs() < 0.1 && (g.origin[2] - 5.0).abs() < 0.1);

        // A drag along X creates one Move feature, then edits it in place.
        let n0 = m.doc.history.len();
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::X));
        assert!(m.update_transform(TransformDelta::Translate([5.0, 0.0, 0.0])));
        assert_eq!(m.doc.history.len(), n0 + 1);
        assert!(m.update_transform(TransformDelta::Translate([8.0, 0.0, 0.0]))); // edits in place
        assert_eq!(m.doc.history.len(), n0 + 1);
        m.finish_transform(true);

        // The committed Move carries the final offset and rebuilds shifted +8 X.
        let moved = m.doc.history.features().iter().any(|f| {
            matches!(&f.kind, FeatureKind::Translate { offset, .. } if (offset.x - 8.0).abs() < 1e-6)
        });
        assert!(moved, "expected a Move with offset x=8");
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((min[0] - 8.0).abs() < 0.1 && (max[0] - 18.0).abs() < 0.1, "x {min:?}..{max:?}");
    }

    #[test]
    fn gizmo_ring_drag_rotates_the_body_about_a_pinned_pivot() {
        use rmf_render::{Axis3, GizmoHandle, TransformDelta};
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        // A 20x10x10 bar so a 90° turn about Z visibly swaps its footprint.
        m.doc.add("Bar", FeatureKind::Box { size: DVec3::new(20.0, 10.0, 10.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);
        let pivot = m.gizmo().unwrap().origin; // AABB center (10,5,5)

        let n0 = m.doc.history.len();
        m.start_transform(GizmoHandle::RotateAxis(Axis3::Z));
        let quarter = std::f64::consts::FRAC_PI_2;
        assert!(m.update_transform(TransformDelta::Rotate {
            axis: [0.0, 0.0, 1.0],
            angle: quarter,
        }));
        assert_eq!(m.doc.history.len(), n0 + 1);

        // The gizmo stays pinned at the original pivot mid-rotation.
        assert_eq!(m.gizmo().unwrap().origin, pivot);
        m.finish_transform(true);

        // The Rotate feature carries the pinned center, and the bar's footprint
        // swapped: now 10 wide in X, 20 deep in Y.
        let rot = m.doc.history.features().iter().find_map(|f| match &f.kind {
            FeatureKind::Rotate { center, angle, .. } => Some((*center, *angle)),
            _ => None,
        });
        let (center, angle) = rot.expect("a Rotate feature");
        assert!((angle - quarter).abs() < 1e-9);
        assert!((center.x - pivot[0]).abs() < 1e-9);
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((max[0] - min[0] - 10.0).abs() < 0.2, "x span {}", max[0] - min[0]);
        assert!((max[1] - min[1] - 20.0).abs() < 0.2, "y span {}", max[1] - min[1]);
    }

    #[test]
    fn gizmo_drag_with_no_movement_is_discarded() {
        use rmf_render::{Axis3, GizmoHandle};
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);
        let n0 = m.doc.history.len();

        // A click on the arrow that never really moved adds then discards.
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::X));
        m.finish_transform(false);
        assert_eq!(m.doc.history.len(), n0);
        assert!(m.transform.is_none());
    }

    #[test]
    fn clicking_a_body_targets_it_for_operations() {
        // Two separate boxes; clicking one should make it the toolbar target.
        let mut doc = Document::new("two");
        let a = doc.add("Box A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b = doc.add("Box B", FeatureKind::Box { size: DVec3::splat(10.0) });
        let mut m = Modeler::new();
        m.doc = doc;
        let _ = m.mesh(); // populate visible + ranges
        assert_eq!(m.ui.visible, vec![a, b]);

        // Face id 0 belongs to the first visible body (box A).
        m.on_pick(Some(Pick::Face(0)), None, false);
        assert_eq!(m.ui.selected, Some(a));
        assert_eq!(m.selected_body(), Some(a));

        // Clicking an edge records its world point as a fillet anchor.
        m.on_pick(Some(Pick::Edge(0)), Some([1.0, 2.0, 3.0]), false);
        assert_eq!(m.selected_body(), Some(a));
        assert_eq!(m.edge_anchors.get(&0), Some(&DVec3::new(1.0, 2.0, 3.0)));

        // Clicking empty space clears the target and the anchors.
        m.on_pick(None, None, false);
        assert_eq!(m.ui.selected, None);
        assert!(m.edge_anchors.is_empty());
    }

    #[test]
    fn additive_click_accumulates_edge_anchors() {
        // One box; ⇧-click two edges, both anchors are retained for a fillet.
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();

        m.on_pick(Some(Pick::Edge(0)), Some([0.0, 5.0, 10.0]), false);
        m.on_pick(Some(Pick::Edge(2)), Some([10.0, 5.0, 10.0]), true); // additive
        assert_eq!(m.selection.selected().len(), 2);
        assert_eq!(m.edge_anchors.len(), 2);

        // Toggling edge 0 back off drops its anchor.
        m.on_pick(Some(Pick::Edge(0)), Some([0.0, 5.0, 10.0]), true);
        assert_eq!(m.selection.selected(), &[Pick::Edge(2)]);
        assert_eq!(m.edge_anchors.len(), 1);
        assert!(m.edge_anchors.contains_key(&2));
    }

    #[test]
    fn filleting_selected_edges_adds_a_feature_that_rebuilds() {
        // A box; fillet two of its top edges in one feature.
        let mut doc = Document::new("one");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let mut m = Modeler::new();
        m.doc = doc;
        let _ = m.mesh();

        // Simulate picking two edges, then invoking the action-bar fillet.
        m.on_pick(Some(Pick::Edge(0)), Some([0.0, 5.0, 10.0]), false);
        m.on_pick(Some(Pick::Edge(2)), Some([10.0, 5.0, 10.0]), true);
        m.ui.selected = Some(b);
        assert!(m.selection.is_edges());
        assert!(m.fillet_selected_edges());

        // The new feature regenerates without error and yields geometry.
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert!(!mesh.vertices.is_empty());
        // Exactly one Fillet feature was added, carrying two edges.
        let fillet = m
            .doc
            .history
            .features()
            .iter()
            .find_map(|f| match &f.kind {
                FeatureKind::Fillet { edges, .. } => Some(edges.len()),
                _ => None,
            });
        assert_eq!(fillet, Some(2));
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
            selection: Selection::default(),
            bodies: Vec::new(),
            face_ranges: Vec::new(),
            edge_ranges: Vec::new(),
            body_centers: Vec::new(),
            edge_anchors: HashMap::new(),
            manipulation: None,
            transform: None,
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
    fn deleting_the_fillet_heals_the_bore_hole() {
        // Reproduces the reported bug: deleting "Fillet edges" must rewire the
        // bore hole to the pre-fillet body, not error with a missing input.
        let mut m = Modeler::new();
        let fillet_id = m.doc.history.features()[2].id; // "Fillet edges"
        m.doc.history.remove(fillet_id);
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors after delete: {:?}", m.ui.errors);
        assert!(!mesh.indices.is_empty());
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
