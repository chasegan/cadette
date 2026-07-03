//! The interactive sketch subsystem: the in-progress [`SketchSession`], its
//! tools and deferred [`SketchAction`]s, the sketch toolbar and canvas overlay,
//! and the session lifecycle (`start_sketch` … `finish_sketch`).

use cdt_core::{Constraint, FeatureId, FeatureKind, LineId, PointId, Sketch2d, SketchPlane};
use cdt_render::egui;
use cdt_render::ViewContext;

use crate::Modeler;

/// What clicking in the viewport does during a sketch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SketchTool {
    /// Place points, chaining line segments.
    Line,
    /// Pick points and lines to constrain; drag points to move them.
    Select,
    /// Click a segment to insert a vertex (split it).
    Insert,
    /// Place anchors: click = corner, click-drag = smooth (pull out a handle).
    Pen,
}

/// An in-progress interactive sketch. Kept out of the document until finished,
/// so the half-drawn (open) profile never triggers regeneration errors.
pub(crate) struct SketchSession {
    pub(crate) plane: SketchPlane,
    pub(crate) sketch: Sketch2d,
    pub(crate) tool: SketchTool,
    /// First point placed (the loop closes back to it).
    pub(crate) start: Option<PointId>,
    /// Most recent point (the next segment starts here).
    pub(crate) last: Option<PointId>,
    /// True once the loop has been closed.
    pub(crate) closed: bool,
    /// Selected entities (for applying constraints).
    pub(crate) selected_points: Vec<PointId>,
    pub(crate) selected_lines: Vec<LineId>,
    /// Selected bezier segments, by index (for straighten).
    pub(crate) selected_beziers: Vec<usize>,
    /// Remaining degrees of freedom from the last solve (for UI feedback).
    pub(crate) dof: usize,
    /// True when the last solve found contradictory constraints (redundant
    /// equations that failed to converge) — surfaced as "over-constrained".
    pub(crate) conflicted: bool,
    /// The feature being re-edited, if this session reopened an existing sketch
    /// — `finish` writes back to it in place. `None` for a brand-new sketch.
    pub(crate) editing: Option<FeatureId>,
    /// When set, this session draws an OPEN sweep PATH (no auto-close); `finish`
    /// builds a `Sweep` of this profile feature along the drawn path.
    pub(crate) sweep_profile: Option<FeatureId>,
    /// The point currently being dragged (Select tool), if any.
    pub(crate) dragging: Option<PointId>,
    /// The bezier control handle being dragged: `(bezier index, is_c1)`.
    pub(crate) dragging_handle: Option<(usize, bool)>,
    /// Whether the handle drag should keep the node smooth (mirror the partner
    /// handle). Captured at grab from the node's current geometry; cleared the
    /// moment Alt is held, so Alt-drag breaks the join into a corner.
    pub(crate) dragging_smooth: bool,
    /// Pen tool: the out-handle of `last` (the next segment's `c1`); `None` at a
    /// corner anchor.
    pub(crate) pen_out: Option<[f64; 2]>,
    /// Pen tool: where a smooth-anchor drag began (the anchor goes here; the
    /// drag pulls out its handle).
    pub(crate) pen_press: Option<[f64; 2]>,
    /// Pen tool: the FIRST anchor's in-handle (mirror of its out-handle), saved
    /// so the closing segment lands on it — a smooth join across the start.
    pub(crate) pen_start_in: Option<[f64; 2]>,
}

/// A deferred edit to the sketch session, collected during egui drawing and
/// applied afterward to keep borrows simple.
pub(crate) enum SketchAction {
    SetTool(SketchTool),
    AddPoint([f64; 2]),
    CloseLoop,
    PickAt(egui::Pos2),
    Apply(Constraint),
    /// Begin dragging an existing point (Select tool).
    GrabPoint(PointId),
    /// Begin dragging a bezier control handle `(bezier index, is_c1)`.
    GrabHandle((usize, bool)),
    /// Move whatever is grabbed (point or handle) to a new plane coordinate.
    /// The bool is whether Alt is held (break a smooth node into a corner).
    DragTo([f64; 2], bool),
    /// End the drag (re-solve).
    ReleaseDrag,
    /// Insert a vertex at the given plane coordinate on `line`, splitting it.
    InsertOnLine(LineId, [f64; 2]),
    /// Insert a vertex on a bezier (index) at parameter `t`, splitting the curve.
    InsertOnBezier(usize, f64),
    /// Delete the selected points, healing the loop.
    DeleteSelected,
    /// Convert the (single) selected line into a bezier.
    CurveSelected,
    /// Convert the (single) selected bezier back into a straight line.
    StraightenSelected,
    /// Re-smooth the (single) selected corner node (symmetrize its handles).
    SmoothSelected,
    /// Pen: a smooth-anchor drag began at this plane coordinate.
    PenPress([f64; 2]),
    /// Pen: place a corner anchor (a plain click) at this coordinate.
    PenCorner([f64; 2]),
    /// Pen: finish a smooth anchor — handle dragged out to this coordinate
    /// (the anchor sits at the earlier `PenPress`).
    PenSmooth([f64; 2]),
    /// Pen: close the loop back to the first anchor.
    PenClose,
    Finish,
    Cancel,
}

impl Modeler {
    /// Start a sketch on the currently selected face's plane.
    pub(crate) fn sketch_on_selected_face(&mut self) {
        if let Some(plane) = self.selection.face_plane {
            self.start_sketch(plane);
            self.selection.clear();
        }
    }

    /// Begin drawing a new sketch on `plane`.
    pub(crate) fn start_sketch(&mut self, plane: SketchPlane) {
        self.sketch_session = Some(SketchSession {
            plane,
            sketch: Sketch2d::new(),
            tool: SketchTool::Line,
            start: None,
            last: None,
            closed: false,
            selected_points: Vec::new(),
            selected_lines: Vec::new(),
            selected_beziers: Vec::new(),
            dof: 0,
            conflicted: false,
            editing: None,
            sweep_profile: None,
            dragging: None,
            dragging_handle: None,
            dragging_smooth: true,
            pen_out: None,
            pen_press: None,
            pen_start_in: None,
        });
    }

    /// Begin drawing a sweep PATH: an open chain on `plane` that, on finish,
    /// sweeps the selected `profile` sketch along it.
    pub(crate) fn start_sweep_path(&mut self, profile: FeatureId, plane: SketchPlane) {
        self.start_sketch(plane);
        if let Some(s) = self.sketch_session.as_mut() {
            s.sweep_profile = Some(profile);
        }
        self.selection.clear();
    }

    /// The selected body's feature id if it can be a sweep PROFILE — any sketch
    /// (constraint or simple) that evaluates to a planar face.
    pub(crate) fn sweepable_profile(&self) -> Option<FeatureId> {
        let id = self.selected_body()?;
        matches!(
            self.doc.history.get(id).map(|f| &f.kind),
            Some(FeatureKind::ConstraintSketch { .. } | FeatureKind::Sketch { .. })
        )
        .then_some(id)
    }

    /// Re-open an existing constraint sketch for editing: load its plane and
    /// geometry into a Select-mode session bound to the feature. Returns true if
    /// the feature was a constraint sketch.
    pub(crate) fn edit_sketch(&mut self, id: FeatureId) -> bool {
        let (plane, sketch) = match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::ConstraintSketch { plane, sketch }) => (*plane, sketch.clone()),
            _ => return false,
        };
        self.sketch_session = Some(SketchSession {
            plane,
            sketch,
            tool: SketchTool::Select,
            start: None,
            last: None,
            closed: true,
            selected_points: Vec::new(),
            selected_lines: Vec::new(),
            selected_beziers: Vec::new(),
            dof: 0,
            conflicted: false,
            editing: Some(id),
            sweep_profile: None,
            dragging: None,
            dragging_handle: None,
            dragging_smooth: true,
            pen_out: None,
            pen_press: None,
            pen_start_in: None,
        });
        self.selection.clear();
        true
    }

    /// The selected body's feature id, if it is a constraint sketch (re-editable).
    pub(crate) fn editable_sketch(&self) -> Option<FeatureId> {
        let id = self.selected_body()?;
        matches!(self.doc.history.get(id).map(|f| &f.kind), Some(FeatureKind::ConstraintSketch { .. }))
            .then_some(id)
    }

    /// The plane of a sketch feature (constraint or simple), else `None`. When the
    /// gizmo target is a sketch, the move edits this plane in place.
    pub(crate) fn sketch_plane_of(&self, id: FeatureId) -> Option<SketchPlane> {
        match self.doc.history.get(id).map(|f| &f.kind) {
            Some(
                FeatureKind::ConstraintSketch { plane, .. } | FeatureKind::Sketch { plane, .. },
            ) => Some(*plane),
            _ => None,
        }
    }

    /// Re-solve the session sketch and record its remaining DOF.
    pub(crate) fn resolve_session(&mut self) {
        if let Some(session) = self.sketch_session.as_mut() {
            let solution = cdt_solver::solve_sketch(&mut session.sketch);
            session.dof = solution.degrees_of_freedom;
            session.conflicted = solution.is_over_constrained() && !solution.report.converged;
        }
    }

    /// Commit the current sketch as a `ConstraintSketch` feature if it forms a
    /// closed loop. Returns whether the document changed.
    pub(crate) fn finish_sketch(&mut self) -> bool {
        let Some(session) = self.sketch_session.take() else {
            return false;
        };
        // A sweep-path session commits a Sweep of its profile along the open path.
        if let Some(profile) = session.sweep_profile {
            if session.sketch.path_elements().is_none() {
                return false; // not a single open chain — discard
            }
            let before = self.doc.clone();
            let id = self.doc.add_numbered(
                "Sweep",
                FeatureKind::Sweep {
                    profile,
                    path: cdt_core::SweepPath::Planar {
                        plane: session.plane,
                        sketch: session.sketch,
                    },
                },
            );
            self.record_undo(before);
            self.ui.selected = Some(id);
            return true;
        }
        if session.sketch.profile_elements().is_none() {
            // Not a closed loop — discard rather than commit something invalid.
            return false;
        }
        let before = self.doc.clone();
        let kind = FeatureKind::ConstraintSketch {
            plane: session.plane,
            sketch: session.sketch,
        };
        let id = match session.editing {
            // Re-editing: overwrite the existing feature so downstream rebuilds.
            Some(id) => {
                let Some(feature) = self.doc.history.get_mut(id) else {
                    return false;
                };
                feature.kind = kind;
                id
            }
            None => self.doc.add_numbered("Sketch", kind),
        };
        self.record_undo(before);
        self.ui.selected = Some(id);
        true
    }

    /// The sketch toolbar (top panel): tool toggle, constraint buttons gated by
    /// the current selection, and status. Reads state; pushes actions.
    pub(crate) fn sketch_top_bar(&self, ctx: &egui::Context, actions: &mut Vec<SketchAction>) {
        let Some(session) = self.sketch_session.as_ref() else {
            return;
        };
        let np = session.selected_points.len();
        let nl = session.selected_lines.len();
        let nb = session.selected_beziers.len();

        // Delete/Backspace removes selected points.
        if np >= 1
            && ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            actions.push(SketchAction::DeleteSelected);
        }

        #[allow(deprecated)]
        egui::Panel::top("sketch_toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if session.sweep_profile.is_some() {
                    ui.label(
                        egui::RichText::new(format!("➿ Sweep path ({})", session.plane.label()))
                            .strong(),
                    )
                    .on_hover_text("Draw an OPEN curve; Finish sweeps the profile along it");
                    ui.separator();
                }
                if ui.button("✓ Finish").clicked() {
                    actions.push(SketchAction::Finish);
                }
                if ui.button("✗ Cancel").clicked() {
                    actions.push(SketchAction::Cancel);
                }
                ui.separator();
                // Drawing tools (Line/Pen) extend the boundary, so they're off once
                // the loop is closed; Insert edits a closed loop, so it's off until
                // then. Select is always available.
                for (tool, label, enabled) in [
                    (SketchTool::Line, "✏ Line", !session.closed),
                    (SketchTool::Pen, "✒ Pen", !session.closed),
                    (SketchTool::Select, "◉ Select", true),
                    (SketchTool::Insert, "✚ Insert", session.closed),
                ] {
                    let w = egui::SelectableLabel::new(session.tool == tool, label);
                    if ui.add_enabled(enabled, w).clicked() {
                        actions.push(SketchAction::SetTool(tool));
                    }
                }
                if ui
                    .add_enabled(np >= 1, egui::Button::new("🗑 Delete"))
                    .on_hover_text("Delete the selected point(s) and reconnect the loop")
                    .clicked()
                {
                    actions.push(SketchAction::DeleteSelected);
                }
                // One button toggles an edge between straight and curved, based on
                // what's selected: a line → "Curve", a bezier → "Straighten".
                if nb == 1 && nl == 0 {
                    if ui
                        .add_enabled(true, egui::Button::new("╱ Straighten"))
                        .on_hover_text("Turn the selected curve back into a straight line")
                        .clicked()
                    {
                        actions.push(SketchAction::StraightenSelected);
                    }
                } else if ui
                    .add_enabled(nl == 1 && nb == 0, egui::Button::new("⤸ Curve"))
                    .on_hover_text("Turn the selected line into a bezier — then drag its handles")
                    .clicked()
                {
                    actions.push(SketchAction::CurveSelected);
                }
                // Re-smooth: only when a single selected point is a corner junction
                // of two beziers (the inverse of an Alt-drag break).
                let is_corner = (np == 1)
                    .then(|| session.selected_points.first().copied())
                    .flatten()
                    .is_some_and(|p| session.sketch.node_smoothness(p) == Some(false));
                if is_corner
                    && ui
                        .add_enabled(true, egui::Button::new("⤿ Smooth"))
                        .on_hover_text("Make this corner smooth — align its two handles")
                        .clicked()
                {
                    actions.push(SketchAction::SmoothSelected);
                }
                ui.separator();
                let dof = session.dof;
                let status = if session.conflicted {
                    egui::RichText::new("over-constrained")
                        .color(self.ui.theme.palette().error)
                } else if dof == 0 {
                    egui::RichText::new("fully constrained").color(self.ui.theme.palette().positive)
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
                    SketchTool::Select => {
                        "Select: click an edge then Curve/Straighten it; drag a point or handle to move it (Alt-drag a handle to break a smooth node into a corner)."
                    }
                    SketchTool::Insert => "Insert: click a segment to add a vertex on it.",
                    SketchTool::Pen => {
                        "Pen: click for a corner, click-drag for a smooth curve; click the first point to close."
                    }
                })
                .small()
                .weak(),
            );
        });
    }

    /// The sketch canvas overlay: draws entities (selection highlighted) and
    /// turns clicks into actions.
    pub(crate) fn sketch_overlay(
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
        // Neutral guide strokes (constraint arms, rubber-band lines) need to read
        // against the viewport background, which follows the theme. On the light
        // background the dark-mode grays wash out, so use Deep Space-tinted slate.
        let (arm_gray, line_gray) = if self.ui.theme.is_dark() {
            (egui::Color32::from_gray(120), egui::Color32::from_gray(150))
        } else {
            (egui::Color32::from_rgb(110, 118, 140), egui::Color32::from_rgb(78, 88, 112))
        };

        #[allow(deprecated)]
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let response =
                    ui.interact(rect, ui.id().with("sketch_canvas"), egui::Sense::click_and_drag());
                let painter = ui.painter_at(rect);

                // Nearest sketch point to a screen position, within a grab radius.
                let nearest_point = |pos: egui::Pos2| -> Option<PointId> {
                    let mut best = (10.0_f32, None);
                    for (i, p) in session.sketch.points.iter().enumerate() {
                        if let Some(sp) = proj(p.x, p.y) {
                            let d = sp.distance(pos);
                            if d < best.0 {
                                best = (d, Some(PointId(i)));
                            }
                        }
                    }
                    best.1
                };
                // Nearest line segment to a screen position (+ its distance).
                let nearest_line = |pos: egui::Pos2| -> Option<(LineId, f32)> {
                    let mut best = (8.0_f32, None);
                    for (i, l) in session.sketch.lines.iter().enumerate() {
                        let a = session.sketch.point(l.a);
                        let b = session.sketch.point(l.b);
                        if let (Some(pa), Some(pb)) = (proj(a.x, a.y), proj(b.x, b.y)) {
                            let d = dist_to_segment(pos, pa, pb);
                            if d < best.0 {
                                best = (d, Some(LineId(i)));
                            }
                        }
                    }
                    best.1.map(|id| (id, best.0))
                };
                // Nearest bezier to a screen position: its index, parameter t, and
                // distance (found by sampling the curve).
                let nearest_bezier = |pos: egui::Pos2| -> Option<(usize, f64, f32)> {
                    let mut best = (8.0_f32, None);
                    for (bi, bz) in session.sketch.beziers.iter().enumerate() {
                        for k in 0..=24 {
                            let t = k as f64 / 24.0;
                            let uv = Sketch2d::bezier_at(bz, &session.sketch.points, t);
                            if let Some(sp) = proj(uv[0], uv[1]) {
                                let d = sp.distance(pos);
                                if d < best.0 {
                                    best = (d, Some((bi, t)));
                                }
                            }
                        }
                    }
                    best.1.map(|(bi, t)| (bi, t, best.0))
                };

                // Nearest bezier control handle to a screen position.
                let nearest_handle = |pos: egui::Pos2| -> Option<(usize, bool)> {
                    let mut best = (10.0_f32, None);
                    for (i, bz) in session.sketch.beziers.iter().enumerate() {
                        for (is_c1, c) in [(true, bz.c1), (false, bz.c2)] {
                            if let Some(sp) = proj(c[0], c[1]) {
                                let d = sp.distance(pos);
                                if d < best.0 {
                                    best = (d, Some((i, is_c1)));
                                }
                            }
                        }
                    }
                    best.1
                };

                for (i, line) in session.sketch.lines.iter().enumerate() {
                    let a = session.sketch.point(line.a);
                    let b = session.sketch.point(line.b);
                    if let (Some(pa), Some(pb)) = (proj(a.x, a.y), proj(b.x, b.y)) {
                        let sel = session.selected_lines.contains(&LineId(i));
                        let color = if sel { selected } else { accent };
                        painter.line_segment([pa, pb], egui::Stroke::new(if sel { 2.5 } else { 1.5 }, color));
                    }
                }
                // Bezier segments (sampled polyline) + their control handles.
                let handle_col = egui::Color32::from_rgb(150, 220, 130);
                for (bi, bz) in session.sketch.beziers.iter().enumerate() {
                    let sel = session.selected_beziers.contains(&bi);
                    let col = if sel { selected } else { accent };
                    let width = if sel { 2.5 } else { 1.5 };
                    let mut prev: Option<egui::Pos2> = None;
                    for k in 0..=24 {
                        let uv = Sketch2d::bezier_at(bz, &session.sketch.points, k as f64 / 24.0);
                        if let Some(sp) = proj(uv[0], uv[1]) {
                            if let Some(pp) = prev {
                                painter.line_segment([pp, sp], egui::Stroke::new(width, col));
                            }
                            prev = Some(sp);
                        }
                    }
                    let a = session.sketch.point(bz.a);
                    let b = session.sketch.point(bz.b);
                    let arm = egui::Stroke::new(0.8, arm_gray);
                    if let (Some(pa), Some(c)) = (proj(a.x, a.y), proj(bz.c1[0], bz.c1[1])) {
                        painter.line_segment([pa, c], arm);
                        painter.circle_filled(c, 3.0, handle_col);
                    }
                    if let (Some(pb), Some(c)) = (proj(b.x, b.y), proj(bz.c2[0], bz.c2[1])) {
                        painter.line_segment([pb, c], arm);
                        painter.circle_filled(c, 3.0, handle_col);
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
                            painter.line_segment([a, cursor], egui::Stroke::new(1.0, line_gray));
                        }
                    }
                    if let Some(start) = session.start {
                        let s = session.sketch.point(start);
                        if let Some(sp) = proj(s.x, s.y) {
                            painter.circle_stroke(sp, 6.0, egui::Stroke::new(1.5, accent));
                        }
                    }
                }

                // Select tool: drag a bezier handle (priority) or an existing point.
                if session.tool == SketchTool::Select {
                    if response.drag_started() {
                        let pos = response.interact_pointer_pos();
                        if let Some(h) = pos.and_then(nearest_handle) {
                            actions.push(SketchAction::GrabHandle(h));
                        } else if let Some(pid) = pos.and_then(nearest_point) {
                            actions.push(SketchAction::GrabPoint(pid));
                        }
                    }
                    if response.dragged() {
                        if let Some(uv) = response
                            .interact_pointer_pos()
                            .and_then(|pos| view.cursor_on_plane(pos, o, xd, yd, nd))
                        {
                            let alt = response.ctx.input(|i| i.modifiers.alt);
                            actions.push(SketchAction::DragTo(uv, alt));
                        }
                    }
                    if response.drag_stopped() {
                        actions.push(SketchAction::ReleaseDrag);
                    }
                }

                // Pen tool: rubber-band + symmetric-handle preview; drag places a
                // smooth anchor (pull a handle out), click places a corner.
                if session.tool == SketchTool::Pen {
                    let preview = egui::Stroke::new(1.0, line_gray);
                    let draw_curve = |p0, p1, p2, p3, stroke| {
                        let mut prev = p0;
                        for k in 1..=20 {
                            let pt = cubic_screen(p0, p1, p2, p3, k as f32 / 20.0);
                            painter.line_segment([prev, pt], stroke);
                            prev = pt;
                        }
                    };
                    match (session.pen_press, response.hover_pos()) {
                        // Click-dragging a smooth anchor: the cursor is the HANDLE,
                        // not the next anchor — so preview the actual curve from
                        // `last` to the new anchor at `press`, plus the handle arms.
                        (Some(press), Some(cursor)) => {
                            if let Some(pp) = proj(press[0], press[1]) {
                                // The drag is the out-handle; the incoming segment
                                // ends on its mirror (this anchor's in-handle).
                                let in_handle = egui::pos2(2.0 * pp.x - cursor.x, 2.0 * pp.y - cursor.y);
                                if let Some(p0) = session
                                    .last
                                    .map(|l| session.sketch.point(l))
                                    .and_then(|lp| proj(lp.x, lp.y))
                                {
                                    let p1 = session.pen_out.and_then(|h| proj(h[0], h[1])).unwrap_or(p0);
                                    draw_curve(p0, p1, in_handle, pp, egui::Stroke::new(1.5, accent));
                                }
                                let arm = egui::Stroke::new(0.8, arm_gray);
                                painter.line_segment([pp, cursor], arm);
                                painter.line_segment([pp, in_handle], arm);
                                let hc = egui::Color32::from_rgb(150, 220, 130);
                                painter.circle_filled(cursor, 3.0, hc);
                                painter.circle_filled(in_handle, 3.0, hc);
                                painter.circle_filled(pp, 3.0, accent);
                            }
                        }
                        // Hovering (not dragging): rubber-band to where a click
                        // would drop the next corner — a curve if the previous
                        // anchor left an out-handle, else a straight line.
                        (None, Some(cursor)) => {
                            if let Some(p0) = session
                                .last
                                .map(|l| session.sketch.point(l))
                                .and_then(|lp| proj(lp.x, lp.y))
                            {
                                match session.pen_out.and_then(|h| proj(h[0], h[1])) {
                                    Some(p1) => draw_curve(p0, p1, cursor, cursor, preview),
                                    None => {
                                        painter.line_segment([p0, cursor], preview);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    if let Some(start) = session.start {
                        let s = session.sketch.point(start);
                        if let Some(sp) = proj(s.x, s.y) {
                            painter.circle_stroke(sp, 6.0, egui::Stroke::new(1.5, accent));
                        }
                    }
                    if response.drag_started() {
                        if let Some(uv) = response
                            .interact_pointer_pos()
                            .and_then(|p| view.cursor_on_plane(p, o, xd, yd, nd))
                        {
                            actions.push(SketchAction::PenPress(uv));
                        }
                    }
                    if response.drag_stopped() {
                        if let Some(uv) = response
                            .interact_pointer_pos()
                            .and_then(|p| view.cursor_on_plane(p, o, xd, yd, nd))
                        {
                            actions.push(SketchAction::PenSmooth(uv));
                        }
                    }
                }

                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        match session.tool {
                            SketchTool::Line => {
                                let near_start = !session.closed && session.sweep_profile.is_none()
                                    && session
                                        .start
                                        .map(|id| session.sketch.point(id))
                                        .and_then(|p| proj(p.x, p.y))
                                        .is_some_and(|sp| sp.distance(pos) < 10.0);
                                let n_segs = session.sketch.lines.len()
                                    + session.sketch.beziers.len();
                                if near_start && n_segs >= 2 {
                                    actions.push(SketchAction::CloseLoop);
                                } else if let Some(uv) = view.cursor_on_plane(pos, o, xd, yd, nd) {
                                    actions.push(SketchAction::AddPoint(uv));
                                }
                            }
                            SketchTool::Select => actions.push(SketchAction::PickAt(pos)),
                            SketchTool::Insert => {
                                // Split whichever segment (line or bezier) is nearer:
                                // a line splits into two lines, a bezier into two
                                // beziers (so the curve is preserved).
                                let line = nearest_line(pos);
                                let bez = nearest_bezier(pos);
                                let bez_nearer = match (line, bez) {
                                    (Some((_, ld)), Some((_, _, bd))) => bd <= ld,
                                    (None, Some(_)) => true,
                                    _ => false,
                                };
                                if bez_nearer {
                                    if let Some((bi, t, _)) = bez {
                                        actions.push(SketchAction::InsertOnBezier(bi, t));
                                    }
                                } else if let (Some((line, _)), Some(uv)) =
                                    (line, view.cursor_on_plane(pos, o, xd, yd, nd))
                                {
                                    actions.push(SketchAction::InsertOnLine(line, uv));
                                }
                            }
                            SketchTool::Pen => {
                                let near_start = !session.closed && session.sweep_profile.is_none()
                                    && session
                                        .start
                                        .map(|id| session.sketch.point(id))
                                        .and_then(|p| proj(p.x, p.y))
                                        .is_some_and(|sp| sp.distance(pos) < 10.0);
                                let n_segs =
                                    session.sketch.lines.len() + session.sketch.beziers.len();
                                if near_start && n_segs >= 2 {
                                    actions.push(SketchAction::PenClose);
                                } else if let Some(uv) = view.cursor_on_plane(pos, o, xd, yd, nd) {
                                    actions.push(SketchAction::PenCorner(uv));
                                }
                            }
                        }
                    }
                }
            });
    }

    /// Apply collected sketch actions. Returns whether the document changed.
    pub(crate) fn apply_sketch_actions(&mut self, actions: Vec<SketchAction>, view: &ViewContext) -> bool {
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
                        s.selected_beziers.clear();
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
                SketchAction::GrabPoint(pid) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.dragging = Some(pid);
                    }
                }
                SketchAction::GrabHandle(h) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.dragging_handle = Some(h);
                        // Remember whether this node is smooth right now, so a
                        // plain drag preserves its smoothness as it moves.
                        s.dragging_smooth = s.sketch.node_is_smooth(h.0, h.1);
                    }
                }
                SketchAction::DragTo([u, v], alt) => {
                    // Move whatever is grabbed — a bezier handle takes priority
                    // over a point. (The session is a working copy; the doc only
                    // changes on Finish.)
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let Some((i, is_c1)) = s.dragging_handle {
                            // Alt breaks the join into a corner (handles move
                            // independently); once broken it stays broken for the
                            // rest of the drag.
                            if alt {
                                s.dragging_smooth = false;
                            }
                            if let Some(bz) = s.sketch.beziers.get_mut(i) {
                                if is_c1 {
                                    bz.c1 = [u, v];
                                } else {
                                    bz.c2 = [u, v];
                                }
                            }
                            // Keep an adjacent bezier's handle mirrored → smooth
                            // node, unless this is a corner (Alt, or already split).
                            if s.dragging_smooth {
                                s.sketch.mirror_partner_handle(i, is_c1);
                            }
                        } else if let Some(pid) = s.dragging {
                            // Carry the anchor's handles with it (keeps the curve).
                            s.sketch.move_point(pid, u, v);
                        }
                    }
                }
                SketchAction::ReleaseDrag => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.dragging = None;
                        s.dragging_handle = None;
                    }
                    self.resolve_session(); // settle any constraints
                }
                SketchAction::InsertOnLine(line, [u, v]) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.sketch.split_line(line, u, v);
                        s.selected_points.clear();
                        s.selected_lines.clear();
                        s.selected_beziers.clear();
                    }
                }
                SketchAction::InsertOnBezier(idx, t) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.sketch.split_bezier(idx, t);
                        s.selected_points.clear();
                        s.selected_lines.clear();
                        s.selected_beziers.clear();
                    }
                }
                SketchAction::DeleteSelected => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        let remove = std::mem::take(&mut s.selected_points);
                        s.sketch.delete_points(&remove);
                        s.selected_lines.clear();
                        s.selected_beziers.clear();
                    }
                    self.resolve_session();
                }
                SketchAction::CurveSelected => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let Some(&line) = s.selected_lines.first() {
                            s.sketch.curve_line(line);
                        }
                        s.selected_lines.clear();
                        s.selected_points.clear();
                        s.selected_beziers.clear();
                    }
                }
                SketchAction::StraightenSelected => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let Some(&bez) = s.selected_beziers.first() {
                            s.sketch.straighten_bezier(bez);
                        }
                        s.selected_lines.clear();
                        s.selected_points.clear();
                        s.selected_beziers.clear();
                    }
                }
                SketchAction::SmoothSelected => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let Some(&p) = s.selected_points.first() {
                            s.sketch.smooth_node(p);
                        }
                    }
                }
                SketchAction::PenPress(uv) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.pen_press = Some(uv);
                    }
                }
                SketchAction::PenCorner([u, v]) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        let pid = s.sketch.add_point(u, v);
                        if let Some(last) = s.last {
                            connect_pen(&mut s.sketch, last, pid, s.pen_out, None);
                        }
                        if s.start.is_none() {
                            s.start = Some(pid);
                        }
                        s.last = Some(pid);
                        s.pen_out = None; // a corner anchor has no out-handle
                    }
                }
                SketchAction::PenSmooth([hu, hv]) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let Some([pu, pv]) = s.pen_press.take() {
                            let pid = s.sketch.add_point(pu, pv);
                            // The in-handle mirrors the out-handle (symmetric node).
                            let in_handle = [2.0 * pu - hu, 2.0 * pv - hv];
                            if let Some(last) = s.last {
                                connect_pen(&mut s.sketch, last, pid, s.pen_out, Some(in_handle));
                            } else {
                                // First anchor: its incoming segment is the eventual
                                // close, so save the in-handle for then.
                                s.pen_start_in = Some(in_handle);
                            }
                            if s.start.is_none() {
                                s.start = Some(pid);
                            }
                            s.last = Some(pid);
                            s.pen_out = Some([hu, hv]); // out-handle for the next segment
                        }
                    }
                }
                SketchAction::PenClose => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        if let (Some(last), Some(start)) = (s.last, s.start) {
                            // The closing segment lands on the first anchor's saved
                            // in-handle → smooth across the start point.
                            connect_pen(&mut s.sketch, last, start, s.pen_out, s.pen_start_in);
                            s.closed = true;
                            s.tool = SketchTool::Select;
                            s.pen_out = None;
                            s.pen_press = None;
                            s.pen_start_in = None;
                        }
                    }
                    self.resolve_session();
                }
                SketchAction::Apply(constraint) => {
                    if let Some(s) = self.sketch_session.as_mut() {
                        s.sketch.add_constraint(constraint);
                        s.selected_points.clear();
                        s.selected_lines.clear();
                        s.selected_beziers.clear();
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
        // Nearest bezier (sampled), competing with the nearest line — pick the
        // edge actually closest to the click, whichever kind.
        let mut best_bez: (f32, Option<usize>) = (8.0, None);
        for (i, bz) in s.sketch.beziers.iter().enumerate() {
            for k in 0..=24 {
                let uv = Sketch2d::bezier_at(bz, &s.sketch.points, k as f64 / 24.0);
                if let Some(sp) = proj(uv[0], uv[1]) {
                    let d = sp.distance(pos);
                    if d < best_bez.0 {
                        best_bez = (d, Some(i));
                    }
                }
            }
        }
        let bez_nearer = best_bez.0 <= best_line.0;
        if bez_nearer {
            if let Some(i) = best_bez.1 {
                toggle(&mut s.selected_beziers, i);
                return;
            }
        }
        if let Some(id) = best_line.1 {
            toggle(&mut s.selected_lines, id);
            return;
        }

        s.selected_points.clear();
        s.selected_lines.clear();
        s.selected_beziers.clear();
    }
}

/// Pen tool: connect anchor `a` to `b` as a straight line, or a bezier when
/// either end carries a handle (`a`'s out-handle / `b`'s in-handle). A missing
/// handle collapses onto its own endpoint.
pub(crate) fn connect_pen(
    sketch: &mut Sketch2d,
    a: PointId,
    b: PointId,
    a_out: Option<[f64; 2]>,
    b_in: Option<[f64; 2]>,
) {
    if a_out.is_none() && b_in.is_none() {
        sketch.add_line(a, b);
    } else {
        let ap = sketch.point(a);
        let bp = sketch.point(b);
        let c1 = a_out.unwrap_or([ap.x, ap.y]);
        let c2 = b_in.unwrap_or([bp.x, bp.y]);
        sketch.add_bezier(a, b, c1, c2);
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

/// A point on the cubic bezier `p0..p3` at `t`, in screen space (for previews).
pub(crate) fn cubic_screen(p0: egui::Pos2, p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2, t: f32) -> egui::Pos2 {
    let l = |a: egui::Pos2, b: egui::Pos2| egui::pos2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    let (q0, q1, q2) = (l(p0, p1), l(p1, p2), l(p2, p3));
    let (r0, r1) = (l(q0, q1), l(q1, q2));
    l(r0, r1)
}
