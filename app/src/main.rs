//! Cadette — Phase 1 interactive shell.
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

use cdt_core::{
    BooleanOp, Constraint, Document, FaceAnchor, FeatureId, FeatureKind, MissingReason,
    Profile, RegenCache, RegenError, Sketch2d, SketchPlane, SolveStatus, DVec3,
};
use cdt_solver::LmSolver;
use cdt_interaction::Selection;
use cdt_kernel::{Body, KernelBackend};
use cdt_render::egui;
use egui_phosphor::regular as phi;
use cdt_render::{
    Axis3, CameraPose, Controller, Gizmo, GizmoHandle, Highlights, MeshData, Pick, ResizeBox,
    TransformDelta, ViewContext, ViewportTheme,
};
use cdt_ui::{history_panel, HistoryState};

mod demos;
#[cfg(target_os = "macos")]
mod macos;
mod commands;
mod drags;
mod prefs;
mod project;
mod sketch;
use drags::{axis_get, axis_set};
use drags::{Manipulation, ResizeDrag, ResizeMode, TransformDrag};
use prefs::Prefs;
use sketch::{SketchAction, SketchSession};

const DEFLECTION_MM: f64 = 0.1;
/// Smallest box dimension (mm) a resize drag may produce, so it can't collapse.
const MIN_DIM: f64 = 0.1;
/// Marker written to the OS clipboard on body copy. Copying seeds the clipboard
/// so egui emits a Paste event on Cmd/Ctrl+V; paste acts only when the incoming
/// payload is exactly this marker, so text copied in another app can't trigger
/// a stale body paste.
const CLIPBOARD_SENTINEL: &str = "cadette:body";

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

/// Tape-measure mode: the distance from a surface point pinned by a click to
/// the surface point currently under the cursor, read out live.
#[derive(Clone, Copy, Default)]
struct TapeState {
    /// The pinned start point (first click on a surface; a later click re-pins).
    pin: Option<DVec3>,
    /// The surface point under the cursor this frame, if any.
    hover: Option<DVec3>,
}

/// Ties the document, the OCCT backend, and the history panel together as a
/// [`Controller`] the viewport drives.
struct Modeler {
    doc: Document,
    backend: KernelBackend,
    /// Incremental-regen cache: only features whose params or inputs changed are
    /// rebuilt, so a push/pull drag recomputes one op instead of the whole tree.
    regen_cache: RegenCache<Body>,
    /// Last transient status message (e.g. an export result), shown in the UI.
    status: Option<String>,
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
    bodies: Vec<Body>,
    face_ranges: Vec<u32>,
    edge_ranges: Vec<u32>,
    /// Bounding-box center of each visible body (aligned with `ui.visible`), the
    /// gizmo origin for that body.
    body_centers: Vec<[f64; 3]>,
    /// World-space AABB `(min, max)` of each visible body (aligned with
    /// `ui.visible`) — drives the resize grips' placement and extents.
    body_bounds: Vec<([f64; 3], [f64; 3])>,
    /// Memoized OCCT volumes by visible-body id, so the selection read-out
    /// doesn't call GProp every frame. Cleared whenever geometry regenerates.
    volume_cache: HashMap<FeatureId, f64>,
    /// World-space anchor point for each selected edge (keyed by its global pick
    /// id), captured at click time so each edge in a multi-selection keeps its
    /// own durable fillet anchor.
    edge_anchors: HashMap<u32, DVec3>,
    /// Tape-measure mode, when engaged (toolbar toggle / Esc to exit).
    tape: Option<TapeState>,
    /// The Build Graph row under the cursor + when that hover began — the
    /// ghost preview waits out a short debounce so scanning the list doesn't
    /// churn tessellations.
    graph_hover: Option<(FeatureId, std::time::Instant)>,
    /// The step whose ghost is showing (mesh in `ghost_cache`); `ghost_key`
    /// bumps on every visible change so the viewport re-uploads exactly then.
    ghost_step: Option<FeatureId>,
    ghost_key: u64,
    /// Tessellated hover-ghost meshes by feature id, from the regen cache's
    /// intermediate solids. Cleared whenever geometry regenerates.
    ghost_cache: HashMap<FeatureId, MeshData>,
    /// The camera pose seen this frame (from `ViewContext`), captured so a
    /// project save can persist the view into `view.json`.
    camera_pose: Option<CameraPose>,
    /// A pose for the viewport to apply (a loaded project's saved view);
    /// drained one-shot by `requested_camera`.
    camera_request: Option<CameraPose>,
    /// Active push/pull drag, if any.
    manipulation: Option<Manipulation>,
    /// Active gizmo move drag, if any.
    transform: Option<TransformDrag>,
    /// Active resize drag (bounding-box grip), if any.
    resize: Option<ResizeDrag>,
    /// The body feature copied to the clipboard (Cmd/Ctrl+C), pasted as an
    /// independent duplicate by Cmd/Ctrl+V. `None` until something is copied.
    clipboard: Option<FeatureId>,
    /// The file this project is bound to (set on Open/Save). The window title
    /// shows its name — "Untitled" until the first save.
    current_path: Option<std::path::PathBuf>,
    /// Unsaved-changes flag, shown as a leading "●" in the window title. Derived
    /// (in `mesh`, whenever the doc changes) by comparing the document against
    /// [`Self::saved_sig`], so undoing back to the saved state clears it.
    dirty: bool,
    /// Signature of the document as last saved/opened (or the initial baseline);
    /// `dirty` is `doc_signature(&doc) != saved_sig`.
    saved_sig: u64,
}

/// A cheap content signature of the document (a hash of its serialized form),
/// for detecting whether it differs from the last saved/baseline state.
fn doc_signature(doc: &Document) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_vec(doc).unwrap_or_default().hash(&mut h);
    h.finish()
}

impl Modeler {
    fn new() -> Self {
        // Store the solved doc so its baseline signature matches what `mesh`
        // hashes (which is post-solve).
        let mut doc = build_document();
        doc.solve_sketches(&mut LmSolver);
        let saved_sig = doc_signature(&doc);
        Self {
            doc,
            backend: KernelBackend::default(),
            regen_cache: RegenCache::new(),
            status: None,
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
            body_bounds: Vec::new(),
            volume_cache: HashMap::new(),
            edge_anchors: HashMap::new(),
            tape: None,
            graph_hover: None,
            ghost_step: None,
            ghost_key: 0,
            ghost_cache: HashMap::new(),
            camera_pose: None,
            camera_request: None,
            manipulation: None,
            transform: None,
            resize: None,
            clipboard: None,
            current_path: None,
            dirty: false,
            saved_sig,
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

    /// A one-line measurement of the current selection for the status bar: the
    /// bounding-box W×H×D (and volume) of the single body behind a selected face
    /// — plus that face's exact area — or a body picked in the Build Graph; a
    /// single selected edge shows its exact length instead. `None` if the
    /// selection isn't a single face/edge/body. Volume is memoized so this
    /// stays cheap per frame (a single face/edge GProp query already is).
    fn selection_measurement(&mut self) -> Option<String> {
        // A single selected edge: its exact length.
        if let [Pick::Edge(g)] = self.selection.selected() {
            let (i, local) = self.locate_edge(*g)?;
            let len = self.bodies.get(i)?.edge_length(local).ok()?;
            return Some(format!("{}  edge {len:.2} mm", phi::RULER));
        }
        // Resolve a single target body index (and picked face) from the selection.
        let (idx, face) = match self.selection.selected() {
            [Pick::Face(g)] => {
                let (i, local) = self.locate_face(*g)?;
                (i, Some(local))
            }
            _ => {
                // Nothing picked in the viewport — fall back to a body selected in
                // the Build Graph (if it's currently visible).
                let sel = self.ui.selected?;
                (self.ui.visible.iter().position(|&f| f == sel)?, None)
            }
        };
        let (min, max) = self.body_bounds.get(idx).copied()?;
        let (w, h, d) = (max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        // Ignore a truly empty body (a sketch's flat face still has useful W×H).
        if w.max(h).max(d) < 1e-6 {
            return None;
        }
        let mut s = format!("{}  {w:.2} × {h:.2} × {d:.2} mm", phi::RULER);
        // Volume, only for real solids (a sketch face encloses none).
        if let Some(&id) = self.ui.visible.get(idx) {
            let vol = match self.volume_cache.get(&id) {
                Some(&v) => Some(v),
                None => {
                    let v = self.bodies.get(idx).and_then(|b| b.volume().ok());
                    if let Some(v) = v {
                        self.volume_cache.insert(id, v);
                    }
                    v
                }
            };
            if let Some(v) = vol {
                if v > 1e-6 {
                    s.push_str(&format!("     vol {v:.0} mm³"));
                }
            }
        }
        // The picked face's exact area (not just its bbox footprint).
        if let Some(local) = face {
            if let Some(a) = self.bodies.get(idx).and_then(|b| b.face_area(local).ok()) {
                if a > 1e-9 {
                    s.push_str(&format!("     face {a:.1} mm²"));
                }
            }
        }
        Some(s)
    }

    /// The tape-measure read-out: pin→cursor distance with per-axis deltas, or
    /// a prompt for the next step. `None` when the tool isn't engaged.
    fn tape_measurement(&self) -> Option<String> {
        let tape = self.tape.as_ref()?;
        Some(match (tape.pin, tape.hover) {
            (Some(a), Some(b)) => {
                let d = b - a;
                format!(
                    "{}  tape {:.2} mm     Δ {:+.2}, {:+.2}, {:+.2}",
                    phi::RULER,
                    d.length(),
                    d.x,
                    d.y,
                    d.z
                )
            }
            (Some(_), None) => format!("{}  tape: hover a surface to measure", phi::RULER),
            _ => format!("{}  tape: click a surface to pin the start point", phi::RULER),
        })
    }

    /// Viewport overlay for tape-measure mode: the pinned point, a line to the
    /// hovered surface point, and the distance at the line's midpoint.
    fn tape_overlay(&self, ctx: &egui::Context, view: &ViewContext, tape: TapeState) {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("tape_overlay"),
        ));
        // One saturated hue that reads on both themes and matches no axis color.
        let color = egui::Color32::from_rgb(0xff, 0x9f, 0x1c);
        let Some(pin) = tape.pin else {
            return;
        };
        let Some(a) = view.project(pin.to_array()) else {
            return;
        };
        painter.circle_filled(a, 4.0, color);
        let Some(hover) = tape.hover else {
            return;
        };
        let Some(b) = view.project(hover.to_array()) else {
            return;
        };
        painter.line_segment([a, b], egui::Stroke::new(2.0, color));
        painter.text(
            egui::pos2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5) + egui::vec2(10.0, -10.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{:.2} mm", (hover - pin).length()),
            egui::FontId::monospace(13.0),
            color,
        );
    }

    /// Drive the Build Graph hover preview: after a short debounce, ghost the
    /// hovered step's own (cached) result in the viewport. Runs every frame.
    fn update_graph_hover(&mut self) {
        /// Steady hover time before a ghost tessellates, so scanning the list
        /// stays cheap.
        const GHOST_DELAY_MS: u128 = 150;
        let want = match self.ui.hovered_step {
            Some(id) => {
                if self.graph_hover.map(|(h, _)| h) != Some(id) {
                    self.graph_hover = Some((id, std::time::Instant::now()));
                }
                let (_, since) = self.graph_hover.unwrap();
                (since.elapsed().as_millis() >= GHOST_DELAY_MS && self.ensure_ghost_mesh(id))
                    .then_some(id)
            }
            None => {
                self.graph_hover = None;
                None
            }
        };
        if self.ghost_step != want {
            self.ghost_step = want;
            self.ghost_key += 1;
        }
    }

    /// Tessellate (and memoize) the hover ghost for `id` from its cached
    /// intermediate solid. False when nothing is available — the feature is
    /// rolled back, suppressed, errored, or was never built.
    fn ensure_ghost_mesh(&mut self, id: FeatureId) -> bool {
        if let Some(mesh) = self.ghost_cache.get(&id) {
            return !mesh.indices.is_empty();
        }
        let Some(solid) = self.regen_cache.cached_body(id) else {
            return false;
        };
        let mesh = match solid.tessellate(DEFLECTION_MM) {
            Ok(part) => MeshData {
                vertices: cdt_render::interleave(
                    &part.positions,
                    &part.normals,
                    &part.face_ids,
                    cdt_render::NO_COLOR,
                ),
                indices: part.indices,
                ..Default::default()
            },
            // Cache the failure as an empty mesh so it isn't retried per frame.
            Err(_) => MeshData::default(),
        };
        let ok = !mesh.indices.is_empty();
        self.ghost_cache.insert(id, mesh);
        ok
    }

    /// The global face-id range of the visible body the hovered Build Graph
    /// step feeds into — the viewport tints it while the row is hovered.
    fn hover_body_range(&self) -> Option<(u32, u32)> {
        let (id, _) = self.graph_hover?;
        let vis = self.downstream_visible(id)?;
        let i = self.ui.visible.iter().position(|&f| f == vis)?;
        let start = self.face_ranges.get(i).copied()?;
        let end = self.face_ranges.get(i + 1).copied().unwrap_or(u32::MAX);
        Some((start, end))
    }

    /// Walk consumers from `id` down to the visible body it feeds into.
    /// `None` when the chain dead-ends before a visible body (rolled back).
    fn downstream_visible(&self, id: FeatureId) -> Option<FeatureId> {
        let mut cur = id;
        for _ in 0..self.doc.history.len() {
            if self.ui.visible.contains(&cur) {
                return Some(cur);
            }
            cur = self
                .doc
                .history
                .features()
                .iter()
                .find(|f| f.kind.inputs().contains(&cur))?
                .id;
        }
        None
    }

    /// The body feature id behind the current viewport selection, if any.
    fn selected_body(&self) -> Option<FeatureId> {
        let body_index = match self.selection.primary()? {
            Pick::Face(g) => self.locate_face(g)?.0,
            Pick::Edge(g) => self.locate_edge(g)?.0,
        };
        self.ui.visible.get(body_index).copied()
    }


    /// The plane of a picked face, if it is planar.
    fn face_plane(&self, global: u32) -> Option<SketchPlane> {
        let (i, local) = self.locate_face(global)?;
        let (origin, x, y) = self.bodies.get(i)?.face_plane(local).ok().flatten()?;
        Some(SketchPlane::from_frame(origin.into(), x.into(), y.into()))
    }

    /// Record the pre-edit document on the undo stack and drop the redo stack.
    fn record_undo(&mut self, before: Document) {
        if self.undo.len() >= UNDO_LIMIT {
            self.undo.remove(0);
        }
        self.undo.push(before);
        self.redo.clear();
        // `dirty` is recomputed in `mesh` (whenever the doc changes), so undoing
        // back to the saved state clears it — no one-way latch here.
    }

    /// Mark the current document as the clean baseline (on save/open): its
    /// signature becomes the reference `dirty` is measured against.
    fn mark_saved(&mut self) {
        self.doc.solve_sketches(&mut LmSolver); // hash the same (solved) state `mesh` will
        self.saved_sig = doc_signature(&self.doc);
        self.dirty = false;
    }

    /// Recompute the unsaved-changes flag against the saved baseline.
    fn refresh_dirty(&mut self) {
        self.dirty = doc_signature(&self.doc) != self.saved_sig;
    }

    /// True while a continuous edit is reshaping the document every frame — a
    /// parameter-slider drag, push/pull, gizmo transform, or resize drag.
    fn interacting(&self) -> bool {
        self.editing
            || self.manipulation.is_some()
            || self.transform.is_some()
            || self.resize.is_some()
    }

    /// Restore the previous document state. Returns whether anything changed.
    fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.doc, prev));
                self.drop_transient_state();
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
                self.drop_transient_state();
                true
            }
            None => false,
        }
    }

    /// Drop state tied to the document that was just swapped out from under us
    /// (undo/redo): the viewport selection and edge anchors hold face/edge ids
    /// that are only valid for one regeneration, and an in-flight drag would
    /// commit against the wrong body. A sketch session survives unless the
    /// feature it edits (or sweeps from) no longer exists.
    fn drop_transient_state(&mut self) {
        self.selection.clear();
        self.edge_anchors.clear();
        self.manipulation = None;
        self.transform = None;
        self.resize = None;
        let session_target_gone = self.sketch_session.as_ref().is_some_and(|s| {
            let gone = |id: FeatureId| self.doc.history.get(id).is_none();
            s.editing.is_some_and(gone) || s.sweep_profile.is_some_and(gone)
        });
        if session_target_gone {
            self.sketch_session = None;
        }
    }
}

/// A full-width contextual action button: Phosphor `icon` + `label`, `tip` on
/// hover. Used in the right Actions panel.
fn action_btn(ui: &mut egui::Ui, icon: &str, label: &str, tip: &str, enabled: bool) -> bool {
    let w = ui.available_width();
    ui.add_enabled(
        enabled,
        egui::Button::new(egui::RichText::new(format!("{icon}  {label}")))
            .min_size(egui::vec2(w, 0.0)),
    )
    .on_hover_text(tip)
    .clicked()
}

impl Controller for Modeler {
    fn ui(&mut self, ctx: &egui::Context, view: &ViewContext) -> bool {
        use egui::Key;
        // Apply the active look-and-feel to the surrounding chrome each frame.
        // The 3D viewport renders behind a transparent CentralPanel, so this
        // never touches the scene.
        ctx.set_visuals(self.ui.theme.visuals());
        // Keep the latest camera pose at hand so a Save can persist the view.
        self.camera_pose = Some(view.camera_pose());
        let mut changed = false;
        let mut sketch_actions: Vec<SketchAction> = Vec::new();

        // One undo entry per continuous edit (a Properties drag or a Build Graph
        // edit): snapshot the doc BEFORE any panel mutates it, then commit that
        // snapshot below iff something changed. `editing` collapses a multi-frame
        // drag into a single entry. Discrete actions (fillet, move, …) record
        // their own undo, so they're excluded here.
        let snapshot = (!self.editing).then(|| self.doc.clone());
        let mut ui_edited = false;

        // --- Right dock: contextual Actions for the picked entity + the
        // selected feature's Properties (replaces the old across-the-top strips).
        {
            let sel = self.selection.selected();
            let single_face = self.sketch_session.is_none() && matches!(sel, [Pick::Face(_)]);
            let multi_face = self.sketch_session.is_none()
                && sel.len() >= 2
                && sel.iter().all(|p| matches!(p, Pick::Face(_)));
            let edges = self.sketch_session.is_none() && self.selection.is_edges();
            let n_sel = sel.len();

            // Single-face gating.
            let can_sketch = single_face && self.selection.can_sketch_on_face();
            let can_mirror = single_face && self.selection.face_plane.is_some();
            let can_ungroup = single_face
                && self.selected_body().is_some_and(|b| self.doc.group_and_chain(b).is_some());
            let edit_sketch_id = single_face.then(|| self.editable_sketch()).flatten();
            let sweep_profile_id = single_face.then(|| self.sweepable_profile()).flatten();
            // Multi-face + edge gating.
            let n_bodies = if multi_face { self.selected_bodies().len() } else { 0 };
            let can_revolve = edges && self.revolvable_profile().is_some();

            let (mut sketch, mut edit, mut fillet_face, mut mirror, mut ungroup) =
                (false, false, false, false, false);
            // "Fillet every edge" of the selected body (was a toolbar button).
            let mut fillet_body = false;
            // A sweep request carries the path plane chosen at that moment.
            let mut sweep_plane_choice: Option<SketchPlane> = None;
            let mut flip: Option<Axis3> = None;
            let (mut group, mut fillet_edges, mut revolve) = (false, false, false);
            let mut extrude_face = false;
            let mut editor_changed = false;
            // The Properties pane's "Edit…" button for a complex step (sketch/sweep).
            let mut edit_step = false;
            // A multi-body alignment request (axis + which box edge/center).
            let mut align: Option<(Axis3, commands::AlignMode)> = None;

            // Top-level `Panel::show` is deprecated in favor of `show_inside`,
            // but migrating means restructuring the whole frame layout — done
            // together in a dedicated egui panel pass.
            #[allow(deprecated)]
            egui::Panel::right("actions_panel")
                .resizable(true)
                .default_size(236.0)
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Actions").strong());
                    ui.add_space(2.0);
                    if single_face {
                        if edit_sketch_id.is_some() {
                            edit = action_btn(ui, phi::NOTE_PENCIL, "Edit sketch", "Re-open this sketch to move its points", true);
                        } else if can_sketch {
                            sketch = action_btn(ui, phi::PENCIL_SIMPLE, "Sketch on face", "Draw a 2D profile on this face", true);
                        }
                        if sweep_profile_id.is_some() {
                            // The face's body is a sketch → offer to pull it into a
                            // solid right here (also available on the top toolbar).
                            extrude_face = action_btn(ui, phi::ARROW_LINE_UP, "Extrude", "Pull this sketch into a solid (toward you)", true);
                            // Sweep: choose the path's plane at this moment (menu).
                            ui.menu_button(format!("{}  Sweep", phi::PATH), |ui| {
                                ui.label("Path plane:");
                                for (p, item) in [
                                    (SketchPlane::Xy, "XY — top"),
                                    (SketchPlane::Xz, "XZ — front"),
                                    (SketchPlane::Yz, "YZ — side"),
                                ] {
                                    if ui.button(item).clicked() {
                                        sweep_plane_choice = Some(p);
                                        ui.close_menu();
                                    }
                                }
                            })
                            .response
                            .on_hover_text("Sweep this profile along a path you draw — choose the path's plane");
                        }
                        fillet_face = action_btn(ui, phi::FRAME_CORNERS, "Fillet face edges", "Round all edges bounding this face", true);
                        fillet_body = action_btn(ui, phi::BOUNDING_BOX, "Fillet every edge", "Round every edge of this body", true);
                        if can_mirror {
                            mirror = action_btn(ui, phi::FLIP_HORIZONTAL, "Mirror across face", "Reflect this body across the face — keeps the original", true);
                        }
                        if can_ungroup {
                            ungroup = action_btn(ui, phi::STACK_MINUS, "Ungroup", "Split this group back into its members (⇧⌘G)", true);
                        }
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Flip").on_hover_text("Mirror this body in place about its center");
                            for (l, ax) in [("X", Axis3::X), ("Y", Axis3::Y), ("Z", Axis3::Z)] {
                                if ui.small_button(l).clicked() {
                                    flip = Some(ax);
                                }
                            }
                        });
                    } else if multi_face {
                        ui.label(egui::RichText::new(format!("{n_sel} faces · {n_bodies} bodies")).small().weak());
                        ui.add_space(2.0);
                        if n_bodies == 1 {
                            fillet_edges = action_btn(ui, phi::FRAME_CORNERS, "Fillet face edges", "Round all edges of the selected faces", true);
                        }
                        if n_bodies >= 2 {
                            group = action_btn(ui, phi::STACK_PLUS, &format!("Group {n_bodies} bodies"), "Combine into one unit that moves/rotates together (⌘G)", true);
                            // Align the bodies to their combined bounding box: one
                            // row per axis, with min / center / max on that axis.
                            ui.add_space(4.0);
                            ui.label(egui::RichText::new("Align").small().weak());
                            for (label, axis, lo, hi) in [
                                ("X", Axis3::X, "Left", "Right"),
                                ("Y", Axis3::Y, "Front", "Back"),
                                ("Z", Axis3::Z, "Bottom", "Top"),
                            ] {
                                ui.horizontal(|ui| {
                                    ui.monospace(label);
                                    if ui.small_button("Min").on_hover_text(lo).clicked() {
                                        align = Some((axis, commands::AlignMode::Min));
                                    }
                                    if ui.small_button("Mid").on_hover_text("Center").clicked() {
                                        align = Some((axis, commands::AlignMode::Center));
                                    }
                                    if ui.small_button("Max").on_hover_text(hi).clicked() {
                                        align = Some((axis, commands::AlignMode::Max));
                                    }
                                });
                            }
                        }
                        ui.add_space(2.0);
                        ui.weak("⇧/⌘-click to add faces");
                    } else if edges {
                        let plural = if n_sel == 1 { "" } else { "s" };
                        ui.label(egui::RichText::new(format!("{n_sel} edge{plural}")).small().weak());
                        ui.add_space(2.0);
                        fillet_edges = action_btn(ui, phi::FRAME_CORNERS,
                            if n_sel == 1 { "Fillet this edge" } else { "Fillet these edges" },
                            "Round the selected edge(s)", true);
                        if can_revolve {
                            revolve = action_btn(ui, phi::ARROW_CLOCKWISE, "Revolve about edge", "Spin the sketch profile a full turn about this edge", true);
                        }
                        ui.add_space(2.0);
                        ui.weak("⇧/⌘-click to add edges");
                    } else {
                        ui.weak("Select a face or edge in the viewport for contextual actions.");
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("Properties").strong());
                    ui.add_space(2.0);
                    if let Some(id) = self.ui.selected {
                        let editor = cdt_ui::selected_editor(ui, &mut self.doc, id);
                        editor_changed |= editor.changed;
                        edit_step |= editor.edit;
                    } else {
                        ui.weak("Select a feature to edit its parameters.");
                    }
                    if !self.ui.errors.is_empty() {
                        ui.separator();
                        cdt_ui::error_list(ui, &self.doc, &self.ui);
                    }
                });

            // Apply the collected actions.
            if edit {
                if let Some(id) = edit_sketch_id {
                    self.edit_sketch(id);
                }
            }
            // The Properties "Edit…" button re-opens the selected complex step in
            // its proper editor (dispatched by kind).
            if edit_step {
                if let Some(id) = self.ui.selected {
                    match self.doc.history.get(id).map(|f| &f.kind) {
                        Some(FeatureKind::ConstraintSketch { .. }) => {
                            self.edit_sketch(id);
                        }
                        Some(FeatureKind::Sweep { .. }) => {
                            self.edit_sweep_path(id);
                        }
                        _ => {}
                    }
                }
            }
            if sketch {
                self.sketch_on_selected_face();
            }
            if extrude_face {
                if let Some(id) = sweep_profile_id {
                    changed |= self.extrude_sketch(id, view.eye());
                }
            }
            if let Some(plane) = sweep_plane_choice {
                if let Some(p) = sweep_profile_id {
                    self.start_sweep_path(p, plane);
                }
            }
            if fillet_face {
                changed |= self.fillet_selected_face_edges();
            }
            if fillet_body {
                changed |= self.fillet_all_selected_body();
            }
            if mirror {
                changed |= self.mirror_across_selected_face();
            }
            if ungroup {
                changed |= self.ungroup_selected();
            }
            if let Some(ax) = flip {
                changed |= self.flip_in_place(ax);
            }
            if group {
                changed |= self.group_selected();
            }
            if let Some((axis, mode)) = align {
                changed |= self.align_selected(axis, mode);
            }
            if fillet_edges {
                changed |= if multi_face {
                    self.fillet_selected_face_edges()
                } else {
                    self.fillet_selected_edges()
                };
            }
            if revolve {
                changed |= self.revolve_selected_profile();
            }
            ui_edited |= editor_changed;
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
                            // Extend (never wrap) so a value like "-135°" stays on
                            // one line — the anchored Area is otherwise clamped to
                            // the screen and wraps the text character by character.
                            ui.add(
                                egui::Label::new(egui::RichText::new(text).strong().size(15.0))
                                    .extend(),
                            );
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
        let resp = history_panel(ctx, &mut self.doc, &mut self.ui);
        ui_edited |= resp.changed;
        // Snap the grid spacing up to the finest value the renderer will draw
        // at this extent, so the field never shows a spacing the grid ignores
        // (shrink the grid size to go finer).
        let min_spacing = cdt_render::min_grid_spacing((self.ui.grid_extent / 2.0) as f32) as f64;
        if self.ui.grid_spacing < min_spacing {
            self.ui.grid_spacing = min_spacing;
        }
        // Tape measure: toolbar toggle or Esc while engaged. Not while sketching
        // (viewport clicks belong to the sketch there).
        let tape_esc = self.tape.is_some() && ctx.input(|i| i.key_pressed(Key::Escape));
        if (resp.tape_toggle && self.sketch_session.is_none()) || tape_esc {
            self.tape = if self.tape.is_some() { None } else { Some(TapeState::default()) };
            self.ui.tape_active = self.tape.is_some();
        }
        // Build Graph hover: ghost the hovered step's own result (debounced)
        // and tint the visible body it feeds into (via `highlights`).
        self.update_graph_hover();
        if resp.theme_changed || resp.tint_changed {
            prefs::save(&Prefs { theme: self.ui.theme, body_tint: self.ui.body_tint });
        }
        // Commit the pre-edit snapshot if any continuous UI edit (Properties or
        // Build Graph) touched the doc this frame.
        if ui_edited {
            if let Some(before) = snapshot {
                self.record_undo(before);
            }
            self.editing = true;
            changed = true;
        } else if !ctx.input(|i| i.pointer.any_down()) {
            // Only end the edit session once the mouse is released: a drag frame
            // where the value didn't change (the pointer paused or moved
            // off-axis) must not split one drag into several undo entries.
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

        // Copy/paste a body (not while sketching — those keys are the sketch's).
        // egui-winit turns Cmd/Ctrl+C/V into Copy/Paste *events* (not Key presses
        // like Cmd+Z), so detect those.
        let (copy_key, paste_key) = ctx.input(|i| {
            (
                i.events.iter().any(|e| matches!(e, egui::Event::Copy)),
                // Only our own sentinel counts: pasting text copied in another
                // app must not resurrect a stale body.
                i.events
                    .iter()
                    .any(|e| matches!(e, egui::Event::Paste(t) if t == CLIPBOARD_SENTINEL)),
            )
        });
        // Group ⌘G / Ungroup ⇧⌘G (plain key presses — egui only special-cases
        // C/X/V, not G).
        let (group_key, ungroup_key) = ctx.input(|i| {
            let cmd_g = i.modifiers.command && i.key_pressed(Key::G);
            (cmd_g && !i.modifiers.shift, cmd_g && i.modifiers.shift)
        });
        if self.sketch_session.is_none() {
            if copy_key {
                self.copy_selected();
                // Seed the OS clipboard so the NEXT Cmd/Ctrl+V produces an egui
                // Paste event — egui only emits one when the OS clipboard isn't
                // empty, and copying a shape otherwise leaves it untouched.
                if self.clipboard.is_some() {
                    ctx.copy_text(CLIPBOARD_SENTINEL.to_owned());
                }
            }
            if paste_key {
                changed |= self.paste();
            }
            if group_key {
                if self.group_selected() {
                    changed = true;
                } else {
                    self.status = Some("Select faces on two or more bodies to group".into());
                }
            }
            if ungroup_key {
                if self.ungroup_selected() {
                    changed = true;
                } else {
                    self.status = Some("Select a group to ungroup".into());
                }
            }
        }

        // The "Sketch" tool in the Add panel enters interactive draw mode on
        // the chosen origin plane.
        if let Some(plane) = resp.start_sketch {
            if self.sketch_session.is_none() {
                self.start_sketch(plane);
            }
        }
        if resp.export_stl {
            self.export_stl();
        }
        if resp.new_project {
            changed |= self.new_document();
        }
        if resp.save_project {
            self.save_project();
        }
        if resp.open_project {
            changed |= self.open_project();
        }
        if let Some(op) = resp.boolean {
            changed |= self.apply_boolean(op);
        }
        if let Some(id) = resp.delete {
            changed |= self.delete_feature(id);
        }
        if resp.tidy {
            let before = self.doc.clone();
            let merged = self.doc.tidy();
            if merged > 0 {
                self.record_undo(before);
                // The merged-away Moves may include the panel selection.
                if self.ui.selected.is_some_and(|id| self.doc.history.get(id).is_none()) {
                    self.ui.selected = None;
                }
                changed = true;
                self.status = Some(format!(
                    "Tidied: merged {merged} stacked move{}",
                    if merged == 1 { "" } else { "s" }
                ));
            } else {
                self.status = Some("History already tidy".into());
            }
        }
        if let Some(source) = resp.extrude {
            changed |= self.extrude_sketch(source, view.eye());
        }

        // --- Status line at the bottom: the tape read-out while measuring
        // (else a live measurement of the selection) on the left, plus any
        // transient status (export result, etc.) on the right.
        let measurement = self
            .tape_measurement()
            .or_else(|| self.selection_measurement());
        if measurement.is_some() || self.status.is_some() {
            #[allow(deprecated)]
            egui::Panel::bottom("status_bar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(m) = &measurement {
                        ui.monospace(m).on_hover_text(
                            "Bounding-box width × height × depth, volume, and exact \
                             face area of the selection; a selected edge shows its length",
                        );
                    }
                    if let Some(status) = &self.status {
                        if measurement.is_some() {
                            ui.separator();
                        }
                        ui.label(status);
                    }
                });
            });
        }

        // --- Tape-measure overlay: pin marker, line, and distance label ---
        if let Some(tape) = self.tape {
            self.tape_overlay(ctx, view, tape);
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
        // Solves the constraint sketches in place first (a sketch that can't be
        // solved is a per-feature error, like any other failed step).
        let regen =
            self.regen_cache.regenerate_solved(&mut self.doc, &mut self.backend, &mut LmSolver);
        // The viewer calls this exactly when the doc changed (edit / undo / redo /
        // drag), so it's the natural place to refresh the unsaved-changes flag.
        // Compared post-solve, matching the state a save would serialize.
        // Skipped mid-interaction: the signature serializes the whole document,
        // which is too heavy for a per-mouse-move frame — the end-of-drag mesh()
        // call refreshes it.
        if !self.interacting() {
            self.refresh_dirty();
        }
        self.ui.errors = regen
            .errors()
            .iter()
            .map(|e| (e.feature(), error_message(e, &self.doc)))
            .collect();
        self.ui.visible = regen.visible().to_vec();
        let bodies = regen.into_visible_bodies();

        // Each feature's Build Graph lane — the palette fallback colour for
        // bodies with no user colour.
        let lane_of: std::collections::HashMap<FeatureId, usize> = self
            .doc
            .history
            .build_graph()
            .rows
            .iter()
            .map(|r| (r.id, r.lane))
            .collect();

        let mut mesh = MeshData::default();
        let mut face_offset = 0u32; // keep face/edge ids unique across bodies
        let mut edge_offset = 0u32;
        self.face_ranges.clear();
        self.edge_ranges.clear();
        self.body_centers.clear();
        self.body_bounds.clear();
        for (idx, body) in bodies.iter().enumerate() {
            self.face_ranges.push(face_offset);
            self.edge_ranges.push(edge_offset);
            // This body's Build Graph colour: its user colour (walking down
            // the input chain) or its lane's palette colour. The shader mixes
            // it over the theme grey by the tint dial, so the graph is always
            // a legend for the scene.
            let color = match self.ui.visible.get(idx) {
                Some(&id) => self.doc.history.resolve_color(id).unwrap_or_else(|| {
                    let lane = lane_of.get(&id).copied().unwrap_or(0);
                    cdt_render::srgb_to_linear(
                        cdt_core::LANE_PALETTE[lane % cdt_core::LANE_PALETTE.len()],
                    )
                }),
                None => cdt_render::NO_COLOR,
            };
            // Exact bounds from the geometry (not the deflection-dependent
            // mesh), so the status bar's W×H×D and the resize grips are true.
            let (bmin, bmax) = body.bbox().unwrap_or(([0.0; 3], [0.0; 3]));
            self.body_centers.push([
                (bmin[0] + bmax[0]) * 0.5,
                (bmin[1] + bmax[1]) * 0.5,
                (bmin[2] + bmax[2]) * 0.5,
            ]);
            self.body_bounds.push((bmin, bmax));
            match body.tessellate(DEFLECTION_MM) {
                Ok(part) => {
                    let base = mesh.vertices.len() as u32;
                    let face_ids: Vec<u32> =
                        part.face_ids.iter().map(|f| f + face_offset).collect();
                    mesh.vertices.extend(cdt_render::interleave(
                        &part.positions,
                        &part.normals,
                        &face_ids,
                        color,
                    ));
                    mesh.indices.extend(part.indices.iter().map(|i| i + base));
                    face_offset += part.face_ids.iter().copied().max().unwrap_or(0) + 1;

                    // Crisp edges (edge ids offset to stay unique per body).
                    let edge_base = mesh.edge_vertices.len() as u32;
                    let edge_ids: Vec<u32> =
                        part.edge_ids.iter().map(|e| e + edge_offset).collect();
                    mesh.edge_vertices.extend(cdt_render::interleave_edges(
                        &part.edge_positions,
                        &edge_ids,
                    ));
                    mesh.edge_indices
                        .extend(part.edge_indices.iter().map(|i| i + edge_base));
                    edge_offset += part.edge_ids.iter().copied().max().unwrap_or(0) + 1;
                }
                Err(e) => {
                    let id = self.ui.visible.get(idx).copied().unwrap_or(cdt_core::FeatureId(0));
                    self.ui.errors.push((id, format!("could not tessellate: {e}")));
                }
            }
        }
        self.bodies = bodies;
        self.volume_cache.clear(); // geometry changed — stale volumes
        // Ghost previews are stale too; the key bump clears the viewport now,
        // and a still-held hover re-tessellates on the next frame.
        self.ghost_cache.clear();
        self.ghost_key += 1;
        mesh
    }

    fn highlights(&self) -> Highlights {
        Highlights {
            selected: self.selection.selected().to_vec(),
            hovered: self.selection.hovered,
            hover_body: self.hover_body_range(),
        }
    }

    fn ghost(&mut self) -> Option<(u64, &MeshData)> {
        let id = self.ghost_step?;
        Some((self.ghost_key, self.ghost_cache.get(&id)?))
    }

    fn on_pick(&mut self, pick: Option<Pick>, point: Option<[f64; 3]>, additive: bool) {
        // Tape mode swallows clicks: a surface hit pins (or re-pins) the start
        // point, and the selection is left untouched.
        if let Some(tape) = &mut self.tape {
            if let Some(p) = point {
                tape.pin = Some(DVec3::from_array(p));
            }
            return;
        }
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

    fn on_hover(&mut self, pick: Option<Pick>, point: Option<[f64; 3]>) {
        if let Some(tape) = &mut self.tape {
            tape.hover = point.map(DVec3::from_array);
        }
        self.selection.hover(pick);
    }

    fn requested_camera(&mut self) -> Option<CameraPose> {
        self.camera_request.take()
    }

    fn wants_picking(&self) -> bool {
        // In sketch mode, viewport clicks draw/select 2D entities instead.
        self.sketch_session.is_none()
    }

    fn title(&self) -> String {
        // App name first, then the open file (Untitled until first saved), with a
        // leading "●" on the file when there are unsaved changes.
        let name = self
            .current_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Untitled".to_string());
        let mark = if self.dirty { "● " } else { "" };
        format!("Cadette — {mark}{name}")
    }

    fn confirm_close(&mut self) -> bool {
        self.confirm_discard()
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
                let id = self.doc.add_numbered(
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
        let (src, anchor, distance) = match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::PushPull { source, anchor, distance }) => (*source, *anchor, *distance),
            _ => return,
        };
        // Discard a cancelled or no-op push/pull (and the undo it recorded).
        if !commit || distance.abs() < 1e-3 {
            self.doc.history.remove(id);
            self.undo.pop();
            return;
        }

        // Coalesce consecutive push/pulls of the SAME face. Each drag otherwise
        // adds a fresh PushPull, and stacking them fuses/cuts a separate prism per
        // step — which on a face with a hole or fillet leaves coplanar bands that
        // OCCT's same-domain merge can't rejoin, so the wall splits into segments.
        // Folding this push's distance into the previous one keeps the body a
        // single clean prism (a single push of the net distance never splits).
        let prev = match self.doc.history.get(src).map(|f| &f.kind) {
            Some(FeatureKind::PushPull { anchor, distance, .. }) => Some((*anchor, *distance)),
            _ => None,
        };
        if let Some((prev_anchor, prev_distance)) = prev {
            // Same face = same outward normal, and this push's face sits exactly
            // where the previous push left it (its plane offset along the normal).
            let same_normal = anchor.normal.dot(prev_anchor.normal) > 0.999;
            let prev_face = prev_anchor.point.dot(prev_anchor.normal) + prev_distance;
            let this_face = anchor.point.dot(anchor.normal);
            if same_normal && (this_face - prev_face).abs() < 0.05 {
                if let Some(f) = self.doc.history.get_mut(src) {
                    if let FeatureKind::PushPull { distance: d, .. } = &mut f.kind {
                        *d = prev_distance + distance;
                    }
                }
                self.doc.history.remove(id);
                if self.ui.selected == Some(id) {
                    self.ui.selected = Some(src);
                }
            }
        }
    }

    fn grid(&self) -> cdt_render::GridSpec {
        cdt_render::GridSpec {
            spacing: self.ui.grid_spacing as f32,
            half_extent: (self.ui.grid_extent / 2.0) as f32,
            visible: self.ui.show_grid,
            xz: self.ui.show_xz,
            yz: self.ui.show_yz,
            snap: self.ui.snap_to_grid,
        }
    }

    fn viewport_theme(&self) -> ViewportTheme {
        let mut theme = if self.ui.theme.is_dark() {
            ViewportTheme::DARK
        } else {
            ViewportTheme::LIGHT
        };
        theme.body_tint = self.ui.body_tint;
        theme
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
        // A sketch moves in place (transform its plane); everything else adds a
        // Translate/Rotate feature.
        let orig_plane = self.sketch_plane_of(source);
        // A translate drag on a body already ending in a Translate edits that
        // feature (the delta adds onto its offset) instead of stacking a second
        // one — a body keeps at most one trailing Move.
        let reuse = match (handle, self.doc.history.get(source).map(|f| &f.kind)) {
            (
                GizmoHandle::TranslateAxis(_) | GizmoHandle::TranslatePlane(_),
                Some(&FeatureKind::Translate { source: input, offset }),
            ) => Some((input, offset)),
            _ => None,
        };
        let (source, feature, base_offset) = match reuse {
            Some((input, offset)) => (input, Some(source), offset),
            None => (source, None, DVec3::ZERO),
        };
        self.transform = Some(TransformDrag {
            source,
            feature,
            pivot,
            base_offset,
            orig_plane,
            started: false,
        });
    }

    fn update_transform(&mut self, delta: TransformDelta) -> bool {
        let Some(mut t) = self.transform else {
            return false;
        };

        // A sketch moves in place: transform its plane so it stays an editable,
        // extrudable sketch rather than becoming a generic transformed body.
        if let Some(orig) = t.orig_plane {
            let significant = match delta {
                TransformDelta::Translate(o) => DVec3::from_array(o).length() >= 1e-3,
                TransformDelta::Rotate { angle, .. } => angle.abs() >= 1e-4,
            };
            if !significant {
                return true; // sub-threshold jitter: no undo, no edit yet
            }
            if !t.started {
                self.record_undo(self.doc.clone());
                t.started = true;
            }
            let moved = match delta {
                TransformDelta::Translate(o) => orig.translated(DVec3::from_array(o)),
                TransformDelta::Rotate { axis, angle } => {
                    let pivot = DVec3::from_array(t.pivot.unwrap_or([0.0; 3]));
                    orig.rotated(pivot, DVec3::from_array(axis), angle)
                }
            };
            if let Some(f) = self.doc.history.get_mut(t.source) {
                match &mut f.kind {
                    FeatureKind::ConstraintSketch { plane, .. }
                    | FeatureKind::Sketch { plane, .. } => *plane = moved,
                    _ => {}
                }
            }
            self.transform = Some(t);
            return true;
        }

        let (name, kind) = match delta {
            TransformDelta::Translate(offset) => (
                "Move",
                FeatureKind::Translate {
                    source: t.source,
                    offset: t.base_offset + DVec3::from_array(offset),
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
                // First real drag: one undo entry, then splice the transform in
                // so moving a body that feeds other steps propagates downstream
                // (rather than branching off an unused copy). For a free-standing
                // body this is identical to a plain add.
                self.record_undo(self.doc.clone());
                let id = self.doc.insert_transform_numbered(name, kind);
                t.feature = Some(id);
                t.started = true;
                self.ui.selected = Some(id);
            }
            Some(id) => {
                // The first edit of a REUSED tip Translate records the drag's
                // one undo (the spliced path records it on creation above).
                if !t.started {
                    self.record_undo(self.doc.clone());
                    t.started = true;
                }
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
        // Sketch moved in place: on cancel, restore the pre-drag snapshot; on
        // commit, keep the edit (and its single undo entry).
        if t.orig_plane.is_some() {
            if t.started && !commit {
                if let Some(prev) = self.undo.pop() {
                    self.doc = prev;
                }
            }
            return;
        }
        if !t.started {
            return; // never dragged: nothing recorded, nothing to discard
        }
        let Some(id) = t.feature else {
            return;
        };
        // Discard a cancelled or no-op transform. Restore the pre-drag snapshot
        // rather than `history.remove` — the transform was SPLICED (its source's
        // consumers were redirected to it, and the rollback bar advanced), and
        // only a full snapshot restore reliably undoes all of that (removing the
        // feature alone would leave the bar one step too high). A reused tip
        // Translate is a no-op when its offset is back at the grab value.
        let nonzero = match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::Translate { offset, .. }) => {
                (*offset - t.base_offset).length() >= 1e-3
            }
            Some(FeatureKind::Rotate { angle, .. }) => angle.abs() >= 1e-4,
            _ => true,
        };
        if !commit || !nonzero {
            if let Some(prev) = self.undo.pop() {
                self.doc = prev;
            }
        }
    }

    fn resize_box(&self) -> Option<ResizeBox> {
        // Grips show on the bounding box of any single-face-selected body.
        if !matches!(self.selection.selected(), [Pick::Face(_)]) {
            return None;
        }
        let body = self.selected_body()?;
        let idx = self.ui.visible.iter().position(|&f| f == body)?;
        let (min, max) = self.body_bounds.get(idx).copied()?;
        // Skip a degenerate (zero-thickness) body — nothing to grab.
        let span = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
        (span >= 1e-6).then_some(ResizeBox { min, max })
    }

    fn start_resize(&mut self, axis: Axis3, positive: bool) {
        let Some(source) = self.selected_body() else {
            return;
        };
        let Some(idx) = self.ui.visible.iter().position(|&f| f == source) else {
            return;
        };
        let Some((min, max)) = self.body_bounds.get(idx).copied() else {
            return;
        };
        let mesh_extent = DVec3::new(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        // A Scale anchors at the FIXED (opposite) face: min for a +face drag,
        // max for a −face drag, along the dragged axis (other axes scale by 1).
        let mut scale_anchor = DVec3::from_array(min);
        if !positive {
            axis_set(&mut scale_anchor, axis, axis_get(DVec3::from_array(max), axis));
        }
        let radial = matches!(axis, Axis3::X | Axis3::Y); // a cylinder's circular section
        // A bare primitive edits its real dimension(s) parametrically (exact base
        // extents, so cancel restores precisely) for the cases that keep a face
        // anchored naturally: a Box/cylinder-height from its +face, or a centered
        // radius (cylinder X/Y, sphere any). A body already carrying a Scale edits
        // that Scale (never stack two). Everything else — a −face of a corner-
        // anchored primitive, or any non-primitive — gets/extends a Scale.
        let (mode, base_extent, base_factors, anchor, feature) =
            match self.doc.history.get(source).map(|f| &f.kind) {
                Some(FeatureKind::Scale { factors, anchor, .. }) => {
                    (ResizeMode::EditScale, mesh_extent, *factors, *anchor, Some(source))
                }
                Some(FeatureKind::Box { size }) if positive => {
                    (ResizeMode::BoxDim, *size, DVec3::ONE, DVec3::from_array(min), None)
                }
                Some(FeatureKind::Cylinder { radius, height }) if radial || positive => {
                    let e = DVec3::new(2.0 * radius, 2.0 * radius, *height);
                    (ResizeMode::CylDim, e, DVec3::ONE, DVec3::from_array(min), None)
                }
                Some(FeatureKind::Sphere { radius }) => {
                    let e = DVec3::splat(2.0 * radius);
                    (ResizeMode::SphereDim, e, DVec3::ONE, DVec3::from_array(min), None)
                }
                _ => (ResizeMode::NewScale, mesh_extent, DVec3::ONE, scale_anchor, None),
            };
        self.resize = Some(ResizeDrag {
            source,
            mode,
            base_extent,
            base_factors,
            anchor,
            feature,
            changed: false,
        });
    }

    fn update_resize(&mut self, axis: Axis3, extent: f64) -> bool {
        let Some(mut r) = self.resize.take() else {
            return false;
        };
        let target = extent.max(MIN_DIM);
        let changed = match r.mode {
            ResizeMode::BoxDim => {
                let cur = match self.doc.history.get(r.source).map(|f| &f.kind) {
                    Some(FeatureKind::Box { size }) => *size,
                    _ => {
                        self.resize = Some(r);
                        return false;
                    }
                };
                let mut new_size = cur;
                axis_set(&mut new_size, axis, target);
                if cur.abs_diff_eq(new_size, 1e-9) {
                    false
                } else {
                    if !r.changed {
                        self.record_undo(self.doc.clone());
                        r.changed = true;
                    }
                    if let Some(FeatureKind::Box { size }) =
                        self.doc.history.get_mut(r.source).map(|f| &mut f.kind)
                    {
                        *size = new_size;
                    }
                    true
                }
            }
            ResizeMode::CylDim => {
                let (cur_r, cur_h) = match self.doc.history.get(r.source).map(|f| &f.kind) {
                    Some(FeatureKind::Cylinder { radius, height }) => (*radius, *height),
                    _ => {
                        self.resize = Some(r);
                        return false;
                    }
                };
                // X/Y grip → radius (grip tracks the cursor: the +face sits at
                // the radius, half the bbox span); Z grip → height (anchored at
                // the base, 1:1 like a box).
                let base = axis_get(r.base_extent, axis);
                let (mut new_r, mut new_h) = (cur_r, cur_h);
                match axis {
                    Axis3::X | Axis3::Y => new_r = (target - base / 2.0).max(MIN_DIM),
                    Axis3::Z => new_h = target,
                }
                if (new_r - cur_r).abs() < 1e-9 && (new_h - cur_h).abs() < 1e-9 {
                    false
                } else {
                    if !r.changed {
                        self.record_undo(self.doc.clone());
                        r.changed = true;
                    }
                    if let Some(FeatureKind::Cylinder { radius, height }) =
                        self.doc.history.get_mut(r.source).map(|f| &mut f.kind)
                    {
                        *radius = new_r;
                        *height = new_h;
                    }
                    true
                }
            }
            ResizeMode::SphereDim => {
                let cur_r = match self.doc.history.get(r.source).map(|f| &f.kind) {
                    Some(FeatureKind::Sphere { radius }) => *radius,
                    _ => {
                        self.resize = Some(r);
                        return false;
                    }
                };
                // Any grip sets the one radius (the +face sits at the radius).
                let base = axis_get(r.base_extent, axis);
                let new_r = (target - base / 2.0).max(MIN_DIM);
                if (new_r - cur_r).abs() < 1e-9 {
                    false
                } else {
                    if !r.changed {
                        self.record_undo(self.doc.clone());
                        r.changed = true;
                    }
                    if let Some(FeatureKind::Sphere { radius }) =
                        self.doc.history.get_mut(r.source).map(|f| &mut f.kind)
                    {
                        *radius = new_r;
                    }
                    true
                }
            }
            ResizeMode::NewScale => {
                let base = axis_get(r.base_extent, axis);
                if base < 1e-6 {
                    self.resize = Some(r);
                    return false; // can't scale a zero-thickness axis
                }
                let mut factors = DVec3::ONE;
                axis_set(&mut factors, axis, (target / base).max(1e-4));
                match r.feature {
                    None if factors.abs_diff_eq(DVec3::ONE, 1e-9) => false, // no-op yet
                    None => {
                        self.record_undo(self.doc.clone());
                        r.changed = true;
                        // Splice so resizing a body that feeds other steps
                        // propagates downstream instead of leaving a copy.
                        let id = self.doc.insert_transform_numbered(
                            "Resize",
                            FeatureKind::Scale { source: r.source, factors, anchor: r.anchor },
                        );
                        r.feature = Some(id);
                        self.ui.selected = Some(id);
                        true
                    }
                    Some(id) => {
                        if let Some(FeatureKind::Scale { factors: f, .. }) =
                            self.doc.history.get_mut(id).map(|f| &mut f.kind)
                        {
                            *f = factors;
                        }
                        true
                    }
                }
            }
            ResizeMode::EditScale => {
                let base = axis_get(r.base_extent, axis);
                let id = r.feature.expect("EditScale has its Scale feature");
                if base < 1e-6 {
                    self.resize = Some(r);
                    return false;
                }
                // Multiply the additional scale this drag onto the grab factors,
                // so we edit the one Scale rather than stacking another.
                let add = target / base;
                let mut factors = r.base_factors;
                axis_set(&mut factors, axis, axis_get(r.base_factors, axis) * add.max(1e-4));
                if factors.abs_diff_eq(r.base_factors, 1e-9) {
                    false
                } else {
                    if !r.changed {
                        self.record_undo(self.doc.clone());
                        r.changed = true;
                    }
                    if let Some(FeatureKind::Scale { factors: f, .. }) =
                        self.doc.history.get_mut(id).map(|f| &mut f.kind)
                    {
                        *f = factors;
                    }
                    true
                }
            }
        };
        self.resize = Some(r);
        changed
    }

    fn finish_resize(&mut self, commit: bool) {
        let Some(r) = self.resize.take() else {
            return;
        };
        if !r.changed {
            return; // a grip click that never dragged
        }
        match r.mode {
            ResizeMode::BoxDim => {
                if !commit {
                    // Cancel: restore the size (== base extents) and drop the undo.
                    if let Some(FeatureKind::Box { size }) =
                        self.doc.history.get_mut(r.source).map(|f| &mut f.kind)
                    {
                        *size = r.base_extent;
                    }
                    self.undo.pop();
                }
            }
            ResizeMode::CylDim => {
                if !commit {
                    // Restore radius/height from the exact base extents.
                    if let Some(FeatureKind::Cylinder { radius, height }) =
                        self.doc.history.get_mut(r.source).map(|f| &mut f.kind)
                    {
                        *radius = r.base_extent.x / 2.0;
                        *height = r.base_extent.z;
                    }
                    self.undo.pop();
                }
            }
            ResizeMode::SphereDim => {
                if !commit {
                    if let Some(FeatureKind::Sphere { radius }) =
                        self.doc.history.get_mut(r.source).map(|f| &mut f.kind)
                    {
                        *radius = r.base_extent.x / 2.0;
                    }
                    self.undo.pop();
                }
            }
            ResizeMode::NewScale => {
                if let Some(id) = r.feature {
                    // Discard a cancelled or no-op (≈identity) freshly-added scale.
                    let noop = match self.doc.history.get(id).map(|f| &f.kind) {
                        Some(FeatureKind::Scale { factors, .. }) => {
                            factors.abs_diff_eq(DVec3::ONE, 1e-6)
                        }
                        _ => true,
                    };
                    if !commit || noop {
                        // The Scale was SPLICED (consumers redirected, bar
                        // advanced) — restore the pre-drag snapshot to undo all
                        // of it, not just remove the feature.
                        if let Some(prev) = self.undo.pop() {
                            self.doc = prev;
                        }
                    }
                }
            }
            ResizeMode::EditScale => {
                if !commit {
                    // Cancel: restore the existing Scale's factors and drop undo.
                    if let Some(id) = r.feature {
                        if let Some(FeatureKind::Scale { factors, .. }) =
                            self.doc.history.get_mut(id).map(|f| &mut f.kind)
                        {
                            *factors = r.base_factors;
                        }
                    }
                    self.undo.pop();
                }
            }
        }
    }
}

fn error_message(err: &RegenError<cdt_kernel::KernelError>, doc: &Document) -> String {
    match err {
        RegenError::Backend { source, .. } => source.to_string(),
        RegenError::MissingInput { input, reason, .. } => {
            let name = doc.history.get(*input).map(|f| f.name.as_str()).unwrap_or("?");
            match reason {
                MissingReason::Failed => format!("needs '{name}', which failed"),
                MissingReason::Suppressed => format!("needs '{name}' (suppressed)"),
                MissingReason::RolledBack => format!("needs '{name}' (rolled back)"),
                MissingReason::Later => format!("needs '{name}', which comes later in history"),
                MissingReason::Deleted => "needs a step that was deleted".to_string(),
            }
        }
        RegenError::Invalid { reason, .. } => reason.to_string(),
        RegenError::UnsolvedSketch { status, .. } => match status {
            SolveStatus::Contradictory { .. } => {
                "sketch constraints contradict each other — remove one".to_string()
            }
            SolveStatus::Stalled { .. } => {
                "sketch constraints can't all be satisfied — check the dimensions".to_string()
            }
            SolveStatus::Converged => unreachable!("a converged sketch is not an error"),
        },
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut modeler = Modeler::new();
    // Restore the saved look-and-feel. `CADETTE_THEME=light|dark` overrides it
    // for a single run (handy for screenshots) without touching saved prefs.
    let saved = prefs::load();
    modeler.ui.theme = match std::env::var("CADETTE_THEME").as_deref() {
        Ok("light") => cdt_ui::Theme::Light,
        Ok("dark") => cdt_ui::Theme::Dark,
        _ => saved.theme,
    };
    // `CADETTE_TINT=0..100` likewise overrides the body-tint dial for one run.
    modeler.ui.body_tint = match std::env::var("CADETTE_TINT").ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(pct) => (pct / 100.0).clamp(0.0, 1.0),
        None => saved.body_tint,
    };

    // Headless verification/demo harnesses (`--sketch-demo`, `--screenshot`,
    // `--bench-regen`, ...). If one ran, the process is done; otherwise
    // continue to the interactive window.
    let Some(modeler) = demos::run_if_flagged(modeler)? else {
        return Ok(());
    };

    println!("Cadette — interactive modeler");
    println!("  history panel: edit/suppress/reorder features, drag the rollback bar");
    println!("  viewport: left-drag orbit, right/middle-drag pan, scroll zoom\n");

    // Name the app for the macOS menu bar + About panel before the event loop
    // builds the menu (dev binaries would otherwise show "cdt-app").
    #[cfg(target_os = "macos")]
    macos::set_macos_app_identity("Cadette", env!("CARGO_PKG_VERSION"));

    cdt_render::run(modeler)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! These exercise the Controller::mesh path through real OCCT, simulating
    //! the document mutations the history panel performs.
    use super::*;

    use crate::commands::PASTE_OFFSET;
    use crate::sketch::{connect_pen, cubic_screen};
    use cdt_core::LineId;

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
            regen_cache: RegenCache::new(),
            status: None,
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
            body_bounds: Vec::new(),
            volume_cache: HashMap::new(),
            edge_anchors: HashMap::new(),
            tape: None,
            graph_hover: None,
            ghost_step: None,
            ghost_key: 0,
            ghost_cache: HashMap::new(),
            camera_pose: None,
            camera_request: None,
            manipulation: None,
            transform: None,
            resize: None,
            clipboard: None,
            current_path: None,
            dirty: false,
            saved_sig: 0,
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
    fn cubic_screen_hits_endpoints_and_midpoint() {
        let (p0, p1, p2, p3) = (
            egui::pos2(0.0, 0.0),
            egui::pos2(0.0, 10.0),
            egui::pos2(10.0, 10.0),
            egui::pos2(10.0, 0.0),
        );
        assert_eq!(cubic_screen(p0, p1, p2, p3, 0.0), p0);
        assert_eq!(cubic_screen(p0, p1, p2, p3, 1.0), p3);
        // Symmetric control net → midpoint at x=5, y=7.5 (Bernstein at t=0.5).
        let m = cubic_screen(p0, p1, p2, p3, 0.5);
        assert!((m.x - 5.0).abs() < 1e-4 && (m.y - 7.5).abs() < 1e-4, "mid {m:?}");
    }

    #[test]
    fn window_title_tracks_the_file_and_dirty_state() {
        let mut m = Modeler::new();
        assert_eq!(m.title(), "Cadette — Untitled", "no file yet");
        m.dirty = true;
        assert_eq!(m.title(), "Cadette — ● Untitled", "unsaved changes marked");
        m.current_path = Some(std::path::PathBuf::from("/tmp/bracket.cdt"));
        m.dirty = false;
        assert_eq!(m.title(), "Cadette — bracket.cdt", "app name first, then file, clean");
        m.dirty = true;
        assert_eq!(m.title(), "Cadette — ● bracket.cdt");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_app_identity_renames_the_process_without_crashing() {
        // Exercises the objc message sends headlessly: if the info-dictionary
        // mutation were to raise (immutable dict), this aborts the test.
        macos::set_macos_app_identity("Cadette", "9.9.9");
        use objc2_foundation::NSProcessInfo;
        let name = NSProcessInfo::processInfo().processName().to_string();
        assert_eq!(name, "Cadette", "process name drives the macOS app menu");
    }

    #[test]
    fn dirty_clears_when_undoing_back_to_the_saved_state() {
        let mut m = Modeler::new();
        m.refresh_dirty();
        assert!(!m.dirty, "a fresh doc matches its baseline");

        // Edit: snapshot, mutate, record.
        let before = m.doc.clone();
        m.doc
            .add_numbered("Box", FeatureKind::Box { size: DVec3::splat(5.0) });
        m.record_undo(before);
        m.refresh_dirty();
        assert!(m.dirty, "after an edit");

        // Undo back to the baseline → clean again (the reported bug).
        assert!(m.undo());
        m.refresh_dirty();
        assert!(!m.dirty, "undoing to the saved state clears dirty");

        // Redo re-applies the change → dirty again.
        assert!(m.redo());
        m.refresh_dirty();
        assert!(m.dirty, "redo re-dirties");

        // Saving here rebaselines; undoing away from it is dirty again.
        m.mark_saved();
        assert!(!m.dirty, "save marks clean");
        assert!(m.undo());
        m.refresh_dirty();
        assert!(m.dirty, "undo away from the saved state is dirty");
    }

    #[test]
    fn new_document_resets_to_a_blank_project() {
        let mut m = Modeler::new();
        // Content + a bound file, then baseline it clean so New doesn't prompt.
        let id = m
            .doc
            .add_numbered("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.ui.selected = Some(id);
        m.current_path = Some(std::path::PathBuf::from("/tmp/x.cdt"));
        m.record_undo(m.doc.clone());
        m.mark_saved();
        assert!(!m.dirty);

        assert!(m.new_document(), "a blank project starts");
        assert!(m.doc.history.is_empty(), "history is blank");
        assert!(m.current_path.is_none(), "unbound to any file");
        assert!(!m.dirty, "the blank doc is its own clean baseline");
        assert!(m.ui.selected.is_none(), "selection cleared");
        assert!(m.undo.is_empty() && m.redo.is_empty(), "undo/redo cleared");
    }

    #[test]
    fn undo_and_redo_drop_stale_viewport_state() {
        let mut m = Modeler::new();
        m.doc = Document::new("t");
        m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.record_undo(m.doc.clone());
        m.doc.add("B", FeatureKind::Box { size: DVec3::splat(5.0) });

        // Face/edge pick ids are only valid for one regeneration; after the
        // document is swapped they'd resolve to different geometry.
        m.selection.select(Some(Pick::Face(0)), false, None);
        m.edge_anchors.insert(3, DVec3::ZERO);

        assert!(m.undo());
        assert!(m.selection.selected().is_empty(), "undo clears the selection");
        assert!(m.edge_anchors.is_empty(), "undo clears edge anchors");

        m.selection.select(Some(Pick::Face(1)), false, None);
        assert!(m.redo());
        assert!(m.selection.selected().is_empty(), "redo clears the selection");
    }

    #[test]
    fn moving_a_sketch_keeps_it_a_sketch() {
        let mut m = Modeler::new();
        m.doc = Document::new("mv");
        let sk = m.doc.add(
            "Sketch",
            FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: constraint_rectangle(20.0, 20.0) },
        );
        let n0 = m.doc.history.len();

        // Drive a gizmo translate of the sketch (in-place plane edit).
        m.transform = Some(TransformDrag {
            source: sk,
            feature: None,
            pivot: None,
            base_offset: DVec3::ZERO,
            orig_plane: Some(SketchPlane::Xy),
            started: false,
        });
        m.update_transform(TransformDelta::Translate([0.0, 0.0, 5.0]));
        m.finish_transform(true);

        assert_eq!(m.doc.history.len(), n0, "moved in place — no Translate feature added");
        match m.doc.history.get(sk).map(|f| &f.kind) {
            Some(FeatureKind::ConstraintSketch { plane, .. }) => {
                assert!(
                    (plane.origin() - DVec3::new(0.0, 0.0, 5.0)).length() < 1e-9,
                    "sketch plane moved +Z5, got {:?}",
                    plane.origin()
                );
            }
            _ => panic!("still a ConstraintSketch after moving"),
        }
        assert!(m.undo(), "the move is one undo step");
    }

    #[test]
    fn sweep_feature_builds_a_tube_along_its_path() {
        // Circle profile on XY at the origin, swept up a straight +Z path.
        let mut doc = Document::new("sweep");
        let prof = doc.add(
            "Profile",
            FeatureKind::Sketch { plane: SketchPlane::Xy, profile: Profile::Circle { radius: 2.0 } },
        );
        // The path lives on the XZ plane (local v -> world Z), so the open chain
        // (0,0)->(0,10) runs straight up from the origin — tangent +Z at the
        // start, matching the profile normal.
        let mut path = Sketch2d::new();
        let a = path.add_point(0.0, 0.0);
        let b = path.add_point(0.0, 10.0);
        path.add_line(a, b);
        doc.add(
            "Sweep",
            FeatureKind::Sweep {
                profile: prof,
                path: cdt_core::SweepPath::Planar { plane: SketchPlane::Xz, sketch: path },
            },
        );

        let mut m = Modeler::new();
        m.doc = doc;
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!(min[2].abs() < 0.5 && (max[2] - 10.0).abs() < 0.5, "tube spans z 0..10");
        assert!((span(&mesh, 0) - 4.0).abs() < 0.5, "tube ⌀4 in x: {}", span(&mesh, 0));
    }

    #[test]
    fn sweep_path_session_commits_a_sweep_feature() {
        // Select a circle profile, start a path session, draw an open chain, and
        // finishing must add a Sweep (not a sketch) that regenerates into a tube.
        let mut m = Modeler::new();
        m.doc = Document::new("sweepui");
        let prof = m.doc.add(
            "Profile",
            FeatureKind::Sketch { plane: SketchPlane::Xy, profile: Profile::Circle { radius: 2.0 } },
        );
        m.ui.sketch_plane = SketchPlane::Xz;
        let n0 = m.doc.history.len();

        m.start_sweep_path(prof, SketchPlane::Xz);
        let s = m.sketch_session.as_mut().expect("path session");
        assert!(s.sweep_profile.is_some(), "session is in sweep-path mode");
        // Draw an open two-point path going up (XZ plane: local v -> world Z).
        let a = s.sketch.add_point(0.0, 0.0);
        let b = s.sketch.add_point(0.0, 10.0);
        s.sketch.add_line(a, b);

        assert!(m.finish_sketch(), "finishing commits the sweep");
        assert!(m.sketch_session.is_none(), "session ends");
        assert_eq!(m.doc.history.len(), n0 + 1, "exactly one new feature (the Sweep)");
        let tip = *m.ui.visible.last().unwrap_or(&prof);
        let _ = tip;
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!(min[2].abs() < 0.5 && (max[2] - 10.0).abs() < 0.5, "tube spans z 0..10");
    }

    #[test]
    fn editing_a_sweep_path_reshapes_it_in_place() {
        // A Sweep's path is editable: re-open it, lengthen it, and finishing
        // overwrites the SAME feature (no new step) with a longer tube.
        let mut m = Modeler::new();
        m.doc = Document::new("sweepedit");
        let prof = m.doc.add(
            "Profile",
            FeatureKind::Sketch { plane: SketchPlane::Xy, profile: Profile::Circle { radius: 2.0 } },
        );
        let mut path = Sketch2d::new();
        let a = path.add_point(0.0, 0.0);
        let b = path.add_point(0.0, 10.0);
        path.add_line(a, b);
        let sweep = m.doc.add(
            "Sweep",
            FeatureKind::Sweep {
                profile: prof,
                path: cdt_core::SweepPath::Planar { plane: SketchPlane::Xz, sketch: path },
            },
        );
        let n0 = m.doc.history.len();
        let _ = m.mesh();

        // Re-open the path and move its far endpoint from z=10 to z=20.
        assert!(m.edit_sweep_path(sweep), "sweep path re-opens");
        let s = m.sketch_session.as_mut().expect("path session");
        assert!(s.sweep_profile.is_some() && s.editing == Some(sweep));
        if let Some(p) = s.sketch.points.iter_mut().find(|p| p.y > 5.0) {
            p.y = 20.0;
        }
        assert!(m.finish_sketch(), "finishing rewrites the sweep");
        assert_eq!(m.doc.history.len(), n0, "no new feature — edited in place");
        assert_eq!(m.ui.selected, Some(sweep), "the same Sweep stays selected");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (_, max) = bounds(&mesh);
        assert!((max[2] - 20.0).abs() < 0.5, "tube now spans to z≈20: {}", max[2]);
    }

    #[test]
    fn push_pull_feature_extends_a_box() {
        use cdt_core::FaceAnchor;
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
    fn repeated_pushes_of_one_face_coalesce_instead_of_splitting() {
        use cdt_core::FaceAnchor;
        use std::collections::BTreeSet;
        // The real starting shape: a filleted block with a bored hole. Stacking a
        // fresh PushPull per drag leaves coplanar bands the kernel can't remerge
        // (the hole defeats UnifySameDomain), so the wall splits into "segments".
        // finish_manipulation must instead fold each push into the previous one.
        let mut m = Modeler::new();
        m.doc = crate::build_document();
        m.mesh(); // populate ui.visible
        let body = *m.ui.visible.first().expect("one starting body");

        let distinct = |m: &mut Modeler| {
            m.mesh().vertices.iter().map(|v| v.face_id).collect::<BTreeSet<u32>>().len()
        };

        // Drive the manipulation lifecycle for several separate pushes of the top
        // face (centered shape → top at z=40; click off-center, clear of the bore).
        // After the first push, every later one folds into it (the base feature).
        let mut base: Option<FeatureId> = None;
        let mut z = 40.0;
        for d in [-1.0_f64, -2.0, -2.0, 3.0, -2.0] {
            let src = base.unwrap_or(body);
            let anchor = FaceAnchor { point: DVec3::new(10.0, 10.0, z), normal: DVec3::Z };
            let id = m.doc.add("Push/Pull", FeatureKind::PushPull { source: src, anchor, distance: d });
            m.undo.push(m.doc.clone()); // each drag records one undo entry
            m.manipulation = Some(Manipulation { source: src, anchor, feature: Some(id) });
            m.finish_manipulation(true);
            base.get_or_insert(id); // the first push is the coalescing target
            z += d;
        }

        // History collapsed to a single push, and the body is one clean prism
        // (~31 faces: the inset ledge), not the ~36 of stacked segments.
        let pushes = m.doc.history.features().iter().filter(|f|
            matches!(f.kind, FeatureKind::PushPull { .. })).count();
        assert_eq!(pushes, 1, "consecutive pushes fold into one feature");
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert!(distinct(&mut m) <= 31, "wall must not split into segments");
    }

    #[test]
    fn push_pull_and_sketch_work_on_a_resized_body() {
        // A non-uniform Scale turns planar faces into b-splines; without the
        // planarity fallback in the kernel, push/pull and sketch-on-face stop
        // recognizing them. Here the +Z face of a scaled box must still push.
        use cdt_core::FaceAnchor;
        let mut doc = Document::new("ppscale");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let s = doc.add(
            "Resize",
            FeatureKind::Scale {
                source: b,
                factors: DVec3::new(2.0, 1.0, 1.0),
                anchor: DVec3::ZERO,
            },
        );
        // Scaled body: x in [0,20], z in [0,10]. Push the top face up by 5.
        doc.add(
            "Push/Pull",
            FeatureKind::PushPull {
                source: s,
                anchor: FaceAnchor { point: DVec3::new(10.0, 5.0, 10.0), normal: DVec3::Z },
                distance: 5.0,
            },
        );
        let mut m = Modeler::new();
        m.doc = doc;
        let mesh = m.mesh();
        assert!(
            m.ui.errors.is_empty(),
            "push/pull on a resized body should rebuild: {:?}",
            m.ui.errors
        );
        let (_min, max) = bounds(&mesh);
        assert!((max[2] - 15.0).abs() < 0.5, "top pushed to z=15, got {}", max[2]);
        // (push/pull rebuilding at all proves the kernel re-found the scaled
        // body's now-b-spline face as planar — same path sketch-on-face uses.)
    }

    #[test]
    fn editing_a_sketch_moves_a_point_in_place_and_rebuilds_downstream() {
        let mut m = Modeler::new();
        m.doc = Document::new("sk");
        // A right-triangle sketch on XY, extruded into a prism.
        let mut s = Sketch2d::new();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(20.0, 0.0);
        let p2 = s.add_point(0.0, 20.0);
        s.add_line(p0, p1);
        s.add_line(p1, p2);
        s.add_line(p2, p0);
        let sk =
            m.doc.add("Sketch", FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: s });
        m.doc.add("Extrude", FeatureKind::Extrude { source: sk, distance: 10.0 });
        let mesh = m.mesh();
        let (_, max0) = bounds(&mesh);
        assert!((max0[0] - 20.0).abs() < 0.5, "prism starts 20 wide");

        // Re-open the sketch for editing (loads a working copy bound to `sk`).
        assert!(m.edit_sketch(sk));
        assert_eq!(m.editable_sketch(), None, "selection cleared on edit");
        assert!(m.sketch_session.is_some());
        let n0 = m.doc.history.len();

        // Move the (20,0) corner out to (40,0), as a point drag would.
        if let Some(sess) = m.sketch_session.as_mut() {
            sess.sketch.points[p1.0].x = 40.0;
        }
        assert!(m.finish_sketch());
        assert_eq!(m.doc.history.len(), n0, "edit writes back in place — no new feature");
        assert!(m.sketch_session.is_none(), "session ends on finish");

        // The extrude downstream rebuilds wider (footprint now reaches x=40).
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (_, max) = bounds(&mesh);
        assert!((max[0] - 40.0).abs() < 0.5, "moved corner widened the prism: {}", max[0]);
    }

    #[test]
    fn pen_connect_makes_lines_for_corners_and_beziers_for_handles() {
        // Three corner anchors → a triangle of straight lines.
        let mut s = Sketch2d::new();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(20.0, 0.0);
        let c = s.add_point(10.0, 20.0);
        connect_pen(&mut s, a, b, None, None);
        connect_pen(&mut s, b, c, None, None);
        connect_pen(&mut s, c, a, None, None);
        assert_eq!(s.lines.len(), 3);
        assert!(s.beziers.is_empty());
        assert!(s.profile_elements().is_some(), "a closed triangle loop");

        // A segment with handles becomes a bezier carrying them.
        let mut s2 = Sketch2d::new();
        let a = s2.add_point(0.0, 0.0);
        let b = s2.add_point(20.0, 0.0);
        connect_pen(&mut s2, a, b, Some([5.0, 5.0]), Some([15.0, -5.0]));
        assert_eq!(s2.beziers.len(), 1);
        assert_eq!(s2.beziers[0].c1, [5.0, 5.0]);
        assert_eq!(s2.beziers[0].c2, [15.0, -5.0]);
    }

    #[test]
    fn pen_close_lands_on_the_first_anchors_in_handle_for_a_smooth_start() {
        // A smooth first anchor P0=(0,0) drags out handle (0,5): its out-handle
        // (0,5) feeds the first segment's c1; its in-handle (0,-5) is saved for
        // the close. After closing, the two segments at P0 must be mirror
        // symmetric through P0 → smooth.
        let mut s = Sketch2d::new();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(10.0, 0.0);
        let p2 = s.add_point(5.0, 10.0);
        connect_pen(&mut s, p0, p1, Some([0.0, 5.0]), None); // first segment, c1 = out-handle
        connect_pen(&mut s, p1, p2, None, None);
        connect_pen(&mut s, p2, p0, None, Some([0.0, -5.0])); // close, c2 = saved in-handle

        let out_c1 = s.beziers.iter().find(|b| b.a == p0).map(|b| b.c1).unwrap();
        let in_c2 = s.beziers.iter().find(|b| b.b == p0).map(|b| b.c2).unwrap();
        assert_eq!(out_c1, [0.0, 5.0]);
        assert_eq!(in_c2, [-out_c1[0], -out_c1[1]], "handles opposed through P0 → smooth");
    }

    #[test]
    fn a_sketch_with_a_bezier_side_extrudes_into_a_curved_solid() {
        let mut m = Modeler::new();
        m.doc = Document::new("bez");
        // A square whose bottom edge bulges down as a cubic bezier.
        let mut s = Sketch2d::new();
        let p: Vec<_> = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
            .iter()
            .map(|&(x, y)| s.add_point(x, y))
            .collect();
        s.add_bezier(p[0], p[1], [5.0, -10.0], [15.0, -10.0]);
        s.add_line(p[1], p[2]);
        s.add_line(p[2], p[3]);
        s.add_line(p[3], p[0]);
        let sk =
            m.doc.add("Sketch", FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: s });
        m.doc.add("Extrude", FeatureKind::Extrude { source: sk, distance: 10.0 });

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "bezier profile rebuilds: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((max[1] - 20.0).abs() < 0.3, "straight top at y=20");
        assert!(min[1] < -5.0, "bezier bottom bulges past y=−5, got {}", min[1]);
    }

    #[test]
    fn editing_a_sketch_inserts_and_deletes_points() {
        let mut m = Modeler::new();
        m.doc = Document::new("sk");
        // A 20×20 square, extruded.
        let mut s = Sketch2d::new();
        let p: Vec<_> = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
            .iter()
            .map(|&(x, y)| s.add_point(x, y))
            .collect();
        for i in 0..4 {
            s.add_line(p[i], p[(i + 1) % 4]);
        }
        let sk =
            m.doc.add("Sketch", FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: s });
        m.doc.add("Extrude", FeatureKind::Extrude { source: sk, distance: 10.0 });
        let _ = m.mesh();

        // Insert a vertex on the bottom edge and pull it down into a notch.
        assert!(m.edit_sketch(sk));
        {
            let sess = m.sketch_session.as_mut().unwrap();
            let new = sess.sketch.split_line(LineId(0), 10.0, 0.0).unwrap();
            sess.sketch.points[new.0].y = -5.0;
        }
        assert!(m.finish_sketch());
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "insert rebuild: {:?}", m.ui.errors);
        let (min, _) = bounds(&mesh);
        assert!((min[1] + 5.0).abs() < 0.5, "inserted vertex extends y to −5: {}", min[1]);

        // Re-open and delete a corner — still a valid loop that rebuilds.
        assert!(m.edit_sketch(sk));
        {
            let sess = m.sketch_session.as_mut().unwrap();
            let order = sess.sketch.loop_order().unwrap();
            assert_eq!(order.len(), 5, "the inserted vertex persisted");
            assert!(sess.sketch.delete_points(&[order[2]]));
        }
        assert!(m.finish_sketch());
        let _mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "delete rebuild: {:?}", m.ui.errors);
    }

    #[test]
    fn revolve_a_sketch_profile_makes_a_solid_of_revolution() {
        use cdt_core::EdgeAnchor;
        use std::f64::consts::TAU;
        // A 10×20 rectangle on the XZ plane with its left edge (u=0) on the Z
        // axis — plane coords (u,v) map to world (u, 0, v).
        let mut s = Sketch2d::new();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(10.0, 0.0);
        let p2 = s.add_point(10.0, 20.0);
        let p3 = s.add_point(0.0, 20.0);
        s.add_line(p0, p1);
        s.add_line(p1, p2);
        s.add_line(p2, p3);
        s.add_line(p3, p0); // close the loop

        let mut doc = Document::new("rev");
        let sk = doc.add(
            "Profile",
            FeatureKind::ConstraintSketch { plane: SketchPlane::Xz, sketch: s },
        );
        // Revolve a full turn about the left edge (a point on it: world (0,0,10)).
        doc.add(
            "Revolve",
            FeatureKind::Revolve {
                source: sk,
                axis: EdgeAnchor { point: DVec3::new(0.0, 0.0, 10.0) },
                angle: TAU,
            },
        );

        let mut m = Modeler::new();
        m.doc = doc;
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "revolve should build: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        // A solid cylinder radius 10 about Z, height 20: Ø20 footprint, z in [0,20].
        assert!((max[0] - min[0] - 20.0).abs() < 0.6, "x footprint {min:?}..{max:?}");
        assert!((max[1] - min[1] - 20.0).abs() < 0.6, "y footprint {min:?}..{max:?}");
        assert!(min[2].abs() < 0.5 && (max[2] - 20.0).abs() < 0.5, "z {}..{}", min[2], max[2]);
    }

    #[test]
    fn paste_duplicates_the_selected_body_offset_and_intact() {
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();

        // Copy the selected box, then paste.
        m.ui.selected = Some(b);
        m.copy_selected();
        assert_eq!(m.clipboard, Some(b));
        assert!(m.paste());

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        // Two visible bodies now: the original plus the pasted copy.
        assert_eq!(m.ui.visible.len(), 2, "original + paste");
        // Original box stays at [0,10]; the paste is offset by PASTE_OFFSET, so
        // the combined footprint reaches 10 + PASTE_OFFSET on X and Y.
        let (min, max) = bounds(&mesh);
        assert!(min[0].abs() < 0.1 && min[1].abs() < 0.1, "original still at origin");
        assert!(
            (max[0] - (10.0 + PASTE_OFFSET as f32)).abs() < 0.2,
            "paste offset on X: {}",
            max[0]
        );
        assert!((max[1] - (10.0 + PASTE_OFFSET as f32)).abs() < 0.2, "paste offset on Y");

        let pasted = m.ui.selected.unwrap();
        assert_ne!(pasted, b, "paste selected a new feature");

        // Paste is a single undo entry: undo restores the lone original.
        assert!(m.undo());
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 1, "undo removes the whole paste");
    }

    #[test]
    fn pasting_a_moved_body_folds_the_offset_into_its_translate() {
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let mv = m
            .doc
            .add("Move", FeatureKind::Translate { source: b, offset: DVec3::new(30.0, 0.0, 0.0) });
        let _ = m.mesh();

        m.ui.selected = Some(mv);
        m.copy_selected();
        let n0 = m.doc.history.len();
        assert!(m.paste());
        // The clone's own Translate absorbed the paste offset: only the two
        // cloned features were added, no extra "Paste" step on top.
        assert_eq!(m.doc.history.len(), n0 + 2, "clone only: Box + its Move");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 2, "original + paste");
        let (_, max) = bounds(&mesh);
        assert!(
            (max[0] - (40.0 + PASTE_OFFSET as f32)).abs() < 0.2,
            "paste offset folded into the clone's Move: max x {}",
            max[0]
        );
    }

    #[test]
    fn mirror_across_a_face_reflects_a_copy_keeping_the_original() {
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();

        // Select a (planar) face; mirror the body across it.
        m.on_pick(Some(Pick::Face(0)), None, false);
        assert!(m.selection.face_plane.is_some(), "face 0 is planar");
        let n0 = m.doc.history.len();
        assert!(m.mirror_across_selected_face());
        assert!(m.doc.history.len() > n0, "mirror added the clone + Mirror feature");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 2, "original + mirrored copy");

        // Reflecting across one of the box's own (axis-aligned) faces doubles
        // exactly one axis to 20mm; the other two stay 10mm.
        let (min, max) = bounds(&mesh);
        let spans = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let doubled = spans.iter().filter(|&&s| (s - 20.0).abs() < 0.5).count();
        assert_eq!(doubled, 1, "exactly one axis doubled by the mirror: {spans:?}");

        // One undo entry: undo restores the lone original.
        assert!(m.undo());
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 1, "undo removes the whole mirror");
    }

    #[test]
    fn flip_mirrors_a_body_in_place_keeping_its_bounds() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        // A stepped (asymmetric) block: a small box on top of one end of a big one.
        let big = m.doc.add("Big", FeatureKind::Box { size: DVec3::new(20.0, 10.0, 10.0) });
        let small = m.doc.add("Small", FeatureKind::Box { size: DVec3::new(6.0, 10.0, 6.0) });
        let small = m.doc.add(
            "Lift",
            FeatureKind::Translate { source: small, offset: DVec3::new(0.0, 0.0, 10.0) },
        );
        m.doc.add(
            "Step",
            FeatureKind::Boolean { op: BooleanOp::Union, target: big, tool: small },
        );
        let mesh = m.mesh();
        let (min0, max0) = bounds(&mesh);

        m.on_pick(Some(Pick::Face(0)), None, false);
        let n0 = m.doc.history.len();
        assert!(m.flip_in_place(Axis3::X));
        assert_eq!(m.doc.history.len(), n0 + 1, "flip added one Mirror feature (no copy)");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 1, "flip is in place — still one body");
        // Mirroring about the body's own center preserves its bounding box.
        let (min1, max1) = bounds(&mesh);
        for k in 0..3 {
            assert!(
                (min1[k] - min0[k]).abs() < 0.2 && (max1[k] - max0[k]).abs() < 0.2,
                "bounds preserved on axis {k}: {:?}..{:?} vs {:?}..{:?}",
                min0,
                max0,
                min1,
                max1
            );
        }
        assert!(m.undo(), "flip is one undo entry");
    }

    #[test]
    fn fillet_every_edge_of_the_selected_body() {
        // The Actions-pane "Fillet every edge" (moved off the toolbar): picking
        // one face of a body and rounding ALL its edges → one FilletAll feature.
        let mut m = Modeler::new();
        m.doc = Document::new("box");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(20.0) });
        let _ = m.mesh();

        m.on_pick(Some(Pick::Face(0)), None, false);
        let n0 = m.doc.history.len();
        assert!(m.fillet_all_selected_body(), "a body is selected");
        assert_eq!(m.doc.history.len(), n0 + 1, "one FilletAll feature");
        assert!(matches!(
            m.doc.history.get(m.ui.selected.unwrap()).map(|f| &f.kind),
            Some(FeatureKind::FilletAll { .. })
        ));

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "fillet-all rebuilds: {:?}", m.ui.errors);
        let faces: std::collections::BTreeSet<u32> = mesh.vertices.iter().map(|v| v.face_id).collect();
        assert!(faces.len() > 6, "rounding every edge adds faces: {}", faces.len());
    }

    #[test]
    fn fillet_multiple_selected_faces_in_one_feature() {
        let mut m = Modeler::new();
        m.doc = Document::new("box");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(20.0) });
        let _ = m.mesh();

        // Select two faces of the one box.
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.on_pick(Some(Pick::Face(1)), None, true); // additive
        assert_eq!(m.selected_bodies().len(), 1, "both faces on one body");

        let n0 = m.doc.history.len();
        assert!(m.fillet_selected_face_edges());
        assert_eq!(m.doc.history.len(), n0 + 1, "one Fillet feature for all the faces");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "multi-face fillet rebuilds: {:?}", m.ui.errors);
        let faces: std::collections::BTreeSet<u32> = mesh
            .vertices
            .iter()
            .map(|v| v.face_id)
            .collect();
        assert!(faces.len() > 6, "filleting two faces adds rounded faces: {}", faces.len());
    }

    #[test]
    fn group_and_ungroup_bodies_via_multi_face_selection() {
        let mut m = Modeler::new();
        m.doc = Document::new("two");
        m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b0 = m.doc.add("B0", FeatureKind::Box { size: DVec3::splat(10.0) });
        // Offset B so the two are separate bodies.
        m.doc.add("B", FeatureKind::Translate { source: b0, offset: DVec3::new(20.0, 0.0, 0.0) });
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 2, "two independent bodies");

        // Shift-select a face on each body (body B's faces start at face_ranges[1]).
        let fb = m.face_ranges[1];
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.on_pick(Some(Pick::Face(fb)), None, true); // additive
        assert_eq!(m.selected_bodies().len(), 2, "two distinct bodies selected");

        assert!(m.group_selected());
        let _ = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 1, "grouped into one compound body");

        // The whole group still spans both boxes: x from 0 to 30.
        let mesh = m.mesh();
        let (min, max) = bounds(&mesh);
        assert!(min[0].abs() < 0.2 && (max[0] - 30.0).abs() < 0.2, "group spans both: {min:?}..{max:?}");

        // Ungroup (select the group's face) frees both members back to visible.
        m.on_pick(Some(Pick::Face(0)), None, false);
        assert!(m.ungroup_selected());
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 2, "ungroup restores the two bodies");
    }

    #[test]
    fn selection_measurement_reports_bbox_and_volume() {
        let mut m = Modeler::new();
        m.doc = Document::new("meas");
        m.doc
            .add("Box", FeatureKind::Box { size: DVec3::new(10.0, 20.0, 30.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);
        let s = m.selection_measurement().expect("a body is selected");
        assert!(s.contains("10.00 × 20.00 × 30.00"), "bbox W×H×D: {s}");
        assert!(s.contains("6000") && s.contains("mm³"), "volume 6000 mm³: {s}");
        // The picked face's exact area: one of the box's 200/300/600 faces.
        assert!(s.contains("mm²"), "face area shown: {s}");
        assert!(
            ["200.0", "300.0", "600.0"].iter().any(|a| s.contains(a)),
            "a box face area: {s}"
        );
    }

    #[test]
    fn selection_measurement_reports_an_edge_length() {
        let mut m = Modeler::new();
        m.doc = Document::new("meas");
        m.doc
            .add("Box", FeatureKind::Box { size: DVec3::new(10.0, 20.0, 30.0) });
        let _ = m.mesh();
        // Any single box edge measures exactly 10, 20, or 30 mm.
        m.on_pick(Some(Pick::Edge(0)), Some([0.0, 0.0, 0.0]), false);
        let s = m.selection_measurement().expect("an edge is selected");
        assert!(s.contains("edge"), "edge readout: {s}");
        assert!(
            ["10.00 mm", "20.00 mm", "30.00 mm"].iter().any(|l| s.contains(l)),
            "an exact box edge length: {s}"
        );
    }

    #[test]
    fn tape_mode_pins_and_measures_without_touching_the_selection() {
        let mut m = Modeler::new();
        m.doc = Document::new("t");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();

        m.tape = Some(TapeState::default());
        // A click in tape mode pins the start point instead of selecting.
        m.on_pick(Some(Pick::Face(0)), Some([0.0, 0.0, 10.0]), false);
        assert!(m.selection.selected().is_empty(), "tape click doesn't select");
        assert!(m.tape.unwrap().pin.is_some(), "the click pinned a point");

        // Hovering another surface point reads the exact distance.
        m.on_hover(Some(Pick::Face(1)), Some([10.0, 0.0, 10.0]));
        let s = m.tape_measurement().expect("tape engaged");
        assert!(s.contains("10.00 mm"), "distance: {s}");

        // A second click re-pins; leaving the mode ends the read-out.
        m.on_pick(Some(Pick::Face(1)), Some([10.0, 0.0, 10.0]), false);
        m.on_hover(Some(Pick::Face(0)), Some([10.0, 10.0, 10.0]));
        let s = m.tape_measurement().expect("tape engaged");
        assert!(s.contains("10.00 mm"), "re-pinned distance: {s}");
        m.tape = None;
        assert!(m.tape_measurement().is_none(), "read-out gone once exited");
    }

    #[test]
    fn tidy_merges_an_old_projects_stacked_moves_without_moving_anything() {
        // Mimic a pre-coalescing project: a box under two stacked Moves.
        let mut m = Modeler::new();
        m.doc = Document::new("old");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let t1 = m
            .doc
            .add("Move 1", FeatureKind::Translate { source: b, offset: DVec3::new(5.0, 0.0, 0.0) });
        m.doc
            .add("Move 2", FeatureKind::Translate { source: t1, offset: DVec3::new(3.0, 4.0, 0.0) });
        let (min0, max0) = bounds(&m.mesh());
        let n0 = m.doc.history.len();

        assert_eq!(m.doc.tidy(), 1, "one stacked Move merged away");
        assert_eq!(m.doc.history.len(), n0 - 1);
        let (min1, max1) = bounds(&m.mesh());
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        for k in 0..3 {
            assert!(
                (min0[k] - min1[k]).abs() < 1e-4 && (max0[k] - max1[k]).abs() < 1e-4,
                "geometry unchanged: {min0:?}..{max0:?} vs {min1:?}..{max1:?}"
            );
        }
    }

    #[test]
    fn aligning_bodies_shares_a_bounding_box_edge() {
        let mut m = Modeler::new();
        m.doc = Document::new("align");
        m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b0 = m.doc.add("B0", FeatureKind::Box { size: DVec3::splat(10.0) });
        // Offset B in X (separate body) and in Z (misaligned along Z).
        m.doc
            .add("B", FeatureKind::Translate { source: b0, offset: DVec3::new(20.0, 0.0, 5.0) });
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 2, "two independent bodies");
        // They start misaligned along Z.
        assert!(
            (m.body_bounds[0].0[2] - m.body_bounds[1].0[2]).abs() > 1.0,
            "bodies start with different min Z"
        );

        // Select a face on each body.
        let fb = m.face_ranges[1];
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.on_pick(Some(Pick::Face(fb)), None, true);
        assert_eq!(m.selected_bodies().len(), 2);

        let undo_before = m.undo.len();
        assert!(m.align_selected(Axis3::Z, commands::AlignMode::Min), "align moved a body");
        assert_eq!(m.undo.len(), undo_before + 1, "one undo entry for the whole align");

        let _ = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        // Every body now shares the same low Z edge.
        let z0 = m.body_bounds[0].0[2];
        for (min, _max) in &m.body_bounds {
            assert!((min[2] - z0).abs() < 0.2, "all bodies share min Z: {} vs {z0}", min[2]);
        }
    }

    #[test]
    fn ungrouping_a_moved_group_bakes_the_move_into_every_member() {
        let mut m = Modeler::new();
        m.doc = Document::new("two");
        let a = m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) }); // [0,10]
        let b0 = m.doc.add("B0", FeatureKind::Box { size: DVec3::splat(10.0) });
        // B at x[20,30].
        m.doc.add("B", FeatureKind::Translate { source: b0, offset: DVec3::new(20.0, 0.0, 0.0) });
        let b = m.doc.history.features().last().unwrap().id;
        let g = m.doc.add("Group", FeatureKind::Group { members: vec![a, b] });
        // Move the whole group +5 in Z.
        m.doc.add("Move", FeatureKind::Translate { source: g, offset: DVec3::new(0.0, 0.0, 5.0) });
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 1, "one (moved) group body");

        // Ungroup the moved group.
        m.on_pick(Some(Pick::Face(0)), None, false);
        assert!(m.ungroup_selected());
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 2, "two members after ungroup");

        // The +5 Z move is baked into BOTH members — combined z stays [5,15] (it
        // would be [0,15] if the move only landed on the first member), and both
        // members are present (x spans the full [0,30]).
        let (min, max) = bounds(&mesh);
        assert!((min[2] - 5.0).abs() < 0.2, "both kept the baked move: z min {}", min[2]);
        assert!((max[2] - 15.0).abs() < 0.2, "z max {}", max[2]);
        assert!(min[0].abs() < 0.2 && (max[0] - 30.0).abs() < 0.2, "both members present: {min:?}..{max:?}");
    }

    #[test]
    fn deleting_a_group_step_ungroups_instead_of_rewiring_dependents() {
        let mut m = Modeler::new();
        m.doc = Document::new("two");
        let a = m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b0 = m.doc.add("B0", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.doc.add("B", FeatureKind::Translate { source: b0, offset: DVec3::new(20.0, 0.0, 0.0) });
        let b = m.doc.history.features().last().unwrap().id;
        let g = m.doc.add("Group", FeatureKind::Group { members: vec![a, b] });
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 1);

        // The history panel's delete button on the Group row.
        assert!(m.delete_feature(g));
        let _ = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 2, "both members are independent bodies again");
        assert!(m.doc.history.get(g).is_none());
        // One undo entry restores the group.
        assert!(m.undo());
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 1);

        // Deleting an ordinary step heals dependents onto its input as before.
        m.doc = Document::new("one");
        let bx = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let f = m.doc.add("Fillet", FeatureKind::FilletAll { source: bx, radius: 1.0 });
        m.doc.add("Move", FeatureKind::Translate { source: f, offset: DVec3::new(1.0, 0.0, 0.0) });
        assert!(m.delete_feature(f));
        let _ = m.mesh();
        assert!(m.ui.errors.is_empty(), "the move now reads the box directly");
        assert!(!m.delete_feature(f), "already gone");
    }

    #[test]
    fn gizmo_shows_on_face_selection_and_drag_moves_the_body() {
        use cdt_render::{Axis3, GizmoHandle, TransformDelta};
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let _b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
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
    fn a_second_move_drag_edits_the_one_translate_instead_of_stacking() {
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        // First drag: one Move spliced in.
        let n0 = m.doc.history.len();
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::X));
        assert!(m.update_transform(TransformDelta::Translate([5.0, 0.0, 0.0])));
        m.finish_transform(true);
        assert_eq!(m.doc.history.len(), n0 + 1);
        let _ = m.mesh();

        // Second drag: reuses that Move (the offsets sum), no new feature.
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::X));
        assert!(m.update_transform(TransformDelta::Translate([3.0, 0.0, 0.0])));
        m.finish_transform(true);
        assert_eq!(m.doc.history.len(), n0 + 1, "second move reuses the one Translate");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!(
            (min[0] - 8.0).abs() < 0.1 && (max[0] - 18.0).abs() < 0.1,
            "offsets summed: x {min:?}..{max:?}"
        );

        // A cancelled third drag restores the committed position, and undoing
        // once steps back exactly one drag (to x=5), not to some mid-drag state.
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::X));
        assert!(m.update_transform(TransformDelta::Translate([4.0, 0.0, 0.0])));
        m.finish_transform(false);
        let mesh = m.mesh();
        let (min, _) = bounds(&mesh);
        assert!((min[0] - 8.0).abs() < 0.1, "cancel restored x=8: {min:?}");
        assert!(m.undo());
        let mesh = m.mesh();
        let (min, _) = bounds(&mesh);
        assert!((min[0] - 5.0).abs() < 0.1, "one undo = one drag: {min:?}");
    }

    /// Build A ∪ B, then move base A while rolled back. Returns the modeler set
    /// up with A, B0, B(=B0 offset +5X), Fuse(A,B), rolled back so A and B show.
    fn fused_pair_rolled_back() -> (Modeler, FeatureId) {
        let mut m = Modeler::new();
        m.doc = Document::new("t");
        let a = m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b0 = m.doc.add("B0", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b = m.doc
            .add("B", FeatureKind::Translate { source: b0, offset: DVec3::new(5.0, 0.0, 0.0) });
        m.doc
            .add("Fuse", FeatureKind::Boolean { op: BooleanOp::Union, target: a, tool: b });
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 1, "at the tip: one fused body");
        m.doc.set_rollback(3); // A, B0, B active; the Union rolled away
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 2, "rolled back: A and B are separate");
        (m, a)
    }

    #[test]
    fn moving_a_base_body_propagates_into_the_boolean_no_orphan() {
        // The fix: moving a base body that feeds a Boolean must shift the whole
        // part (the move splices into the chain) — not leave a floating copy.
        let (mut m, _a) = fused_pair_rolled_back();

        // Gizmo-move base A up +20 in Y.
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::Y));
        assert!(m.update_transform(TransformDelta::Translate([0.0, 20.0, 0.0])));
        m.finish_transform(true);

        m.doc.rollback_to_tip();
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 1, "still ONE body — the move propagated, no orphan copy");
        // A (inside the union) moved to y[20,30], so the part now reaches y≈30.
        let (_, max) = bounds(&mesh);
        assert!((max[1] - 30.0).abs() < 0.5, "the boolean followed the moved base: max y {}", max[1]);
    }

    #[test]
    fn cancelling_a_rolled_back_move_restores_the_bar() {
        // Reviewer finding #3: a spliced transform advances the rollback bar, so
        // cancel must restore the bar (via snapshot), not just drop the feature —
        // else a rolled-away step pops into view after a cancelled drag.
        let (mut m, _a) = fused_pair_rolled_back();
        let len0 = m.doc.history.len();
        let rb0 = m.doc.rollback();

        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_transform(GizmoHandle::TranslateAxis(Axis3::Y));
        assert!(m.update_transform(TransformDelta::Translate([0.0, 20.0, 0.0])));
        assert_eq!(m.doc.history.len(), len0 + 1, "the drag spliced a transform");
        assert_eq!(m.doc.rollback(), rb0 + 1, "and advanced the bar");

        m.finish_transform(false); // cancel

        assert_eq!(m.doc.history.len(), len0, "the spliced transform is gone");
        assert_eq!(m.doc.rollback(), rb0, "the rollback bar is restored");
        m.doc.history.validate().expect("the graph is intact after cancel");
        assert!(
            m.doc.active_features().iter().all(|f| !matches!(f.kind, FeatureKind::Boolean { .. })),
            "the rolled-away Union did not pop active after a cancelled drag"
        );
    }

    #[test]
    fn resize_grips_show_on_a_box_and_a_drag_grows_one_dimension() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::new(10.0, 10.0, 10.0) });
        let _ = m.mesh();

        // No grips with nothing selected; selecting a face exposes the AABB.
        assert!(m.resize_box().is_none());
        m.on_pick(Some(Pick::Face(0)), None, false);
        let rb = m.resize_box().expect("resize box on a primitive face");
        assert_eq!(rb.min, [0.0; 3]);
        assert!((rb.max[0] - 10.0).abs() < 1e-9);

        // A drag along +X edits the Box size in place (one undo entry).
        let n0 = m.doc.history.len();
        let undo0 = m.undo.len();
        m.start_resize(Axis3::X, true);
        assert!(m.update_resize(Axis3::X, 25.0));
        assert!(m.update_resize(Axis3::X, 30.0)); // edits in place, same undo
        assert_eq!(m.doc.history.len(), n0, "resize edits the Box, adds no feature");
        assert_eq!(m.undo.len(), undo0 + 1, "the whole drag is one undo entry");
        m.finish_resize(true);

        // X grew to 30; Y/Z untouched; the rebuilt body spans 30 along X.
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((max[0] - min[0] - 30.0).abs() < 0.1, "x span {min:?}..{max:?}");
        assert!((max[1] - min[1] - 10.0).abs() < 0.1, "y span unchanged");
    }

    #[test]
    fn cancelled_resize_restores_the_original_size_and_undo() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let id = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        let undo0 = m.undo.len();
        m.start_resize(Axis3::Z, true);
        assert!(m.update_resize(Axis3::Z, 40.0));
        m.finish_resize(false); // cancel

        assert_eq!(m.undo.len(), undo0, "the cancelled drag's undo entry is dropped");
        let restored = matches!(
            &m.doc.history.get(id).unwrap().kind,
            FeatureKind::Box { size } if (size.z - 10.0).abs() < 1e-9
        );
        assert!(restored, "Z restored to its pre-drag value");
    }

    #[test]
    fn resizing_a_cylinder_edits_radius_and_height_staying_circular() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("cyl");
        let c = m.doc.add("Cyl", FeatureKind::Cylinder { radius: 10.0, height: 20.0 });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        // Dragging the +X grip edits the radius in place — no Scale feature, and
        // the X/Y span stays equal (a circle, not an ellipse).
        let n0 = m.doc.history.len();
        m.start_resize(Axis3::X, true);
        // The +X grip tracks the cursor: radius = grip's new x = extent − base/2.
        // base (X span) is 2r = 20, so dragging the grip to extent 25 → radius 15.
        assert!(m.update_resize(Axis3::X, 25.0));
        assert_eq!(m.doc.history.len(), n0, "edits the Cylinder, adds no feature");
        m.finish_resize(true);
        let radius = match &m.doc.history.get(c).unwrap().kind {
            FeatureKind::Cylinder { radius, .. } => *radius,
            _ => panic!("still a cylinder"),
        };
        assert!((radius - 15.0).abs() < 1e-9, "radius now 15, got {radius}");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        let (sx, sy) = (max[0] - min[0], max[1] - min[1]);
        assert!((sx - 30.0).abs() < 0.3 && (sx - sy).abs() < 0.3, "stays circular: {sx} vs {sy}");

        // The +Z grip edits height (anchored at the base, 1:1).
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_resize(Axis3::Z, true);
        assert!(m.update_resize(Axis3::Z, 50.0));
        m.finish_resize(true);
        let height = match &m.doc.history.get(c).unwrap().kind {
            FeatureKind::Cylinder { height, .. } => *height,
            _ => unreachable!(),
        };
        assert!((height - 50.0).abs() < 1e-9, "height now 50, got {height}");
    }

    #[test]
    fn resizing_a_box_from_the_negative_face_pins_the_opposite_side() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::new(10.0, 10.0, 10.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        // The box is [0,10] in X. Dragging the −X grip keeps the +X face (x=10)
        // pinned and extends into −X — a corner-anchored Box can't do that
        // parametrically, so it gets a Scale anchored at the +X face.
        let n0 = m.doc.history.len();
        m.start_resize(Axis3::X, false);
        // base X span 10; widen to 16 → −X face moves to x = 10 − 16 = −6.
        assert!(m.update_resize(Axis3::X, 16.0));
        assert_eq!(m.doc.history.len(), n0 + 1, "a Scale was added for the −face");
        m.finish_resize(true);

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((max[0] - 10.0).abs() < 0.2, "+X face stays at 10, got {}", max[0]);
        assert!((min[0] + 6.0).abs() < 0.2, "−X face now at −6, got {}", min[0]);
    }

    #[test]
    fn resizing_a_cylinder_from_a_radial_negative_face_stays_parametric() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("cyl");
        let c = m.doc.add("Cyl", FeatureKind::Cylinder { radius: 10.0, height: 20.0 });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        // The −X grip of a cylinder still edits the (centered) radius in place —
        // no Scale, no ellipse.
        let n0 = m.doc.history.len();
        m.start_resize(Axis3::X, false);
        assert!(m.update_resize(Axis3::X, 25.0)); // radius 15
        assert_eq!(m.doc.history.len(), n0, "radial −face edits radius, adds no feature");
        m.finish_resize(true);
        let radius = match &m.doc.history.get(c).unwrap().kind {
            FeatureKind::Cylinder { radius, .. } => *radius,
            _ => panic!("still a cylinder"),
        };
        assert!((radius - 15.0).abs() < 1e-9, "radius now 15, got {radius}");
    }

    #[test]
    fn resizing_a_sphere_edits_its_one_radius() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("sph");
        let s = m.doc.add("Sph", FeatureKind::Sphere { radius: 10.0 });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        let n0 = m.doc.history.len();
        m.start_resize(Axis3::Y, true);
        // Grip tracks the cursor: radius = extent − base/2 = 25 − 10 = 15.
        assert!(m.update_resize(Axis3::Y, 25.0));
        assert_eq!(m.doc.history.len(), n0, "edits the Sphere, adds no feature");
        m.finish_resize(true);
        let radius = match &m.doc.history.get(s).unwrap().kind {
            FeatureKind::Sphere { radius } => *radius,
            _ => panic!("still a sphere"),
        };
        assert!((radius - 15.0).abs() < 1e-9, "radius now 15, got {radius}");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        // A sphere: all three spans grow together to ~30.
        for k in 0..3 {
            assert!((max[k] - min[k] - 30.0).abs() < 0.5, "axis {k} span {}", max[k] - min[k]);
        }
    }

    #[test]
    fn a_non_primitive_body_resizes_via_a_scale_feature() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        // Wrap it in a Move so the tip feature is a Translate, not a bare Box.
        m.doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::new(5.0, 0.0, 0.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        // Grips still show (on the moved body's bbox: x in [5,15]).
        let rb = m.resize_box().expect("grips on any body's bbox");
        assert!((rb.min[0] - 5.0).abs() < 0.1 && (rb.max[0] - 15.0).abs() < 0.1);

        // A drag adds a Scale feature (not a dimension edit) and rebuilds bigger.
        let n0 = m.doc.history.len();
        m.start_resize(Axis3::X, true);
        assert!(m.update_resize(Axis3::X, 20.0)); // factor 2.0 about x=5
        assert_eq!(m.doc.history.len(), n0 + 1, "a Scale feature was added");
        let added_scale = m
            .doc
            .history
            .features()
            .iter()
            .any(|f| matches!(&f.kind, FeatureKind::Scale { .. }));
        assert!(added_scale, "the new feature is a Scale");
        m.finish_resize(true);

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        // Anchored at x=5, factor 2 → x now spans [5,25] = 20 wide; y unchanged.
        assert!((max[0] - min[0] - 20.0).abs() < 0.2, "x span {min:?}..{max:?}");
        assert!((max[1] - min[1] - 10.0).abs() < 0.2, "y span unchanged");
        assert!((min[0] - 5.0).abs() < 0.2, "anchored at the -x face");
    }

    #[test]
    fn a_second_resize_edits_the_one_scale_instead_of_stacking() {
        // Regression: two GTransforms in a row segfault OCCT on complex bodies,
        // so a second resize must reuse the existing Scale, not add another.
        use cdt_render::Axis3;
        let mut m = Modeler::new(); // the default filleted + bored part
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        // First resize (X) adds a Scale.
        m.start_resize(Axis3::X, true);
        assert!(m.update_resize(Axis3::X, 60.0));
        m.finish_resize(true);
        let after_first = m.doc.history.len();
        let _ = m.mesh(); // tip is now the Scale
        assert!(m.ui.errors.is_empty(), "errors after first: {:?}", m.ui.errors);

        // Second resize on a DIFFERENT axis must NOT add a feature (would stack a
        // second GTransform → OCCT segfault); it edits the existing Scale.
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_resize(Axis3::Y, true);
        assert!(m.update_resize(Axis3::Y, 70.0));
        m.finish_resize(true);
        assert_eq!(m.doc.history.len(), after_first, "no second Scale was stacked");
        let scale_count = m
            .doc
            .history
            .features()
            .iter()
            .filter(|f| matches!(&f.kind, FeatureKind::Scale { .. }))
            .count();
        assert_eq!(scale_count, 1, "exactly one Scale feature");

        // And it rebuilds (through real OCCT) without crashing, both axes grown.
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors after second: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((max[0] - min[0] - 60.0).abs() < 0.5, "x grown {min:?}..{max:?}");
        assert!((max[1] - min[1] - 70.0).abs() < 0.5, "y grown {min:?}..{max:?}");
    }

    #[test]
    fn a_cancelled_scale_resize_is_discarded() {
        use cdt_render::Axis3;
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::new(5.0, 0.0, 0.0) });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);

        let n0 = m.doc.history.len();
        let undo0 = m.undo.len();
        m.start_resize(Axis3::Z, true);
        assert!(m.update_resize(Axis3::Z, 30.0));
        m.finish_resize(false); // cancel

        assert_eq!(m.doc.history.len(), n0, "the Scale feature is removed");
        assert_eq!(m.undo.len(), undo0, "its undo entry is dropped");
    }

    #[test]
    fn gizmo_ring_drag_rotates_the_body_about_a_pinned_pivot() {
        use cdt_render::{Axis3, GizmoHandle, TransformDelta};
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
        use cdt_render::{Axis3, GizmoHandle};
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
    fn fillet_face_edges_rounds_the_face_perimeter() {
        let mut m = Modeler::new();
        m.doc = Document::new("one");
        m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _ = m.mesh();

        // Select a face, then fillet all its edges.
        m.selection.select(Some(Pick::Face(0)), false, None);
        assert!(m.fillet_selected_face_edges());
        let edges = m.doc.history.features().iter().find_map(|f| match &f.kind {
            FeatureKind::Fillet { edges, .. } => Some(edges.len()),
            _ => None,
        });
        assert_eq!(edges, Some(4), "a box face is bounded by 4 edges");

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert!(!mesh.vertices.is_empty());
    }

    #[test]
    fn extrude_goes_toward_the_camera() {
        // A rectangle sketch on XY (normal +Z). The extrude sign should follow
        // which side the eye is on.
        let mut m = Modeler::new();
        m.doc = Document::new("sk");
        let s = m.doc.add(
            "Sketch",
            FeatureKind::Sketch {
                plane: SketchPlane::Xy,
                profile: Profile::Rectangle { width: 10.0, height: 10.0 },
            },
        );

        // Eye above (+Z): extrude upward, toward the eye → positive distance.
        assert!(m.extrude_sketch(s, [0.0, 0.0, 100.0]));
        let dist = m.doc.history.features().iter().find_map(|f| match f.kind {
            FeatureKind::Extrude { distance, .. } => Some(distance),
            _ => None,
        });
        assert_eq!(dist, Some(20.0));

        // Undo, then extrude from below (-Z): toward the eye is now -Z → negative.
        m.undo();
        assert!(m.extrude_sketch(s, [0.0, 0.0, -100.0]));
        let dist = m.doc.history.features().iter().find_map(|f| match f.kind {
            FeatureKind::Extrude { distance, .. } => Some(distance),
            _ => None,
        });
        assert_eq!(dist, Some(-20.0));
    }

    #[test]
    fn subtract_removes_the_selected_body_from_the_other() {
        use cdt_core::BooleanOp;
        // Box A spans x[0,10]; box B spans x[5,15] (overlapping in x[5,10]).
        let mut m = Modeler::new();
        m.doc = Document::new("two");
        let a = m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let bb = m.doc.add("Bbox", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b = m.doc.add(
            "B",
            FeatureKind::Translate { source: bb, offset: DVec3::new(5.0, 0.0, 0.0) },
        );
        let _ = m.mesh();
        assert_eq!(m.ui.visible, vec![a, b]);

        // Select A as the carver and subtract — A is removed from B (B - A),
        // the opposite of the old "keep the selection" behavior.
        m.ui.selected = Some(a);
        assert!(m.apply_boolean(BooleanOp::Subtract));
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        let (min, max) = bounds(&mesh);
        assert!((min[0] - 10.0).abs() < 0.2, "min x {} (B-A starts where A ends)", min[0]);
        assert!((max[0] - 15.0).abs() < 0.2, "max x {}", max[0]);
    }

    #[test]
    fn subtract_carves_the_selection_out_of_every_other_body() {
        use cdt_core::BooleanOp;
        // A carver at the origin overlapping two separate boxes.
        let mut m = Modeler::new();
        m.doc = Document::new("three");
        let carver = m.doc.add("Carver", FeatureKind::Box { size: DVec3::splat(20.0) });
        let l1 = m.doc.add("L1", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b1 = m.doc.add(
            "B1",
            FeatureKind::Translate { source: l1, offset: DVec3::new(15.0, 0.0, 0.0) },
        );
        let l2 = m.doc.add("L2", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b2 = m.doc.add(
            "B2",
            FeatureKind::Translate { source: l2, offset: DVec3::new(0.0, 15.0, 0.0) },
        );
        let _ = m.mesh();
        assert_eq!(m.ui.visible, vec![carver, b1, b2]);

        m.ui.selected = Some(carver);
        assert!(m.apply_boolean(BooleanOp::Subtract));
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        // Both other bodies were carved -> two results remain, carver consumed.
        assert_eq!(m.ui.visible.len(), 2, "both bodies carved");
        assert!(!mesh.vertices.is_empty());
    }

    #[test]
    fn union_combines_only_the_selected_bodies() {
        use cdt_core::BooleanOp;
        // A and B overlap; C sits far away and must NOT be swallowed.
        let mut m = Modeler::new();
        m.doc = Document::new("u");
        m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let lb = m.doc.add("Lb", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.doc.add(
            "B",
            FeatureKind::Translate { source: lb, offset: DVec3::new(5.0, 0.0, 0.0) },
        );
        let lc = m.doc.add("Lc", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.doc.add(
            "C",
            FeatureKind::Translate { source: lc, offset: DVec3::new(40.0, 0.0, 0.0) },
        );
        let _ = m.mesh();
        assert_eq!(m.ui.visible.len(), 3, "three independent bodies");

        // Shift-select a face on A and a face on B (B's faces start at
        // face_ranges[1]), then union.
        let fb = m.face_ranges[1];
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.on_pick(Some(Pick::Face(fb)), None, true); // additive
        assert_eq!(m.selected_bodies().len(), 2, "two distinct bodies selected");
        assert!(m.apply_boolean(BooleanOp::Union));

        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert_eq!(m.ui.visible.len(), 2, "A∪B is one body; C stays independent");
        // The scene still spans C: x reaches 50, not just A∪B's 15.
        let (_, max) = bounds(&mesh);
        assert!((max[0] - 50.0).abs() < 0.2, "C untouched: max x {}", max[0]);
    }

    #[test]
    fn union_refuses_without_a_two_body_selection() {
        use cdt_core::BooleanOp;
        let mut m = Modeler::new();
        m.doc = Document::new("u");
        let a = m.doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let lb = m.doc.add("Lb", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.doc.add(
            "B",
            FeatureKind::Translate { source: lb, offset: DVec3::new(5.0, 0.0, 0.0) },
        );
        let _ = m.mesh();
        let n0 = m.doc.history.len();

        // Only a Build-Graph selection (the old "fold with all others" trigger):
        // no viewport faces picked, so the union is refused with a hint.
        m.ui.selected = Some(a);
        assert!(!m.apply_boolean(BooleanOp::Union), "no multi-selection: refused");
        assert_eq!(m.doc.history.len(), n0, "document unchanged");
        assert!(
            m.status.as_deref().is_some_and(|s| s.contains("two or more bodies")),
            "a hint explains what to select: {:?}",
            m.status
        );

        // A single body's face selected in the viewport is refused too.
        m.on_pick(Some(Pick::Face(0)), None, false);
        assert!(!m.apply_boolean(BooleanOp::Union), "one body selected: refused");
        assert_eq!(m.doc.history.len(), n0, "document unchanged");
    }

    #[test]
    fn project_round_trips_through_an_cdt_file() {
        let path = std::env::temp_dir().join("cdt-roundtrip-test.cdt");
        let path = path.to_str().unwrap();
        let mut m = Modeler::new();
        let (features, name) = (m.doc.history.len(), m.doc.name.clone());

        m.save_project_to(path).unwrap();
        // Replace the document, then load it back from disk.
        m.doc = Document::new("scratch");
        m.load_project_from(path).unwrap();

        assert_eq!(m.doc.history.len(), features);
        assert_eq!(m.doc.name, name);
        // The loaded document regenerates without error.
        let mesh = m.mesh();
        assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
        assert!(!mesh.vertices.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn graph_hover_ghosts_an_intermediate_step_after_the_debounce() {
        let mut m = Modeler::new();
        m.doc = Document::new("g");
        let b = m.doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        m.doc
            .add("Move", FeatureKind::Translate { source: b, offset: DVec3::new(30.0, 0.0, 0.0) });
        let _ = m.mesh();

        // Hovering the consumed Box row: nothing until the debounce elapses.
        m.ui.hovered_step = Some(b);
        m.update_graph_hover();
        assert_eq!(m.ghost_step, None, "debounce holds the ghost back");
        // Backdate the hover start and run another frame: the ghost appears.
        m.graph_hover = Some((b, std::time::Instant::now() - std::time::Duration::from_secs(1)));
        m.update_graph_hover();
        assert_eq!(m.ghost_step, Some(b), "ghost after the debounce");
        let (_, ghost) = m.ghost().expect("a ghost mesh");
        assert!(!ghost.vertices.is_empty() && !ghost.indices.is_empty());

        // The tint targets the visible body the step feeds into (the Move).
        assert_eq!(m.downstream_visible(b), Some(m.ui.visible[0]));
        assert!(m.highlights().hover_body.is_some(), "fed body tinted");

        // Regeneration invalidates the cached ghost but a held hover restores
        // it on the next frame.
        let _ = m.mesh();
        assert!(m.ghost().is_none(), "stale ghost cleared by regen");
        m.update_graph_hover();
        assert!(m.ghost().is_some(), "held hover re-tessellates");

        // Hover off: the ghost clears and the key bumps so the viewport does too.
        let k = m.ghost_key;
        m.ui.hovered_step = None;
        m.update_graph_hover();
        assert_eq!(m.ghost_step, None);
        assert!(m.ghost().is_none());
        assert_eq!(m.ghost_key, k + 1, "key change tells the viewport to clear");
    }

    #[test]
    fn view_state_round_trips_through_the_project_file() {
        let path = std::env::temp_dir().join("cdt-view-roundtrip-test.cdt");
        let path = path.to_str().unwrap();
        let mut m = Modeler::new();
        m.ui.grid_spacing = 2.5;
        m.ui.grid_extent = 300.0;
        m.ui.show_xz = true;
        m.ui.snap_to_grid = false;
        let pose = CameraPose { target: [1.0, 2.0, 3.0], yaw: 0.5, pitch: -0.25, distance: 77.0 };
        m.camera_pose = Some(pose);
        m.save_project_to(path).unwrap();

        let mut m2 = Modeler::new();
        m2.load_project_from(path).unwrap();
        assert_eq!(m2.ui.grid_spacing, 2.5);
        assert_eq!(m2.ui.grid_extent, 300.0);
        assert!(m2.ui.show_xz && !m2.ui.snap_to_grid, "grid toggles restored");
        assert_eq!(m2.requested_camera(), Some(pose), "saved camera handed to the viewport");
        assert_eq!(m2.requested_camera(), None, "the request is one-shot");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_project_without_view_state_still_loads() {
        // Mimic a pre-view .cdt: a zip holding only document.json.
        let path = std::env::temp_dir().join("cdt-no-view-test.cdt");
        let m = Modeler::new();
        cdt_project::write_path(&path, &cdt_project::ProjectFile::new(m.doc.clone())).unwrap();

        let mut m2 = Modeler::new();
        let (spacing, extent) = (m2.ui.grid_spacing, m2.ui.grid_extent);
        m2.load_project_from(path.to_str().unwrap()).unwrap();
        assert_eq!(m2.requested_camera(), None, "no saved view: camera untouched");
        assert_eq!((m2.ui.grid_spacing, m2.ui.grid_extent), (spacing, extent), "grid defaults kept");
        let _ = std::fs::remove_file(&path);
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
        use cdt_core::{Profile, SketchPlane};
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
            regen_cache: RegenCache::new(),
            status: None,
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
            body_bounds: Vec::new(),
            volume_cache: HashMap::new(),
            edge_anchors: HashMap::new(),
            tape: None,
            graph_hover: None,
            ghost_step: None,
            ghost_key: 0,
            ghost_cache: HashMap::new(),
            camera_pose: None,
            camera_request: None,
            manipulation: None,
            transform: None,
            resize: None,
            clipboard: None,
            current_path: None,
            dirty: false,
            saved_sig: 0,
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

#[cfg(test)]
mod resize_cache {
    use super::*;
    use cdt_render::Axis3;

    /// Regression: resizing an extruded body (which goes through a Scale on a
    /// CACHED extrude) must not "float" the top face off the walls. The cause
    /// was tessellation meshing the shared TShape in place, baking a stale
    /// triangulation the downstream GTransform then reused.
    #[test]
    fn resizing_an_extruded_body_keeps_a_consistent_mesh() {
        let mut m = Modeler::new();
        m.doc = Document::new("ext");
        let mut s = Sketch2d::new();
        let p: Vec<_> = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
            .iter()
            .map(|&(x, y)| s.add_point(x, y))
            .collect();
        for i in 0..4 {
            s.add_line(p[i], p[(i + 1) % 4]);
        }
        let sk =
            m.doc.add("Sketch", FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: s });
        m.doc.add("Extrude", FeatureKind::Extrude { source: sk, distance: 20.0 });
        let _ = m.mesh();
        m.on_pick(Some(Pick::Face(0)), None, false);
        m.start_resize(Axis3::Z, true);

        // Drag the top up through several heights; each rebuild must be a clean
        // prism — every wall face reaches the body's top z.
        for target in [22.0, 25.0, 28.0, 30.0] {
            assert!(m.update_resize(Axis3::Z, target));
            let mesh = m.mesh();
            assert!(m.ui.errors.is_empty(), "errors: {:?}", m.ui.errors);
            use std::collections::BTreeMap;
            let mut face_z: BTreeMap<u32, (f32, f32)> = BTreeMap::new();
            for v in &mesh.vertices {
                let e = face_z.entry(v.face_id).or_insert((f32::MAX, f32::MIN));
                e.0 = e.0.min(v.position[2]);
                e.1 = e.1.max(v.position[2]);
            }
            let zmax = mesh.vertices.iter().map(|v| v.position[2]).fold(f32::MIN, f32::max);
            // Wall/bottom faces start at z≈0; their highest point must equal the
            // body's top (it was 42 vs 22 with the bug — top floating).
            let wall_top = face_z
                .values()
                .filter(|(lo, _)| *lo < 1.0)
                .map(|(_, hi)| *hi)
                .fold(f32::MIN, f32::max);
            assert!(
                (wall_top - zmax).abs() < 0.5,
                "floating top at target {target}: walls reach {wall_top}, body top {zmax}"
            );
        }
    }
}

