//! # cdt-ui
//!
//! Minimal-chrome egui panels. This first panel hosts the **Build Graph**: the
//! feature history drawn as a git-style lane graph (a lane per body, merging at
//! booleans/groups), with selection, suppression, reordering, a rollback bar,
//! and an inline editor for the selected feature's parameters.
//!
//! The panel mutates the [`Document`] directly (all the structural operations
//! are already validated in `cdt-core`) and reports whether anything changed so
//! the host can regenerate geometry. It depends only on egui + core — no wgpu,
//! no kernel.

use egui::{Color32, Context, RichText, Ui};
use egui_phosphor::regular as icon;
use cdt_core::{
    BooleanOp, Constraint, Document, FeatureId, FeatureKind, Helix, PatternLayout, Profile, SketchPlane, DVec3,
};

mod theme;
pub use theme::{Palette, Theme};

/// An icon-only toolbar button: the Phosphor `glyph` with `tip` as its tooltip.
pub fn icon_btn(ui: &mut Ui, glyph: &str, tip: &str, enabled: bool) -> bool {
    ui.add_enabled(enabled, egui::Button::new(RichText::new(glyph).size(18.0)))
        .on_hover_text(tip)
        .clicked()
}

/// Persistent UI state owned by the host across frames.
pub struct HistoryState {
    /// Currently selected feature, if any.
    pub selected: Option<FeatureId>,
    /// Per-feature regeneration errors, set by the host after each rebuild.
    pub errors: Vec<(FeatureId, String)>,
    /// Feature ids whose bodies are currently visible, set by the host after
    /// each rebuild. Operations added from the toolbar reference these.
    pub visible: Vec<FeatureId>,
    /// Whether undo/redo are available, set by the host from its undo stacks.
    pub can_undo: bool,
    pub can_redo: bool,
    /// Grid cell spacing (mm) — the line density and snap increment.
    pub grid_spacing: f64,
    /// Total grid size (mm) across — the workplane's extent.
    pub grid_extent: f64,
    /// Whether to draw the XY ground plane (the primary workplane).
    pub show_grid: bool,
    /// Whether to draw the XZ origin plane (front) as a construction guide.
    pub show_xz: bool,
    /// Whether to draw the YZ origin plane (side) as a construction guide.
    pub show_yz: bool,
    /// Whether moves/resizes snap to the grid.
    pub snap_to_grid: bool,
    /// How strongly bodies wear their Build Graph colour in the viewport:
    /// 0.0 = the neutral default grey, 1.0 = exactly the graph's lane colour.
    pub body_tint: f32,
    /// The plane a new toolbar sketch is started on.
    pub sketch_plane: SketchPlane,
    /// Whether the About window is open.
    pub about_open: bool,
    /// Whether the tape-measure tool is engaged (set by the host; the toolbar
    /// button reflects and toggles it).
    pub tape_active: bool,
    /// The Build Graph row under the cursor this frame (recomputed whenever a
    /// pointer exists), for the host's viewport hover preview.
    pub hovered_step: Option<FeatureId>,
    /// The active look-and-feel (light/dark). The host applies it as egui
    /// visuals each frame and persists it when it changes.
    pub theme: Theme,
}

impl Default for HistoryState {
    fn default() -> Self {
        Self {
            selected: None,
            errors: Vec::new(),
            visible: Vec::new(),
            can_undo: false,
            can_redo: false,
            grid_spacing: 1.0,
            grid_extent: 200.0,
            show_grid: true,
            show_xz: false,
            show_yz: false,
            snap_to_grid: true,
            body_tint: 0.68,
            sketch_plane: SketchPlane::Xy,
            about_open: false,
            tape_active: false,
            hovered_step: None,
            theme: Theme::default(),
        }
    }
}

/// What the history panel reports back to the host for one frame.
#[derive(Default)]
pub struct HistoryResponse {
    /// The document changed and geometry should be regenerated.
    pub changed: bool,
    /// The undo button was clicked.
    pub undo: bool,
    /// The redo button was clicked.
    pub redo: bool,
    /// The "Sketch" tool was clicked — the host should enter sketch-draw mode
    /// on this plane.
    pub start_sketch: Option<SketchPlane>,
    /// "Export STL" was clicked — the host should write the model to a file.
    pub export_stl: bool,
    /// "Export STEP" was clicked — the host should write the model as STEP.
    pub export_step: bool,
    /// "Import STEP" was clicked — the host should pick a STEP file and add
    /// it as an Import feature.
    pub import_step: bool,
    /// "New" was clicked — the host should reset to a blank project (confirming
    /// first if there are unsaved changes).
    pub new_project: bool,
    /// "Save" was clicked — the host should write the project to a file.
    pub save_project: bool,
    /// "Open" was clicked — the host should load a project from a file.
    pub open_project: bool,
    /// A boolean op was clicked. Subtract carves the selected body out of the
    /// other visible bodies; union/intersect combine exactly the bodies behind
    /// the viewport multi-selection (the host refuses with a hint otherwise).
    pub boolean: Option<BooleanOp>,
    /// "Extrude" was clicked — the host extrudes this sketch (toward the camera).
    pub extrude: Option<FeatureId>,
    /// The theme toggle was clicked — `state.theme` changed and the host should
    /// persist it.
    pub theme_changed: bool,
    /// The tape-measure button was clicked — the host toggles the tool.
    pub tape_toggle: bool,
    /// "Tidy history" was clicked — the host merges stacked Translates (one
    /// undo entry) and reports the result in the status bar.
    pub tidy: bool,
    /// A step's delete button was clicked — the host removes it (ungrouping a
    /// Group so its members keep their places).
    pub delete: Option<FeatureId>,
    /// The body-tint dial moved — `state.body_tint` changed and the host
    /// should persist it (rendering picks it up per frame, no regen needed).
    pub tint_changed: bool,
}

impl HistoryState {
    fn error_for(&self, id: FeatureId) -> Option<&str> {
        self.errors
            .iter()
            .find(|(e, _)| *e == id)
            .map(|(_, m)| m.as_str())
    }

    /// The body a new unary op (move, fillet) should act on: the selection if
    /// it is currently visible, otherwise the most recently visible body.
    fn unary_source(&self) -> Option<FeatureId> {
        if let Some(sel) = self.selected {
            if self.visible.contains(&sel) {
                return Some(sel);
            }
        }
        self.visible.last().copied()
    }

    /// Whether a boolean is available: a body is selected, and there's at least
    /// one other visible body to combine it with. For subtract the selection is
    /// the carver; union/intersect further require faces selected on two or
    /// more bodies, which only the host can see — it refuses with a status
    /// hint when the click can't be honoured.
    fn can_boolean(&self) -> bool {
        self.selected.is_some_and(|s| self.visible.contains(&s)) && self.visible.len() >= 2
    }
}

/// Draw the history side panel and report what the user did this frame.
pub fn history_panel(
    ctx: &Context,
    doc: &mut Document,
    state: &mut HistoryState,
) -> HistoryResponse {
    let mut resp = HistoryResponse::default();

    // Persistent commands live in the top toolbar; the left panel is the Build
    // Graph + workplane settings + the selected-feature editor.
    command_bar(ctx, doc, state, &mut resp);

    // egui 0.34 is mid-migration to a unified `Panel`; the context-level
    // `.show(ctx)` is deprecated in favor of `show_inside(ui)`, but at the top
    // of a frame we only hold a `&Context`. `.show(ctx)` still works, so allow
    // it until the replacement for context-level panels settles.
    #[allow(deprecated)]
    egui::Panel::left("history_panel")
        .resizable(true)
        .default_size(300.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Grid");
                ui.add(
                    egui::DragValue::new(&mut state.grid_spacing)
                        .speed(0.1)
                        .range(0.05..=1000.0)
                        .suffix(" mm"),
                )
                .on_hover_text(
                    "Cell spacing (snap increment). The finest drawable spacing \
                     depends on the grid size — shrink the grid to go finer.",
                );
                ui.add(
                    egui::DragValue::new(&mut state.grid_extent)
                        .speed(5.0)
                        .range(10.0..=5000.0)
                        .suffix(" mm"),
                )
                .on_hover_text("Total grid size (extent)");
                ui.label("Tint");
                let mut pct = (state.body_tint * 100.0).round();
                if ui
                    .add(egui::DragValue::new(&mut pct).speed(1.0).range(0.0..=100.0).suffix("%"))
                    .on_hover_text(
                        "How strongly bodies wear their Build Graph colour: \
                         0% = neutral grey, 100% = exactly the graph's colours",
                    )
                    .changed()
                {
                    state.body_tint = (pct as f32 / 100.0).clamp(0.0, 1.0);
                    resp.tint_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Planes");
                ui.checkbox(&mut state.show_grid, "XY")
                    .on_hover_text("Ground plane (workplane)");
                ui.checkbox(&mut state.show_xz, "XZ")
                    .on_hover_text("Front origin plane (guide)");
                ui.checkbox(&mut state.show_yz, "YZ")
                    .on_hover_text("Side origin plane (guide)");
                ui.checkbox(&mut state.snap_to_grid, "Snap")
                    .on_hover_text("Snap moves and resizes to the grid (Alt overrides)");
            });
            ui.separator();

            resp.changed |= rollback_controls(ui, doc);
            ui.separator();

            ui.horizontal(|ui| {
                let steps = doc.history.len();
                ui.label(RichText::new(format!("Build Graph ({steps} steps)")).strong())
                    .on_hover_text("Each body is a lane; lanes merge at a boolean or group.");
                // Reorder the SELECTED step (backward-only refs may reject a move).
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(icon::BROOM).small())
                        .on_hover_text(
                            "Tidy history: merge stacked Moves (from older versions) \
                             into one per body — geometry is unchanged",
                        )
                        .clicked()
                    {
                        resp.tidy = true;
                    }
                    let sel_idx = state
                        .selected
                        .and_then(|id| doc.history.features().iter().position(|f| f.id == id));
                    let last = doc.history.len().saturating_sub(1);
                    if ui
                        .add_enabled(sel_idx.is_some_and(|i| i < last), egui::Button::new("🔽").small())
                        .on_hover_text("Move the selected step down")
                        .clicked()
                    {
                        if let (Some(id), Some(i)) = (state.selected, sel_idx) {
                            let _ = doc.history.reorder(id, i + 1);
                            resp.changed = true;
                        }
                    }
                    if ui
                        .add_enabled(sel_idx.is_some_and(|i| i > 0), egui::Button::new("🔼").small())
                        .on_hover_text("Move the selected step up")
                        .clicked()
                    {
                        if let (Some(id), Some(i)) = (state.selected, sel_idx) {
                            let _ = doc.history.reorder(id, i - 1);
                            resp.changed = true;
                        }
                    }
                });
            });

            // The selected-feature editor + problems now live in the right
            // Actions panel; the Build Graph fills the rest of this panel.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    resp.changed |= build_graph(ui, doc, state, &mut resp);
                });
        });

    resp
}

/// The persistent top command bar: file, edit/export, and create/modify ops as
/// icon buttons (tooltips carry the labels). Primitives are always available;
/// unary ops (Move, Fillet) and booleans enable only when valid inputs exist.
/// New features are selected on creation.
fn command_bar(
    ctx: &Context,
    doc: &mut Document,
    state: &mut HistoryState,
    resp: &mut HistoryResponse,
) {
    // Operation inputs up front (immutable borrows) so the mutating `add` closure
    // can take exclusive borrows afterward.
    let unary = state.unary_source();
    let can_bool = state.can_boolean();
    let extrude_source = unary.filter(|id| {
        matches!(
            doc.history.get(*id).map(|f| &f.kind),
            Some(FeatureKind::Sketch { .. } | FeatureKind::ConstraintSketch { .. })
        )
    });

    // Top-level `Panel::show` is deprecated in favor of `show_inside`, but
    // migrating means restructuring the whole frame layout — done together in
    // a dedicated egui panel pass.
    #[allow(deprecated)]
    egui::Panel::top("command_bar").show(ctx, |ui| {
        ui.add_space(2.0);
        ui.horizontal_wrapped(|ui| {
            // File.
            if icon_btn(ui, icon::FILE_PLUS, "New project", true) {
                resp.new_project = true;
            }
            if icon_btn(ui, icon::FOLDER_OPEN, "Open project (.cdt)", true) {
                resp.open_project = true;
            }
            if icon_btn(ui, icon::FLOPPY_DISK, "Save project (.cdt)", true) {
                resp.save_project = true;
            }
            if icon_btn(ui, icon::FILE_ARROW_DOWN, "Import a STEP file as a new body", true) {
                resp.import_step = true;
            }
            ui.separator();
            // Edit + export.
            if icon_btn(ui, icon::ARROW_ARC_LEFT, "Undo (⌘Z)", state.can_undo) {
                resp.undo = true;
            }
            if icon_btn(ui, icon::ARROW_ARC_RIGHT, "Redo (⇧⌘Z)", state.can_redo) {
                resp.redo = true;
            }
            if icon_btn(ui, icon::EXPORT, "Export the model as an STL for printing", true) {
                resp.export_stl = true;
            }
            if icon_btn(ui, icon::FILE_ARROW_UP, "Export the model as a STEP file for other CAD", true) {
                resp.export_step = true;
            }
            ui.separator();

            // Create (always) + modify (selection-gated). `add` passes `state`
            // per call so `plane_picker` can also borrow it between calls.
            let mut changed = false;
            let mut add = |state: &mut HistoryState, base: &str, kind: FeatureKind| {
                let id = doc.add_numbered(base, kind);
                state.selected = Some(id);
                changed = true;
            };
            if icon_btn(ui, icon::CUBE, "Box", true) {
                add(state, "Box", FeatureKind::Box { size: DVec3::splat(20.0) });
            }
            if icon_btn(ui, icon::CYLINDER, "Cylinder", true) {
                add(state, "Cylinder", FeatureKind::Cylinder { radius: 10.0, height: 20.0 });
            }
            if icon_btn(ui, icon::SPHERE, "Sphere", true) {
                add(state, "Sphere", FeatureKind::Sphere { radius: 10.0 });
            }
            // Sketch: the plane is chosen at the moment of starting (a menu), not
            // from a persistent global picker — each sketch names its own plane.
            // Sized like `icon_btn`'s glyphs — a bare str would render at the
            // default font size and leave this button shorter than its row.
            ui.menu_button(RichText::new(icon::PENCIL_SIMPLE).size(18.0), |ui| {
                ui.label("Sketch on plane:");
                for (p, item) in [
                    (SketchPlane::Xy, "XY — top"),
                    (SketchPlane::Xz, "XZ — front"),
                    (SketchPlane::Yz, "YZ — side"),
                ] {
                    if ui.button(item).clicked() {
                        resp.start_sketch = Some(p);
                        state.sketch_plane = p; // remember as the default elsewhere
                        ui.close_menu();
                    }
                }
            })
            .response
            .on_hover_text("Sketch a 2D profile — choose the plane");
            ui.separator();

            let extrude_ok = extrude_source.is_some();
            if icon_btn(ui, icon::ARROW_LINE_UP, "Extrude the selected/last sketch into a solid", extrude_ok) {
                resp.extrude = extrude_source;
            }
            // Move is done directly with the gizmo, and "Fillet every edge" lives
            // in the Actions pane when a body is selected — neither needs a
            // persistent toolbar button.
            for (glyph, op, tip) in [
                (icon::UNITE, BooleanOp::Union, "Union: merge the selected bodies into one (select faces on each)"),
                (icon::SUBTRACT, BooleanOp::Subtract, "Subtract: carve the selected body out of the others"),
                (icon::INTERSECT, BooleanOp::Intersect, "Intersect: keep only where the selected bodies overlap (select faces on each)"),
            ] {
                if icon_btn(ui, glyph, tip, can_bool) {
                    resp.boolean = Some(op);
                }
            }
            ui.separator();
            // Tape measure: a mode toggle, so it renders pressed while active.
            let tape_tip = if state.tape_active {
                "Tape measure (active) — click a surface to pin, hover to read; Esc or click here to exit"
            } else {
                "Tape measure: pin a point on a surface, hover to read the distance"
            };
            if ui
                .selectable_label(state.tape_active, RichText::new(icon::RULER).size(18.0))
                .on_hover_text(tape_tip)
                .clicked()
            {
                resp.tape_toggle = true;
            }
            resp.changed |= changed;

            // About + theme toggle, pushed to the far right.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if icon_btn(ui, icon::INFO, "About Cadette", true) {
                    state.about_open = !state.about_open;
                }
                // Sun in dark mode (switch to light), moon in light mode.
                let (glyph, tip) = if state.theme.is_dark() {
                    (icon::SUN, "Switch to light theme")
                } else {
                    (icon::MOON, "Switch to dark theme")
                };
                if icon_btn(ui, glyph, tip, true) {
                    state.theme = state.theme.toggled();
                    resp.theme_changed = true;
                }
            });
        });
    });

    about_window(ctx, state);
}

/// The About window: app name, version, and links. Toggled by the ⓘ button.
fn about_window(ctx: &Context, state: &mut HistoryState) {
    let mut open = state.about_open;
    egui::Window::new("About Cadette")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.heading("Cadette");
            ui.label(RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION"))).weak());
            ui.add_space(6.0);
            ui.hyperlink_to("cadette.org", "https://cadette.org");
            ui.hyperlink_to("Source on GitHub", "https://github.com/chasegan/cadette");
            ui.add_space(6.0);
            ui.weak("A parametric, direct-manipulation 3D modeler for printing.");
        });
    state.about_open = open;
}

fn rollback_controls(ui: &mut Ui, doc: &mut Document) -> bool {
    let mut changed = false;
    let len = doc.history.len();
    let mut position = doc.rollback();

    ui.horizontal(|ui| {
        ui.label("Rollback");
        if ui
            .add(egui::Slider::new(&mut position, 0..=len).show_value(true))
            .changed()
        {
            doc.set_rollback(position);
            changed = true;
        }
        if ui.button("⤓").on_hover_text("Roll to tip").clicked() && !doc.is_at_tip() {
            doc.rollback_to_tip();
            changed = true;
        }
    });

    changed
}

const LANE_W: f32 = 14.0;
const GRAPH_ROW_H: f32 = 22.0;
const NODE_R: f32 = 4.0;

/// A distinct colour per lane so parallel body lines read apart — used when a
/// body has no user colour. The palette lives in cdt-core (`LANE_PALETTE`) so
/// the viewport tints bodies with EXACTLY these colours: the Build Graph is
/// the scene's legend.
fn lane_color(lane: usize) -> Color32 {
    let [r, g, b] = cdt_core::LANE_PALETTE[lane % cdt_core::LANE_PALETTE.len()];
    Color32::from_rgb(r, g, b)
}

/// A body's linear-RGB colour as an egui (sRGB) colour for the lane graph.
fn linear_to_color32(c: [f32; 3]) -> Color32 {
    Color32::from(egui::Rgba::from_rgb(c[0], c[1], c[2]))
}

/// The history as a git-style lane graph: a column per body line, a node per
/// feature, lanes converging at a merge (boolean/group) and splitting at a
/// branch. Each row keeps the list's interactions (suppress / select / delete /
/// reorder); the gutter on the left is the graph.
fn build_graph(
    ui: &mut Ui,
    doc: &mut Document,
    state: &mut HistoryState,
    resp: &mut HistoryResponse,
) -> bool {
    let mut changed = false;
    // Recompute the hovered row from scratch each frame — but only while a
    // pointer exists, so a headless pass (screenshots) keeps a pre-arranged
    // hover instead of wiping it.
    if ui.ctx().pointer_latest_pos().is_some() {
        state.hovered_step = None;
    }
    let error_color = state.theme.palette().error;
    let graph = doc.history.build_graph();
    let gutter_w = graph.lane_count.max(1) as f32 * LANE_W;
    let active = doc.rollback();

    // Snapshot rows (with their lane layout) so we can mutate the doc in the loop.
    let rows: Vec<_> = doc
        .history
        .features()
        .iter()
        .zip(graph.rows.iter())
        .map(|(f, gr)| (f.id, f.name.clone(), f.suppressed, gr.clone()))
        .collect();

    // A body's canonical colour: its user colour (walking down the input
    // chain) or the palette keyed by its lane — ONE formula, shared with the
    // viewport's body tinting, so the graph is an exact legend for the scene.
    // Ribbons are painted per SEGMENT by the body flowing through them, so a
    // freed-and-reused column shows each body's own colour on its stretch.
    let mut body_colors: std::collections::HashMap<FeatureId, Color32> =
        std::collections::HashMap::new();
    for gr in graph.rows.iter() {
        for &occ in gr.occupants.iter().chain(std::iter::once(&gr.id)) {
            body_colors.entry(occ).or_insert_with(|| match doc.history.resolve_color(occ) {
                Some(c) => linear_to_color32(c),
                None => lane_color(
                    graph.rows.iter().find(|r| r.id == occ).map_or(0, |r| r.lane),
                ),
            });
        }
    }
    let body_color = |id: FeatureId| -> Color32 {
        body_colors.get(&id).copied().unwrap_or_else(|| lane_color(0))
    };

    // Contiguous rows so the lane lines join across rows.
    ui.spacing_mut().item_spacing.y = 0.0;

    // Colours of the lanes flowing INTO the current row (the previous row's
    // outgoing occupants, keyed by lane).
    let mut top_colors: std::collections::HashMap<usize, Color32> =
        std::collections::HashMap::new();
    for (index, (id, name, suppressed, gr)) in rows.into_iter().enumerate() {
        let node_color = body_color(id);
        let bot_colors: std::collections::HashMap<usize, Color32> = gr
            .open
            .iter()
            .zip(gr.occupants.iter())
            .map(|(&l, &occ)| (l, body_color(occ)))
            .collect();
        let rolled_back = index >= active;
        let error = state.error_for(id).map(str::to_owned);
        let selected = state.selected == Some(id);

        let row = ui.horizontal(|ui| {
            // --- Lane gutter -------------------------------------------------
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(gutter_w, GRAPH_ROW_H),
                egui::Sense::hover(),
            );
            let p = ui.painter().clone();
            let (top, bot, mid) = (rect.top(), rect.bottom(), rect.center().y);
            let lane_x = |l: usize| rect.left() + l as f32 * LANE_W + LANE_W * 0.5;
            let node_x = lane_x(gr.lane);
            let dim = rolled_back || suppressed;
            let stroke = |c: Color32| egui::Stroke::new(1.5, if dim { c.gamma_multiply(0.4) } else { c });
            let top_col = |l: usize| top_colors.get(&l).copied().unwrap_or_else(|| lane_color(l));
            let bot_col = |l: usize| bot_colors.get(&l).copied().unwrap_or_else(|| lane_color(l));
            // Top half: each incoming lane connects to the node, others pass
            // down. An input lane that STAYS OPEN below (a body feeding more
            // consumers further down, e.g. one carver subtracted from several
            // bodies) FORKS: one branch dives into the node, one continues
            // straight down its own column — not either/or, which left a gap
            // and a floating stub where the body's line should flow through.
            for &l in &gr.top_open {
                let feeds_node = l == gr.lane || gr.input_lanes.contains(&l);
                if feeds_node {
                    p.line_segment([egui::pos2(lane_x(l), top), egui::pos2(node_x, mid)], stroke(top_col(l)));
                }
                if !feeds_node || (l != gr.lane && gr.open.contains(&l)) {
                    p.line_segment([egui::pos2(lane_x(l), top), egui::pos2(lane_x(l), mid)], stroke(top_col(l)));
                }
            }
            // Bottom half: the node's lane (and any other open lane) flows down.
            for &l in &gr.open {
                let x = if l == gr.lane { node_x } else { lane_x(l) };
                p.line_segment([egui::pos2(x, mid), egui::pos2(x, bot)], stroke(bot_col(l)));
            }
            // The node: a ring for a merge, a dot otherwise.
            let node_c = if error.is_some() {
                error_color
            } else if dim {
                Color32::from_gray(120)
            } else {
                node_color
            };
            let center = egui::pos2(node_x, mid);
            if gr.is_merge {
                p.circle(center, NODE_R, Color32::from_gray(40), egui::Stroke::new(2.0, node_c));
            } else {
                p.circle_filled(center, NODE_R, node_c);
            }
            if selected {
                p.circle_stroke(center, NODE_R + 3.0, egui::Stroke::new(1.5, Color32::WHITE));
            }

            // --- Row controls ------------------------------------------------
            let mut sup = suppressed;
            if ui
                .add(egui::Checkbox::without_text(&mut sup))
                .on_hover_text("Suppress")
                .changed()
            {
                doc.history.set_suppressed(id, sup);
                changed = true;
            }

            let mut text = RichText::new(name.clone());
            if error.is_some() {
                text = text.color(error_color);
            } else if dim {
                text = text.weak();
            }
            let response = ui.selectable_label(selected, text);
            if let Some(msg) = &error {
                response.clone().on_hover_text(msg);
            }
            if response.clicked() {
                state.selected = if selected { None } else { Some(id) };
            }

            // Reordering moved to the external Up/Down buttons on the Build Graph
            // header (they act on the selected step); only delete stays per-row.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("🗑").on_hover_text("Delete").clicked() {
                    // The host deletes: a Group needs its transform chain baked
                    // onto the members first (see the app's ungroup command),
                    // which a bare `history.remove` would silently skip.
                    resp.delete = Some(id);
                }
            });
        });
        // Report the row under the cursor for the host's viewport
        // hover preview (ghost + fed-body tint).
        if row.response.contains_pointer() {
            state.hovered_step = Some(id);
        }
        top_colors = bot_colors;
    }

    changed
}

/// What the Properties editor reported this frame.
#[derive(Default)]
pub struct EditorResponse {
    /// A parameter value changed — the host should regenerate.
    pub changed: bool,
    /// The "Edit…" button for a complex step (sketch/sweep) was clicked — the
    /// host should re-open that step's proper editor.
    pub edit: bool,
}

pub fn selected_editor(ui: &mut Ui, doc: &mut Document, id: FeatureId) -> EditorResponse {
    let mut changed = false;
    let Some(feature) = doc.history.get_mut(id) else {
        return EditorResponse::default();
    };

    // Name is primary; the operation kind is a dim secondary (it's dropped from
    // the Build Graph rows, so this is where you confirm what a step actually is).
    ui.horizontal(|ui| {
        ui.label(RichText::new(&feature.name).strong());
        ui.label(RichText::new(feature.kind.type_name()).weak());
    });
    ui.add_space(2.0);

    // A complex step's full state can't be listed as fields — instead it offers
    // an "Edit…" button (set here, drawn after the match) that re-opens its
    // proper editor. `None` = the fields above fully describe the step.
    let mut edit_label: Option<&str> = None;

    match &mut feature.kind {
        FeatureKind::Box { size } => {
            changed |= drag(ui, "X", &mut size.x);
            changed |= drag(ui, "Y", &mut size.y);
            changed |= drag(ui, "Z", &mut size.z);
        }
        FeatureKind::Cylinder { radius, height } => {
            changed |= drag(ui, "Radius", radius);
            changed |= drag(ui, "Height", height);
        }
        FeatureKind::Sphere { radius } => {
            changed |= drag(ui, "Radius", radius);
        }
        FeatureKind::Import { asset } => {
            ui.horizontal(|ui| {
                ui.label("File");
                ui.label(asset.as_str());
            });
            ui.label("The STEP file travels inside the project. To use a different file, import it again.");
        }
        FeatureKind::Sketch { plane, profile } => {
            changed |= plane_combo(ui, plane);
            changed |= profile_combo(ui, profile);
            match profile {
                Profile::Rectangle { width, height } => {
                    changed |= drag(ui, "Width", width);
                    changed |= drag(ui, "Height", height);
                }
                Profile::Circle { radius } => {
                    changed |= drag(ui, "Radius", radius);
                }
            }
        }
        FeatureKind::ConstraintSketch { plane, sketch } => {
            changed |= plane_combo(ui, plane);
            ui.label(
                RichText::new(format!(
                    "{} points · {} lines · {} constraints",
                    sketch.points.len(),
                    sketch.lines.len(),
                    sketch.constraints.len()
                ))
                .small()
                .weak(),
            );
            // Editable dimensions: drag a Distance/Radius value and the sketch
            // re-solves. (Drawing entities + geometric constraints land in the
            // next increment.)
            let mut dim = 0;
            for constraint in &mut sketch.constraints {
                match constraint {
                    Constraint::Distance(_, _, value) => {
                        dim += 1;
                        changed |= drag(ui, &format!("Length {dim}"), value);
                    }
                    Constraint::Radius(_, value) => {
                        dim += 1;
                        changed |= drag(ui, &format!("Radius {dim}"), value);
                    }
                    _ => {}
                }
            }
            // Points, lines, curves + geometric constraints are too rich to list;
            // re-open the sketch canvas to edit them.
            edit_label = Some("Edit sketch");
        }
        FeatureKind::Extrude { distance, .. } => {
            changed |= drag_signed(ui, "Distance", distance);
        }
        FeatureKind::PushPull { distance, .. } => {
            changed |= drag_signed(ui, "Distance", distance);
        }
        FeatureKind::Translate { offset, .. } => {
            // Offsets are signed — you can move in either direction on each axis.
            changed |= drag_signed(ui, "dX", &mut offset.x);
            changed |= drag_signed(ui, "dY", &mut offset.y);
            changed |= drag_signed(ui, "dZ", &mut offset.z);
        }
        FeatureKind::FilletAll { radius, .. } | FeatureKind::Fillet { radius, .. } => {
            changed |= drag(ui, "Radius", radius);
        }
        FeatureKind::ChamferAll { distance, .. } | FeatureKind::Chamfer { distance, .. } => {
            changed |= drag(ui, "Distance", distance);
            ui.weak("Measured back along both faces of each edge");
        }
        FeatureKind::Shell { thickness, open, .. } => {
            changed |= drag(ui, "Wall", thickness);
            ui.horizontal(|ui| {
                if open.is_empty() {
                    ui.weak("Closed hollow (no open faces)");
                } else {
                    ui.weak(format!("{} face(s) open", open.len()));
                    if ui
                        .small_button("Close all")
                        .on_hover_text("Keep every face: a sealed hollow body")
                        .clicked()
                    {
                        open.clear();
                        changed = true;
                    }
                }
            });
            ui.weak("The wall grows inward; the outside is unchanged");
        }
        FeatureKind::Rotate { angle, .. } => {
            // Edit the angle in degrees; stored as radians.
            let mut deg = angle.to_degrees();
            if ui
                .add(egui::DragValue::new(&mut deg).speed(1.0).suffix("°"))
                .changed()
            {
                *angle = deg.to_radians();
                changed = true;
            }
        }
        FeatureKind::Boolean { op, .. } => {
            changed |= boolean_op(ui, op);
        }
        FeatureKind::Scale { factors, .. } => {
            // Per-axis scale multipliers (1.0 = unchanged).
            changed |= drag(ui, "Scale X", &mut factors.x);
            changed |= drag(ui, "Scale Y", &mut factors.y);
            changed |= drag(ui, "Scale Z", &mut factors.z);
        }
        FeatureKind::Revolve { angle, .. } => {
            // Edit the revolve angle in degrees; stored as radians.
            let mut deg = angle.to_degrees();
            if ui
                .add(egui::DragValue::new(&mut deg).speed(1.0).range(1.0..=360.0).suffix("°"))
                .changed()
            {
                *angle = deg.to_radians();
                changed = true;
            }
        }
        FeatureKind::Mirror { normal, .. } => {
            // The mirror plane is fixed at creation (from a face/plane); show its
            // normal for reference.
            ui.label(
                RichText::new(format!("plane ⊥ ({:.2}, {:.2}, {:.2})", normal.x, normal.y, normal.z))
                    .small()
                    .weak(),
            );
        }
        FeatureKind::Group { members } => {
            ui.label(RichText::new(format!("{} members", members.len())).small().weak());
        }
        FeatureKind::Sweep { path: cdt_core::SweepPath::Planar { sketch, .. }, .. } => {
            let segs = sketch.lines.len() + sketch.beziers.len();
            ui.label(RichText::new(format!("path: {segs} segments")).small().weak());
            // The path is a line the user can go back and reshape.
            edit_label = Some("Edit path");
        }
        FeatureKind::Sweep { path: cdt_core::SweepPath::Helix { helix }, .. } => {
            // The coil's radius is where the profile sits (it is carried round
            // rigidly), so the helix's own radius isn't offered.
            changed |= helix_editor(ui, helix, false);
            ui.label(RichText::new("radius: the profile's distance from the axis").small().weak());
        }
        FeatureKind::Thread { helix, depth, internal, starts, .. } => {
            changed |= drag(ui, "Pitch", &mut helix.pitch);
            changed |= drag(ui, "Depth", depth);
            changed |= drag(ui, "Length", &mut helix.length);
            changed |= drag(ui, "Radius", &mut helix.radius);
            ui.horizontal(|ui| {
                ui.label("Starts");
                changed |= ui
                    .add(egui::DragValue::new(starts).speed(0.1).range(1..=cdt_core::MAX_THREAD_STARTS))
                    .on_hover_text("Grooves around the axis; many, with a steep pitch, cut both ways, make a knurl")
                    .changed();
            });
            ui.horizontal(|ui| {
                if ui
                    .small_button("ISO")
                    .on_hover_text("Coarse pitch for this diameter, and the ISO depth for it")
                    .clicked()
                {
                    helix.pitch = Helix::iso_pitch(2.0 * helix.radius) * f64::from(*starts);
                    *depth = Helix::iso_depth(helix.pitch / f64::from(*starts));
                    changed = true;
                }
                changed |= ui.checkbox(internal, "Internal").on_hover_text("Cut outward into a hole's wall rather than inward into a shaft").changed();
                changed |= ui.checkbox(&mut helix.left_handed, "Left-hand").changed();
            });
            ui.label(RichText::new(format!("{:.1} turns", helix.turns())).small().weak());
        }
        FeatureKind::Pattern { layout, count, skip, .. } => {
            changed |= count_drag(ui, "Count", count);
            if !skip.is_empty() {
                ui.horizontal(|ui| {
                    let list: Vec<String> = skip.iter().map(|i| i.to_string()).collect();
                    ui.label(RichText::new(format!("skipped: {}", list.join(", "))).small().weak());
                    if ui.small_button("Restore").on_hover_text("Bring back every skipped instance").clicked() {
                        skip.clear();
                        changed = true;
                    }
                });
            }
            match layout {
                PatternLayout::Linear { step } => {
                    // The per-instance step; signed so a pattern can run either way.
                    changed |= drag_signed(ui, "Step X", &mut step.x);
                    changed |= drag_signed(ui, "Step Y", &mut step.y);
                    changed |= drag_signed(ui, "Step Z", &mut step.z);
                }
                PatternLayout::Circular { center, axis, angle } => {
                    // Per-instance angle, with a shortcut to spread the
                    // instances evenly around a full turn.
                    changed |= step_angle(ui, angle, Some(*count));
                    changed |= axis_fields(ui, center, axis);
                }
                PatternLayout::Helical { center, axis, angle, rise } => {
                    changed |= step_angle(ui, angle, None);
                    changed |= drag_signed(ui, "Rise", rise);
                    changed |= axis_fields(ui, center, axis);
                }
            }
        }
    }

    // Body colour: a swatch overriding the theme's default material for this
    // body (and, via the shared colour, its Build Graph lane). Resetting returns
    // it to the theme default. Colour is linear RGB, matching the shader.
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Colour");
        let mut rgb = feature.color.unwrap_or([0.60, 0.60, 0.62]);
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            feature.color = Some(rgb);
            changed = true;
        }
        if feature.color.is_some()
            && ui
                .small_button("Reset")
                .on_hover_text("Use the theme's default colour")
                .clicked()
        {
            feature.color = None;
            changed = true;
        }
    });

    let mut edit = false;
    if let Some(label) = edit_label {
        ui.add_space(4.0);
        edit = ui
            .button(RichText::new(format!("{}  {label}", icon::NOTE_PENCIL)))
            .clicked();
    }

    EditorResponse { changed, edit }
}

pub fn error_list(ui: &mut Ui, doc: &Document, state: &HistoryState) {
    let error_color = state.theme.palette().error;
    ui.label(RichText::new("Problems").color(error_color).strong());
    for (id, msg) in &state.errors {
        let name = doc
            .history
            .get(*id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("{id:?}"));
        ui.label(
            RichText::new(format!("• {name}: {msg}"))
                .small()
                .color(error_color),
        );
    }
}

/// A labeled millimeter drag-value, non-negative. Returns whether it changed.
fn drag(ui: &mut Ui, label: &str, value: &mut f64) -> bool {
    drag_ranged(ui, label, value, 0.0..=10_000.0)
}

/// A labeled millimeter drag-value allowing negative values (e.g. extrude
/// distance, translate offset).
fn drag_signed(ui: &mut Ui, label: &str, value: &mut f64) -> bool {
    drag_ranged(ui, label, value, -10_000.0..=10_000.0)
}

/// The fields of a helix (a coil's path): pitch, length, radius, hand, and
/// the axis — centre plus a world direction.
fn helix_editor(ui: &mut Ui, helix: &mut Helix, with_radius: bool) -> bool {
    let mut changed = false;
    changed |= drag(ui, "Pitch", &mut helix.pitch);
    changed |= drag(ui, "Length", &mut helix.length);
    if with_radius {
        changed |= drag(ui, "Radius", &mut helix.radius);
    }
    changed |= ui.checkbox(&mut helix.left_handed, "Left-hand").changed();
    changed |= axis_fields(ui, &mut helix.center, &mut helix.axis);
    ui.label(RichText::new(format!("{:.1} turns", helix.turns())).small().weak());
    changed
}

/// A pattern's per-instance angle, edited in degrees (stored as radians).
/// With `spread_over` = the instance count, a "Full turn" button sets the
/// angle that spaces them evenly round 360°.
fn step_angle(ui: &mut Ui, angle: &mut f64, spread_over: Option<u32>) -> bool {
    let mut changed = false;
    let mut deg = angle.to_degrees();
    ui.horizontal(|ui| {
        ui.label("Angle");
        if ui
            .add(egui::DragValue::new(&mut deg).speed(1.0).range(-360.0..=360.0).suffix("°"))
            .changed()
        {
            *angle = deg.to_radians();
            changed = true;
        }
        if let Some(count) = spread_over {
            if ui
                .small_button("Full turn")
                .on_hover_text("Spread the instances evenly over 360°")
                .clicked()
            {
                *angle = std::f64::consts::TAU / f64::from(count.max(1));
                changed = true;
            }
        }
    });
    changed
}

/// An axis through a point: the centre's three coordinates and a world axis.
fn axis_fields(ui: &mut Ui, center: &mut DVec3, axis: &mut DVec3) -> bool {
    let mut changed = false;
    changed |= drag_signed(ui, "Center X", &mut center.x);
    changed |= drag_signed(ui, "Center Y", &mut center.y);
    changed |= drag_signed(ui, "Center Z", &mut center.z);
    changed |= axis_combo(ui, axis);
    changed
}

/// An instance count: a whole number from 1 to the regen limit.
fn count_drag(ui: &mut Ui, label: &str, value: &mut u32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui
            .add(egui::DragValue::new(value).speed(0.1).range(1..=cdt_core::MAX_PATTERN_COUNT))
            .changed();
    });
    changed
}

/// Pick a pattern axis: one of the world axes, or keep a custom direction
/// (shown, not editable) that came from a face.
fn axis_combo(ui: &mut Ui, axis: &mut DVec3) -> bool {
    let mut changed = false;
    let named = [("X", DVec3::X), ("Y", DVec3::Y), ("Z", DVec3::Z)];
    let current = named
        .iter()
        .find(|(_, v)| (*v - *axis).length() < 1e-9)
        .map_or_else(|| format!("({:.2}, {:.2}, {:.2})", axis.x, axis.y, axis.z), |(l, _)| l.to_string());
    egui::ComboBox::from_label("Axis")
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (label, v) in named {
                if ui.selectable_label(*axis == v, label).clicked() && *axis != v {
                    *axis = v;
                    changed = true;
                }
            }
        });
    changed
}

fn drag_ranged(
    ui: &mut Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(value)
                .speed(0.2)
                .range(range)
                .suffix(" mm"),
        )
        .changed()
    })
    .inner
}

/// Compact origin-plane chooser (no label) for the Sketch toolbar button.
fn plane_combo(ui: &mut Ui, plane: &mut SketchPlane) -> bool {
    let mut changed = false;
    egui::ComboBox::from_label("Plane")
        .selected_text(plane.label())
        .show_ui(ui, |ui| {
            for option in [SketchPlane::Xy, SketchPlane::Xz, SketchPlane::Yz] {
                if ui
                    .selectable_value(plane, option, option.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });
    changed
}

/// Profile-type selector. Switching type resets to a sensible default size.
fn profile_combo(ui: &mut Ui, profile: &mut Profile) -> bool {
    let mut changed = false;
    egui::ComboBox::from_label("Profile")
        .selected_text(profile.type_name())
        .show_ui(ui, |ui| {
            let is_rect = matches!(profile, Profile::Rectangle { .. });
            if ui.selectable_label(is_rect, "Rectangle").clicked() && !is_rect {
                *profile = Profile::Rectangle {
                    width: 30.0,
                    height: 30.0,
                };
                changed = true;
            }
            let is_circle = matches!(profile, Profile::Circle { .. });
            if ui.selectable_label(is_circle, "Circle").clicked() && !is_circle {
                *profile = Profile::Circle { radius: 15.0 };
                changed = true;
            }
        });
    changed
}

fn boolean_op(ui: &mut Ui, op: &mut BooleanOp) -> bool {
    let mut changed = false;
    let label = match op {
        BooleanOp::Union => "Union",
        BooleanOp::Subtract => "Subtract",
        BooleanOp::Intersect => "Intersect",
    };
    egui::ComboBox::from_label("Operation")
        .selected_text(label)
        .show_ui(ui, |ui| {
            for option in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
                let name = match option {
                    BooleanOp::Union => "Union",
                    BooleanOp::Subtract => "Subtract",
                    BooleanOp::Intersect => "Intersect",
                };
                if ui.selectable_value(op, option, name).changed() {
                    changed = true;
                }
            }
        });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(selected: Option<u64>, visible: &[u64]) -> HistoryState {
        HistoryState {
            selected: selected.map(FeatureId),
            visible: visible.iter().copied().map(FeatureId).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn unary_prefers_visible_selection_else_last_visible() {
        // Selection visible -> use it.
        assert_eq!(state(Some(2), &[1, 2, 3]).unary_source(), Some(FeatureId(2)));
        // Selection not visible -> fall back to last visible.
        assert_eq!(state(Some(9), &[1, 2, 3]).unary_source(), Some(FeatureId(3)));
        // Nothing visible -> none.
        assert_eq!(state(Some(1), &[]).unary_source(), None);
    }

    #[test]
    fn boolean_needs_a_selected_body_and_another() {
        // No selection -> not ready (which body would be the operand?).
        assert!(!state(None, &[1, 2, 3]).can_boolean());
        // Selection not among the visible bodies -> not ready.
        assert!(!state(Some(9), &[1, 2, 3]).can_boolean());
        // A visible selection with only itself visible -> nothing to combine.
        assert!(!state(Some(1), &[1]).can_boolean());
        // A visible selection plus another visible body -> ready.
        assert!(state(Some(2), &[1, 2, 3]).can_boolean());
    }
}
