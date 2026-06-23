//! # rmf-ui
//!
//! Minimal-chrome egui panels. This first panel is the **history tree**: the
//! ordered feature list with selection, suppression, reordering, a rollback
//! bar, and an inline editor for the selected feature's parameters.
//!
//! The panel mutates the [`Document`] directly (all the structural operations
//! are already validated in `rmf-core`) and reports whether anything changed so
//! the host can regenerate geometry. It depends only on egui + core — no wgpu,
//! no kernel.

use egui::{Color32, Context, RichText, Ui};
use rmf_core::{BooleanOp, Document, FeatureId, FeatureKind, Profile, SketchPlane, DVec3};

const ERROR_COLOR: Color32 = Color32::from_rgb(232, 92, 92);

/// Persistent UI state owned by the host across frames.
#[derive(Default)]
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

    /// The (target, tool) a new boolean should combine: the selection (or the
    /// first visible body) as target, and a different visible body as tool.
    fn binary_inputs(&self) -> Option<(FeatureId, FeatureId)> {
        if self.visible.len() < 2 {
            return None;
        }
        let target = self
            .selected
            .filter(|s| self.visible.contains(s))
            .unwrap_or(self.visible[0]);
        let tool = self.visible.iter().rev().find(|v| **v != target).copied()?;
        Some((target, tool))
    }
}

/// Draw the history side panel and report what the user did this frame.
pub fn history_panel(
    ctx: &Context,
    doc: &mut Document,
    state: &mut HistoryState,
) -> HistoryResponse {
    let mut resp = HistoryResponse::default();

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
            ui.heading(&doc.name);
            ui.label(
                RichText::new(format!("{} features", doc.history.len()))
                    .small()
                    .weak(),
            );

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(state.can_undo, egui::Button::new("↶ Undo"))
                    .on_hover_text("Undo (⌘Z)")
                    .clicked()
                {
                    resp.undo = true;
                }
                if ui
                    .add_enabled(state.can_redo, egui::Button::new("↷ Redo"))
                    .on_hover_text("Redo (⇧⌘Z)")
                    .clicked()
                {
                    resp.redo = true;
                }
            });
            ui.separator();

            resp.changed |= add_feature_toolbar(ui, doc, state);
            ui.separator();

            resp.changed |= rollback_controls(ui, doc);
            ui.separator();

            resp.changed |= feature_list(ui, doc, state);

            if let Some(selected) = state.selected {
                ui.separator();
                resp.changed |= selected_editor(ui, doc, selected);
            }

            if !state.errors.is_empty() {
                ui.separator();
                error_list(ui, doc, state);
            }
        });

    resp
}

/// Buttons to create new features. Primitives are always available; unary ops
/// (Move, Fillet) and booleans require visible bodies to act on, so they enable
/// only when valid inputs exist. New features are selected on creation.
fn add_feature_toolbar(ui: &mut Ui, doc: &mut Document, state: &mut HistoryState) -> bool {
    let mut changed = false;

    // Pick operation inputs up front (immutable borrows of doc/state) so the
    // mutating `add` closure can take exclusive borrows afterward.
    let unary = state.unary_source();
    let binary = state.binary_inputs();
    let extrude_source = unary.filter(|id| {
        matches!(
            doc.history.get(*id).map(|f| &f.kind),
            Some(FeatureKind::Sketch { .. })
        )
    });

    let mut add = |state: &mut HistoryState, name: &str, kind: FeatureKind| {
        let id = doc.add(name, kind);
        state.selected = Some(id);
        changed = true;
    };

    ui.label(RichText::new("Add").small().weak());

    ui.horizontal_wrapped(|ui| {
        if ui.button("Box").clicked() {
            add(state, "Box", FeatureKind::Box { size: DVec3::splat(20.0) });
        }
        if ui.button("Cylinder").clicked() {
            add(state, "Cylinder", FeatureKind::Cylinder { radius: 10.0, height: 20.0 });
        }
        if ui.button("Sphere").clicked() {
            add(state, "Sphere", FeatureKind::Sphere { radius: 10.0 });
        }
        if ui.button("Sketch").clicked() {
            add(
                state,
                "Sketch",
                FeatureKind::Sketch {
                    plane: SketchPlane::Xy,
                    profile: Profile::Rectangle {
                        width: 30.0,
                        height: 30.0,
                    },
                },
            );
        }
    });

    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(extrude_source.is_some(), egui::Button::new("Extrude"))
            .on_hover_text("Extrude the selected/last sketch into a solid")
            .clicked()
        {
            add(
                state,
                "Extrude",
                FeatureKind::Extrude {
                    source: extrude_source.unwrap(),
                    distance: 20.0,
                },
            );
        }
        if ui
            .add_enabled(unary.is_some(), egui::Button::new("Move"))
            .on_hover_text("Translate the selected/last body")
            .clicked()
        {
            add(
                state,
                "Move",
                FeatureKind::Translate {
                    source: unary.unwrap(),
                    offset: DVec3::new(10.0, 0.0, 0.0),
                },
            );
        }
        if ui
            .add_enabled(unary.is_some(), egui::Button::new("Fillet"))
            .on_hover_text("Fillet all edges of the selected/last body")
            .clicked()
        {
            add(
                state,
                "Fillet",
                FeatureKind::FilletAll {
                    source: unary.unwrap(),
                    radius: 2.0,
                },
            );
        }
        for (label, op) in [
            ("Union", BooleanOp::Union),
            ("Subtract", BooleanOp::Subtract),
            ("Intersect", BooleanOp::Intersect),
        ] {
            if ui
                .add_enabled(binary.is_some(), egui::Button::new(label))
                .on_hover_text("Combine two visible bodies")
                .clicked()
            {
                let (target, tool) = binary.unwrap();
                add(state, label, FeatureKind::Boolean { op, target, tool });
            }
        }
    });

    changed
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

fn feature_list(ui: &mut Ui, doc: &mut Document, state: &mut HistoryState) -> bool {
    let mut changed = false;

    // Snapshot the rows so we can mutate the document inside the loop.
    let rows: Vec<(FeatureId, String, &'static str, bool)> = doc
        .history
        .features()
        .iter()
        .map(|f| (f.id, f.name.clone(), f.kind.type_name(), f.suppressed))
        .collect();
    let active = doc.rollback();
    let last = rows.len().saturating_sub(1);

    for (index, (id, name, type_name, suppressed)) in rows.into_iter().enumerate() {
        let rolled_back = index >= active;
        let error = state.error_for(id).map(str::to_owned);

        ui.horizontal(|ui| {
            // Suppress toggle.
            let mut sup = suppressed;
            if ui
                .add(egui::Checkbox::without_text(&mut sup))
                .on_hover_text("Suppress")
                .changed()
            {
                doc.history.set_suppressed(id, sup);
                changed = true;
            }

            // Selectable label, colored by state.
            let mut text = RichText::new(format!("{name}  ·  {type_name}"));
            if error.is_some() {
                text = text.color(ERROR_COLOR);
            } else if rolled_back || suppressed {
                text = text.weak();
            }
            let selected = state.selected == Some(id);
            let response = ui.selectable_label(selected, text);
            if let Some(msg) = &error {
                response.clone().on_hover_text(msg);
            }
            if response.clicked() {
                state.selected = if selected { None } else { Some(id) };
            }

            // Reorder + delete on the right.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("🗑").on_hover_text("Delete").clicked() {
                    doc.history.remove(id);
                    if state.selected == Some(id) {
                        state.selected = None;
                    }
                    changed = true;
                }
                if ui
                    .add_enabled(index < last, egui::Button::new("↓").small())
                    .clicked()
                {
                    let _ = doc.history.reorder(id, index + 1);
                    changed = true;
                }
                if ui
                    .add_enabled(index > 0, egui::Button::new("↑").small())
                    .clicked()
                {
                    let _ = doc.history.reorder(id, index - 1);
                    changed = true;
                }
            });
        });
    }

    changed
}

fn selected_editor(ui: &mut Ui, doc: &mut Document, id: FeatureId) -> bool {
    let mut changed = false;
    let Some(feature) = doc.history.get_mut(id) else {
        return false;
    };

    ui.label(RichText::new(format!("Edit · {}", feature.name)).strong());
    ui.add_space(2.0);

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
        FeatureKind::Extrude { distance, .. } => {
            changed |= drag_signed(ui, "Distance", distance);
        }
        FeatureKind::Translate { offset, .. } => {
            changed |= drag(ui, "dX", &mut offset.x);
            changed |= drag(ui, "dY", &mut offset.y);
            changed |= drag(ui, "dZ", &mut offset.z);
        }
        FeatureKind::FilletAll { radius, .. } => {
            changed |= drag(ui, "Radius", radius);
        }
        FeatureKind::Boolean { op, .. } => {
            changed |= boolean_op(ui, op);
        }
    }

    changed
}

fn error_list(ui: &mut Ui, doc: &Document, state: &HistoryState) {
    ui.label(RichText::new("Problems").color(ERROR_COLOR).strong());
    for (id, msg) in &state.errors {
        let name = doc
            .history
            .get(*id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("{id:?}"));
        ui.label(
            RichText::new(format!("• {name}: {msg}"))
                .small()
                .color(ERROR_COLOR),
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
    fn binary_needs_two_distinct_visible_bodies() {
        assert_eq!(state(None, &[7]).binary_inputs(), None);
        // Default: first visible is target, a different visible is tool.
        assert_eq!(
            state(None, &[1, 2, 3]).binary_inputs(),
            Some((FeatureId(1), FeatureId(3)))
        );
        // Visible selection becomes target; tool is a different visible body.
        assert_eq!(
            state(Some(2), &[1, 2, 3]).binary_inputs(),
            Some((FeatureId(2), FeatureId(3)))
        );
    }
}
