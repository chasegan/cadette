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
use rmf_core::{BooleanOp, Document, FeatureId, FeatureKind};

const ERROR_COLOR: Color32 = Color32::from_rgb(232, 92, 92);

/// Persistent UI state owned by the host across frames.
#[derive(Default)]
pub struct HistoryState {
    /// Currently selected feature, if any.
    pub selected: Option<FeatureId>,
    /// Per-feature regeneration errors, set by the host after each rebuild.
    pub errors: Vec<(FeatureId, String)>,
}

impl HistoryState {
    fn error_for(&self, id: FeatureId) -> Option<&str> {
        self.errors
            .iter()
            .find(|(e, _)| *e == id)
            .map(|(_, m)| m.as_str())
    }
}

/// Draw the history side panel. Returns `true` if the document changed in a way
/// that requires regeneration.
pub fn history_panel(ctx: &Context, doc: &mut Document, state: &mut HistoryState) -> bool {
    let mut changed = false;

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
            ui.separator();

            changed |= rollback_controls(ui, doc);
            ui.separator();

            changed |= feature_list(ui, doc, state);

            if let Some(selected) = state.selected {
                ui.separator();
                changed |= selected_editor(ui, doc, selected);
            }

            if !state.errors.is_empty() {
                ui.separator();
                error_list(ui, doc, state);
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

/// A labeled millimeter drag-value. Returns whether the value changed.
fn drag(ui: &mut Ui, label: &str, value: &mut f64) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(value)
                .speed(0.2)
                .range(0.0..=10_000.0)
                .suffix(" mm"),
        )
        .changed()
    })
    .inner
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
