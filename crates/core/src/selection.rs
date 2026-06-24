//! Selection vocabulary shared across the render, interaction, and app layers.

/// A selectable entity in the model, identified by its transient id (face/edge
/// ids are assigned per regeneration, so a `Pick` is only valid until the model
/// rebuilds).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Pick {
    Face(u32),
    Edge(u32),
}
