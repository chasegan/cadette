//! The project document: a named history with a working unit and a rollback bar.
//!
//! The **rollback bar** sits between two features in history. Only features
//! before it are "active" and get regenerated — dragging it back lets you view
//! or edit the model at an earlier state, exactly the "adjustable steps" idea
//! from the project outline. New features are inserted at the bar.

use serde::{Deserialize, Serialize};

use crate::features::{Feature, FeatureId, FeatureKind};
use crate::history::History;
use crate::units::LengthUnit;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub name: String,
    pub units: LengthUnit,
    pub history: History,
    /// Number of leading features that are active. Invariant: `<= history.len()`.
    /// Equals `history.len()` when the bar is at the tip (the common case).
    rollback: usize,
}

impl Document {
    /// A new, empty document in millimeters.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            units: LengthUnit::Millimeter,
            history: History::new(),
            rollback: 0,
        }
    }

    /// Add a feature at the rollback bar (the tip, normally), returning its id.
    /// If the bar is pulled back, the feature is inserted there and the bar
    /// moves just past it, so it becomes the new active end. (Should an input
    /// somehow sit beyond the bar, the insertion — and the bar with it — is
    /// pushed past that input; see [`History::insert`].)
    pub fn add(&mut self, name: impl Into<String>, kind: FeatureKind) -> FeatureId {
        let (index, id) = self.history.insert(self.rollback, name, kind);
        self.rollback = index + 1;
        id
    }

    /// Add a feature with an auto-numbered name `"{base} {n}"`, where `n` climbs
    /// monotonically (see [`History::next_name_number`]) — for interactively
    /// created features so repeats are disambiguated (`Fillet 1`, `Fillet 2`, …).
    pub fn add_numbered(&mut self, base: &str, kind: FeatureKind) -> FeatureId {
        let n = self.history.next_name_number(base);
        self.add(format!("{base} {n}"), kind)
    }

    /// Duplicate the subtree feeding `id` (fresh ids, see
    /// [`History::clone_subtree`]) and return the new tip, leaving the rollback
    /// bar at the document tip so the clone is active. For copy/paste & mirror.
    pub fn duplicate(&mut self, id: FeatureId) -> Option<FeatureId> {
        let new_tip = self.history.clone_subtree(id)?;
        self.rollback_to_tip();
        Some(new_tip)
    }

    /// The active features: those before the rollback bar.
    pub fn active_features(&self) -> &[Feature] {
        let n = self.rollback.min(self.history.len());
        &self.history.features()[..n]
    }

    /// Current rollback position (count of active leading features).
    pub fn rollback(&self) -> usize {
        self.rollback
    }

    /// Move the rollback bar, clamped to `0..=history.len()`.
    pub fn set_rollback(&mut self, position: usize) {
        self.rollback = position.min(self.history.len());
    }

    /// Put the rollback bar at the tip so the whole history is active.
    pub fn rollback_to_tip(&mut self) {
        self.rollback = self.history.len();
    }

    /// True when the bar is at the tip (all features active).
    pub fn is_at_tip(&self) -> bool {
        self.rollback == self.history.len()
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new("Untitled")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn boxx() -> FeatureKind {
        FeatureKind::Box { size: DVec3::splat(1.0) }
    }

    fn translate(source: FeatureId) -> FeatureKind {
        FeatureKind::Translate { source, offset: DVec3::X }
    }

    #[test]
    fn add_at_tip_advances_the_bar() {
        let mut doc = Document::new("t");
        let a = doc.add("A", boxx());
        assert!(doc.is_at_tip());
        assert_eq!(doc.active_features().last().unwrap().id, a);
    }

    #[test]
    fn add_while_rolled_back_inserts_at_the_bar() {
        let mut doc = Document::new("t");
        let a = doc.add("A", boxx());
        let b = doc.add("B", boxx());
        let c = doc.add("C", boxx());
        doc.set_rollback(1); // only A active; B and C rolled back

        let d = doc.add("D", translate(a));

        // D sits right after A; B and C stay rolled back, in order.
        let ids: Vec<_> = doc.history.features().iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![a, d, b, c]);
        assert_eq!(doc.rollback(), 2);
        let active: Vec<_> = doc.active_features().iter().map(|f| f.id).collect();
        assert_eq!(active, vec![a, d], "the new feature is active; B and C are not");
        doc.history.validate().expect("insertion keeps references backward");
    }

    #[test]
    fn add_with_an_input_beyond_the_bar_lands_after_that_input() {
        let mut doc = Document::new("t");
        let a = doc.add("A", boxx());
        let b = doc.add("B", boxx());
        doc.set_rollback(0); // everything rolled back

        // Defensive path: hosts only offer active features as inputs, but if a
        // reference to rolled-back B arrives anyway, the insertion may not land
        // before B or replay would see a forward reference.
        let d = doc.add("D", translate(b));

        let ids: Vec<_> = doc.history.features().iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![a, b, d]);
        assert_eq!(doc.rollback(), 3, "bar moves just past the inserted feature");
        doc.history.validate().expect("insertion keeps references backward");
    }
}
