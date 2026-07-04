//! The feature tree: an ordered list of [`Feature`]s with stable ids.
//!
//! [`History`] owns the features and enforces the one structural invariant that
//! makes replay sound: **every reference points backward**. A feature may only
//! depend on features that appear earlier, so [`crate::regen`] can evaluate in a
//! single forward pass. Reordering and editing are validated against this rule.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::features::{Feature, FeatureId, FeatureKind};

/// Why a history failed dependency validation.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DependencyError {
    #[error("feature {feature:?} references {input:?}, which does not exist")]
    UnknownInput { feature: FeatureId, input: FeatureId },
    #[error("feature {feature:?} references {input:?}, which comes later in history")]
    ForwardReference { feature: FeatureId, input: FeatureId },
}

/// An ordered collection of features with monotonic id allocation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct History {
    features: Vec<Feature>,
    next_id: u64,
}

impl History {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> FeatureId {
        let id = FeatureId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Append a feature, returning its freshly allocated id.
    pub fn add(&mut self, name: impl Into<String>, kind: FeatureKind) -> FeatureId {
        self.insert(self.features.len(), name, kind).1
    }

    /// Insert a feature at `index`, returning where it actually landed and its
    /// freshly allocated id. The index is clamped into `0..=len`, and clamped
    /// forward past the feature's inputs so every reference keeps pointing
    /// backward (the invariant [`crate::regen`] replay relies on).
    pub fn insert(
        &mut self,
        index: usize,
        name: impl Into<String>,
        kind: FeatureKind,
    ) -> (usize, FeatureId) {
        let after_inputs = kind
            .inputs()
            .iter()
            .filter_map(|&input| self.index_of(input))
            .map(|i| i + 1)
            .max()
            .unwrap_or(0);
        let index = index.max(after_inputs).min(self.features.len());
        let id = self.alloc_id();
        self.features.insert(
            index,
            Feature {
                id,
                name: name.into(),
                suppressed: false,
                kind,
            },
        );
        (index, id)
    }

    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Repair the id-allocation invariant that a corrupt or hand-edited file
    /// can violate: `next_id` must be greater than every existing feature id,
    /// or the next [`Self::add`] would mint a duplicate [`FeatureId`] and
    /// references would silently alias. Raises `next_id` to `max(id) + 1`
    /// when it lags behind. Returns `true` if anything was fixed. Safe to
    /// call on any history; a consistent one is left untouched.
    pub fn repair(&mut self) -> bool {
        match self.features.iter().map(|f| f.id.0).max() {
            Some(max) if self.next_id <= max => {
                self.next_id = max + 1;
                true
            }
            _ => false,
        }
    }

    /// The next monotonic number for an auto-named `"{base} {n}"` feature: one
    /// more than the highest `n` currently in use for that `base` (so numbers
    /// climb and a gap left by a deletion is never backfilled). Starts at 1.
    pub fn next_name_number(&self, base: &str) -> u32 {
        self.features
            .iter()
            .filter_map(|f| f.name.strip_prefix(base))
            .filter_map(|rest| rest.trim_start().parse::<u32>().ok())
            .max()
            .map_or(1, |m| m + 1)
    }

    /// Deep-clone the subtree feeding `id` — that feature plus all of its
    /// transitive inputs — with fresh ids, appended in dependency order. Returns
    /// the new tip (the clone of `id`), or `None` if `id` doesn't exist. The
    /// clone references only other clones, so it's a fully independent,
    /// parametrically-editable duplicate (the basis of copy/paste and mirror).
    pub fn clone_subtree(&mut self, id: FeatureId) -> Option<FeatureId> {
        self.get(id)?;
        // The set of features in the subtree (transitive closure of inputs).
        let mut set = HashSet::new();
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            if !set.insert(cur) {
                continue;
            }
            if let Some(f) = self.get(cur) {
                stack.extend(f.kind.inputs());
            }
        }
        // Clone in existing history order so inputs land before their dependents.
        let originals: Vec<Feature> =
            self.features.iter().filter(|f| set.contains(&f.id)).cloned().collect();
        // Fresh ids are all greater than any existing id, so remapping each
        // old→new can't collide with an as-yet-unremapped input.
        let map: HashMap<FeatureId, FeatureId> =
            originals.iter().map(|f| (f.id, self.alloc_id())).collect();
        for f in &originals {
            let mut kind = f.kind.clone();
            for (&old, &new) in &map {
                kind.remap_input(old, new);
            }
            self.features.push(Feature {
                id: map[&f.id],
                name: f.name.clone(),
                suppressed: f.suppressed,
                kind,
            });
        }
        Some(map[&id])
    }

    /// Mutable access to all features in order — for bulk passes such as
    /// re-solving every constraint sketch before regeneration.
    pub fn features_mut(&mut self) -> &mut [Feature] {
        &mut self.features
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn index_of(&self, id: FeatureId) -> Option<usize> {
        self.features.iter().position(|f| f.id == id)
    }

    pub fn get(&self, id: FeatureId) -> Option<&Feature> {
        self.features.iter().find(|f| f.id == id)
    }

    /// Mutable access to a feature for editing its parameters. The caller is
    /// expected to keep references backward-pointing; call [`Self::validate`]
    /// after structural edits.
    pub fn get_mut(&mut self, id: FeatureId) -> Option<&mut Feature> {
        self.features.iter_mut().find(|f| f.id == id)
    }

    /// Remove a feature, healing dependents: references to the removed feature
    /// are rewired to its primary input (e.g. deleting a fillet reconnects its
    /// dependents to the body it was filleting). Dependents of a feature with
    /// no usable input (a primitive or sketch) are left dangling and will
    /// surface as regeneration errors.
    pub fn remove(&mut self, id: FeatureId) -> Option<Feature> {
        let index = self.index_of(id)?;
        let removed = self.features.remove(index);
        if let Some(replacement) = removed.kind.primary_input() {
            for feature in &mut self.features {
                feature.kind.remap_input(id, replacement);
            }
        }
        Some(removed)
    }

    pub fn set_suppressed(&mut self, id: FeatureId, suppressed: bool) {
        if let Some(f) = self.get_mut(id) {
            f.suppressed = suppressed;
        }
    }

    /// Move a feature to a new index. Rejected (and left unchanged) if the move
    /// would make any reference point forward.
    pub fn reorder(&mut self, id: FeatureId, new_index: usize) -> Result<(), Vec<DependencyError>> {
        let Some(from) = self.index_of(id) else {
            return Ok(());
        };
        let mut candidate = self.features.clone();
        let feature = candidate.remove(from);
        let to = new_index.min(candidate.len());
        candidate.insert(to, feature);

        let errors = Self::validate_slice(&candidate);
        if errors.is_empty() {
            self.features = candidate;
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check that every reference is known and backward-pointing.
    pub fn validate(&self) -> Result<(), Vec<DependencyError>> {
        let errors = Self::validate_slice(&self.features);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_slice(features: &[Feature]) -> Vec<DependencyError> {
        let mut errors = Vec::new();
        for (index, feature) in features.iter().enumerate() {
            for input in feature.kind.inputs() {
                match features.iter().position(|f| f.id == input) {
                    None => errors.push(DependencyError::UnknownInput {
                        feature: feature.id,
                        input,
                    }),
                    Some(input_index) if input_index >= index => {
                        errors.push(DependencyError::ForwardReference {
                            feature: feature.id,
                            input,
                        })
                    }
                    Some(_) => {}
                }
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn boxx() -> FeatureKind {
        FeatureKind::Box { size: DVec3::splat(1.0) }
    }

    #[test]
    fn name_numbers_climb_and_never_backfill_gaps() {
        let mut h = History::default();
        // First of a kind starts at 1; each subsequent climbs.
        assert_eq!(h.next_name_number("Fillet"), 1);
        let a = h.add("Fillet 1", boxx());
        let b = h.add("Fillet 2", boxx());
        let _c = h.add("Fillet 3", boxx());
        assert_eq!(h.next_name_number("Fillet"), 4);

        // Deleting a MIDDLE one leaves its gap; the next still climbs past the max.
        h.remove(b);
        assert_eq!(h.next_name_number("Fillet"), 4, "gap at 2 is not backfilled");

        // A different base numbers independently, and unrelated/curated names
        // (e.g. "Fillet edges") don't get miscounted as "Fillet N".
        let _ = a;
        h.add("Fillet edges", boxx());
        assert_eq!(h.next_name_number("Fillet"), 4);
        assert_eq!(h.next_name_number("Move"), 1);
    }

    #[test]
    fn repair_raises_a_lagging_next_id_past_the_max() {
        let mut h = History::new();
        let a = h.add("A", boxx());
        let b = h.add("B", boxx());
        // Simulate a corrupt/hand-edited file whose next_id lags the ids.
        h.next_id = 1;
        assert!(h.repair(), "an inconsistent next_id is reported as fixed");
        let c = h.add("C", boxx());
        assert!(c != a && c != b, "the next add mints a fresh id, not a duplicate");
        assert!(c.0 > b.0, "allocation resumes past the highest existing id");
    }

    #[test]
    fn repair_leaves_a_healthy_history_untouched() {
        let mut h = History::new();
        assert!(!h.repair(), "an empty history needs no repair");
        h.add("A", boxx());
        h.add("B", boxx());
        let before = h.features().to_vec();
        assert!(!h.repair(), "a consistent history needs no repair");
        assert_eq!(h.features(), &before[..]);
        let c = h.add("C", boxx());
        assert_eq!(h.index_of(c), Some(2), "allocation is unaffected");
    }
}
