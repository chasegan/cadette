//! The feature tree: an ordered list of [`Feature`]s with stable ids.
//!
//! [`History`] owns the features and enforces the one structural invariant that
//! makes replay sound: **every reference points backward**. A feature may only
//! depend on features that appear earlier, so [`crate::regen`] can evaluate in a
//! single forward pass. Reordering and editing are validated against this rule.

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
        let id = self.alloc_id();
        self.features.push(Feature {
            id,
            name: name.into(),
            suppressed: false,
            kind,
        });
        id
    }

    pub fn features(&self) -> &[Feature] {
        &self.features
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

    /// Remove a feature. Note this can orphan downstream references; the caller
    /// should validate or handle the resulting regeneration errors.
    pub fn remove(&mut self, id: FeatureId) -> Option<Feature> {
        self.index_of(id).map(|i| self.features.remove(i))
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
