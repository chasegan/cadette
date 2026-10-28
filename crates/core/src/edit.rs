//! Document edits as data: the verbs around [`FeatureKind`]'s nouns.
//!
//! `FeatureKind` already describes every operation (a box, an extrude, a
//! fillet) as serializable data — it is what `.cdt` files store. A [`DocEdit`]
//! is one mutation of the history expressed the same way: add this feature,
//! replace that one's parameters, suppress, remove, reorder. A batch of edits
//! is therefore a request an external caller (a CLI, a script, an AI agent)
//! can send, a log of what it did, and a test vocabulary — without a second
//! ontology of operations.
//!
//! [`Document::apply_edits`] applies a batch transactionally: either every
//! edit lands and the history still validates, or the document is untouched
//! and the failing edit is named.

use serde::{Deserialize, Serialize};

use crate::document::Document;
use crate::features::{FeatureId, FeatureKind, ImportFormat};
use crate::history::DependencyError;

/// One mutation of a document's history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum DocEdit {
    /// Append a feature at the rollback bar. Without a `name` it gets an
    /// auto-numbered one from its kind (`Box 3`, `Fillet 2`, …).
    Add {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        kind: FeatureKind,
    },
    /// Add an in-place transform (translate, rotate, scale, mirror) and splice
    /// it into its source's chain so everything downstream moves with it —
    /// what a drag in the viewport does. See [`Document::insert_transform`].
    InsertTransform {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        kind: FeatureKind,
    },
    /// Store a STEP or STL file in the project and add an `Import` feature
    /// that replays it, at the rollback bar. `file` is the file's name (its
    /// extension says the format; it becomes the asset's name, numbered if
    /// taken); `data` its bytes, base64. The feature is named after the file
    /// unless `name` is given.
    #[serde(alias = "import_step")]
    Import {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        file: String,
        data: String,
    },
    /// Replace a feature's definition (edit its parameters). Its id, name and
    /// position are kept.
    Replace { id: FeatureId, kind: FeatureKind },
    Rename { id: FeatureId, name: String },
    /// Set (or clear) a feature's display colour, linear RGB.
    SetColor {
        id: FeatureId,
        #[serde(default)]
        color: Option<[f32; 3]>,
    },
    Suppress { id: FeatureId, suppressed: bool },
    /// Remove a feature, healing its dependents onto its primary input (see
    /// [`crate::History::remove`]). For a Group prefer [`DocEdit::Ungroup`].
    Remove { id: FeatureId },
    /// Dissolve a Group: its members become independent bodies again with the
    /// group's transforms baked onto each. `id` may be the group or a
    /// transform stacked on it.
    Ungroup { id: FeatureId },
    /// Move a feature to `index` in history (refused if a reference would then
    /// point forward).
    Reorder { id: FeatureId, index: usize },
    /// Clone the subtree feeding `id` with fresh ids; the new tip is returned.
    Duplicate { id: FeatureId },
    /// Move the rollback bar: `position` leading features stay active.
    SetRollback { position: usize },
    RollbackToTip,
    /// Merge stacked translates (see [`Document::tidy`]).
    Tidy,
}

/// Why an edit could not be applied. The document is left as it was.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EditError {
    #[error("no feature with id {0:?}")]
    UnknownFeature(FeatureId),
    #[error("{0:?} is not a group")]
    NotAGroup(FeatureId),
    #[error("{kind} is not an in-place transform")]
    NotATransform { kind: String },
    #[error("the file data is not valid base64: {0}")]
    BadData(String),
    #[error("the file can't be imported: {0}")]
    BadFile(String),
    #[error("the history would no longer be valid: {}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))]
    Dependency(Vec<DependencyError>),
}

/// What one edit produced, so a caller can chain (e.g. add a box, then fillet
/// *that* box by the id it got).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOutcome {
    /// A feature was created with this id.
    Created(FeatureId),
    /// An existing feature (or the document) changed.
    Changed,
    /// Features were removed (a Remove, or a Tidy that merged some away).
    Removed(usize),
}

impl EditOutcome {
    /// The created feature's id, if this edit created one.
    pub fn created(self) -> Option<FeatureId> {
        match self {
            EditOutcome::Created(id) => Some(id),
            _ => None,
        }
    }
}

impl DocEdit {
    /// Apply this one edit. Prefer [`Document::apply_edits`], which is
    /// transactional and validates the result.
    pub fn apply(&self, doc: &mut Document) -> Result<EditOutcome, EditError> {
        let exists = |doc: &Document, id: FeatureId| {
            doc.history.get(id).map(|_| ()).ok_or(EditError::UnknownFeature(id))
        };
        Ok(match self {
            DocEdit::Add { name, kind } => EditOutcome::Created(match name {
                Some(name) => doc.add(name.clone(), kind.clone()),
                None => doc.add_numbered(kind.display_name(), kind.clone()),
            }),
            DocEdit::InsertTransform { name, kind } => {
                if !kind.can_splice() {
                    return Err(EditError::NotATransform { kind: kind.display_name().to_string() });
                }
                if let Some(source) = kind.primary_input() {
                    exists(doc, source)?;
                }
                EditOutcome::Created(match name {
                    Some(name) => doc.insert_transform(name.clone(), kind.clone()),
                    None => doc.insert_transform_numbered(kind.display_name(), kind.clone()),
                })
            }
            DocEdit::Import { name, file, data } => {
                if ImportFormat::of_name(file).is_none() {
                    return Err(EditError::BadFile(format!("{file}: not a STEP or STL file (by its extension)")));
                }
                let bytes = decode_import_data(data)?;
                // A deleted import's orphaned file would otherwise keep the
                // plain name and push this one to `name 2`.
                doc.prune_assets();
                let asset = doc.assets.insert(file, bytes);
                let kind = FeatureKind::Import { asset: asset.clone() };
                let name = match name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
                    Some(given) => given.to_string(),
                    // From the file, not the asset: `servo 2.step` is still a servo.
                    None => import_feature_name(doc, file.rsplit(['/', '\\']).next().unwrap_or(file)),
                };
                EditOutcome::Created(doc.add(name, kind))
            }
            DocEdit::Replace { id, kind } => {
                let f = doc.history.get_mut(*id).ok_or(EditError::UnknownFeature(*id))?;
                f.kind = kind.clone();
                EditOutcome::Changed
            }
            DocEdit::Rename { id, name } => {
                let f = doc.history.get_mut(*id).ok_or(EditError::UnknownFeature(*id))?;
                f.name = name.clone();
                EditOutcome::Changed
            }
            DocEdit::SetColor { id, color } => {
                let f = doc.history.get_mut(*id).ok_or(EditError::UnknownFeature(*id))?;
                f.color = *color;
                EditOutcome::Changed
            }
            DocEdit::Suppress { id, suppressed } => {
                exists(doc, *id)?;
                doc.history.set_suppressed(*id, *suppressed);
                EditOutcome::Changed
            }
            DocEdit::Remove { id } => {
                doc.remove(*id).ok_or(EditError::UnknownFeature(*id))?;
                EditOutcome::Removed(1)
            }
            DocEdit::Ungroup { id } => {
                exists(doc, *id)?;
                let members = doc.ungroup(*id).ok_or(EditError::NotAGroup(*id))?;
                let _ = members;
                EditOutcome::Changed
            }
            DocEdit::Reorder { id, index } => {
                exists(doc, *id)?;
                doc.history.reorder(*id, *index).map_err(EditError::Dependency)?;
                EditOutcome::Changed
            }
            DocEdit::Duplicate { id } => {
                EditOutcome::Created(doc.duplicate(*id).ok_or(EditError::UnknownFeature(*id))?)
            }
            DocEdit::SetRollback { position } => {
                doc.set_rollback(*position);
                EditOutcome::Changed
            }
            DocEdit::RollbackToTip => {
                doc.rollback_to_tip();
                EditOutcome::Changed
            }
            DocEdit::Tidy => EditOutcome::Removed(doc.tidy()),
        })
    }
}

/// The feature name an import gets by default: the file's stem (`servo` for
/// `servo.step`), numbered from 2 when a feature already has it.
pub fn import_feature_name(doc: &Document, file: &str) -> String {
    let stem = match file.rfind('.') {
        Some(dot) if dot > 0 => &file[..dot],
        _ => file,
    }
    .trim();
    if !doc.history.features().iter().any(|f| f.name == stem) {
        return stem.to_string();
    }
    format!("{stem} {}", doc.history.next_name_number(stem).max(2))
}

/// The bytes behind an [`DocEdit::Import`]'s `data`: standard base64,
/// padded or not.
pub fn decode_import_data(data: &str) -> Result<Vec<u8>, EditError> {
    use base64::Engine as _;
    const LENIENT: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        base64::engine::general_purpose::PAD
            .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent),
    );
    LENIENT.decode(data.trim()).map_err(|e| EditError::BadData(e.to_string()))
}

impl DocEdit {
    /// An [`DocEdit::Import`] for a file already read into memory.
    pub fn import_file(file: impl Into<String>, bytes: &[u8], name: Option<String>) -> Self {
        use base64::Engine as _;
        DocEdit::Import {
            name,
            file: file.into(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    }
}

/// A batch that failed: which edit (by index) and why.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("edit {index}: {error}")]
pub struct BatchError {
    pub index: usize,
    pub error: EditError,
}

impl Document {
    /// Apply `edits` in order, transactionally: on any failure — an unknown
    /// id, a refused reorder, or a history that no longer validates once
    /// everything has landed — the document is left exactly as it was and the
    /// failing edit is named. On success, one outcome per edit.
    pub fn apply_edits(&mut self, edits: &[DocEdit]) -> Result<Vec<EditOutcome>, BatchError> {
        let mut working = self.clone();
        let mut outcomes = Vec::with_capacity(edits.len());
        for (index, edit) in edits.iter().enumerate() {
            let outcome = edit.apply(&mut working).map_err(|error| BatchError { index, error })?;
            outcomes.push(outcome);
        }
        working.sanitize();
        if let Err(errors) = working.history.validate() {
            return Err(BatchError { index: edits.len().saturating_sub(1), error: EditError::Dependency(errors) });
        }
        *self = working;
        Ok(outcomes)
    }


    /// Walk down from `tip` through any in-place transforms to the Group that
    /// feeds them. Returns `(group, chain)` where `chain` is the transforms
    /// above the group in bottom-up order (empty if `tip` is the group itself);
    /// `None` if `tip` isn't a (possibly transformed) group, or sits under a
    /// non-transform (e.g. a boolean) that can't be baked through.
    pub fn group_and_chain(&self, tip: FeatureId) -> Option<(FeatureId, Vec<FeatureId>)> {
        let mut chain = Vec::new();
        let mut cur = tip;
        loop {
            let kind = &self.history.get(cur)?.kind;
            match kind {
                FeatureKind::Group { .. } => {
                    chain.reverse(); // collected top-down; want bottom-up
                    return Some((cur, chain));
                }
                _ if kind.is_in_place_transform() => {
                    chain.push(cur);
                    cur = kind.primary_input()?;
                }
                _ => return None,
            }
        }
    }

    /// Dissolve the group at (or under) `tip`: every member gets its own copy
    /// of the group's transform chain, then the chain and the group are
    /// removed, so the members stay where they were as independent bodies.
    /// Returns the members' new tips, or `None` if `tip` isn't a group.
    pub fn ungroup(&mut self, tip: FeatureId) -> Option<Vec<FeatureId>> {
        let (group, chain) = self.group_and_chain(tip)?;
        let members = match &self.history.get(group)?.kind {
            FeatureKind::Group { members } => members.clone(),
            _ => return None,
        };
        let mut tips = Vec::with_capacity(members.len());
        for m in &members {
            // Clone the chain onto this member: each step's source rewired to
            // the previous clone, the first from the group to the member.
            let mut prev = *m;
            let mut prev_old = group;
            for &t in &chain {
                let feature = self.history.get(t)?;
                let name = feature.name.clone();
                let mut kind = feature.kind.clone();
                kind.remap_input(prev_old, prev);
                let id = self.add(name, kind);
                prev_old = t;
                prev = id;
            }
            tips.push(prev);
        }
        // Drop the original chain (top-down, so each is a tip) and the group.
        for &t in chain.iter().rev() {
            self.remove(t);
        }
        self.remove(group);
        self.rollback_to_tip();
        Some(tips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BooleanOp, DVec3};

    fn box_kind(s: f64) -> FeatureKind {
        FeatureKind::Box { size: DVec3::splat(s) }
    }

    #[test]
    fn import_step_stores_the_file_and_adds_a_feature_named_after_it() {
        let mut doc = Document::new("t");
        let edit = DocEdit::Import { name: None, file: "parts/servo.step".into(), data: "SVNPLTEwMzAzLTIxOw==".into() };
        let out = doc.apply_edits(std::slice::from_ref(&edit)).unwrap();
        let id = out[0].created().unwrap();
        let f = doc.history.get(id).unwrap();
        assert_eq!(f.name, "servo");
        assert_eq!(f.kind, FeatureKind::Import { asset: "servo.step".into() });
        assert_eq!(doc.assets.get("servo.step").map(|b| b.as_ref()), Some(&b"ISO-10303-21;"[..]));

        // The same file again shares the asset and numbers the feature; a
        // different file with the same name gets its own asset.
        let out = doc.apply_edits(std::slice::from_ref(&edit)).unwrap();
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(doc.history.get(out[0].created().unwrap()).unwrap().name, "servo 2");
        let other = DocEdit::Import { name: Some("Bolt".into()), file: "servo.step".into(), data: "b3RoZXI=".into() };
        let out = doc.apply_edits(&[other]).unwrap();
        let f = doc.history.get(out[0].created().unwrap()).unwrap();
        assert_eq!(f.name, "Bolt");
        assert_eq!(f.kind, FeatureKind::Import { asset: "servo 2.step".into() });

        // Unpadded base64 and a blank name are both fine.
        let out = doc
            .apply_edits(&[DocEdit::Import { name: Some("  ".into()), file: "servo.step".into(), data: "b3RoZXI".into() }])
            .unwrap();
        let f = doc.history.get(out[0].created().unwrap()).unwrap();
        assert_eq!(f.name, "servo 3");
        assert_eq!(f.kind, FeatureKind::Import { asset: "servo 2.step".into() }, "the same bytes as Bolt, shared");

        // Removing an import and importing a different same-named file: the
        // orphan is pruned first, so the plain name is free again.
        let first = doc.history.features()[0].id;
        doc.apply_edits(&[DocEdit::Remove { id: first }, DocEdit::Remove { id: doc.history.features()[1].id }]).unwrap();
        assert_eq!(doc.history.len(), 2, "{:?}", doc.history.features().iter().map(|f| &f.name).collect::<Vec<_>>());
        let out = doc.apply_edits(&[DocEdit::import_file("servo.step", b"third", None)]).unwrap();
        let f = doc.history.get(out[0].created().unwrap()).unwrap();
        assert_eq!(f.kind, FeatureKind::Import { asset: "servo.step".into() });
        assert!(doc.assets.contains("servo 2.step"), "still used by Bolt");
        assert_eq!(doc.assets.len(), 2);

        // Bad base64 is refused and leaves the document untouched.
        let before = doc.history.len();
        let err = doc
            .apply_edits(&[DocEdit::Import { name: None, file: "x.obj".into(), data: "b3RoZXI=".into() }])
            .unwrap_err();
        assert_eq!(err.error, EditError::BadFile("x.obj: not a STEP or STL file (by its extension)".into()));
        assert!(!doc.assets.contains("x.obj"));
        let err = doc.apply_edits(&[DocEdit::Import { name: None, file: "x.step".into(), data: "@@@".into() }]).unwrap_err();
        assert!(matches!(err.error, EditError::BadData(_)), "{err}");
        assert_eq!(doc.history.len(), before);
        assert!(!doc.assets.contains("x.step"));
    }

    #[test]
    fn a_batch_lands_in_order_and_chains_by_outcome() {
        let mut doc = Document::new("t");
        let out = doc
            .apply_edits(&[DocEdit::Add { name: None, kind: box_kind(10.0) }])
            .unwrap();
        let b = out[0].created().unwrap();
        assert_eq!(doc.history.get(b).unwrap().name, "Box 1", "auto-numbered");
        let out = doc
            .apply_edits(&[
                DocEdit::Add { name: Some("Round".into()), kind: FeatureKind::FilletAll { source: b, radius: 1.0 } },
                DocEdit::Rename { id: b, name: "Base".into() },
                DocEdit::SetColor { id: b, color: Some([1.0, 0.0, 0.0]) },
                DocEdit::Suppress { id: b, suppressed: true },
            ])
            .unwrap();
        assert!(matches!(out[0], EditOutcome::Created(_)));
        assert_eq!(&out[1..], &[EditOutcome::Changed; 3]);
        let base = doc.history.get(b).unwrap();
        assert_eq!(base.name, "Base");
        assert_eq!(base.color, Some([1.0, 0.0, 0.0]));
        assert!(base.suppressed);
        assert_eq!(doc.rollback(), 2);
    }

    #[test]
    fn a_failing_batch_leaves_the_document_untouched() {
        let mut doc = Document::new("t");
        let b = doc.add("Box", box_kind(10.0));
        let before = doc.clone();
        let err = doc
            .apply_edits(&[
                DocEdit::Add { name: None, kind: box_kind(5.0) }, // would land...
                DocEdit::Remove { id: FeatureId(999) },            // ...but this fails
            ])
            .unwrap_err();
        assert_eq!(err.index, 1);
        assert_eq!(err.error, EditError::UnknownFeature(FeatureId(999)));
        assert_eq!(doc.history.features(), before.history.features(), "nothing landed");

        // A batch that leaves a dangling reference is refused by validation.
        let err = doc
            .apply_edits(&[DocEdit::Replace {
                id: b,
                kind: FeatureKind::FilletAll { source: FeatureId(42), radius: 1.0 },
            }])
            .unwrap_err();
        assert!(matches!(err.error, EditError::Dependency(_)), "{err}");
        assert_eq!(doc.history.features(), before.history.features());
    }

    #[test]
    fn transforms_splice_and_other_kinds_are_refused_there() {
        let mut doc = Document::new("t");
        let b = doc.add("Box", box_kind(10.0));
        let f = doc.add("Round", FeatureKind::FilletAll { source: b, radius: 1.0 });
        let out = doc
            .apply_edits(&[DocEdit::InsertTransform {
                name: None,
                kind: FeatureKind::Translate { source: b, offset: DVec3::new(5.0, 0.0, 0.0) },
            }])
            .unwrap();
        let mv = out[0].created().unwrap();
        // The fillet now consumes the moved box, not the original.
        assert_eq!(doc.history.get(f).unwrap().kind.primary_input(), Some(mv));
        assert_eq!(doc.history.get(mv).unwrap().name, "Move 1");
        let err = doc
            .apply_edits(&[DocEdit::InsertTransform {
                name: None,
                kind: FeatureKind::FilletAll { source: b, radius: 2.0 },
            }])
            .unwrap_err();
        assert!(matches!(err.error, EditError::NotATransform { .. }));
    }

    #[test]
    fn remove_keeps_the_rollback_bar_on_the_same_features() {
        let mut doc = Document::new("t");
        let a = doc.add("A", box_kind(1.0));
        let _b = doc.add("B", box_kind(2.0));
        let c = doc.add("C", box_kind(3.0));
        doc.set_rollback(2); // A, B active
        doc.apply_edits(&[DocEdit::Remove { id: a }]).unwrap();
        assert_eq!(doc.rollback(), 1, "B still the only... B active, C not");
        assert_eq!(doc.active_features().len(), 1);
        assert_eq!(doc.history.get(c).unwrap().name, "C");
    }

    #[test]
    fn ungroup_bakes_the_chain_onto_every_member() {
        let mut doc = Document::new("t");
        let a = doc.add("A", box_kind(10.0));
        let b = doc.add("B", box_kind(10.0));
        let g = doc.add("Group", FeatureKind::Group { members: vec![a, b] });
        let mv = doc.add("Move", FeatureKind::Translate { source: g, offset: DVec3::new(0.0, 0.0, 5.0) });
        assert_eq!(doc.group_and_chain(mv), Some((g, vec![mv])));
        let out = doc.apply_edits(&[DocEdit::Ungroup { id: mv }]).unwrap();
        assert_eq!(out, vec![EditOutcome::Changed]);
        assert!(doc.history.get(g).is_none() && doc.history.get(mv).is_none());
        let moves: Vec<_> = doc
            .history
            .features()
            .iter()
            .filter(|f| matches!(f.kind, FeatureKind::Translate { .. }))
            .collect();
        assert_eq!(moves.len(), 2, "one baked move per member");
        assert!(moves.iter().any(|m| m.kind.primary_input() == Some(a)));
        assert!(moves.iter().any(|m| m.kind.primary_input() == Some(b)));
        doc.history.validate().unwrap();

        let err = doc.apply_edits(&[DocEdit::Ungroup { id: a }]).unwrap_err();
        assert_eq!(err.error, EditError::NotAGroup(a));
    }

    #[test]
    fn edits_round_trip_as_tagged_json() {
        let edits = vec![
            DocEdit::Add { name: None, kind: box_kind(1.0) },
            DocEdit::Add {
                name: Some("Cut".into()),
                kind: FeatureKind::Boolean { op: BooleanOp::Subtract, target: FeatureId(1), tool: FeatureId(2) },
            },
            DocEdit::Suppress { id: FeatureId(3), suppressed: true },
            DocEdit::Tidy,
        ];
        let json = serde_json::to_string(&edits).unwrap();
        assert!(json.contains(r#""op":"add""#) && json.contains(r#""op":"tidy""#), "{json}");
        let back: Vec<DocEdit> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, edits);
        // Hand-written form: the tag plus the feature kind, nothing else.
        let hand: DocEdit =
            serde_json::from_str(r#"{"op":"add","kind":{"Cylinder":{"radius":5.0,"height":20.0}}}"#).unwrap();
        assert!(matches!(hand, DocEdit::Add { name: None, kind: FeatureKind::Cylinder { .. } }));
    }
}
