//! Discrete modeling commands driven from the Actions panel and shortcuts:
//! booleans, extrude, fillets, mirror/flip, group/ungroup, revolve, and
//! copy/paste of bodies.

use cdt_core::{BooleanOp, EdgeAnchor, FeatureId, FeatureKind, DVec3};
use cdt_render::{Axis3, Pick};

use crate::Modeler;

/// How far (mm, in X and Y) a paste is offset from its original so it's visibly
/// a separate body, ready to be moved.
pub(crate) const PASTE_OFFSET: f64 = 10.0;

/// Which part of the combined bounding box aligned bodies come to share on an
/// axis: the low edge, the center, or the high edge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AlignMode {
    Min,
    Center,
    Max,
}

impl Modeler {
    /// Apply a boolean. Subtract carves the selected body out of every other
    /// visible body (a cut against a body it doesn't overlap is a no-op, so
    /// this is "removed from any it overlaps"); union/intersect fold exactly
    /// the bodies behind the multi-face selection into one, in click order,
    /// and refuse (with a status hint) unless faces on at least two bodies
    /// are selected. Returns true if the document changed.
    pub(crate) fn apply_boolean(&mut self, op: BooleanOp) -> bool {
        match op {
            BooleanOp::Subtract => {
                let Some(tool) = self.ui.selected.filter(|s| self.ui.visible.contains(s)) else {
                    return false;
                };
                let others: Vec<FeatureId> = self
                    .ui
                    .visible
                    .iter()
                    .copied()
                    .filter(|&v| v != tool)
                    .collect();
                if others.is_empty() {
                    return false;
                }
                self.record_undo(self.doc.clone());
                // Carve the selected body out of each other body independently.
                for target in others {
                    self.doc
                        .add_numbered("Subtract", FeatureKind::Boolean { op, target, tool });
                }
                self.ui.selected = None; // the carver is consumed
            }
            BooleanOp::Union | BooleanOp::Intersect => {
                let union = matches!(op, BooleanOp::Union);
                let mut bodies = self.selected_bodies().into_iter();
                let (Some(first), Some(second)) = (bodies.next(), bodies.next()) else {
                    self.status = Some(format!(
                        "Select faces on two or more bodies to {}",
                        if union { "union" } else { "intersect" }
                    ));
                    return false;
                };
                self.record_undo(self.doc.clone());
                let name = if union { "Union" } else { "Intersect" };
                let mut acc = first;
                for other in std::iter::once(second).chain(bodies) {
                    acc = self
                        .doc
                        .add_numbered(name, FeatureKind::Boolean { op, target: acc, tool: other });
                }
                self.ui.selected = Some(acc);
            }
        }
        self.selection.clear();
        true
    }

    /// Extrude the sketch `source` into a solid, toward the camera: the extrude
    /// is along the sketch plane's normal, so flip the sign when that normal
    /// points away from the eye. Returns true if a feature was added.
    pub(crate) fn extrude_sketch(&mut self, source: FeatureId, eye: [f64; 3]) -> bool {
        let plane = match self.doc.history.get(source).map(|f| &f.kind) {
            Some(FeatureKind::Sketch { plane, .. })
            | Some(FeatureKind::ConstraintSketch { plane, .. }) => *plane,
            _ => return false,
        };
        let eye = DVec3::from_array(eye);
        let toward_eye = plane.normal().dot(eye - plane.origin()) >= 0.0;
        let distance = if toward_eye { 20.0 } else { -20.0 };
        self.record_undo(self.doc.clone());
        let id = self
            .doc
            .add_numbered("Extrude", FeatureKind::Extrude { source, distance });
        self.ui.selected = Some(id);
        true
    }

    /// Fillet all edges bounding the selected face(s), as one feature. Every
    /// selected face must be on the same body (the kernel dedupes shared edges).
    /// Returns true if a feature was added.
    /// Round EVERY edge of the selected body (the body behind the picked face)
    /// with a default radius. Distinct from `fillet_selected_face_edges`, which
    /// only rounds the selected face's own edges.
    pub(crate) fn fillet_all_selected_body(&mut self) -> bool {
        let Some(source) = self.selected_body() else {
            return false;
        };
        self.record_undo(self.doc.clone());
        let id = self
            .doc
            .add_numbered("Fillet", FeatureKind::FilletAll { source, radius: 2.0 });
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    pub(crate) fn fillet_selected_face_edges(&mut self) -> bool {
        // All selected faces must belong to one body.
        let bodies = self.selected_bodies();
        if bodies.len() != 1 {
            return false;
        }
        let source = bodies[0];
        let Some(body_index) = self.ui.visible.iter().position(|&f| f == source) else {
            return false;
        };
        let Some(solid) = self.bodies.get(body_index) else {
            return false;
        };
        // Gather a point on every edge of every selected face on this body.
        let mut edges: Vec<EdgeAnchor> = Vec::new();
        for pick in self.selection.selected() {
            if let Pick::Face(g) = pick {
                if let Some((idx, local)) = self.locate_face(*g) {
                    if idx == body_index {
                        if let Ok(points) = solid.face_edge_midpoints(local) {
                            edges.extend(
                                points.into_iter().map(|p| EdgeAnchor { point: DVec3::from_array(p) }),
                            );
                        }
                    }
                }
            }
        }
        if edges.is_empty() {
            return false;
        }
        self.record_undo(self.doc.clone());
        let id = self
            .doc
            .add_numbered("Fillet", FeatureKind::Fillet { source, edges, radius: 2.0 });
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// Mirror the selected face's body across that face's plane, keeping the
    /// original (a mirror-copy of a duplicated body). Returns true if added.
    pub(crate) fn mirror_across_selected_face(&mut self) -> bool {
        let Some(Pick::Face(global)) = self.selection.primary() else {
            return false;
        };
        let Some(plane) = self.selection.face_plane else {
            return false; // only a planar face defines a mirror plane
        };
        let Some((body_index, _)) = self.locate_face(global) else {
            return false;
        };
        let Some(source) = self.ui.visible.get(body_index).copied() else {
            return false;
        };
        self.record_undo(self.doc.clone());
        // Duplicate the body, reflect the clone across the face plane, keep the
        // original — so a half becomes a symmetric whole.
        let Some(clone_tip) = self.doc.duplicate(source) else {
            self.undo.pop();
            return false;
        };
        let id = self.doc.add_numbered(
            "Mirror",
            FeatureKind::Mirror {
                source: clone_tip,
                origin: plane.origin(),
                normal: plane.normal(),
            },
        );
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// Flip (mirror) the selected body in place across the plane through its own
    /// bbox center, perpendicular to `axis` — reverses handedness without moving
    /// it. No copy. Returns true if a feature was added.
    pub(crate) fn flip_in_place(&mut self, axis: Axis3) -> bool {
        let Some(body) = self.selected_body() else {
            return false;
        };
        let Some(idx) = self.ui.visible.iter().position(|&f| f == body) else {
            return false;
        };
        let Some(center) = self.body_centers.get(idx).copied() else {
            return false;
        };
        let normal = match axis {
            Axis3::X => DVec3::X,
            Axis3::Y => DVec3::Y,
            Axis3::Z => DVec3::Z,
        };
        self.record_undo(self.doc.clone());
        // Splice so flipping a body that feeds other steps propagates downstream.
        let id = self.doc.insert_transform_numbered(
            "Flip",
            FeatureKind::Mirror { source: body, origin: DVec3::from_array(center), normal },
        );
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// The distinct visible bodies behind the currently selected faces, in
    /// first-clicked order — the candidate members of a group.
    pub(crate) fn selected_bodies(&self) -> Vec<FeatureId> {
        let mut bodies = Vec::new();
        for pick in self.selection.selected() {
            if let Pick::Face(g) = pick {
                if let Some((idx, _)) = self.locate_face(*g) {
                    if let Some(&body) = self.ui.visible.get(idx) {
                        if !bodies.contains(&body) {
                            bodies.push(body);
                        }
                    }
                }
            }
        }
        bodies
    }

    /// Align every body behind the multi-face selection along one axis, to the
    /// combined selection's bounding box. `mode` picks which edge/center of the
    /// box they share. The whole alignment is one undo entry (a Translate per
    /// body that needs to move). Returns true if anything moved.
    pub(crate) fn align_selected(&mut self, axis: Axis3, mode: AlignMode) -> bool {
        let bodies = self.selected_bodies();
        if bodies.len() < 2 {
            return false;
        }
        let ax = match axis {
            Axis3::X => 0,
            Axis3::Y => 1,
            Axis3::Z => 2,
        };
        // Each selected body's extent on the axis, from its world AABB.
        let mut items: Vec<(FeatureId, f64, f64)> = Vec::new();
        for &b in &bodies {
            let Some(idx) = self.ui.visible.iter().position(|&f| f == b) else {
                continue;
            };
            if let Some((min, max)) = self.body_bounds.get(idx).copied() {
                items.push((b, min[ax], max[ax]));
            }
        }
        if items.len() < 2 {
            return false;
        }
        let overall_min = items.iter().map(|i| i.1).fold(f64::INFINITY, f64::min);
        let overall_max = items.iter().map(|i| i.2).fold(f64::NEG_INFINITY, f64::max);
        let overall_center = 0.5 * (overall_min + overall_max);

        // Precompute every body's axis offset before touching the document (each
        // Translate consumes its body and reorders `visible`, staling the bounds).
        let mut moves: Vec<(FeatureId, f64)> = Vec::new();
        for (b, mn, mx) in items {
            let delta = match mode {
                AlignMode::Min => overall_min - mn,
                AlignMode::Center => overall_center - 0.5 * (mn + mx),
                AlignMode::Max => overall_max - mx,
            };
            if delta.abs() >= 1e-6 {
                moves.push((b, delta));
            }
        }
        if moves.is_empty() {
            return false; // already aligned
        }
        self.record_undo(self.doc.clone());
        for (b, delta) in moves {
            let mut offset = DVec3::ZERO;
            offset[ax] = delta;
            // A body already ending in a Translate is aligned by adding to that
            // feature's offset. Otherwise splice a move into its chain, so
            // aligning bodies that feed downstream steps repositions the model
            // rather than spawning floating copies.
            match self.doc.history.get_mut(b).map(|f| &mut f.kind) {
                Some(FeatureKind::Translate { offset: o, .. }) => *o += offset,
                _ => {
                    self.doc.insert_transform_numbered(
                        "Align",
                        FeatureKind::Translate { source: b, offset },
                    );
                }
            }
        }
        self.selection.clear();
        true
    }

    /// Group the bodies behind the multi-face selection into one unit (a
    /// compound that moves/rotates/scales together). Returns true if grouped.
    pub(crate) fn group_selected(&mut self) -> bool {
        let members = self.selected_bodies();
        if members.len() < 2 {
            return false;
        }
        self.record_undo(self.doc.clone());
        let id = self.doc.add_numbered("Group", FeatureKind::Group { members });
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// Walk down from `tip` through any world-space transforms to the Group that
    /// feeds them. Returns `(group_id, chain)` where `chain` is the transforms
    /// above the group in bottom-up order (`chain[0].source == group`, the last
    /// is `tip`); empty if `tip` is the group itself. `None` if `tip` isn't a
    /// (possibly transformed) group, or sits under a non-transform (e.g. a
    /// boolean) we can't cleanly bake through.
    pub(crate) fn group_and_chain(&self, tip: FeatureId) -> Option<(FeatureId, Vec<FeatureId>)> {
        let mut chain = Vec::new();
        let mut cur = tip;
        loop {
            match &self.doc.history.get(cur)?.kind {
                FeatureKind::Group { .. } => {
                    chain.reverse(); // collected top-down; want bottom-up
                    return Some((cur, chain));
                }
                FeatureKind::Translate { source, .. }
                | FeatureKind::Rotate { source, .. }
                | FeatureKind::Scale { source, .. }
                | FeatureKind::Mirror { source, .. } => {
                    chain.push(cur);
                    cur = *source;
                }
                _ => return None,
            }
        }
    }

    /// Clone the transform `chain` (each step's `source` rewired to the previous
    /// clone, the first from `group` to `base`) onto `base`, returning the new
    /// tip. Bakes the group's accumulated transform onto one member.
    fn bake_chain_onto(
        &mut self,
        group: FeatureId,
        chain: &[FeatureId],
        base: FeatureId,
    ) -> Option<FeatureId> {
        let mut prev = base;
        let mut prev_old = group;
        for &t in chain {
            let feature = self.doc.history.get(t)?;
            let name = feature.name.clone();
            let mut kind = feature.kind.clone();
            kind.remap_input(prev_old, prev);
            let id = self.doc.add(name, kind);
            prev_old = t;
            prev = id;
        }
        Some(prev)
    }

    /// Ungroup the selected group, freeing its members back to independent
    /// visible bodies. Any transform applied to the group (move/rotate/scale/
    /// mirror) is BAKED onto each member first, so they keep their places.
    /// Returns true if ungrouped.
    pub(crate) fn ungroup_selected(&mut self) -> bool {
        let Some(tip) = self.selected_body() else {
            return false;
        };
        self.ungroup(tip)
    }

    /// Delete a history step (one undo entry). A Group is ungrouped instead —
    /// its members get the group's transforms baked on and become independent
    /// bodies again — because `History::remove` would otherwise rewire every
    /// dependent onto the first member alone. Returns true if the document
    /// changed.
    pub(crate) fn delete_feature(&mut self, id: FeatureId) -> bool {
        let is_group =
            matches!(self.doc.history.get(id).map(|f| &f.kind), Some(FeatureKind::Group { .. }));
        if is_group {
            return self.ungroup(id);
        }
        if self.doc.history.get(id).is_none() {
            return false;
        }
        self.record_undo(self.doc.clone());
        self.doc.history.remove(id);
        if self.ui.selected == Some(id) {
            self.ui.selected = None;
        }
        true
    }

    /// Ungroup the (possibly transformed) group at `tip`; see
    /// [`Self::ungroup_selected`]. Returns true if ungrouped.
    pub(crate) fn ungroup(&mut self, tip: FeatureId) -> bool {
        let Some((group, chain)) = self.group_and_chain(tip) else {
            return false;
        };
        let members = match self.doc.history.get(group).map(|f| &f.kind) {
            Some(FeatureKind::Group { members }) => members.clone(),
            _ => return false,
        };
        self.record_undo(self.doc.clone());
        // Give each member its own copy of the group's transform chain (consumes
        // the member, so only the transformed copy stays visible).
        for m in &members {
            self.bake_chain_onto(group, &chain, *m);
        }
        // Drop the original chain (top-down, so each is a tip) and the group.
        for &t in chain.iter().rev() {
            self.doc.history.remove(t);
        }
        self.doc.history.remove(group);
        self.doc.rollback_to_tip();
        self.ui.selected = None;
        self.selection.clear();
        true
    }

    /// Fillet the currently selected edges (each anchored at its clicked point),
    /// as one feature. Returns true if a feature was added.
    pub(crate) fn fillet_selected_edges(&mut self) -> bool {
        let Some(source) = self.selected_body() else {
            return false;
        };
        // Gather an anchor per selected edge, in click order.
        let edges: Vec<EdgeAnchor> = self
            .selection
            .selected()
            .iter()
            .filter_map(|pick| match pick {
                Pick::Edge(g) => self.edge_anchors.get(g).copied(),
                _ => None,
            })
            .map(|point| EdgeAnchor { point })
            .collect();
        if edges.is_empty() {
            return false;
        }

        self.record_undo(self.doc.clone());
        let id = self
            .doc
            .add_numbered("Fillet", FeatureKind::Fillet { source, edges, radius: 2.0 });
        self.ui.selected = Some(id);
        self.selection.clear();
        self.edge_anchors.clear();
        true
    }

    /// If a single edge of a planar sketch profile is selected, the profile
    /// feature to revolve about that edge. (Model-face profiles come later.)
    pub(crate) fn revolvable_profile(&self) -> Option<FeatureId> {
        if self.selection.selected().len() != 1 {
            return None;
        }
        let source = self.selected_body()?;
        matches!(
            self.doc.history.get(source).map(|f| &f.kind),
            Some(FeatureKind::Sketch { .. }) | Some(FeatureKind::ConstraintSketch { .. })
        )
        .then_some(source)
    }

    /// Revolve the selected sketch profile a full turn about the selected edge.
    /// The angle is then editable in the panel. Returns true if a feature added.
    pub(crate) fn revolve_selected_profile(&mut self) -> bool {
        let Some(source) = self.revolvable_profile() else {
            return false;
        };
        let Some(Pick::Edge(g)) = self.selection.primary() else {
            return false;
        };
        let Some(point) = self.edge_anchors.get(&g).copied() else {
            return false;
        };
        self.record_undo(self.doc.clone());
        let id = self.doc.add_numbered(
            "Revolve",
            FeatureKind::Revolve {
                source,
                axis: EdgeAnchor { point },
                angle: std::f64::consts::TAU,
            },
        );
        self.ui.selected = Some(id);
        self.selection.clear();
        self.edge_anchors.clear();
        true
    }

    /// The body to copy/duplicate: the viewport-selected body, else the
    /// panel-selected feature.
    fn copy_target(&self) -> Option<FeatureId> {
        self.selected_body().or(self.ui.selected)
    }

    /// Copy the selected body to the clipboard.
    pub(crate) fn copy_selected(&mut self) {
        self.clipboard = self.copy_target();
        self.status = Some(match self.clipboard.and_then(|id| self.doc.history.get(id)) {
            Some(f) => format!("Copied '{}'", f.name),
            None => "Select a body to copy".into(),
        });
    }

    /// Paste the clipboard body as an independent parametric duplicate, offset
    /// from the original and selected. Returns true if a body was pasted.
    pub(crate) fn paste(&mut self) -> bool {
        let Some(source) = self.clipboard else {
            return false;
        };
        if self.doc.history.get(source).is_none() {
            self.status = Some("Nothing to paste".into()); // the copied body is gone
            return false;
        }
        self.record_undo(self.doc.clone());
        let Some(clone_tip) = self.doc.duplicate(source) else {
            self.undo.pop();
            return false;
        };
        let offset = DVec3::new(PASTE_OFFSET, PASTE_OFFSET, 0.0);
        // The clone of a moved body already ends in its own Translate: fold the
        // paste offset into it instead of stacking a "Paste" step on top, so a
        // body keeps at most one trailing Move.
        let pasted = match self.doc.history.get_mut(clone_tip).map(|f| &mut f.kind) {
            Some(FeatureKind::Translate { offset: o, .. }) => {
                *o += offset;
                clone_tip
            }
            _ => self
                .doc
                .add_numbered("Move", FeatureKind::Translate { source: clone_tip, offset }),
        };
        self.ui.selected = Some(pasted);
        self.selection.clear();
        self.status = Some("Pasted a copy".into());
        true
    }
}
