//! Discrete modeling commands driven from the Actions panel and shortcuts:
//! booleans, extrude, fillets, mirror/flip, group/ungroup, revolve, and
//! copy/paste of bodies.

use cdt_core::{BooleanOp, EdgeAnchor, FeatureId, FeatureKind, Helix, PatternLayout, DVec3};
use cdt_render::{Axis3, Pick};

use crate::Modeler;

/// How far (mm, in X and Y) a paste is offset from its original so it's visibly
/// a separate body, ready to be moved.
pub(crate) const PASTE_OFFSET: f64 = 10.0;

/// Which starting layout a "Pattern" action uses (refined in Properties).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PatternKind {
    Linear,
    Circular,
    Helical,
}

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

    /// Cut an ISO thread into the selected cylindrical face: the coarse pitch
    /// for its diameter, the ISO depth for that pitch, inward on a shaft and
    /// outward in a hole. The helix overshoots the face by a pitch at each end
    /// so the groove runs cleanly off (it only cuts air there). Returns true if
    /// a feature was added.
    pub(crate) fn thread_selected_face(&mut self) -> bool {
        let Some(cyl) = self.selection.face_cylinder else {
            return false;
        };
        let Some(body) = self.selected_body() else {
            return false;
        };
        // A hole is modelled at its minor (tap-drill) diameter, a shaft at its
        // major: pick the standard size accordingly.
        let pitch = if cyl.convex {
            Helix::iso_pitch(2.0 * cyl.radius)
        } else {
            Helix::iso_pitch_for_minor(2.0 * cyl.radius)
        };
        let helix = Helix {
            center: cyl.start - cyl.axis * pitch,
            axis: cyl.axis,
            radius: cyl.radius,
            pitch,
            length: cyl.length + 2.0 * pitch,
            left_handed: false,
        };
        self.record_undo(self.doc.clone());
        let id = self.doc.add_numbered(
            "Thread",
            FeatureKind::Thread {
                source: body,
                helix,
                depth: Helix::iso_depth(pitch),
                internal: !cyl.convex,
                starts: 1,
            },
        );
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// Sweep the selected sketch profile along a helix into a coil. The helix
    /// axis is the sketch plane's in-plane Y (so a section drawn on the front
    /// plane coils about Z), through the plane origin, at the profile's
    /// distance from that axis; the pitch is twice the profile's size and the
    /// coil five turns long — all editable in Properties. Returns true if added.
    pub(crate) fn sweep_along_helix(&mut self, profile: FeatureId) -> bool {
        let plane = match self.doc.history.get(profile).map(|f| &f.kind) {
            Some(FeatureKind::Sketch { plane, .. })
            | Some(FeatureKind::ConstraintSketch { plane, .. }) => *plane,
            _ => return false,
        };
        let Some(idx) = self.ui.visible.iter().position(|&f| f == profile) else {
            return false;
        };
        let Some((min, max)) = self.body_bounds.get(idx).copied() else {
            return false;
        };
        let center_of = DVec3::new(0.5 * (min[0] + max[0]), 0.5 * (min[1] + max[1]), 0.5 * (min[2] + max[2]));
        let axis = plane.y_dir().normalize();
        let origin = plane.origin();
        let to_body = center_of - origin;
        let radial = to_body - axis * to_body.dot(axis);
        let size = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]).max(1.0);
        let radius = if radial.length() > 1e-6 { radial.length() } else { 2.0 * size };
        let pitch = 2.0 * size;
        let helix = Helix { center: origin, axis, radius, pitch, length: 5.0 * pitch, left_handed: false };
        self.record_undo(self.doc.clone());
        let id = self.doc.add_numbered(
            "Coil",
            FeatureKind::Sweep { profile, path: cdt_core::SweepPath::Helix { helix } },
        );
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// The pattern instance behind the selected face, if the selected body is
    /// a Pattern: `(pattern id, instance index)`. Instances are compounded in
    /// order, each with the seed's face count, so the local face index maps to
    /// the k-th built instance, which maps back through the skip list.
    pub(crate) fn selected_pattern_instance(&self) -> Option<(FeatureId, u32)> {
        let Pick::Face(global) = self.selection.primary()? else {
            return None;
        };
        let (body_index, local) = self.locate_face(global)?;
        // The pattern may sit under in-place transforms (a dragged pattern is
        // a Translate of it); those keep the compound's face layout, so walk
        // down the chain to the Pattern itself.
        let mut id = *self.ui.visible.get(body_index)?;
        let (count, skip) = loop {
            let kind = &self.doc.history.get(id)?.kind;
            match kind {
                FeatureKind::Pattern { count, skip, .. } => break (*count, skip.as_slice()),
                k if k.is_in_place_transform() => id = k.primary_input()?,
                _ => return None,
            }
        };
        let built = cdt_core::pattern_instances(count, skip);
        let faces = self.bodies.get(body_index)?.face_count().ok()? as u32;
        if built.is_empty() || faces == 0 || faces % built.len() as u32 != 0 {
            return None;
        }
        let per_instance = faces / built.len() as u32;
        built.get((local / per_instance) as usize).map(|&i| (id, i))
    }

    /// Leave the pattern instance behind the selected face out of the pattern.
    /// Returns true if an instance was skipped.
    pub(crate) fn skip_selected_instance(&mut self) -> bool {
        let Some((id, instance)) = self.selected_pattern_instance() else {
            return false;
        };
        self.record_undo(self.doc.clone());
        if let Some(FeatureKind::Pattern { skip, .. }) = self.doc.history.get_mut(id).map(|f| &mut f.kind) {
            if !skip.contains(&instance) {
                skip.push(instance);
                skip.sort_unstable();
            }
        }
        self.ui.selected = Some(id);
        self.selection.clear();
        true
    }

    /// Repeat the selected body in a pattern, spliced into its chain so a
    /// patterned tool repeats its cut downstream. `Linear` steps along the
    /// selected face's normal by the body's own extent plus a gap, so the
    /// instances land side by side; `Circular` spreads them around the world
    /// Z axis through the origin. Both are starting points for the Properties
    /// pane. Returns true if a feature was added.
    pub(crate) fn pattern_selected_body(&mut self, kind: PatternKind) -> bool {
        let Some(body) = self.selected_body() else {
            return false;
        };
        let Some(idx) = self.ui.visible.iter().position(|&f| f == body) else {
            return false;
        };
        let Some((min, max)) = self.body_bounds.get(idx).copied() else {
            return false;
        };
        let (layout, count) = match kind {
            PatternKind::Linear => {
                // Along the picked face's normal (falling back to +X), by the
                // body's extent on that axis plus a half-extent gap.
                let dir = self
                    .selection
                    .face_plane
                    .map(|p| p.normal())
                    .filter(|n| n.length() > 1e-9)
                    .unwrap_or(DVec3::X)
                    .normalize();
                let extent = DVec3::new(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
                let along = extent.abs().dot(dir.abs()).max(1.0);
                (PatternLayout::Linear { step: dir * (along * 1.5) }, 3)
            }
            PatternKind::Circular | PatternKind::Helical => {
                // Around the world Z axis through the origin — unless the body
                // sits on that axis, where every instance would coincide; then
                // orbit it about a centre off to its side instead.
                let body_center = DVec3::new(
                    0.5 * (min[0] + max[0]),
                    0.5 * (min[1] + max[1]),
                    0.5 * (min[2] + max[2]),
                );
                let on_axis = body_center.truncate().length() < 1e-6;
                let center = if on_axis {
                    DVec3::new(body_center.x + 1.5 * (max[0] - min[0]).max(1.0), body_center.y, 0.0)
                } else {
                    DVec3::ZERO
                };
                if kind == PatternKind::Circular {
                    let count = 6;
                    (
                        PatternLayout::Circular {
                            center,
                            axis: DVec3::Z,
                            angle: std::f64::consts::TAU / f64::from(count),
                        },
                        count,
                    )
                } else {
                    // Twelve instances a turn, each lifted by the body's own
                    // height — a spiral staircase of the body up the axis.
                    (
                        PatternLayout::Helical {
                            center,
                            axis: DVec3::Z,
                            angle: std::f64::consts::TAU / 12.0,
                            rise: (max[2] - min[2]).max(1.0),
                        },
                        12,
                    )
                }
            }
        };
        self.record_undo(self.doc.clone());
        let id = self.doc.insert_transform_numbered(
            "Pattern",
            FeatureKind::Pattern { source: body, layout, count, skip: Vec::new() },
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
        if self.doc.group_and_chain(tip).is_none() {
            return false;
        }
        self.record_undo(self.doc.clone());
        if self.doc.ungroup(tip).is_none() {
            self.undo.pop();
            return false;
        }
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
