//! The current viewport selection: what's clicked, what's hovered, and the
//! plane of a selected planar face (which enables sketch-on-face).
//!
//! Selection is a **set**: ⇧/⌘-click adds or removes an entity, a plain click
//! replaces. A set is kept homogeneous (all faces, or all edges) — picking a
//! different kind replaces, since mixed selections map to no real operation.

use rmf_core::{Pick, SketchPlane};

/// What the user has selected and is hovering in the viewport.
#[derive(Clone, Default)]
pub struct Selection {
    /// The clicked entities (strong highlight), in click order. Homogeneous.
    selected: Vec<Pick>,
    /// The entity under the cursor (subtle pre-highlight).
    pub hovered: Option<Pick>,
    /// The plane of the (single) selected face, if exactly one planar face is
    /// selected — the host computes this (it needs the kernel) and stores it.
    pub face_plane: Option<SketchPlane>,
}

/// Two picks are the same kind if both are faces or both are edges.
fn same_kind(a: &Pick, b: &Pick) -> bool {
    matches!(
        (a, b),
        (Pick::Face(_), Pick::Face(_)) | (Pick::Edge(_), Pick::Edge(_))
    )
}

impl Selection {
    /// Apply a click. `additive` (⇧/⌘) toggles `pick` in/out of the set; a plain
    /// click replaces the set with just `pick`. A pick of a different kind than
    /// the current set always replaces (the set stays homogeneous). `face_plane`
    /// is the clicked face's plane, kept only when the result is a single face.
    pub fn select(&mut self, pick: Option<Pick>, additive: bool, face_plane: Option<SketchPlane>) {
        match pick {
            None => {
                // A plain click on empty space clears; ⇧/⌘ on empty keeps the set.
                if !additive {
                    self.selected.clear();
                }
            }
            Some(p) => {
                let homogeneous = self.selected.first().is_none_or(|q| same_kind(q, &p));
                if additive && homogeneous {
                    if let Some(i) = self.selected.iter().position(|q| *q == p) {
                        self.selected.remove(i); // toggle off
                    } else {
                        self.selected.push(p); // toggle on
                    }
                } else {
                    self.selected.clear();
                    self.selected.push(p);
                }
            }
        }
        // A face plane is meaningful only for a lone selected face.
        self.face_plane = match self.selected.as_slice() {
            [Pick::Face(_)] => face_plane,
            _ => None,
        };
    }

    /// Set the hovered entity.
    pub fn hover(&mut self, pick: Option<Pick>) {
        self.hovered = pick;
    }

    /// Clear everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The selected entities, in click order.
    pub fn selected(&self) -> &[Pick] {
        &self.selected
    }

    /// The first selected entity — the "primary" for single-target operations.
    pub fn primary(&self) -> Option<Pick> {
        self.selected.first().copied()
    }

    /// True when nothing is selected.
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// True when every selected entity is an edge (and at least one exists) —
    /// i.e. the selection can be filleted.
    pub fn is_edges(&self) -> bool {
        !self.selected.is_empty() && self.selected.iter().all(|p| matches!(p, Pick::Edge(_)))
    }

    /// True when a single planar face is selected — i.e. a sketch can start.
    pub fn can_sketch_on_face(&self) -> bool {
        self.face_plane.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selecting_a_planar_face_enables_sketch_on_face() {
        let mut s = Selection::default();
        assert!(!s.can_sketch_on_face());

        s.select(Some(Pick::Face(2)), false, Some(SketchPlane::Xy));
        assert_eq!(s.primary(), Some(Pick::Face(2)));
        assert!(s.can_sketch_on_face());

        // An edge selection carries no face plane.
        s.select(Some(Pick::Edge(5)), false, None);
        assert!(!s.can_sketch_on_face());
        assert!(s.is_edges());

        s.clear();
        assert!(s.is_empty() && s.hovered.is_none());
    }

    #[test]
    fn additive_click_builds_and_toggles_a_homogeneous_set() {
        let mut s = Selection::default();
        s.select(Some(Pick::Edge(1)), false, None);
        s.select(Some(Pick::Edge(2)), true, None);
        s.select(Some(Pick::Edge(3)), true, None);
        assert_eq!(s.selected(), &[Pick::Edge(1), Pick::Edge(2), Pick::Edge(3)]);

        // Re-clicking a member with the modifier removes it.
        s.select(Some(Pick::Edge(2)), true, None);
        assert_eq!(s.selected(), &[Pick::Edge(1), Pick::Edge(3)]);

        // A plain click replaces the whole set.
        s.select(Some(Pick::Edge(9)), false, None);
        assert_eq!(s.selected(), &[Pick::Edge(9)]);

        // A different-kind pick replaces even with the modifier held.
        s.select(Some(Pick::Face(4)), true, Some(SketchPlane::Xy));
        assert_eq!(s.selected(), &[Pick::Face(4)]);
        assert!(s.can_sketch_on_face());
    }

    #[test]
    fn additive_on_empty_space_keeps_the_set() {
        let mut s = Selection::default();
        s.select(Some(Pick::Edge(1)), false, None);
        s.select(None, true, None); // ⇧-click on background
        assert_eq!(s.selected(), &[Pick::Edge(1)]);
        s.select(None, false, None); // plain click on background clears
        assert!(s.is_empty());
    }
}
