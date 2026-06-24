//! The current viewport selection: what's clicked, what's hovered, and the
//! plane of a selected planar face (which enables sketch-on-face).

use rmf_core::{Pick, SketchPlane};

/// What the user has selected and is hovering in the viewport.
#[derive(Clone, Copy, Default)]
pub struct Selection {
    /// The clicked entity (strong highlight).
    pub selected: Option<Pick>,
    /// The entity under the cursor (subtle pre-highlight).
    pub hovered: Option<Pick>,
    /// The plane of the selected face, if it is planar — the host computes this
    /// (it needs the kernel) and stores it here.
    pub face_plane: Option<SketchPlane>,
}

impl Selection {
    /// Set the clicked entity and the plane of its face (if planar).
    pub fn select(&mut self, pick: Option<Pick>, face_plane: Option<SketchPlane>) {
        self.selected = pick;
        self.face_plane = face_plane;
    }

    /// Set the hovered entity.
    pub fn hover(&mut self, pick: Option<Pick>) {
        self.hovered = pick;
    }

    /// Clear everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// True when a planar face is selected — i.e. a sketch can start on it.
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

        s.select(Some(Pick::Face(2)), Some(SketchPlane::Xy));
        assert_eq!(s.selected, Some(Pick::Face(2)));
        assert!(s.can_sketch_on_face());

        // An edge selection carries no face plane.
        s.select(Some(Pick::Edge(5)), None);
        assert!(!s.can_sketch_on_face());

        s.clear();
        assert!(s.selected.is_none() && s.hovered.is_none());
    }
}
