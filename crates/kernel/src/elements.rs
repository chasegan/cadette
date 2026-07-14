//! Element maps: derivation-based identity for a body's faces and edges.
//!
//! Geometric anchors ([`cdt_core::FaceAnchor`]/[`cdt_core::EdgeAnchor`])
//! re-find a face or edge by *coordinates* on every rebuild, so they break —
//! or worse, silently match the wrong element — when an upstream edit moves
//! geometry. An element map instead gives every face and edge a NAME derived
//! from how it came into existence: minted [`ElementName::Born`] when a body
//! is created (primitive, sketch), carried unchanged through transforms, and
//! composed through topology-changing operations via OCCT's evolution
//! histories (splits/merges/generation — later slices; the mechanism is
//! proven in `tests/history_spike.rs`). A reference stores a name; resolving
//! it is a map lookup, immune to motion.
//!
//! Indices are TopExp map order — the SAME order as tessellation
//! `face_ids`/`edge_ids`, GPU picking, and `face_area`/`edge_length` — so a
//! picked face index converts to a name (and back) without translation.

/// Face or edge — which element list a name or index refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElemKind {
    Face = 0,
    Edge = 1,
}

/// The derivation-based name of one face or edge.
///
/// Only the bedrock variant exists yet: operations that transform identity
/// (splits, merges, generated fillet faces, section edges) add composing
/// variants in later slices. A name never encodes geometry — only lineage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementName {
    /// The `index`-th face/edge (TopExp order) of the body born at feature
    /// `origin` (a [`cdt_core::FeatureId`]'s raw value) — the bedrock every
    /// derivation chain bottoms out on. Stable because a primitive's topology
    /// is a deterministic function of its parameters. (A constraint sketch's
    /// boundary-edge indices can still shift under sketch edits; deriving
    /// those names from sketch segment ids is the planned refinement when
    /// extrude naming lands.)
    Born {
        origin: u64,
        kind: ElemKind,
        index: u32,
    },
}

/// The element map of one body: a name per face and per edge, index-aligned
/// with TopExp order. An EMPTY map is meaningful — it marks a body from an
/// operation that doesn't propagate identity yet; lookups miss and callers
/// fall back to geometric anchors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementMap {
    pub faces: Vec<ElementName>,
    pub edges: Vec<ElementName>,
}

impl ElementMap {
    /// Mint the bedrock map for a newly born body: every current face and
    /// edge is [`ElementName::Born`] under the creating feature `origin`.
    pub fn born(origin: u64, faces: usize, edges: usize) -> Self {
        let name = |kind, index: usize| ElementName::Born {
            origin,
            kind,
            index: index as u32,
        };
        ElementMap {
            faces: (0..faces).map(|i| name(ElemKind::Face, i)).collect(),
            edges: (0..edges).map(|i| name(ElemKind::Edge, i)).collect(),
        }
    }

    /// The name of face `index`, if this body is mapped.
    pub fn face(&self, index: u32) -> Option<&ElementName> {
        self.faces.get(index as usize)
    }

    /// The name of edge `index`, if this body is mapped.
    pub fn edge(&self, index: u32) -> Option<&ElementName> {
        self.edges.get(index as usize)
    }

    /// The current face index carrying `name` (resolution), if present.
    pub fn find_face(&self, name: &ElementName) -> Option<u32> {
        self.faces.iter().position(|n| n == name).map(|i| i as u32)
    }

    /// The current edge index carrying `name` (resolution), if present.
    pub fn find_edge(&self, name: &ElementName) -> Option<u32> {
        self.edges.iter().position(|n| n == name).map(|i| i as u32)
    }

    /// True for the map of an identity-untracked body (see type docs).
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty() && self.edges.is_empty()
    }
}
