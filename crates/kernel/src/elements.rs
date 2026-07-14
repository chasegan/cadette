//! Element maps: derivation-based identity for a body's faces and edges.
//!
//! Geometric anchors ([`cdt_core::FaceAnchor`]/[`cdt_core::EdgeAnchor`])
//! re-find a face or edge by *coordinates* on every rebuild, so they break —
//! or worse, silently match the wrong element — when an upstream edit moves
//! geometry. An element map instead gives every face and edge a NAME derived
//! from how it came into existence: minted [`ElementName::Born`] when a body
//! is created (primitive, sketch), carried unchanged through transforms, and
//! composed through booleans via OCCT's evolution histories (proven in
//! `tests/history_spike.rs`; generative ops like extrude/fillet are later
//! slices). A reference stores a name; resolving it is a map lookup, immune
//! to motion.
//!
//! Indices are TopExp map order — the SAME order as tessellation
//! `face_ids`/`edge_ids`, GPU picking, and `face_area`/`edge_length` — so a
//! picked face index converts to a name (and back) without translation.

use std::collections::HashMap;

/// Face or edge — which element list a name or index refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElemKind {
    Face = 0,
    Edge = 1,
}

/// The derivation-based name of one face or edge. A name never encodes
/// geometry — only lineage. Survivors of an operation keep their name
/// VERBATIM (that's what makes references stable across parameter edits);
/// names compose only at topology-changing events.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// The `ordinal`-th piece of `parent` after an operation divided it —
    /// pieces are ranked by centroid (x, then y, then z), a deterministic,
    /// geometry-anchored tie-break. If a later edit un-splits the parent, the
    /// name dangles and the referencing feature fails loudly (never a silent
    /// re-guess).
    Split {
        parent: Box<ElementName>,
        ordinal: u32,
    },
    /// One element unified from several same-domain parents (a fuse merging
    /// coplanar faces, unify removing their seam). Parents are sorted, so the
    /// name is order-independent.
    Merged { parents: Vec<ElementName> },
    /// A NEW edge born where faces intersect (the rim circle where a bore
    /// meets a box top). Parents are the intersecting faces' names, sorted;
    /// an untracked parent face is simply absent, so the name may carry one
    /// parent instead of two.
    Intersection { parents: Vec<ElementName> },
}

/// One (input element → result element) lineage pair reported by a tracked
/// operation. A split input yields several pairs; `output: None` means the
/// element was consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lineage {
    /// 0 = the first operand, 1 = the second.
    pub input: u8,
    /// Element index in that operand (TopExp map order).
    pub index: u32,
    /// Element index in the result, or `None` if deleted.
    pub output: Option<u32>,
}

/// The element map of one body: a name per face and per edge, index-aligned
/// with TopExp order. `None` marks an element whose lineage is unknown (its
/// parent body predates name tracking, or the producing operation doesn't
/// propagate identity yet); lookups miss and callers fall back to geometric
/// anchors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementMap {
    pub faces: Vec<Option<ElementName>>,
    pub edges: Vec<Option<ElementName>>,
}

impl ElementMap {
    /// Mint the bedrock map for a newly born body: every current face and
    /// edge is [`ElementName::Born`] under the creating feature `origin`.
    pub fn born(origin: u64, faces: usize, edges: usize) -> Self {
        let name = |kind, index: usize| {
            Some(ElementName::Born {
                origin,
                kind,
                index: index as u32,
            })
        };
        ElementMap {
            faces: (0..faces).map(|i| name(ElemKind::Face, i)).collect(),
            edges: (0..edges).map(|i| name(ElemKind::Edge, i)).collect(),
        }
    }

    /// The name of face `index`, if known.
    pub fn face(&self, index: u32) -> Option<&ElementName> {
        self.faces.get(index as usize)?.as_ref()
    }

    /// The name of edge `index`, if known.
    pub fn edge(&self, index: u32) -> Option<&ElementName> {
        self.edges.get(index as usize)?.as_ref()
    }

    /// The current face index carrying `name` (resolution), if present.
    pub fn find_face(&self, name: &ElementName) -> Option<u32> {
        self.faces
            .iter()
            .position(|n| n.as_ref() == Some(name))
            .map(|i| i as u32)
    }

    /// The current edge index carrying `name` (resolution), if present.
    pub fn find_edge(&self, name: &ElementName) -> Option<u32> {
        self.edges
            .iter()
            .position(|n| n.as_ref() == Some(name))
            .map(|i| i as u32)
    }

    /// True when the map carries no names at all — the body of an
    /// identity-untracked operation (see type docs).
    pub fn is_empty(&self) -> bool {
        self.faces.iter().flatten().next().is_none()
            && self.edges.iter().flatten().next().is_none()
    }

    /// Compose a boolean result's map from the operands' maps and the
    /// kernel's evolution report (`Solid::boolean_tracked`).
    ///
    /// Rules, per result element from its contributing parents:
    /// - one parent, sole piece → the parent's name VERBATIM (survivor);
    /// - one parent among several pieces → [`ElementName::Split`], ordinal by
    ///   centroid rank;
    /// - several parents → [`ElementName::Merged`] of the (possibly Split)
    ///   contributions, sorted;
    /// - no known parent → `None` (untracked operand).
    ///
    /// New section edges get [`ElementName::Intersection`] of their parent
    /// faces' names — only where edge lineage left a gap (survival wins).
    ///
    /// `centroid(kind, result_index)` supplies the split tie-break key.
    #[allow(clippy::too_many_arguments)]
    pub fn compose_boolean(
        a: &ElementMap,
        b: &ElementMap,
        face_lineage: &[Lineage],
        edge_lineage: &[Lineage],
        sections: &[(u8, u32, u32)],
        out_faces: usize,
        out_edges: usize,
        centroid: &mut dyn FnMut(ElemKind, u32) -> [f64; 3],
    ) -> ElementMap {
        let faces = compose_kind(a, b, ElemKind::Face, face_lineage, out_faces, centroid);
        let mut edges = compose_kind(a, b, ElemKind::Edge, edge_lineage, out_edges, centroid);

        // Section edges fill the gaps the edge lineage leaves: a genuinely new
        // curve has no edge parent, only face parents.
        let mut parents_of: HashMap<u32, Vec<ElementName>> = HashMap::new();
        for &(which, face_index, out_edge) in sections {
            let map = if which == 0 { a } else { b };
            if let Some(name) = map.face(face_index) {
                parents_of.entry(out_edge).or_default().push(name.clone());
            }
        }
        for (out_edge, mut parents) in parents_of {
            if let Some(slot @ None) = edges.get_mut(out_edge as usize) {
                parents.sort();
                parents.dedup();
                *slot = Some(ElementName::Intersection { parents });
            }
        }

        ElementMap { faces, edges }
    }
}

/// The survivor/split/merge composition for one element kind — see
/// [`ElementMap::compose_boolean`].
fn compose_kind(
    a: &ElementMap,
    b: &ElementMap,
    kind: ElemKind,
    lineage: &[Lineage],
    out_len: usize,
    centroid: &mut dyn FnMut(ElemKind, u32) -> [f64; 3],
) -> Vec<Option<ElementName>> {
    let parent_name = |which: u8, index: u32| -> Option<&ElementName> {
        let map = if which == 0 { a } else { b };
        match kind {
            ElemKind::Face => map.face(index),
            ElemKind::Edge => map.edge(index),
        }
    };

    // Each parent's surviving pieces (for split detection), and each result
    // element's contributing parents (for merge detection).
    let mut pieces_of: HashMap<(u8, u32), Vec<u32>> = HashMap::new();
    let mut contributors: Vec<Vec<(u8, u32)>> = vec![Vec::new(); out_len];
    for l in lineage {
        let Some(out) = l.output else { continue };
        pieces_of.entry((l.input, l.index)).or_default().push(out);
        if let Some(c) = contributors.get_mut(out as usize) {
            c.push((l.input, l.index));
        }
    }

    // Memoized centroid ranking (x, then y, then z; total_cmp so even a NaN
    // from a failed measure stays deterministic).
    let mut keys: HashMap<u32, [f64; 3]> = HashMap::new();
    let mut key = |out: u32, centroid: &mut dyn FnMut(ElemKind, u32) -> [f64; 3]| {
        *keys.entry(out).or_insert_with(|| centroid(kind, out))
    };
    let rank = |p: [f64; 3], q: [f64; 3]| {
        p[0].total_cmp(&q[0])
            .then(p[1].total_cmp(&q[1]))
            .then(p[2].total_cmp(&q[2]))
    };

    contributors
        .iter()
        .enumerate()
        .map(|(out, contribs)| {
            let mut parts: Vec<ElementName> = Vec::new();
            for &(which, index) in contribs {
                let Some(base) = parent_name(which, index) else {
                    continue;
                };
                let pieces = &pieces_of[&(which, index)];
                let name = if pieces.len() <= 1 {
                    base.clone()
                } else {
                    let mut ranked = pieces.clone();
                    ranked.sort_unstable();
                    ranked.dedup();
                    ranked.sort_by(|&p, &q| rank(key(p, centroid), key(q, centroid)));
                    let ordinal =
                        ranked.iter().position(|&p| p == out as u32).unwrap_or(0) as u32;
                    ElementName::Split {
                        parent: Box::new(base.clone()),
                        ordinal,
                    }
                };
                if !parts.contains(&name) {
                    parts.push(name);
                }
            }
            match parts.len() {
                0 => None,
                1 => parts.pop(),
                _ => {
                    parts.sort();
                    Some(ElementName::Merged { parents: parts })
                }
            }
        })
        .collect()
}
