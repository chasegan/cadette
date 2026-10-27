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
    /// parent instead of two. Also names a generative op's lateral edges by
    /// the (result) faces they separate.
    Intersection { parents: Vec<ElementName> },
    /// A face a generative op (extrude/revolve/sweep) swept out of profile
    /// boundary edge `parent` — the side wall an extrude drags out of a
    /// sketch segment.
    Swept { parent: Box<ElementName> },
    /// The far-end image a generative op creates opposite the profile: the
    /// end cap from the profile FACE, or an end-cap rim edge from a profile
    /// EDGE. (The profile-side images keep the profile elements' names
    /// verbatim — they ARE those elements.)
    Cap { parent: Box<ElementName> },
    /// The face a fillet rolled out of edge `parent` — "the rounding of that
    /// edge". The trimmed faces around it keep their names verbatim.
    Rounded { parent: Box<ElementName> },
    /// The flat face a chamfer cut across edge `parent` — "the bevel of that
    /// edge". As with fillets, the trimmed neighbours keep their names.
    Chamfered { parent: Box<ElementName> },
    /// The inner wall a shell offset from face `parent` — "the inside of
    /// that face". The outer faces keep their names verbatim.
    Offset { parent: Box<ElementName> },
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

/// One lineage record from a generative op (extrude/revolve/sweep) — the
/// parsed form of the kernel's `[tag, in, out]` triples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenRecord {
    /// The profile face's image at the sweep START (the profile-side cap) —
    /// it keeps the profile face's name verbatim.
    StartCapFace { out_face: u32 },
    /// The profile face's image at the END (the far cap).
    EndCapFace { out_face: u32 },
    /// A side face swept from profile edge `edge`.
    SideFace { edge: u32, out_face: u32 },
    /// Profile edge `edge`'s image on the start side (name verbatim).
    StartEdge { edge: u32, out_edge: u32 },
    /// Profile edge `edge`'s image on the end cap.
    EndEdge { edge: u32, out_edge: u32 },
}

impl GenRecord {
    /// Parse one kernel `[tag, in, out]` triple; `None` for an unknown tag.
    pub fn parse(tag: i32, input: u32, out: u32) -> Option<GenRecord> {
        Some(match tag {
            0 => GenRecord::StartCapFace { out_face: out },
            1 => GenRecord::EndCapFace { out_face: out },
            2 => GenRecord::SideFace { edge: input, out_face: out },
            3 => GenRecord::StartEdge { edge: input, out_edge: out },
            4 => GenRecord::EndEdge { edge: input, out_edge: out },
            _ => return None,
        })
    }
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
        let mut faces = compose_kind(a, b, ElemKind::Face, face_lineage, out_faces, centroid);
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

        // Ambiguity is never guessed away (e.g. the same body fed to both
        // operand slots): clear any name that landed on two elements.
        clear_duplicates(&mut faces);
        clear_duplicates(&mut edges);

        ElementMap { faces, edges }
    }

    /// Compose a generative op's map (extrude/revolve/sweep) from the profile
    /// body's map and the kernel's lineage report.
    ///
    /// Rules: the profile face's START image keeps its name verbatim (it IS
    /// that face); its END image becomes [`ElementName::Cap`]; each profile
    /// edge sweeps a [`ElementName::Swept`] side face (several pieces — a
    /// corner-transition sweep — become `Split`s of it by centroid rank);
    /// start/end rim edges mirror the face rule. Lateral edges (swept from
    /// vertices, which maps don't track) are then named by ADJACENCY —
    /// [`ElementName::Intersection`] of the named result faces they separate.
    /// Finally, any name that ended up on two elements is cleared from both:
    /// ambiguity is never guessed away.
    pub fn compose_generated(
        profile: &ElementMap,
        records: &[GenRecord],
        out_faces: usize,
        out_edges: usize,
        adjacency: &[(u32, u32)],
        centroid: &mut dyn FnMut(ElemKind, u32) -> [f64; 3],
    ) -> ElementMap {
        let mut faces: Vec<Option<ElementName>> = vec![None; out_faces];
        let mut edges: Vec<Option<ElementName>> = vec![None; out_edges];
        // First write wins: a full revolution's start and end images are the
        // SAME seam element, and the verbatim start name (reported first) is
        // the truthful one — the seam IS the profile element.
        let set = |slots: &mut Vec<Option<ElementName>>, i: u32, name: Option<ElementName>| {
            if let Some(slot @ None) = slots.get_mut(i as usize) {
                *slot = name;
            }
        };

        // Side faces first, grouped per profile edge so multi-piece sweeps
        // (RoundCorner transitions) get Split ordinals instead of duplicates.
        let mut sides_of: HashMap<u32, Vec<u32>> = HashMap::new();
        for r in records {
            if let GenRecord::SideFace { edge, out_face } = r {
                sides_of.entry(*edge).or_default().push(*out_face);
            }
        }
        for (edge, mut outs) in sides_of {
            let Some(parent) = profile.edge(edge) else { continue };
            let swept = ElementName::Swept {
                parent: Box::new(parent.clone()),
            };
            outs.sort_unstable();
            outs.dedup();
            if outs.len() == 1 {
                set(&mut faces, outs[0], Some(swept));
            } else {
                let mut ranked = outs;
                ranked.sort_by(|&p, &q| {
                    let (a, b) = (centroid(ElemKind::Face, p), centroid(ElemKind::Face, q));
                    a[0].total_cmp(&b[0])
                        .then(a[1].total_cmp(&b[1]))
                        .then(a[2].total_cmp(&b[2]))
                });
                for (ordinal, &out) in ranked.iter().enumerate() {
                    set(
                        &mut faces,
                        out,
                        Some(ElementName::Split {
                            parent: Box::new(swept.clone()),
                            ordinal: ordinal as u32,
                        }),
                    );
                }
            }
        }
        for r in records {
            match *r {
                GenRecord::StartCapFace { out_face } => {
                    set(&mut faces, out_face, profile.face(0).cloned());
                }
                GenRecord::EndCapFace { out_face } => {
                    let name = profile.face(0).map(|p| ElementName::Cap {
                        parent: Box::new(p.clone()),
                    });
                    set(&mut faces, out_face, name);
                }
                GenRecord::StartEdge { edge, out_edge } => {
                    set(&mut edges, out_edge, profile.edge(edge).cloned());
                }
                GenRecord::EndEdge { edge, out_edge } => {
                    let name = profile.edge(edge).map(|p| ElementName::Cap {
                        parent: Box::new(p.clone()),
                    });
                    set(&mut edges, out_edge, name);
                }
                GenRecord::SideFace { .. } => {} // handled above
            }
        }

        fill_edges_by_adjacency(&faces, &mut edges, adjacency);

        // Ambiguity is never guessed away: clear any name that landed on two
        // elements of the same kind.
        clear_duplicates(&mut faces);
        clear_duplicates(&mut edges);

        ElementMap { faces, edges }
    }

    /// Compose a modifying op's map (fillet, chamfer, shell) from the single
    /// source body's map and the kernel's lineage: surviving faces/edges keep
    /// their names (splits/merges as in booleans), each face a source element
    /// `generated` is named by `generated_name` of that element's name — e.g.
    /// [`ElementName::Rounded`] of a filleted edge — (several pieces →
    /// centroid-ranked `Split`s of it), and the op's new boundary edges are
    /// named by adjacency. `generated_from` says which element list the
    /// generating indices refer to (edges for blends, faces for a shell).
    #[allow(clippy::too_many_arguments)]
    pub fn compose_modified(
        source: &ElementMap,
        face_lineage: &[Lineage],
        edge_lineage: &[Lineage],
        generated: &[(u32, u32)],
        generated_from: ElemKind,
        generated_name: &dyn Fn(Box<ElementName>) -> ElementName,
        out_faces: usize,
        out_edges: usize,
        adjacency: &[(u32, u32)],
        centroid: &mut dyn FnMut(ElemKind, u32) -> [f64; 3],
    ) -> ElementMap {
        let empty = ElementMap::default();
        let mut faces =
            compose_kind(source, &empty, ElemKind::Face, face_lineage, out_faces, centroid);
        let mut edges =
            compose_kind(source, &empty, ElemKind::Edge, edge_lineage, out_edges, centroid);

        // Generated faces, grouped per source element (one edge can round
        // into several pieces) — the same Split policy as swept side walls.
        let mut generated_of: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(elem, out_face) in generated {
            generated_of.entry(elem).or_default().push(out_face);
        }
        for (elem, mut outs) in generated_of {
            let parent = match generated_from {
                ElemKind::Face => source.face(elem),
                ElemKind::Edge => source.edge(elem),
            };
            let Some(parent) = parent else { continue };
            let name = generated_name(Box::new(parent.clone()));
            outs.sort_unstable();
            outs.dedup();
            let set = |slots: &mut Vec<Option<ElementName>>, i: u32, n: ElementName| {
                if let Some(slot @ None) = slots.get_mut(i as usize) {
                    *slot = Some(n);
                }
            };
            if outs.len() == 1 {
                set(&mut faces, outs[0], name);
            } else {
                let mut ranked = outs;
                ranked.sort_by(|&p, &q| {
                    let (a, b) = (centroid(ElemKind::Face, p), centroid(ElemKind::Face, q));
                    a[0].total_cmp(&b[0])
                        .then(a[1].total_cmp(&b[1]))
                        .then(a[2].total_cmp(&b[2]))
                });
                for (ordinal, &out) in ranked.iter().enumerate() {
                    set(
                        &mut faces,
                        out,
                        ElementName::Split {
                            parent: Box::new(name.clone()),
                            ordinal: ordinal as u32,
                        },
                    );
                }
            }
        }

        fill_edges_by_adjacency(&faces, &mut edges, adjacency);
        clear_duplicates(&mut faces);
        clear_duplicates(&mut edges);
        ElementMap { faces, edges }
    }

    /// Name still-unnamed edges by the (named) faces they separate, then
    /// re-clear duplicates — the gap-filling pass a composite op (push/pull)
    /// runs after boolean composition.
    pub fn name_edges_by_adjacency(&mut self, adjacency: &[(u32, u32)]) {
        fill_edges_by_adjacency(&self.faces, &mut self.edges, adjacency);
        clear_duplicates(&mut self.edges);
    }
}

/// Fill `None` edge slots with the [`ElementName::Intersection`] of the named
/// faces bounding them — only when EVERY bounding face is named (a partial
/// parent list could collide with a different edge's).
fn fill_edges_by_adjacency(
    faces: &[Option<ElementName>],
    edges: &mut [Option<ElementName>],
    adjacency: &[(u32, u32)],
) {
    let mut bounds_of: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(edge, face) in adjacency {
        bounds_of.entry(edge).or_default().push(face);
    }
    for (edge, bounds) in bounds_of {
        let Some(slot) = edges.get_mut(edge as usize) else { continue };
        if slot.is_some() {
            continue;
        }
        let mut parents: Vec<ElementName> = Vec::with_capacity(bounds.len());
        let all_named = bounds.iter().all(|&f| {
            faces
                .get(f as usize)
                .and_then(|n| n.as_ref())
                .map(|n| parents.push(n.clone()))
                .is_some()
        });
        if all_named && !parents.is_empty() {
            parents.sort();
            parents.dedup();
            *slot = Some(ElementName::Intersection { parents });
        }
    }
}

/// Clear every name that appears on more than one element — an ambiguous name
/// must not resolve at all rather than resolve arbitrarily.
fn clear_duplicates(slots: &mut [Option<ElementName>]) {
    let mut seen: HashMap<&ElementName, usize> = HashMap::new();
    for name in slots.iter().flatten() {
        *seen.entry(name).or_default() += 1;
    }
    let dups: Vec<ElementName> = seen
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(name, _)| name.clone())
        .collect();
    if dups.is_empty() {
        return;
    }
    for slot in slots.iter_mut() {
        if slot.as_ref().is_some_and(|n| dups.contains(n)) {
            *slot = None;
        }
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
