//! Lane layout for the "Build Graph" (the history as a git-style graph).
//!
//! Turns the feature DAG into a column ("lane") per body line, like a git commit
//! graph: features are rows in history order, each on a lane; an op that consumes
//! ≥2 inputs is a MERGE where lanes converge, and a body feeding two consumers is
//! a BRANCH where a lane splits. This is pure presentation data — the renderer in
//! `cdt-ui` draws nodes + lane ribbons from it; nothing here touches geometry.

use std::collections::HashMap;

use crate::features::FeatureId;
use crate::history::History;

/// The lane colour palette (sRGB), shared by the Build Graph's ribbons and
/// the viewport's body tinting so the graph is always a legend for the scene:
/// a body with no user colour takes `LANE_PALETTE[lane % LANE_PALETTE.len()]`,
/// and the viewport shows that colour at the user's tint strength.
pub const LANE_PALETTE: [[u8; 3]; 6] = [
    [120, 170, 255], // blue
    [140, 210, 140], // green
    [230, 170, 110], // orange
    [200, 140, 220], // purple
    [220, 200, 120], // yellow
    [150, 210, 210], // teal
];

/// One feature's placement in the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildGraphRow {
    pub id: FeatureId,
    /// The lane (column) the feature's node sits on.
    pub lane: usize,
    /// Lanes of this feature's inputs (for the connectors feeding the node).
    pub input_lanes: Vec<usize>,
    /// Lanes occupied entering this row from above (the previous row's `open`).
    pub top_open: Vec<usize>,
    /// Lanes occupied leaving this row downward (open for consumers below).
    pub open: Vec<usize>,
    /// True if the feature consumes two or more inputs (a merge node).
    pub is_merge: bool,
}

/// The whole laid-out history graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildGraph {
    pub rows: Vec<BuildGraphRow>,
    /// Number of lanes (columns) needed.
    pub lane_count: usize,
}

/// One node of the CONDENSED graph: a whole region of the feature DAG folded
/// into the step everything in it flows to.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplayNode {
    /// The representative — the region's final step (its sink), whose body is
    /// what the node stands for.
    pub rep: FeatureId,
    /// Every folded step, history order; the last one is `rep`. Length 1
    /// means nothing folded.
    pub members: Vec<FeatureId>,
}

/// The condensed graph: nodes plus their lane layout (`graph.rows[i]`
/// describes `nodes[i]`). Because the layout runs on the condensed DAG
/// itself — rather than hiding rows of the full layout — the ribbons are
/// consistent by construction, no matter how history interleaves.
#[derive(Clone, Debug, Default)]
pub struct DisplayGraph {
    pub nodes: Vec<DisplayNode>,
    pub graph: BuildGraph,
}

/// What the lane layout needs to know about one node — a raw feature (the
/// full graph) or a folded [`DisplayNode`] (the condensed graph).
struct LayoutNode {
    id: FeatureId,
    inputs: Vec<FeatureId>,
    /// The input whose lane this node continues (its main body line).
    primary: Option<FeatureId>,
}

impl History {
    /// Lay the feature DAG out into lanes for the graph view.
    pub fn build_graph(&self) -> BuildGraph {
        let nodes: Vec<LayoutNode> = self
            .features()
            .iter()
            .map(|f| LayoutNode {
                id: f.id,
                inputs: f.kind.inputs(),
                primary: f.kind.primary_input(),
            })
            .collect();
        layout(&nodes)
    }

    /// Condense the feature DAG for the compact Build Graph, then lay the
    /// condensed DAG out (same lane algorithm as [`Self::build_graph`]).
    ///
    /// One rule, applied to a fixpoint: **a node folds into its sole
    /// consumer, along a PRIMARY edge only** — the consumer must continue
    /// this node's body line (a boolean belongs to its target's line; the
    /// tool is a guest there). So chains and same-line merges fold, no
    /// matter how history interleaves their steps, but every body that ever
    /// existed keeps at least one node: a tool's story folds to its final
    /// step and stops, feeding the boolean visibly. Also visible: live tips,
    /// branch points (a body feeding several consumers), and pinned steps.
    /// Pins: suppressed steps and anything `keep` names (the selected step,
    /// errored steps, expanded folds) never fold — though they can still
    /// absorb; suppressed steps absorb nothing; and no fold crosses the
    /// rollback boundary at `active`.
    pub fn display_graph(&self, active: usize, keep: impl Fn(FeatureId) -> bool) -> DisplayGraph {
        let features = self.features();
        let n = features.len();
        let index_of: HashMap<FeatureId, usize> =
            features.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
        let mut feature_consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, f) in features.iter().enumerate() {
            for input in f.kind.inputs() {
                if let Some(&pi) = index_of.get(&input) {
                    feature_consumers[pi].push(i);
                }
            }
        }

        // `owner[i]` = the node feature `i` currently belongs to, named by its
        // representative's index. Contract one eligible node per pass until
        // nothing changes; n is small (a model has hundreds of steps, not
        // millions), so the O(n²) simplicity wins.
        let mut owner: Vec<usize> = (0..n).collect();
        // A lane's ORIGIN (an input-less step: primitive, sketch) never folds
        // forward: every body line keeps its birth visible alongside its
        // final step, and a display node never swallows a lane start (which
        // would leave the node with no primary line to continue).
        let may_fold = |r: usize| {
            !features[r].suppressed
                && !keep(features[r].id)
                && !features[r].kind.inputs().is_empty()
        };
        let may_absorb = |s: usize| !features[s].suppressed;
        'contract: loop {
            for r in (0..n).rev() {
                if owner[r] != r || !may_fold(r) {
                    continue;
                }
                // A node's outward edges all leave from its representative
                // (every earlier fold pointed the folded node's edges at the
                // absorber), so the rep's consumers ARE the node's consumers.
                let cs = &feature_consumers[r];
                let Some(&first) = cs.first() else {
                    continue; // a live tip — always visible
                };
                let s = owner[first];
                if cs.iter().any(|&c| owner[c] != s) {
                    continue; // a branch point — feeds several nodes
                }
                // Fold along PRIMARY edges only: every consuming step must
                // continue THIS body line. A boolean's tool never folds into
                // it — the tool body keeps a node of its own.
                if !cs
                    .iter()
                    .all(|&c| features[c].kind.primary_input() == Some(features[r].id))
                {
                    continue;
                }
                if !may_absorb(s) || (r >= active) != (s >= active) {
                    continue;
                }
                for o in owner.iter_mut() {
                    if *o == r {
                        *o = s;
                    }
                }
                continue 'contract;
            }
            break;
        }

        // Assemble nodes (ordered by representative, i.e. history order) and
        // their condensed edges for the layout.
        let reps: Vec<usize> = (0..n).filter(|&i| owner[i] == i).collect();
        let nodes: Vec<DisplayNode> = reps
            .iter()
            .map(|&r| DisplayNode {
                rep: features[r].id,
                members: (0..n)
                    .filter(|&m| owner[m] == r)
                    .map(|m| features[m].id)
                    .collect(),
            })
            .collect();
        let layout_nodes: Vec<LayoutNode> = reps
            .iter()
            .map(|&r| {
                // Inputs: every member edge that leaves the node, deduped, as
                // the owning node's representative.
                let mut inputs: Vec<FeatureId> = Vec::new();
                for m in (0..n).filter(|&m| owner[m] == r) {
                    for input in features[m].kind.inputs() {
                        let Some(&pi) = index_of.get(&input) else { continue };
                        if owner[pi] == r {
                            continue;
                        }
                        let rep = features[owner[pi]].id;
                        if !inputs.contains(&rep) {
                            inputs.push(rep);
                        }
                    }
                }
                // Primary: follow the representative's primary chain until it
                // exits the node — that node's line is the one we continue.
                let mut primary = None;
                let mut cur = r;
                while let Some(p) = features[cur].kind.primary_input() {
                    let Some(&pi) = index_of.get(&p) else { break };
                    if owner[pi] == r {
                        cur = pi;
                        continue;
                    }
                    primary = Some(features[owner[pi]].id);
                    break;
                }
                LayoutNode { id: features[r].id, inputs, primary }
            })
            .collect();

        DisplayGraph { graph: layout(&layout_nodes), nodes }
    }
}

/// The lane layout pass, over raw features or condensed display nodes alike.
fn layout(nodes: &[LayoutNode]) -> BuildGraph {
    {
        // How many nodes consume each node's output. `total` is fixed (a
        // body feeding ONE consumer keeps its lane; feeding several, each consumer
        // branches off); `remaining` counts down so a lane frees after its last
        // consumer.
        let mut total: HashMap<FeatureId, usize> = HashMap::new();
        for f in nodes {
            for inp in &f.inputs {
                *total.entry(*inp).or_default() += 1;
            }
        }
        let mut remaining = total.clone();

        // `lanes[i]` = the feature whose output currently flows down lane `i`,
        // available to consumers below.
        let mut lanes: Vec<Option<FeatureId>> = Vec::new();
        let lane_of = |lanes: &[Option<FeatureId>], id: FeatureId| {
            lanes.iter().position(|l| *l == Some(id))
        };
        let alloc = |lanes: &mut Vec<Option<FeatureId>>| -> usize {
            match lanes.iter().position(|l| l.is_none()) {
                Some(i) => i,
                None => {
                    lanes.push(None);
                    lanes.len() - 1
                }
            }
        };

        let mut rows = Vec::with_capacity(nodes.len());
        let mut lane_count = 0usize;
        let mut prev_open: Vec<usize> = Vec::new();

        for f in nodes {
            let inputs = &f.inputs;
            let is_merge = inputs.len() >= 2;
            let input_lanes: Vec<usize> =
                inputs.iter().filter_map(|inp| lane_of(&lanes, *inp)).collect();

            // The primary input continues this node's lane (its main body
            // line); a node with no primary STARTS a line — it must not annex
            // an input's lane (that input is a guest, e.g. a folded body
            // consuming its tool).
            let primary_lane = f.primary.and_then(|p| lane_of(&lanes, p));

            // This node now uses up one consumer slot of each input.
            for inp in inputs {
                if let Some(r) = remaining.get_mut(inp) {
                    *r = r.saturating_sub(1);
                }
            }

            // Place the node: continue the primary lane if that body is fully
            // consumed, else branch onto a fresh lane (the primary still feeds
            // others below).
            let node_lane = match primary_lane {
                Some(pl) => {
                    let pid = lanes[pl].expect("primary lane occupied");
                    if total.get(&pid).copied().unwrap_or(0) == 1 {
                        pl // sole consumer → continue the body's lane
                    } else {
                        alloc(&mut lanes) // body feeds others too → branch off
                    }
                }
                None => alloc(&mut lanes),
            };

            // Free any secondary input lanes whose body is now fully consumed.
            for &li in &input_lanes {
                if li == node_lane {
                    continue;
                }
                if let Some(lid) = lanes[li] {
                    if remaining.get(&lid).copied().unwrap_or(0) == 0 {
                        lanes[li] = None;
                    }
                }
            }

            lanes[node_lane] = Some(f.id);
            let open: Vec<usize> =
                lanes.iter().enumerate().filter_map(|(i, l)| l.map(|_| i)).collect();
            lane_count = lane_count.max(lanes.len());

            rows.push(BuildGraphRow {
                id: f.id,
                lane: node_lane,
                input_lanes,
                top_open: std::mem::take(&mut prev_open),
                open: open.clone(),
                is_merge,
            });
            prev_open = open;

            // A feature with no consumers is a LIVE body: its lane stays
            // occupied all the way down, so later independent chains can't
            // move into its column. (Freeing it here was the bug that put a
            // new primitive on the same line as a still-visible sweep.)
            while lanes.last() == Some(&None) {
                lanes.pop();
            }
        }

        BuildGraph { rows, lane_count }
    }
}

#[cfg(test)]
mod tests {
    use crate::document::Document;
    use crate::features::{BooleanOp, FeatureKind};
    use crate::DVec3;

    #[test]
    fn a_linear_chain_is_one_lane() {
        let mut doc = Document::new("chain");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m = doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::X });
        doc.add("Fillet", FeatureKind::FilletAll { source: m, radius: 1.0 });

        let g = doc.history.build_graph();
        assert_eq!(g.lane_count, 1, "a straight chain stays on one lane");
        assert!(g.rows.iter().all(|r| r.lane == 0));
    }

    #[test]
    fn a_boolean_is_a_two_lane_merge() {
        let mut doc = Document::new("bool");
        let a = doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let c = doc.add("C", FeatureKind::Cylinder { radius: 2.0, height: 20.0 });
        let u = doc.add(
            "U",
            FeatureKind::Boolean { op: BooleanOp::Union, target: a, tool: c },
        );

        let g = doc.history.build_graph();
        assert_eq!(g.lane_count, 2, "two source bodies → two lanes");
        // The cylinder opens a second lane; the union merges it back in.
        assert_eq!(g.rows[1].lane, 1, "cylinder on lane 1");
        let union_row = g.rows.iter().find(|r| r.id == u).unwrap();
        assert!(union_row.is_merge, "the boolean is a merge node");
        assert_eq!(union_row.input_lanes.len(), 2, "two inputs feed it");
        assert_eq!(union_row.lane, 0, "it continues the target's lane");
    }

    #[test]
    fn a_body_feeding_two_consumers_branches() {
        // One sketch extruded, and the same sketch also referenced by a second
        // feature → the sketch's lane must branch (feed two).
        let mut doc = Document::new("branch");
        let s = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _m1 = doc.add("M1", FeatureKind::Translate { source: s, offset: DVec3::X });
        let _m2 = doc.add("M2", FeatureKind::Translate { source: s, offset: DVec3::Y });

        let g = doc.history.build_graph();
        // The source feeds two: it keeps lane 0 and the consumers branch off
        // onto lanes of their own. Both results are live bodies, so both
        // branch lanes stay open — three lanes total.
        assert_eq!(g.lane_count, 3, "source lane + one lane per live branch");
        assert_eq!(g.rows[0].lane, 0, "the source holds lane 0");
        assert!(g.rows[1].lane != 0 && g.rows[2].lane != 0, "both consumers branch off");
        assert_ne!(g.rows[1].lane, g.rows[2].lane, "each live body has its own lane");
        let _ = (s, _m1, _m2);
    }

    #[test]
    fn a_tool_carving_two_bodies_flows_until_its_last_cut() {
        // One carver A subtracted from B and then C: A's lane must stay open
        // through the first Subtract (the renderer forks it there) and close
        // only after the second — no gap, no floating stub.
        let mut doc = Document::new("fork");
        let a = doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b = doc.add("B", FeatureKind::Box { size: DVec3::splat(10.0) });
        let c = doc.add("C", FeatureKind::Box { size: DVec3::splat(10.0) });
        let s1 = doc.add(
            "S1",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: b, tool: a },
        );
        let s2 = doc.add(
            "S2",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: c, tool: a },
        );

        let g = doc.history.build_graph();
        let row = |id| g.rows.iter().find(|r| r.id == id).unwrap();
        let la = row(a).lane;
        assert!(row(s1).input_lanes.contains(&la), "A feeds the first cut");
        assert!(
            row(s1).open.contains(&la),
            "A's lane still flows below the first cut (the fork's through-branch)"
        );
        assert!(row(s2).input_lanes.contains(&la), "A feeds the second cut");
        assert!(
            !row(s2).open.contains(&la),
            "A's lane terminates at its last consumer"
        );
    }

    #[test]
    fn a_linear_chain_condenses_to_one_node() {
        let mut doc = Document::new("chain");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m = doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::X });
        let f = doc.add("Fillet", FeatureKind::FilletAll { source: m, radius: 1.0 });

        let d = doc.history.display_graph(doc.history.len(), |_| false);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![b, f], "a body shows its birth and its final step");
        assert_eq!(d.nodes[0].members, vec![b], "the origin never folds forward");
        assert_eq!(d.nodes[1].members, vec![m, f]);
        assert_eq!(d.graph.lane_count, 1, "still one body line");
    }

    #[test]
    fn every_body_keeps_a_node_of_its_own() {
        // Folds follow PRIMARY edges only: the target's chain folds through
        // the boolean it owns (interleaving notwithstanding), but the TOOL —
        // a separate body — folds to its own final step and stops there,
        // feeding the merge visibly.
        let mut doc = Document::new("merge");
        let a = doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let t = doc.add("Tool", FeatureKind::Cylinder { radius: 2.0, height: 20.0 });
        let am = doc.add("Move A", FeatureKind::Translate { source: a, offset: DVec3::X });
        let tm = doc.add("Move T", FeatureKind::Translate { source: t, offset: DVec3::Y });
        let u = doc.add(
            "Cut",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: am, tool: tm },
        );
        let f = doc.add("Fillet", FeatureKind::FilletAll { source: u, radius: 1.0 });

        let d = doc.history.display_graph(doc.history.len(), |_| false);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![a, t, tm, f], "each body: its birth + its final step");
        assert_eq!(d.nodes[2].members, vec![tm], "the tool's chain folds to its final");
        assert_eq!(
            d.nodes[3].members,
            vec![am, u, f],
            "the target's chain folds THROUGH its boolean"
        );
        // Two body lines: the main chain continues its origin's lane, and the
        // tool's ribbon connects into it from a lane of its own.
        assert_eq!(d.graph.lane_count, 2);
        let main = &d.graph.rows[3];
        assert_eq!(main.lane, d.graph.rows[0].lane, "main continues its origin's lane");
        assert!(main.input_lanes.contains(&d.graph.rows[2].lane), "the tool connects");
        assert_ne!(main.lane, d.graph.rows[2].lane, "the tool keeps its own lane");
    }

    #[test]
    fn branch_points_and_shared_tools_stay_visible() {
        // One carver subtracted from two bodies: the carver feeds two
        // consumers, so it must remain its own node, feeding both cut nodes.
        let mut doc = Document::new("fork");
        let a = doc.add("Carver", FeatureKind::Box { size: DVec3::splat(10.0) });
        let b = doc.add("B", FeatureKind::Box { size: DVec3::splat(10.0) });
        let c = doc.add("C", FeatureKind::Box { size: DVec3::splat(10.0) });
        let s1 = doc.add(
            "S1",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: b, tool: a },
        );
        let s2 = doc.add(
            "S2",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: c, tool: a },
        );

        let d = doc.history.display_graph(doc.history.len(), |_| false);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![a, b, c, s1, s2], "carver + both targets' births + cuts");
        // Both cut nodes list the carver's node among their inputs — the
        // ribbons connect to a VISIBLE node, never into thin air.
        for row in &d.graph.rows[3..] {
            assert_eq!(row.input_lanes.len(), 2, "each cut connects target + carver");
        }
    }

    #[test]
    fn a_diamond_keeps_the_branch_and_the_second_arm() {
        // s branches into two arms that remerge (the mirror-copy pattern).
        // The union belongs to m1's line, so m1 folds through it; m2 is a
        // body of its own and stays; s feeds two nodes and stays.
        let mut doc = Document::new("diamond");
        let s = doc.add("S", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m1 = doc.add("M1", FeatureKind::Translate { source: s, offset: DVec3::X });
        let m2 = doc.add("M2", FeatureKind::Translate { source: s, offset: DVec3::Y });
        let u = doc.add(
            "U",
            FeatureKind::Boolean { op: BooleanOp::Union, target: m1, tool: m2 },
        );

        let d = doc.history.display_graph(doc.history.len(), |_| false);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![s, m2, u], "origin, second arm, main line");
        assert_eq!(d.nodes[2].members, vec![m1, u], "the owning arm folds through");
        let _ = ();
    }

    #[test]
    fn pins_split_folds_and_boundaries_hold() {
        let mut doc = Document::new("pins");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m1 = doc.add("M1", FeatureKind::Translate { source: b, offset: DVec3::X });
        let m2 = doc.add("M2", FeatureKind::Translate { source: m1, offset: DVec3::Y });
        let f = doc.add("Fillet", FeatureKind::FilletAll { source: m2, radius: 1.0 });
        let len = doc.history.len();

        // keep(m1): it stays a node of its own, splitting the chain.
        let d = doc.history.display_graph(len, |id| id == m1);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![b, m1, f]);
        assert_eq!(d.nodes[2].members, vec![m2, f]);

        // No fold crosses the rollback boundary (active = 2: m2 dims, m1 not).
        let d = doc.history.display_graph(2, |_| false);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![b, m1, f], "the boundary splits the body's story");

        // A suppressed step neither folds nor absorbs — it interrupts the
        // story (upstream steps can't leapfrog it into a later consumer).
        doc.history.set_suppressed(m2, true);
        let d = doc.history.display_graph(len, |_| false);
        let reps: Vec<_> = d.nodes.iter().map(|n| n.rep).collect();
        assert_eq!(reps, vec![b, m1, m2, f], "the suppressed step stands alone");
    }

    #[test]
    fn independent_chains_keep_separate_lanes() {
        // Two unrelated bodies (a swept part, then a new box) must not share a
        // line: the first chain's tip is a live body, so its lane stays
        // occupied to the bottom of the graph.
        let mut doc = Document::new("two-chains");
        let a = doc.add("A", FeatureKind::Box { size: DVec3::splat(10.0) });
        let am = doc.add("Move A", FeatureKind::Translate { source: a, offset: DVec3::X });
        let b = doc.add("B", FeatureKind::Box { size: DVec3::splat(5.0) });
        let bm = doc.add("Move B", FeatureKind::Translate { source: b, offset: DVec3::Y });

        let g = doc.history.build_graph();
        assert_eq!(g.lane_count, 2, "two live chains → two lanes");
        let lane = |id| g.rows.iter().find(|r| r.id == id).unwrap().lane;
        assert_eq!(lane(a), 0);
        assert_eq!(lane(am), 0);
        assert_eq!(lane(b), 1, "the second chain must not reuse the live first lane");
        assert_eq!(lane(bm), 1);
        // The live first chain's lane stays open under the second chain's rows.
        let last = g.rows.last().unwrap();
        assert!(last.open.contains(&0) && last.open.contains(&1));
    }
}
