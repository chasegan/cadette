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

/// The condensed view of a [`BuildGraph`]: linear runs of chain steps folded
/// into the run's FINAL step — each body line shows only the version a merge
/// consumes (or the live tip), plus branch points and merges. Folding a
/// chain-interior row is layout-safe: its lane occupancy is identical entering
/// and leaving (`top_open == open`), so remaining rows' ribbons still connect.
#[derive(Clone, Debug, Default)]
pub struct Condensed {
    /// Aligned with `BuildGraph::rows`: true when the view should fold this
    /// step into its consumer's run.
    pub hidden: Vec<bool>,
    /// For each VISIBLE row index, the steps folded into it (oldest first).
    /// Rows that head no run are absent.
    pub runs: HashMap<usize, Vec<FeatureId>>,
}

impl BuildGraph {
    /// Fold linear runs. A step folds when its whole story continues in one
    /// successor — exactly ONE consumer, which takes exactly ONE input and
    /// stays on the SAME lane — and nothing pins it visible: suppressed steps
    /// and steps whose dim state differs from their consumer's (the rollback
    /// boundary at `active` runs between them) never fold, and `keep` lets
    /// the caller pin more (the selected step, errored steps, expanded runs).
    pub fn condensed(
        &self,
        history: &History,
        active: usize,
        keep: impl Fn(FeatureId) -> bool,
    ) -> Condensed {
        let features = history.features();
        debug_assert_eq!(features.len(), self.rows.len());
        let index_of: HashMap<FeatureId, usize> =
            features.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
        let mut consumers: HashMap<FeatureId, Vec<usize>> = HashMap::new();
        for (i, f) in features.iter().enumerate() {
            for input in f.kind.inputs() {
                consumers.entry(input).or_default().push(i);
            }
        }

        let hidden: Vec<bool> = features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if f.suppressed || keep(f.id) {
                    return false;
                }
                let Some(cs) = consumers.get(&f.id) else {
                    return false; // a live tip — always a run's visible end
                };
                let &[ci] = cs.as_slice() else {
                    return false; // a branch point — feeds several consumers
                };
                features[ci].kind.inputs().len() == 1
                    && self.rows[ci].lane == self.rows[i].lane
                    && (i >= active) == (ci >= active)
            })
            .collect();

        // Attribute each folded chain to the visible step that ends it, by
        // walking single inputs upward while they stay folded.
        let mut runs: HashMap<usize, Vec<FeatureId>> = HashMap::new();
        for i in 0..features.len() {
            if hidden[i] {
                continue;
            }
            let mut run = Vec::new();
            let mut j = i;
            loop {
                let inputs = features[j].kind.inputs();
                let &[input] = inputs.as_slice() else { break };
                let Some(&pi) = index_of.get(&input) else { break };
                if !hidden[pi] {
                    break;
                }
                run.push(features[pi].id);
                j = pi;
            }
            if !run.is_empty() {
                run.reverse();
                runs.insert(i, run);
            }
        }

        Condensed { hidden, runs }
    }
}

impl History {
    /// Lay the feature DAG out into lanes for the graph view.
    pub fn build_graph(&self) -> BuildGraph {
        let features = self.features();

        // How many features consume each feature's output. `total` is fixed (a
        // body feeding ONE consumer keeps its lane; feeding several, each consumer
        // branches off); `remaining` counts down so a lane frees after its last
        // consumer.
        let mut total: HashMap<FeatureId, usize> = HashMap::new();
        for f in features {
            for inp in f.kind.inputs() {
                *total.entry(inp).or_default() += 1;
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

        let mut rows = Vec::with_capacity(features.len());
        let mut lane_count = 0usize;
        let mut prev_open: Vec<usize> = Vec::new();

        for f in features {
            let inputs = f.kind.inputs();
            let is_merge = inputs.len() >= 2;
            let input_lanes: Vec<usize> =
                inputs.iter().filter_map(|inp| lane_of(&lanes, *inp)).collect();

            // The primary input continues this feature's lane (its main body line).
            let primary = f.kind.primary_input().or_else(|| inputs.first().copied());
            let primary_lane = primary.and_then(|p| lane_of(&lanes, p));

            // This feature now uses up one consumer slot of each input.
            for inp in &inputs {
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
    fn a_linear_chain_condenses_to_its_final_step() {
        let mut doc = Document::new("chain");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m = doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::X });
        let f = doc.add("Fillet", FeatureKind::FilletAll { source: m, radius: 1.0 });

        let g = doc.history.build_graph();
        let c = g.condensed(&doc.history, doc.history.len(), |_| false);
        assert_eq!(c.hidden, vec![true, true, false], "only the tip stays");
        assert_eq!(c.runs[&2], vec![b, m], "the run folds into the tip, oldest first");
        let _ = f;
    }

    #[test]
    fn merges_and_their_final_inputs_stay_visible() {
        // box → move feeds a boolean as target; the tool is a bare cylinder.
        // `move` is the target body's FINAL version → visible; only `box`
        // folds. The merge and the tool stay.
        let mut doc = Document::new("merge");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m = doc.add("Move", FeatureKind::Translate { source: b, offset: DVec3::X });
        let t = doc.add("Tool", FeatureKind::Cylinder { radius: 2.0, height: 20.0 });
        let u = doc.add(
            "Cut",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: m, tool: t },
        );

        let g = doc.history.build_graph();
        let c = g.condensed(&doc.history, doc.history.len(), |_| false);
        assert_eq!(c.hidden, vec![true, false, false, false]);
        assert_eq!(c.runs[&1], vec![b], "box folds into the pre-merge move");
        let _ = u;
    }

    #[test]
    fn branch_points_stay_visible() {
        let mut doc = Document::new("branch");
        let s = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let _m1 = doc.add("M1", FeatureKind::Translate { source: s, offset: DVec3::X });
        let _m2 = doc.add("M2", FeatureKind::Translate { source: s, offset: DVec3::Y });

        let g = doc.history.build_graph();
        let c = g.condensed(&doc.history, doc.history.len(), |_| false);
        assert_eq!(
            c.hidden,
            vec![false, false, false],
            "a body feeding two consumers never folds"
        );
        assert!(c.runs.is_empty());
    }

    #[test]
    fn keep_splits_a_run_and_suppression_or_rollback_pin_steps() {
        let mut doc = Document::new("pins");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::splat(10.0) });
        let m1 = doc.add("M1", FeatureKind::Translate { source: b, offset: DVec3::X });
        let m2 = doc.add("M2", FeatureKind::Translate { source: m1, offset: DVec3::Y });
        let f = doc.add("Fillet", FeatureKind::FilletAll { source: m2, radius: 1.0 });
        let g = doc.history.build_graph();
        let len = doc.history.len();

        // keep(m1) splits box→m1 and m2→fillet into two runs.
        let c = g.condensed(&doc.history, len, |id| id == m1);
        assert_eq!(c.hidden, vec![true, false, true, false]);
        assert_eq!(c.runs[&1], vec![b]);
        assert_eq!(c.runs[&3], vec![m2]);

        // The rollback boundary between m1 and m2 pins m1 (its consumer dims,
        // it doesn't) — the run may not straddle the boundary invisibly.
        let c = g.condensed(&doc.history, 2, |_| false);
        assert!(!c.hidden[1], "the last active step stays visible");

        // A suppressed step never folds.
        doc.history.set_suppressed(m2, true);
        let c = doc.history.build_graph().condensed(&doc.history, len, |_| false);
        assert!(!c.hidden[2], "suppressed steps stay visible");
        let _ = f;
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
