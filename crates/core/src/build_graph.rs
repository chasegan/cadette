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

            // A leaf (no consumers, i.e. a visible body) ends its lane here.
            if remaining.get(&f.id).copied().unwrap_or(0) == 0 {
                lanes[node_lane] = None;
            }
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
        // The source feeds two: it keeps lane 0 and the consumers branch off it.
        assert_eq!(g.lane_count, 2, "a branch opens a second lane");
        assert_eq!(g.rows[0].lane, 0, "the source holds lane 0");
        assert!(g.rows[1].lane != 0 && g.rows[2].lane != 0, "both consumers branch off");
        let _ = (s, _m1, _m2);
    }
}
