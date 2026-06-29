//! 2D sketch primitives: the planar profiles that become solids via extrude.
//!
//! This is the MVP seed of the sketcher. A sketch lives on a [`SketchPlane`] and
//! carries one closed [`Profile`]. Later phases grow this into multi-curve
//! sketches with a constraint graph (line/arc/spline + dimensions); for now a
//! rectangle or circle is enough to drive a real sketch → extrude pipeline.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// The plane a sketch lives on: one of the world base planes, or a custom frame
/// (e.g. a selected face).
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum SketchPlane {
    /// X-Y plane, normal +Z (the default; matches the Z-up world).
    Xy,
    /// X-Z plane, normal -Y.
    Xz,
    /// Y-Z plane, normal +X.
    Yz,
    /// An arbitrary plane: origin plus in-plane x/y axes (from a face).
    Custom {
        origin: DVec3,
        x_dir: DVec3,
        y_dir: DVec3,
    },
}

impl SketchPlane {
    /// A custom plane from a face frame.
    pub fn from_frame(origin: DVec3, x_dir: DVec3, y_dir: DVec3) -> Self {
        SketchPlane::Custom {
            origin,
            x_dir,
            y_dir,
        }
    }

    /// Origin of the sketch frame.
    pub fn origin(self) -> DVec3 {
        match self {
            SketchPlane::Custom { origin, .. } => origin,
            _ => DVec3::ZERO,
        }
    }

    /// In-plane "right" axis (local +x).
    pub fn x_dir(self) -> DVec3 {
        match self {
            SketchPlane::Xy | SketchPlane::Xz => DVec3::X,
            SketchPlane::Yz => DVec3::Y,
            SketchPlane::Custom { x_dir, .. } => x_dir,
        }
    }

    /// In-plane "up" axis (local +y).
    pub fn y_dir(self) -> DVec3 {
        match self {
            SketchPlane::Xy => DVec3::Y,
            SketchPlane::Xz | SketchPlane::Yz => DVec3::Z,
            SketchPlane::Custom { y_dir, .. } => y_dir,
        }
    }

    /// Plane normal (right-handed: `x_dir × y_dir`). Extrude defaults to this.
    pub fn normal(self) -> DVec3 {
        self.x_dir().cross(self.y_dir())
    }

    /// Short label for UI display.
    pub fn label(self) -> &'static str {
        match self {
            SketchPlane::Xy => "XY",
            SketchPlane::Xz => "XZ",
            SketchPlane::Yz => "YZ",
            SketchPlane::Custom { .. } => "Face",
        }
    }
}

/// A closed 2D profile, centered on the sketch origin.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum Profile {
    /// Axis-aligned rectangle, centered on the origin.
    Rectangle { width: f64, height: f64 },
    /// Circle centered on the origin.
    Circle { radius: f64 },
}

impl Profile {
    pub fn type_name(self) -> &'static str {
        match self {
            Profile::Rectangle { .. } => "Rectangle",
            Profile::Circle { .. } => "Circle",
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint-based 2D sketch
//
// The richer sketcher: points, lines, and circles related by geometric and
// dimensional constraints, solved numerically (see the `rmf-solver` crate).
// `Profile` above is the simple shortcut still used by the current extrude
// path; this model is what grows into real CAD sketching.
// ---------------------------------------------------------------------------

/// Index of a point in [`Sketch2d::points`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct PointId(pub usize);

/// Index of a line in [`Sketch2d::lines`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct LineId(pub usize);

/// Index of a circle in [`Sketch2d::circles`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct CircleId(pub usize);

/// A sketch point. `x`/`y` are the current/initial position in sketch-plane
/// coordinates; the solver updates them to satisfy the constraints.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchPoint {
    pub x: f64,
    pub y: f64,
}

/// A straight segment between two points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SketchLine {
    pub a: PointId,
    pub b: PointId,
}

/// A circle defined by a center point and a radius.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchCircle {
    pub center: PointId,
    pub radius: f64,
}

/// A cubic bezier segment between two loop points, with two off-curve control
/// handles `c1` (near `a`) and `c2` (near `b`) in plane coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchBezier {
    pub a: PointId,
    pub b: PointId,
    pub c1: [f64; 2],
    pub c2: [f64; 2],
}

/// One segment of a resolved profile boundary, in plane coordinates — a
/// straight edge or a cubic bezier. Consecutive elements share endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProfileElem {
    Line { a: [f64; 2], b: [f64; 2] },
    Bezier { a: [f64; 2], c1: [f64; 2], c2: [f64; 2], b: [f64; 2] },
}

impl ProfileElem {
    /// The segment's start vertex (shared with the previous element's end).
    pub fn start(self) -> [f64; 2] {
        match self {
            ProfileElem::Line { a, .. } | ProfileElem::Bezier { a, .. } => a,
        }
    }
}

/// A geometric or dimensional constraint relating sketch entities.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Pin a point at its current location.
    Fixed(PointId),
    /// Two points share a position.
    Coincident(PointId, PointId),
    /// A line is horizontal (its endpoints share `y`).
    Horizontal(LineId),
    /// A line is vertical (its endpoints share `x`).
    Vertical(LineId),
    /// The distance between two points equals `value`.
    Distance(PointId, PointId, f64),
    /// Two lines are parallel.
    Parallel(LineId, LineId),
    /// Two lines are perpendicular.
    Perpendicular(LineId, LineId),
    /// Two lines have equal length.
    EqualLength(LineId, LineId),
    /// A circle's radius equals `value`.
    Radius(CircleId, f64),
}

/// A 2D constraint sketch: entities plus the constraints relating them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sketch2d {
    pub points: Vec<SketchPoint>,
    pub lines: Vec<SketchLine>,
    pub circles: Vec<SketchCircle>,
    pub constraints: Vec<Constraint>,
    /// Cubic-bezier boundary segments. `#[serde(default)]` so projects saved
    /// before beziers existed still load.
    #[serde(default)]
    pub beziers: Vec<SketchBezier>,
}

impl Sketch2d {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_point(&mut self, x: f64, y: f64) -> PointId {
        self.points.push(SketchPoint { x, y });
        PointId(self.points.len() - 1)
    }

    pub fn add_line(&mut self, a: PointId, b: PointId) -> LineId {
        self.lines.push(SketchLine { a, b });
        LineId(self.lines.len() - 1)
    }

    pub fn add_circle(&mut self, center: PointId, radius: f64) -> CircleId {
        self.circles.push(SketchCircle { center, radius });
        CircleId(self.circles.len() - 1)
    }

    /// Add a cubic-bezier segment between two points, with control handles.
    pub fn add_bezier(&mut self, a: PointId, b: PointId, c1: [f64; 2], c2: [f64; 2]) {
        self.beziers.push(SketchBezier { a, b, c1, c2 });
    }

    /// Convert a straight `line` into a bezier with default handles at 1/3 and
    /// 2/3 along it (so it starts looking straight, ready to be shaped). Returns
    /// false if the id is invalid.
    pub fn curve_line(&mut self, line: LineId) -> bool {
        let Some(l) = self.lines.get(line.0).copied() else {
            return false;
        };
        let (a, b) = (self.point(l.a), self.point(l.b));
        let c1 = [a.x + (b.x - a.x) / 3.0, a.y + (b.y - a.y) / 3.0];
        let c2 = [a.x + 2.0 * (b.x - a.x) / 3.0, a.y + 2.0 * (b.y - a.y) / 3.0];
        self.lines.remove(line.0);
        self.add_bezier(l.a, l.b, c1, c2);
        true
    }

    /// Keep a shared node smooth after moving bezier `bez`'s handle (`is_c1`
    /// selects which end's handle): any OTHER bezier meeting at that node gets
    /// its adjacent handle set diametrically opposite, with equal length —
    /// collinear through the node, so the curve passes through smoothly (C¹).
    pub fn mirror_partner_handle(&mut self, bez: usize, is_c1: bool) {
        let Some(b) = self.beziers.get(bez).copied() else {
            return;
        };
        let (node, handle) = if is_c1 { (b.a, b.c1) } else { (b.b, b.c2) };
        let n = self.points[node.0];
        let mirrored = [2.0 * n.x - handle[0], 2.0 * n.y - handle[1]];
        for (i, other) in self.beziers.iter_mut().enumerate() {
            if i == bez {
                continue;
            }
            if other.a == node {
                other.c1 = mirrored;
            } else if other.b == node {
                other.c2 = mirrored;
            }
        }
    }

    /// A point on the cubic bezier `segment` at parameter `t ∈ [0,1]`.
    pub fn bezier_at(seg: &SketchBezier, points: &[SketchPoint], t: f64) -> [f64; 2] {
        let a = points[seg.a.0];
        let b = points[seg.b.0];
        let u = 1.0 - t;
        let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
        [
            w[0] * a.x + w[1] * seg.c1[0] + w[2] * seg.c2[0] + w[3] * b.x,
            w[0] * a.y + w[1] * seg.c1[1] + w[2] * seg.c2[1] + w[3] * b.y,
        ]
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub fn point(&self, id: PointId) -> SketchPoint {
        self.points[id.0]
    }

    pub fn line(&self, id: LineId) -> SketchLine {
        self.lines[id.0]
    }

    pub fn circle(&self, id: CircleId) -> SketchCircle {
        self.circles[id.0]
    }

    /// Order the line segments into a single closed loop of point coordinates
    /// `[x, y]` (in plane coordinates), ready to become a face. Returns `None`
    /// unless the lines form exactly one closed loop (each visited once,
    /// returning to the start) — the case the MVP extrude path supports.
    pub fn profile_loop(&self) -> Option<Vec<[f64; 2]>> {
        let n = self.lines.len();
        if n < 3 {
            return None;
        }

        let start = self.lines[0].a;
        let mut current = start;
        let mut visited = vec![false; n];
        let mut loop_points = Vec::with_capacity(n);

        for _ in 0..n {
            let p = self.point(current);
            loop_points.push([p.x, p.y]);

            // Follow an unvisited line incident to the current point.
            let next = self
                .lines
                .iter()
                .enumerate()
                .find(|(i, l)| !visited[*i] && (l.a == current || l.b == current));
            let (line_index, line) = next?;
            visited[line_index] = true;
            current = if line.a == current { line.b } else { line.a };
        }

        // A single closed loop returns to its start after visiting every line.
        (current == start).then_some(loop_points)
    }

    /// The closed profile boundary as ordered segments (straight or bezier),
    /// walking the loop over BOTH lines and beziers. `None` unless they form
    /// exactly one closed loop. This is the canonical profile for the kernel.
    pub fn profile_elements(&self) -> Option<Vec<ProfileElem>> {
        struct Conn {
            a: PointId,
            b: PointId,
            bez: Option<([f64; 2], [f64; 2])>,
        }
        let mut conns: Vec<Conn> = Vec::new();
        for l in &self.lines {
            conns.push(Conn { a: l.a, b: l.b, bez: None });
        }
        for bz in &self.beziers {
            conns.push(Conn { a: bz.a, b: bz.b, bez: Some((bz.c1, bz.c2)) });
        }
        let n = conns.len();
        if n < 3 {
            return None;
        }
        let coord = |p: PointId| {
            let q = self.point(p);
            [q.x, q.y]
        };

        let start = conns[0].a;
        let mut current = start;
        let mut visited = vec![false; n];
        let mut elems = Vec::with_capacity(n);
        for _ in 0..n {
            let (idx, conn) = conns
                .iter()
                .enumerate()
                .find(|(i, c)| !visited[*i] && (c.a == current || c.b == current))?;
            visited[idx] = true;
            let forward = conn.a == current;
            let next = if forward { conn.b } else { conn.a };
            let (a, b) = (coord(current), coord(next));
            match conn.bez {
                None => elems.push(ProfileElem::Line { a, b }),
                // Traversing the bezier b→a swaps its handles.
                Some((c1, c2)) => {
                    let (c1, c2) = if forward { (c1, c2) } else { (c2, c1) };
                    elems.push(ProfileElem::Bezier { a, c1, c2, b });
                }
            }
            current = next;
        }
        (current == start).then_some(elems)
    }

    /// The point ids of the single closed loop in traversal order (`None` unless
    /// the lines form exactly one closed loop). Like [`Self::profile_loop`] but
    /// returns the ids, for editing the loop's topology.
    pub fn loop_order(&self) -> Option<Vec<PointId>> {
        let n = self.lines.len();
        if n < 3 {
            return None;
        }
        let start = self.lines[0].a;
        let mut current = start;
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        for _ in 0..n {
            order.push(current);
            let (line_index, line) = self
                .lines
                .iter()
                .enumerate()
                .find(|(i, l)| !visited[*i] && (l.a == current || l.b == current))?;
            visited[line_index] = true;
            current = if line.a == current { line.b } else { line.a };
        }
        (current == start).then_some(order)
    }

    /// Insert a new point at `(x, y)` partway along `line`, splitting it into
    /// two segments (`a→new`, `new→b`). Point/line ids stay stable (the split
    /// line keeps its id as the first half; the second half is appended), so
    /// existing constraints are preserved. Returns the new point.
    pub fn split_line(&mut self, line: LineId, x: f64, y: f64) -> Option<PointId> {
        let l = *self.lines.get(line.0)?;
        let p = self.add_point(x, y);
        self.lines[line.0] = SketchLine { a: l.a, b: p };
        self.lines.push(SketchLine { a: p, b: l.b });
        Some(p)
    }

    /// Delete `remove`d points from the closed loop, reconnecting the survivors
    /// into a clean loop. Requires a single closed loop and at least 3 points
    /// left (a profile needs a triangle). Rebuilds the geometry, so constraints
    /// and circles are dropped. Returns false (unchanged) if not applicable.
    pub fn delete_points(&mut self, remove: &[PointId]) -> bool {
        let Some(order) = self.loop_order() else {
            return false;
        };
        let kept: Vec<SketchPoint> = order
            .iter()
            .filter(|id| !remove.contains(id))
            .map(|&id| self.point(id))
            .collect();
        if kept.len() < 3 || kept.len() == order.len() {
            return false; // nothing removed, or too few points left
        }
        let n = kept.len();
        self.points = kept;
        self.lines = (0..n)
            .map(|i| SketchLine { a: PointId(i), b: PointId((i + 1) % n) })
            .collect();
        self.circles.clear();
        self.constraints.clear();
        true
    }
}

#[cfg(test)]
mod edit_tests {
    use super::*;

    /// A closed square loop of 4 points.
    fn square() -> Sketch2d {
        let mut s = Sketch2d::new();
        let p: Vec<_> = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
            .iter()
            .map(|&(x, y)| s.add_point(x, y))
            .collect();
        for i in 0..4 {
            s.add_line(p[i], p[(i + 1) % 4]);
        }
        s
    }

    #[test]
    fn split_line_inserts_a_vertex_keeping_the_loop_closed() {
        let mut s = square();
        let n_pts = s.points.len();
        let new = s.split_line(LineId(0), 5.0, 0.0).unwrap();
        assert_eq!(new.0, n_pts, "new point appended");
        assert_eq!(s.points.len(), 5);
        assert_eq!(s.lines.len(), 5, "one line became two");
        // Still a single closed loop, now of 5 points.
        assert_eq!(s.loop_order().unwrap().len(), 5);
    }

    #[test]
    fn delete_points_reconnects_the_loop() {
        let mut s = square();
        let to_remove = s.loop_order().unwrap()[1]; // one corner
        assert!(s.delete_points(&[to_remove]));
        assert_eq!(s.points.len(), 3, "a triangle remains");
        assert_eq!(s.lines.len(), 3);
        assert!(s.loop_order().is_some(), "still one closed loop");
    }

    #[test]
    fn curve_line_replaces_a_segment_with_a_bezier() {
        let mut s = square();
        let n_lines = s.lines.len();
        assert!(s.curve_line(LineId(0)));
        assert_eq!(s.lines.len(), n_lines - 1, "the straight line is gone");
        assert_eq!(s.beziers.len(), 1, "replaced by a bezier");
        // Still one closed loop of four segments, one of them curved.
        let elems = s.profile_elements().unwrap();
        assert_eq!(elems.len(), 4);
        assert_eq!(
            elems.iter().filter(|e| matches!(e, ProfileElem::Bezier { .. })).count(),
            1
        );
    }

    #[test]
    fn mirror_partner_handle_makes_a_smooth_node() {
        // Two beziers meeting at a shared node N=(10,0); drag one's handle and
        // the other's handle at N must go diametrically opposite, equal length.
        let mut s = Sketch2d::new();
        let a = s.add_point(0.0, 0.0);
        let n = s.add_point(10.0, 0.0); // the shared node
        let c = s.add_point(20.0, 0.0);
        s.add_bezier(a, n, [3.0, 0.0], [7.0, 4.0]); // bez 0 ends at n (c2 = its handle at n)
        s.add_bezier(n, c, [13.0, 0.0], [17.0, 0.0]); // bez 1 starts at n (c1 = its handle at n)

        // Move bez 0's end handle (c2) up to (7, 4) [already], mirror onto bez 1.
        s.mirror_partner_handle(0, false); // is_c1=false → node = n, handle = c2 = (7,4)
        // Partner = bez 1's c1 = 2*n - (7,4) = (20-7, 0-4) = (13, -4).
        assert_eq!(s.beziers[1].c1, [13.0, -4.0], "handle mirrored through the node");
    }

    #[test]
    fn profile_elements_walks_lines_and_beziers() {
        let mut s = Sketch2d::new();
        let p: Vec<_> = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
            .iter()
            .map(|&(x, y)| s.add_point(x, y))
            .collect();
        // Bottom edge is a bezier; the other three are straight.
        s.add_bezier(p[0], p[1], [5.0, -5.0], [15.0, -5.0]);
        s.add_line(p[1], p[2]);
        s.add_line(p[2], p[3]);
        s.add_line(p[3], p[0]);

        let elems = s.profile_elements().unwrap();
        assert_eq!(elems.len(), 4);
        let beziers = elems.iter().filter(|e| matches!(e, ProfileElem::Bezier { .. })).count();
        assert_eq!(beziers, 1, "one bezier segment in the loop");
        // The loop is continuous: each element's end is the next's start.
        for i in 0..4 {
            let end = match elems[i] {
                ProfileElem::Line { b, .. } | ProfileElem::Bezier { b, .. } => b,
            };
            assert_eq!(end, elems[(i + 1) % 4].start(), "continuity at {i}");
        }
    }

    #[test]
    fn delete_keeps_at_least_a_triangle() {
        let mut s = square();
        let order = s.loop_order().unwrap();
        // Removing two of four points would leave only two — refused.
        assert!(!s.delete_points(&[order[0], order[1]]));
        assert_eq!(s.points.len(), 4, "unchanged");
    }
}
