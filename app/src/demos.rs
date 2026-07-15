//! Headless verification/demo harnesses, each behind a `--flag`.
//!
//! These mirror interactive flows (sketching, constraining, picking, drags)
//! and render one composited frame to `out/*.png` — or print a measurement —
//! so features can be verified without a live window. [`run_if_flagged`] is
//! the single dispatch point `main` consults before starting the event loop.

use cdt_core::{
    regenerate, BooleanOp, Constraint, Document, EdgeAnchor, FeatureKind, Sketch2d, SketchPlane,
    DVec3,
};
use cdt_kernel::KernelBackend;
use cdt_render::{Controller, Pick};

use crate::drags::TransformDrag;
use crate::sketch::SketchTool;
use crate::{Modeler, DEFLECTION_MM};

/// Run the demo matching a known `--flag`, if any. Returns `Ok(None)` when a
/// demo ran (the process is done) and `Ok(Some(modeler))` to continue to the
/// interactive window.
pub fn run_if_flagged(mut modeler: Modeler) -> anyhow::Result<Option<Modeler>> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |f: &str| args.iter().any(|a| a == f);

    // Verification aid: draw a closed triangle and finish it, mirroring the
    // interactive flow, so we can confirm the Extrude button enables afterward.
    // Verification aid: the Build Graph hover ghost — hover the bored-hole
    // cylinder's row and its ghost materializes in the bore, x-ray through
    // the final body, with the fed body tinted.
    if flag("--ghost-demo") {
        let hole = modeler
            .doc
            .history
            .features()
            .iter()
            .find_map(|f| match &f.kind {
                FeatureKind::Boolean { tool, .. } => Some(*tool),
                _ => None,
            })
            .expect("the default part has a boolean");
        let _ = modeler.mesh();
        modeler.ui.hovered_step = Some(hole);
        // Backdate the hover so the debounce has already elapsed.
        modeler.graph_hover =
            Some((hole, std::time::Instant::now() - std::time::Duration::from_secs(1)));
        modeler.update_graph_hover();
        shoot(modeler, "out/ghost-demo.png")?;
        return Ok(None);
    }

    if flag("--sketch-demo") {
        let mut empty = Document::new("draw");
        std::mem::swap(&mut modeler.doc, &mut empty); // start from a clean doc
        modeler.start_sketch(SketchPlane::Xy);
        if let Some(s) = modeler.sketch_session.as_mut() {
            let p0 = s.sketch.add_point(-15.0, -12.0);
            let p1 = s.sketch.add_point(15.0, -12.0);
            let p2 = s.sketch.add_point(0.0, 14.0);
            s.sketch.add_line(p0, p1);
            s.sketch.add_line(p1, p2);
            s.sketch.add_line(p2, p0); // close
            s.start = Some(p0);
            s.last = Some(p2);
        }
        modeler.finish_sketch(); // commits + selects the sketch
        shoot(modeler, "out/sketch-demo.png")?;
        return Ok(None);
    }

    // Verification aid: a skewed quad with Horizontal/Vertical constraints
    // applied should solve to an axis-aligned rectangle.
    if flag("--constrain-demo") {
        // Keep the default part so the camera frames the scene and the sketch
        // overlay (drawn at the part's scale) is visible.
        modeler.start_sketch(SketchPlane::Xy);
        if let Some(s) = modeler.sketch_session.as_mut() {
            let p0 = s.sketch.add_point(-18.0, -15.0);
            let p1 = s.sketch.add_point(16.0, -12.0);
            let p2 = s.sketch.add_point(20.0, 13.0);
            let p3 = s.sketch.add_point(-15.0, 17.0);
            let bottom = s.sketch.add_line(p0, p1);
            let right = s.sketch.add_line(p1, p2);
            let top = s.sketch.add_line(p2, p3);
            let left = s.sketch.add_line(p3, p0);
            s.closed = true;
            s.tool = SketchTool::Select;
            for c in [
                Constraint::Fixed(p0),
                Constraint::Horizontal(bottom),
                Constraint::Vertical(right),
                Constraint::Horizontal(top),
                Constraint::Vertical(left),
            ] {
                s.sketch.add_constraint(c);
            }
            s.selected_lines = vec![bottom, top]; // show selection + enable Equal/Parallel
        }
        modeler.resolve_session();
        shoot(modeler, "out/constrain-demo.png")?;
        return Ok(None);
    }

    // Timing aid: the per-mouse-move cost during a drag, before vs after the
    // incremental-regen cache.
    if flag("--bench-regen") {
        use std::time::Instant;
        let runs = 30;

        // OLD behavior: replay the whole history through OCCT + tessellate, every
        // frame (a fresh backend with no cache).
        let mut fresh = KernelBackend::default();
        let _ = regenerate(&modeler.doc, &mut fresh); // warm OCCT
        let t0 = Instant::now();
        for _ in 0..runs {
            let regen = regenerate(&modeler.doc, &mut fresh);
            for b in regen.visible_bodies() {
                let _ = b.tessellate(DEFLECTION_MM);
            }
        }
        let uncached = t0.elapsed().as_secs_f64() * 1000.0 / runs as f64;

        // NEW behavior: simulate a drag — append a Move at the tip and change its
        // offset each frame. Only that op rebuilds; the cache reuses everything
        // upstream. mesh() also re-tessellates the changed body.
        let src = modeler.doc.history.features().last().unwrap().id;
        let drag = modeler.doc.add(
            "Drag",
            FeatureKind::Translate { source: src, offset: DVec3::ZERO },
        );
        let _ = modeler.mesh(); // warm the cache
        let t0 = Instant::now();
        for i in 0..runs {
            if let Some(FeatureKind::Translate { offset, .. }) =
                modeler.doc.history.get_mut(drag).map(|f| &mut f.kind)
            {
                offset.x = i as f64 * 0.1;
            }
            let _ = modeler.mesh();
        }
        let cached = t0.elapsed().as_secs_f64() * 1000.0 / runs as f64;

        println!(
            "{} features — per-frame drag cost: uncached {:.1}ms, cached {:.1}ms ({:.1}x faster)",
            modeler.doc.history.len(),
            uncached,
            cached,
            uncached / cached,
        );
        return Ok(None);
    }

    // Verification aid: save then reload the default project, no dialogs.
    if flag("--project-test") {
        std::fs::create_dir_all("out")?;
        let path = "out/project-test.cdt";
        let before = modeler.doc.history.len();
        if let Err(e) = modeler.save_project_to(path) {
            println!("save failed: {e}");
            return Ok(None);
        }
        // Wipe the in-memory doc, then load it back from disk.
        modeler.doc = Document::new("scratch");
        match modeler.load_project_from(path) {
            Ok(()) => {
                let after = modeler.doc.history.len();
                let mesh = modeler.mesh();
                println!(
                    "round-trip ok: {before} -> {after} features, name {:?}, {} verts, errors {:?}",
                    modeler.doc.name,
                    mesh.vertices.len(),
                    modeler.ui.errors,
                );
            }
            Err(e) => println!("load failed: {e}"),
        }
        return Ok(None);
    }

    // Verification aid: export the default model to STL without the dialog.
    if flag("--export-test") {
        let _ = modeler.mesh(); // populate visible bodies
        std::fs::create_dir_all("out")?;
        let path = "out/export-test.stl";
        match modeler.export_stl_to(path) {
            Ok(()) => {
                let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                println!("wrote {path} ({bytes} bytes)");
            }
            Err(e) => println!("export failed: {e}"),
        }
        return Ok(None);
    }

    // Verification aid: highlight a face by id to confirm the shader tint path.
    if flag("--highlight-demo") {
        modeler.selection.select(Some(Pick::Face(0)), false, None); // strong face
        modeler.selection.hover(Some(Pick::Edge(0))); // hovered edge
        shoot(modeler, "out/highlight-demo.png")?;
        return Ok(None);
    }

    // Verification aid: an in-progress pen sketch — a smooth closed loop of four
    // cubic beziers (a circle-ish blob), opened for editing.
    if flag("--pen-demo") {
        let mut s = Sketch2d::new();
        let anchors = [(20.0, 0.0), (0.0, 20.0), (-20.0, 0.0), (0.0, -20.0)];
        let ids: Vec<_> = anchors.iter().map(|&(x, y)| s.add_point(x, y)).collect();
        // Tangent handles (~0.55 r) make each node smooth; out/in are mirrored.
        let h = 11.0;
        let handles = [
            ([20.0, h], [h, 20.0]),
            ([-h, 20.0], [-20.0, h]),
            ([-20.0, -h], [-h, -20.0]),
            ([h, -20.0], [20.0, -h]),
        ];
        for i in 0..4 {
            let (c1, c2) = handles[i];
            s.add_bezier(ids[i], ids[(i + 1) % 4], c1, c2);
        }
        let mut doc = Document::new("pen");
        let sk = doc.add("Sketch", FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: s });
        std::mem::swap(&mut modeler.doc, &mut doc);
        let _ = modeler.mesh();
        modeler.edit_sketch(sk);
        shoot(modeler, "out/pen-demo.png")?;
        return Ok(None);
    }

    if flag("--bezier-demo") {
        // A square sketch whose bottom edge is a bezier, opened for editing so
        // the curve + control handles show in the sketch canvas.
        let mut s = Sketch2d::new();
        let p: Vec<_> = [(-20.0, -15.0), (20.0, -15.0), (20.0, 15.0), (-20.0, 15.0)]
            .iter()
            .map(|&(x, y)| s.add_point(x, y))
            .collect();
        s.add_bezier(p[0], p[1], [-10.0, -32.0], [10.0, -32.0]);
        s.add_line(p[1], p[2]);
        s.add_line(p[2], p[3]);
        s.add_line(p[3], p[0]);
        let mut doc = Document::new("bezier");
        let sk = doc.add("Sketch", FeatureKind::ConstraintSketch { plane: SketchPlane::Xy, sketch: s });
        std::mem::swap(&mut modeler.doc, &mut doc);
        let _ = modeler.mesh();
        modeler.edit_sketch(sk);
        shoot(modeler, "out/bezier-demo.png")?;
        return Ok(None);
    }

    if flag("--multiface-demo") {
        // Select several faces of one box; all should highlight (multi-select).
        let mut doc = Document::new("multiface");
        doc.add("Box", FeatureKind::Box { size: DVec3::new(24.0, 18.0, 14.0) });
        std::mem::swap(&mut modeler.doc, &mut doc);
        let _ = modeler.mesh();
        modeler.on_pick(Some(Pick::Face(0)), None, false);
        modeler.on_pick(Some(Pick::Face(2)), None, true); // additive
        modeler.on_pick(Some(Pick::Face(4)), None, true);
        shoot(modeler, "out/multiface-demo.png")?;
        return Ok(None);
    }

    if flag("--group-demo") {
        // Two separated boxes grouped, then the group rotated about its combined
        // center — they turn together as one unit.
        let mut doc = Document::new("group");
        let a = doc.add("A", FeatureKind::Box { size: DVec3::new(20.0, 12.0, 12.0) });
        let b0 = doc.add("B0", FeatureKind::Box { size: DVec3::new(20.0, 12.0, 12.0) });
        let b = doc.add(
            "B",
            FeatureKind::Translate { source: b0, offset: DVec3::new(30.0, 0.0, 0.0) },
        );
        let g = doc.add("Group", FeatureKind::Group { members: vec![a, b] });
        doc.add(
            "Turn",
            FeatureKind::Rotate {
                source: g,
                axis: DVec3::Z,
                angle: 30.0_f64.to_radians(),
                center: DVec3::new(25.0, 6.0, 6.0), // the combined bbox center
            },
        );
        std::mem::swap(&mut modeler.doc, &mut doc);
        let _ = modeler.mesh();
        shoot(modeler, "out/group-demo.png")?;
        return Ok(None);
    }

    if flag("--mirror-demo") {
        // An asymmetric bored block, mirror-copied across its +X face so the
        // reflected hole reads clearly.
        let mut doc = Document::new("mirror");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::new(20.0, 14.0, 12.0) });
        let pin = doc.add("Pin", FeatureKind::Cylinder { radius: 3.0, height: 40.0 });
        let pin = doc.add(
            "Position",
            FeatureKind::Translate { source: pin, offset: DVec3::new(6.0, 7.0, -10.0) },
        );
        let body = doc.add(
            "Bore",
            FeatureKind::Boolean { op: BooleanOp::Subtract, target: b, tool: pin },
        );
        let clone = doc.duplicate(body).expect("duplicate");
        doc.add(
            "Mirror",
            FeatureKind::Mirror {
                source: clone,
                origin: DVec3::new(20.0, 0.0, 0.0),
                normal: DVec3::X,
            },
        );
        std::mem::swap(&mut modeler.doc, &mut doc);
        let _ = modeler.mesh();
        shoot(modeler, "out/mirror-demo.png")?;
        return Ok(None);
    }

    if flag("--revolve-demo") {
        // An L-shaped profile on XZ, its left edge on the Z axis, revolved 270°
        // so the swept cross-section reads clearly as a solid of revolution.
        let mut s = Sketch2d::new();
        let pts = [
            (0.0, 0.0),
            (16.0, 0.0),
            (16.0, 6.0),
            (6.0, 6.0),
            (6.0, 30.0),
            (0.0, 30.0),
        ];
        let ids: Vec<_> = pts.iter().map(|&(x, y)| s.add_point(x, y)).collect();
        for i in 0..ids.len() {
            s.add_line(ids[i], ids[(i + 1) % ids.len()]);
        }
        let mut doc = Document::new("revolve");
        let sk = doc.add(
            "Profile",
            FeatureKind::ConstraintSketch { plane: SketchPlane::Xz, sketch: s },
        );
        doc.add(
            "Revolve",
            FeatureKind::Revolve {
                source: sk,
                axis: EdgeAnchor { point: DVec3::new(0.0, 0.0, 15.0) },
                angle: 270.0_f64.to_radians(),
            },
        );
        std::mem::swap(&mut modeler.doc, &mut doc);
        let _ = modeler.mesh();
        shoot(modeler, "out/revolve-demo.png")?;
        return Ok(None);
    }

    if flag("--planes-demo") {
        // All three origin planes on, as construction guides.
        modeler.ui.show_xz = true;
        modeler.ui.show_yz = true;
        let _ = modeler.mesh();
        shoot(modeler, "out/planes-demo.png")?;
        return Ok(None);
    }

    if flag("--resize-demo") {
        // The default (non-primitive) part, one face selected, so the bbox resize
        // grips show on a filleted/booleaned body alongside the move/rotate gizmo.
        let _ = modeler.mesh();
        modeler.on_pick(Some(Pick::Face(0)), None, false);
        shoot(modeler, "out/resize-demo.png")?;
        return Ok(None);
    }

    // Verification aid: an in-progress rotation, so the gizmo readout (an egui
    // overlay only shown mid-drag) renders headlessly. Uses a negative angle to
    // exercise the worst-case width.
    if flag("--gizmo-readout-demo") {
        let _ = modeler.mesh(); // populate visible + body_centers
        modeler.on_pick(Some(Pick::Face(0)), None, false); // select a face → gizmo
        if let (Some(source), Some(pivot)) = (modeler.selected_body(), modeler.gizmo_origin()) {
            let id = modeler.doc.add(
                "Rotate",
                FeatureKind::Rotate {
                    source,
                    axis: DVec3::Z,
                    angle: (-135.0f64).to_radians(),
                    center: DVec3::from_array(pivot),
                },
            );
            modeler.transform = Some(TransformDrag {
                source,
                feature: Some(id),
                pivot: Some(pivot),
                base_offset: DVec3::ZERO,
                orig_plane: None,
                started: true,
            });
        }
        shoot(modeler, "out/gizmo-readout-demo.png")?;
        return Ok(None);
    }

    if flag("--screenshot") {
        // An optional .cdt path after the flag loads that project first —
        // for shots of a real model (e.g. the home-page hero). Default: the
        // built-in part with its first sketch selected to show the editor.
        let project = args
            .iter()
            .position(|a| a == "--screenshot")
            .and_then(|i| args.get(i + 1))
            .filter(|a| !a.starts_with("--"));
        match project {
            Some(path) => {
                modeler
                    .load_project_from(path)
                    .map_err(|e| anyhow::anyhow!("load {path}: {e}"))?;
                // Select the tip so the Properties panel is populated.
                modeler.ui.selected = modeler.doc.history.features().last().map(|f| f.id);
                // Frame the whole model with extra margin: the automatic
                // screenshot framing fits the full window, but the side
                // panels cover much of its width and crop a wide model.
                let _ = modeler.mesh();
                let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
                for (bmin, bmax) in &modeler.body_bounds {
                    for k in 0..3 {
                        min[k] = min[k].min(bmin[k]);
                        max[k] = max[k].max(bmax[k]);
                    }
                }
                if min[0].is_finite() {
                    let radius = (0..3)
                        .map(|k| (max[k] - min[k]) * 0.5)
                        .map(|h| h * h)
                        .sum::<f64>()
                        .sqrt()
                        .max(1e-3);
                    modeler.camera_request = Some(cdt_render::CameraPose {
                        target: [
                            ((min[0] + max[0]) * 0.5) as f32,
                            ((min[1] + max[1]) * 0.5) as f32,
                            ((min[2] + max[2]) * 0.5) as f32,
                        ],
                        yaw: -0.9,
                        pitch: 0.6,
                        // Tuned for the 1920-wide shot: the fixed-width side
                        // panels leave proportionally more viewport than the
                        // old 1280 window, so frame the part closer.
                        distance: (radius * 3.4) as f32,
                    });
                }
            }
            None => {
                modeler.ui.selected = modeler.doc.history.features().first().map(|f| f.id);
            }
        }
        shoot(modeler, "out/modeler.png")?;
        return Ok(None);
    }

    Ok(Some(modeler))
}

/// Render one composited frame (panel + viewport) to `path` — the shared tail
/// of the screenshot demos.
fn shoot(modeler: Modeler, path: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all("out")?;
    cdt_render::screenshot(modeler, 1920, 1230, path)?;
    println!("wrote {path}");
    Ok(())
}
