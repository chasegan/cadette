//! Riemanifold — Phase 1 spike, now data-driven.
//!
//! The model is no longer built by imperative kernel calls; it's described as a
//! parametric **document** (`rmf-core`) and regenerated against the OCCT
//! **backend** (`rmf-kernel`). This is the Phase 1 spine: features-as-data
//! replayed to produce geometry, which is then tessellated, exported to STL,
//! and shown in the wgpu viewport.
//!
//! The part: a 40 mm cube, all edges filleted at 4 mm, with a 12 mm-diameter
//! hole bored through it — the same shape as the Phase 0 spike, but now every
//! step is an editable, reorderable, serializable history entry.

use anyhow::Context;
use rmf_core::{regenerate, BooleanOp, Document, FeatureKind, DVec3};
use rmf_kernel::KernelBackend;
use rmf_render::interleave;

const OUT_STL: &str = "out/spike.stl";
const DEFLECTION_MM: f64 = 0.1;

/// Build the part as parametric history.
fn build_document() -> Document {
    let mut doc = Document::new("spike-part");

    let cube = doc.add(
        "Box",
        FeatureKind::Box {
            size: DVec3::new(40.0, 40.0, 40.0),
        },
    );
    let rounded = doc.add(
        "Fillet edges",
        FeatureKind::FilletAll {
            source: cube,
            radius: 4.0,
        },
    );
    let drill = doc.add(
        "Drill",
        FeatureKind::Cylinder {
            radius: 6.0,
            height: 60.0,
        },
    );
    let drill = doc.add(
        "Position drill",
        FeatureKind::Translate {
            source: drill,
            offset: DVec3::new(20.0, 20.0, -10.0),
        },
    );
    doc.add(
        "Bore hole",
        FeatureKind::Boolean {
            op: BooleanOp::Subtract,
            target: rounded,
            tool: drill,
        },
    );

    doc
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("Riemanifold — Phase 1 spike (data-driven)\n");

    // 1. Describe the part as a document and validate its history.
    let doc = build_document();
    doc.history
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid history: {e:?}"))?;
    println!(
        "  [1] document \"{}\" — {} features, history valid",
        doc.name,
        doc.history.len()
    );

    // 2. Regenerate: replay the history against the OCCT backend.
    let mut backend = KernelBackend::default();
    let regen = regenerate(&doc, &mut backend);
    if !regen.is_ok() {
        for err in regen.errors() {
            let id = err.feature();
            let name = doc.history.get(id).map(|f| f.name.as_str()).unwrap_or("?");
            eprintln!("    regen error in feature {id:?} ({name}): {err:?}");
        }
        anyhow::bail!("regeneration failed");
    }
    println!(
        "  [2] regenerated -> {} visible body(ies)",
        regen.visible().len()
    );

    // 3. Take the resulting body.
    let body = regen
        .visible_bodies()
        .next()
        .context("regeneration produced no visible body")?;

    // 4. Tessellate for display and export.
    let mesh = body.tessellate(DEFLECTION_MM)?;
    let verts = mesh.positions.len() / 3;
    let tris = mesh.indices.len() / 3;
    println!("  [3] tessellated @ {DEFLECTION_MM} mm -> {verts} vertices, {tris} triangles");

    std::fs::create_dir_all("out")?;
    body.write_stl(OUT_STL, DEFLECTION_MM)?;
    println!("  [4] wrote {OUT_STL} ({} bytes)", std::fs::metadata(OUT_STL)?.len());

    // 5. Show it. `--screenshot` renders one framed view to a PNG (headless);
    //    otherwise open the interactive wgpu viewport.
    let vertices = interleave(&mesh.positions, &mesh.normals);
    let indices = mesh.indices.clone();

    if std::env::args().any(|a| a == "--screenshot") {
        let path = "out/spike.png";
        rmf_render::render_to_png(&vertices, &indices, 1024, 768, path)?;
        println!("  [5] rendered offscreen view -> {path}\n");
        println!("Phase 1 spike: parametric document regenerated through OCCT and rendered. ✅");
    } else {
        println!("\nOpening viewport — left-drag orbit, right/middle-drag pan, scroll zoom.");
        rmf_render::run(vertices, indices)?;
    }

    Ok(())
}
