//! Riemanifold — Phase 0 spike.
//!
//! Proves the full stack the project rests on: drive OpenCASCADE from Rust
//! through our hand-written cxx bridge to build a real B-rep solid, tessellate
//! it, export a printable STL (Milestone A), and display it in a live wgpu
//! viewport with an orbit camera (Milestone B).
//!
//! The model: a 40 mm cube, all edges filleted at 4 mm, with a 12 mm-diameter
//! hole bored straight through it — exactly the "make a solid, round its edges,
//! cut a hole with another shape" gesture from the project outline.

use rmf_kernel::Solid;
use rmf_render::interleave;

const OUT_STL: &str = "out/spike.stl";
const DEFLECTION_MM: f64 = 0.1;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("Riemanifold — Phase 0 spike\n");

    // 1. A 40 mm cube, edges rounded — a clean B-rep fillet on all 12 edges.
    let cube = Solid::cuboid(40.0, 40.0, 40.0)?;
    let rounded = cube.fillet_all_edges(4.0)?;
    println!("  [1] cuboid 40^3, filleted all edges @ 4 mm");

    // 2. A drill: a cylinder long enough to pass fully through, positioned on
    //    the cube's central Z axis and started below the bottom face.
    let drill = Solid::cylinder(6.0, 60.0)?.translate(20.0, 20.0, -10.0)?;
    println!("  [2] cylinder r=6, h=60, centered on the Z axis");

    // 3. Boolean subtract — cut the hole.
    let part = rounded.cut(&drill)?;
    println!("  [3] boolean cut -> rounded cube with a through-hole");

    // 4. Tessellate for display.
    let mesh = part.tessellate(DEFLECTION_MM)?;
    let verts = mesh.positions.len() / 3;
    let tris = mesh.indices.len() / 3;
    println!("  [4] tessellated @ {DEFLECTION_MM} mm -> {verts} vertices, {tris} triangles");

    // 5. Export a printable binary STL.
    std::fs::create_dir_all("out")?;
    part.write_stl(OUT_STL, DEFLECTION_MM)?;
    let bytes = std::fs::metadata(OUT_STL)?.len();
    println!("  [5] wrote {OUT_STL} ({bytes} bytes)");

    // 6. Show it. `--screenshot` renders one framed view to a PNG (headless,
    //    for verification); otherwise open the interactive wgpu viewport.
    let vertices = interleave(&mesh.positions, &mesh.normals);
    let indices = mesh.indices.clone();

    if std::env::args().any(|a| a == "--screenshot") {
        let path = "out/spike.png";
        rmf_render::render_to_png(&vertices, &indices, 1024, 768, path)?;
        let bytes = std::fs::metadata(path)?.len();
        println!("  [6] rendered offscreen view -> {path} ({bytes} bytes)\n");
        println!("Phase 0 spike complete: OCCT geometry rendered through wgpu. ✅");
    } else {
        println!(
            "\nOpening viewport — left-drag orbit, right/middle-drag pan, scroll zoom."
        );
        rmf_render::run(vertices, indices)?;
    }

    Ok(())
}
