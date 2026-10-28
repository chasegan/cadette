# Cadette

**Powerful. Friendly. Free. Pick all three.**

Cadette is free, open-source parametric CAD for makers — Tinkercad's simplicity,
Shapr3D's spunk, FreeCAD's kernel, nobody's subscription. Model a part, change
any step later, and export a clean STL for your printer.

![The Cadette modeler: the Build Graph tracing a 66-step parametric history on the left, a robot gauge bracket in the viewport, the properties panel on the right](docs/modeler-light.png)

More at **[cadette.org](https://cadette.org)**.

## Download

- **macOS (Apple silicon):** [cadette-macos-arm64.dmg](https://github.com/chasegan/cadette/releases/latest/download/cadette-macos-arm64.dmg)
  — the app is ad-hoc signed, so the first launch needs **right-click → Open**.
- **Windows / Linux:** coming soon.

## What's inside

- **Direct modeling** — select a face and push/pull it, resize with handles,
  snap to a grid, group bodies and move them together.
- **A real B-rep kernel** — OpenCASCADE, the engine behind FreeCAD: exact
  solids, fillets, booleans, sweeps and revolves. Real geometry, not meshes.
- **Everything parametric** — every action lives in the Build Graph: re-open,
  tweak, re-order or roll back any step and the model rebuilds.
- **A constraint sketcher** — pen and Bézier tools with dimensions and
  constraints, driven by a pure-Rust Levenberg–Marquardt solver.
- **Print-ready output** — STL export, plus `.cdt` project files that keep the
  full parametric history.
- **STEP in and out** — import a vendor part (a servo, a bearing) as exact
  B-rep and fillet, shell or cut it like anything else; export the model as
  STEP for other CAD. The imported file travels inside the `.cdt`.

## Scripting and AI agents

Everything the window can do to a model is available headlessly through the
`cdt` command line (JSON in, JSON out), built on the same crates the app uses:

```sh
cargo run -p cdt-cli -- info part.cdt              # every step's status, every body's exact bbox/volume/area
cargo run -p cdt-cli -- info part.cdt --detail all # + faces and edges with types, normals and anchors
cargo run -p cdt-cli -- check part.cdt             # exit 2 if a step failed to build
echo '[{"op":"add","kind":{"Box":{"size":[40,13,26]}}}]' | cargo run -p cdt-cli -- apply part.cdt
cargo run -p cdt-cli -- render part.cdt -o iso.png --view top --highlight 12
cargo run -p cdt-cli -- export-stl part.cdt -o part.stl
cargo run -p cdt-cli -- import part.cdt servo.step      # a STEP file as a new body
cargo run -p cdt-cli -- export-step part.cdt -o part.step
```

Edits are `DocEdit` batches — the verbs (`add`, `replace`, `suppress`,
`remove`, `insert_transform`, `ungroup`, …) around the same `FeatureKind`
data a `.cdt` file stores — applied all-or-nothing as one undo step.

**MCP.** `cdt mcp` serves the same operations to an AI agent over the Model
Context Protocol (stdio): tools `cadette_info`, `cadette_apply`,
`cadette_render` (returns the image), `cadette_check`, `cadette_export_stl`,
`cadette_import` and `cadette_export_step` (STEP in and out),
`cadette_new`, and — for a running window — `cadette_state`, `cadette_undo`,
`cadette_redo`, `cadette_save`, `cadette_windows`; plus a `cadette://guide`
resource (also the `cadette_help` tool) that teaches the edit and feature JSON.
The repo's `.mcp.json` registers it for Claude Code; elsewhere use
`{"command": "cdt", "args": ["mcp"]}` with a built `cdt` on your PATH.

If a Cadette window has the file open, `info`, `check` and `apply` go through
that window instead: the edit lands as an undoable step and the viewport
updates live (`--offline` forces the file). `cdt live ...` talks to a window
directly, saved file or not — `cdt live state` says what the user has
selected, `cdt live render -o shot.png` shows what they see, `cdt live apply`
edits it. Windows register in `<config dir>/Cadette/instances/`; the link is a
Unix socket (Windows named pipes are not wired yet).

## Build from source

You need Rust (1.85+) and OpenCASCADE 7.7 or newer.

```sh
# macOS
brew install opencascade
cargo run -p cdt-app
```

On other platforms (or a non-Homebrew OCCT), point the build at your install:

```sh
export CDT_OCCT_PREFIX=/path/to/occt-prefix   # contains include/opencascade and lib/
cargo run -p cdt-app
```

On Windows, `vcpkg install opencascade --triplet x64-windows` and use
`%VCPKG_ROOT%\installed\x64-windows` as the prefix.

## Workspace layout

| Crate                | What it owns                                             |
|----------------------|----------------------------------------------------------|
| `crates/core`        | Document model: features, history, regeneration, sketches |
| `crates/kernel`      | OpenCASCADE FFI (cxx) — solids, booleans, tessellation   |
| `crates/solver`      | Levenberg–Marquardt sketch-constraint solver             |
| `crates/project`     | The `.cdt` project file: versioned document envelope + extras |
| `crates/session`     | Headless session: document + kernel + solver + cache + undo, serializable report |
| `crates/cli`         | `cdt`: inspect, edit, check, render and export projects from the shell (JSON out) |
| `crates/render`      | wgpu viewport: scene, picking, gizmo, window shell       |
| `crates/interaction` | Selection model                                          |
| `crates/ui`          | egui chrome: Build Graph panel, command bar, theming     |
| `app`                | The Cadette application                                  |

`cargo test --workspace` runs the whole suite (the kernel tests exercise real
OCCT; the render tests create a real GPU device).

## License

[MPL-2.0](LICENSE). Release bundles include OpenCASCADE and its dependencies,
which are licensed under LGPL-2.1 with the OCCT exception.
