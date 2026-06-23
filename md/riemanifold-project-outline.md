# Project Outline — Cross-Platform Parametric Modeler for 3D Printing

*Working title: TBD. A Shapr3D-spirited, direct-manipulation solid modeler built in Rust.*

---

## 1. Vision

A native, cross-platform desktop application for designing printable 3D models, prioritizing **fluid direct manipulation** over menu-driven CAD. The feel should be: sketch a profile, grab a face, push it into a solid, cut a hole with another shape, fillet an edge — all with minimal chrome and immediate visual feedback. Industrial-grade geometry under a deliberately light interface.

The differentiator is not raw feature count but **interaction quality**: the right tool surfaces when you select the right thing, the viewport dominates the screen, and the history tree lets you reach back and adjust any step without fear.

---

## 2. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | **Rust** | Native cross-platform binaries, strict typing, clean C/C++ FFI, no VM/runtime to ship |
| Geometry kernel | **OpenCASCADE (OCCT)** via `opencascade-rs` / custom FFI | Industrial B-rep solid modeling: booleans, fillets, NURBS, STEP/STL/3MF |
| Rendering | **wgpu** | Portable GPU layer (Vulkan/Metal/DX12), responsive viewport, WASM/WebGPU path later |
| UI framework | **egui** (MVP) → evaluate **Slint** for polish | Immediate-mode UI ideal for tool-heavy, contextual interfaces; both fully typed |
| Constraint solver | Wrap **planegcs** (FreeCAD, C++) via FFI, or hand-rolled for MVP | Sketch constraints (parallel, tangent, equal, coincident) |
| Mesh ops (fast path) | **Manifold** (optional) | Fast, robust watertight mesh booleans for the print-export pipeline |
| Serialization | `serde` + custom document format | History tree, project files, undo state |

**Cross-platform reach:** native Windows/macOS/Linux from one codebase; potential browser deployment later via WASM + WebGPU with no JavaScript authored by hand.

---

## 3. Architecture (Module Structure)

```
app/
├── core/                  # Pure Rust domain logic, no UI, no GPU
│   ├── document/          # Document model, project file, save/load
│   ├── history/           # Feature tree, replay engine, rollback
│   ├── sketch/            # 2D sketch model + constraint graph
│   ├── features/          # Extrude, revolve, fillet, boolean, etc. as data
│   └── units/             # Unit system, dimension values, expressions
├── kernel/                # OCCT FFI boundary — the only place that touches C++
│   ├── ffi/               # Raw bindings
│   ├── solids/            # Safe Rust wrappers: make box, boolean, fillet...
│   ├── tessellate/        # B-rep → mesh for display + export
│   └── io/                # STEP / STL / 3MF / OBJ import & export
├── solver/                # Constraint solver FFI wrapper (planegcs)
├── render/                # wgpu viewport, camera, picking, gizmos, grid
│   ├── scene/             # Render graph, mesh buffers, materials
│   ├── picking/           # Ray-cast selection of faces/edges/vertices
│   └── overlays/          # Dimensions, snap hints, measurement HUD
├── interaction/           # The UX brain: tool state machine, snapping, gestures
│   ├── tools/             # Each tool (sketch, push-pull, fillet...) as a state
│   ├── selection/         # Selection model + filters (face/edge/vertex/body)
│   └── snapping/          # Inference engine: midpoints, axes, faces, grid
├── ui/                    # egui panels, contextual toolbars, history panel
└── main.rs                # Wiring, event loop, app shell
```

**Key boundary discipline:** `core/` is pure and testable with no kernel dependency — features are described as *data* (e.g. `Extrude { profile, distance, direction }`). The `kernel/` layer is the **only** module that crosses into C++. The `history/` engine replays the feature list against the kernel to regenerate geometry, which is what makes editable, adjustable steps possible.

---

## 4. The Interaction Model (the heart of it)

The whole app is a **tool state machine** driven by **selection context**. The pattern that makes Shapr3D feel good:

1. **Selection drives available tools.** Select a planar face → push/pull, sketch-on-face, and offset become primary. Select an edge → fillet/chamfer surface. Select a body → move, scale, boolean.
2. **Contextual toolbar** appears near the cursor/selection with only the relevant actions; everything else stays hidden.
3. **Direct manipulation first.** Whenever possible, an operation is a drag with live preview and a typed numeric override (drag to ~12mm, or type `12` to lock it exactly).
4. **Inference & snapping** constantly suggest meaningful targets (midpoints, centers, tangents, coplanar faces, axis alignment) with visual hints, so precision comes for free.
5. **Everything is reversible and editable** via the history tree.

---

## 5. Feature & Interaction Catalog

This is the elaborate list you asked for, grouped by domain. Items are tagged **[MVP]**, **[v1]**, or **[later]** to suggest sequencing.

### 5.1 Primitives & Shape Creation
- **[MVP]** Box, cylinder, sphere, cone, torus, wedge — created by **drag-on-plane then drag-height**, with live dimensions you can type to override.
- **[MVP]** Place primitive by clicking a base point; numeric entry for exact dimensions at creation.
- **[v1]** Parametric primitives that remain editable in history (change a cylinder's radius after the fact).
- **[v1]** Construction geometry: reference planes, axes, points (offset plane from a face, plane through 3 points, mid-plane).
- **[later]** Pattern primitives on creation (e.g. an array of holes as one feature).

### 5.2 Sketching (2D → 3D foundation)
- **[MVP]** Sketch on any plane or planar face; auto-orient camera to the sketch.
- **[MVP]** Line, rectangle, circle, arc, polyline, ellipse, polygon.
- **[MVP]** Snapping/inference: endpoint, midpoint, center, intersection, tangent, perpendicular, horizontal/vertical, grid.
- **[MVP]** Dimensional constraints (length, radius, angle) — type a value, geometry updates.
- **[v1]** Geometric constraints: coincident, parallel, perpendicular, equal, tangent, symmetric, concentric, collinear, fix.
- **[v1]** Constraint solver feedback — under/fully/over-constrained shown by color; drag under-constrained geometry to explore.
- **[v1]** Splines (Bézier / NURBS) with editable control points.
- **[v1]** Trim, extend, offset, fillet/chamfer **in 2D**, mirror, project edges from existing 3D geometry onto the sketch.
- **[later]** Text-to-sketch (outlines from fonts) for embossing/engraving.
- **[later]** Image-as-reference underlay for tracing.

### 5.3 Solid Creation from Profiles
- **[MVP]** **Extrude** — push/pull a profile; symmetric, one-side, two-side; to-distance or up-to-face/up-to-body.
- **[MVP]** **Extrude-cut** — same gesture, subtract instead of add (detected automatically by direction/context).
- **[v1]** **Revolve** around an axis (full or partial angle).
- **[v1]** **Loft** between two or more profiles on different planes.
- **[v1]** **Sweep** a profile along a path.
- **[later]** **Helix/coil** (threads, springs) with pitch/turns/taper.

### 5.4 Direct Solid Editing (Push/Pull family — the signature interaction)
- **[MVP]** **Push/Pull a face** — drag any planar face to add or remove material; live preview, typed override.
- **[v1]** **Offset face** — move a face while maintaining adjacent geometry.
- **[v1]** **Move/rotate face** — reshape a solid by manipulating one face.
- **[v1]** **Press-pull on edges** — drag an edge to reshape.
- **[v1]** **Delete face** — remove and heal (cap or extend neighbors).
- **[later]** **Replace face** — swap a face's surface with another.

### 5.5 Boolean & Combination Operations
- **[MVP]** **Union** (merge bodies).
- **[MVP]** **Subtract** — use one shape to cut a hole/pocket in another (your "shape to create a hole").
- **[MVP]** **Intersect** — keep only overlapping volume.
- **[v1]** **Keep tools** option (non-destructive booleans that preserve the cutting body).
- **[v1]** **Split body** with a plane or surface.
- **[v1]** **Combine with preview** — scrub between union/subtract/intersect before committing.

### 5.6 Edge & Face Treatments
- **[MVP]** **Fillet** — select one or many edges, drag radius with live preview, type exact value.
- **[MVP]** **Chamfer** — distance or distance-angle.
- **[v1]** **Variable-radius fillet** (different radius at each end).
- **[v1]** **Full-round fillet** across three faces.
- **[v1]** **Shell** — hollow a body to a wall thickness, optionally removing chosen faces (great for printable enclosures).
- **[v1]** **Draft** — angle faces for moldability/printability.
- **[later]** **Rib / boss** helpers for functional parts.

### 5.7 Surface/Face-Targeted Interactions (answering your question directly)
Yes — faces and surfaces are first-class interaction targets, not just edges. Beyond push/pull and offset above:
- **[v1]** **Sketch on face** — start a new sketch directly on a selected face.
- **[v1]** **Measure** between faces (distance, angle, area).
- **[v1]** **Thicken a surface** into a solid.
- **[v1]** **Patch / fill** an open region with a surface.
- **[later]** **Surface trim/extend/stitch** for advanced shaping.
- **[later]** **Project to face** / wrap a sketch onto a curved face (for embossed text on a cylinder, etc.).

### 5.8 Transforms & Duplication
- **[MVP]** Move/rotate with an on-screen **gizmo** (axis handles), plus numeric entry.
- **[MVP]** Scale (uniform / non-uniform).
- **[v1]** **Align** — snap a face/edge/point of one body to another.
- **[v1]** **Linear pattern**, **circular pattern**, **mirror** — all editable in history.
- **[v1]** **Pattern along a path/sketch.**
- **[later]** Pattern driven by a table/expression (e.g. count tied to a parameter).

### 5.9 History, Parametrics & Editing
- **[MVP]** **Feature/history tree** listing every operation in order.
- **[MVP]** **Edit any step** — reopen a feature, change its parameters, regenerate downstream.
- **[MVP]** **Rollback bar** — drag to a point in history to see/insert at an earlier state ("adjustable steps").
- **[v1]** **Suppress/enable** a feature without deleting it.
- **[v1]** **Reorder** features (with dependency validation).
- **[v1]** **Named parameters & expressions** — define `wall = 3mm`, reference it across features; change once, propagate everywhere.
- **[v1]** **Robust references** — tolerate topology changes so edits don't orphan downstream features (the hard, important reliability problem).
- **[later]** **Configurations/variants** — multiple parameter sets of one model.

### 5.10 Selection & Navigation
- **[MVP]** Click to select; ray-cast picking of vertex/edge/face/body.
- **[MVP]** **Selection filters** (toggle what's pickable).
- **[MVP]** Orbit/pan/zoom; **view cube** or named views (front/top/iso); zoom-to-fit and zoom-to-selection.
- **[v1]** **Box / lasso** select; window vs. crossing semantics.
- **[v1]** **Smart selection** — select tangent-connected edges, all edges of a face, all faces of a feature, loops.
- **[v1]** **Hover highlighting** with pre-selection preview.
- **[v1]** **Hide/isolate/show** bodies; section/clipping plane to see inside.

### 5.11 Measurement, Analysis & Print-Readiness
- **[MVP]** Measure distance/length/radius/angle.
- **[v1]** Mass/volume/bounding-box/center-of-mass readout.
- **[v1]** **Wall-thickness analysis** (flag too-thin regions for printing).
- **[v1]** **Watertight/manifold check** before export.
- **[v1]** **Overhang highlight** relative to a chosen print orientation.
- **[later]** Draft analysis, interference/clearance checks between bodies.

### 5.12 Import / Export
- **[MVP]** **Import STL** (display + use as reference or convert to body).
- **[MVP]** **Export STL** with **adjustable mesh resolution** (deflection/chord tolerance controls surface smoothness).
- **[v1]** **Export 3MF** (modern print format, units + color).
- **[v1]** **STEP import/export** (precise B-rep interchange with other CAD).
- **[v1]** OBJ import/export.
- **[v1]** Mesh repair on import (fix non-manifold/flipped normals) and **mesh-to-BRep** conversion.
- **[later]** Direct "send to slicer" handoff.

### 5.13 UX / App-Level Qualities
- **[MVP]** **Minimal chrome** — viewport-dominant layout, contextual toolbar near selection.
- **[MVP]** **Live preview** for every operation before commit.
- **[MVP]** **Typed numeric override** during any drag (drag approximate, type exact).
- **[MVP]** **Undo/redo** at every step.
- **[v1]** **Inference HUD** — visual hints for snaps/alignments as you move.
- **[v1]** **Keyboard-driven** tool invocation + customizable shortcuts.
- **[v1]** **Adaptive units** (mm/cm/inch) and grid.
- **[v1]** **Autosave & crash recovery.**
- **[later]** Dark/light theming; onboarding tooltips; in-app tutorials.
- **[later]** Cross-device project sync (the genuinely expensive, post-product-market-fit feature).

---

## 6. Suggested Build Sequence

**Phase 0 — Spike (prove the risky boundary).**
Get OCCT callable from Rust: make a box, boolean two boxes, fillet an edge, tessellate, render in a wgpu window, export STL. This de-risks the entire project before any UI investment.

**Phase 1 — MVP modeler.**
Sketch (line/rect/circle + dimensions) → extrude/cut → primitives → booleans → fillet/chamfer → push/pull faces → linear history with edit + rollback → STL import/export with resolution control → minimal contextual UI.

**Phase 2 — Parametric depth.**
Full constraint solver, named parameters/expressions, robust references, revolve/loft/sweep, shell/draft, patterns/mirror, STEP/3MF, measurement & analysis.

**Phase 3 — Polish & reach.**
Slint UI evaluation, print-readiness analysis suite, advanced surface tools, theming/onboarding, WASM/WebGPU browser build, (eventually) sync.

---

## 7. Hard Problems to Respect Early

- **Robust references** across topology changes — the difference between a toy and a real parametric modeler.
- **The OCCT FFI boundary** — memory ownership, error handling across the C++ line; keep it small and well-tested.
- **Constraint solver integration** — numerically finicky; budget real time for it.
- **Tessellation quality vs. performance** — adaptive meshing so the viewport stays fluid on large models.
- **History replay performance** — caching/incremental regeneration so edits to early steps don't stall.

---

*This is a living outline. The tagged phases are a suggestion, not a contract — the Phase 0 spike is the one piece I'd insist on doing first, because it tells you whether the whole stack hangs together before you commit to it.*
