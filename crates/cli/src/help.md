# Cadette for agents

Cadette is parametric CAD. A project (`.cdt`) is a **history** of features;
regenerating replays them in order through the OpenCASCADE kernel. Every
feature has an integer `id` (stable), a `name`, `suppressed`, and a `kind` —
the operation as data. Later features reference earlier ones by id; a feature
that consumes an input (an extrude consumes its sketch, a boolean its two
operands) makes that input invisible, so the **visible bodies** are the
leftovers. Units are millimetres, world is Z-up, Y forward, X right.

## The loop

1. `cadette_info` — read the report: every feature's `status` (`built`,
   `failed`, `suppressed`, `rolled_back`) and `error`, the visible `bodies`
   with exact `bbox`, `volume`, `surface_area`, `centroid`, `faces`, `edges`,
   `valid`. Ask for `detail` to get each face (surface type, area, centroid,
   outward normal, radius) and edge (curve type, length, endpoints, an
   on-curve `midpoint` you can use as an anchor; cylindrical faces also carry
   `axis_point` + `axis`).
2. `cadette_apply` — send a batch of edits. It is all-or-nothing; the reply
   carries the outcomes (new ids) and the fresh report. Read the report's
   `ok` / `errors`: a step that fails is reported, not fatal.
3. `cadette_render` — look, when geometry matters visually (named views, or
   `as_seen` for the user's own camera in a running window). Prefer the
   numbers in the report for verification; use images for sanity.

If a Cadette window has the file open, `info`/`apply` go through the window:
the edit shows up live as one undoable step. `cadette_state` tells you what
the user has selected (`picks`: body feature id + local face/edge index) and
whether they are mid-edit (`busy` — edits are refused then; retry).

## Edits (`cadette_apply` → `edits`)

Each edit is an object with an `op`:

- `{"op":"add","kind":KIND}` (+ optional `"name"`) → creates a feature at the
  end; the outcome is `{"created": id}`.
- `{"op":"insert_transform","kind":KIND}` — for Translate/Rotate/Scale/Mirror
  and Pattern only: splices the transform into its source's chain so
  everything downstream moves too (what dragging in the viewport does). Use
  `add` for a transform when you want a *branch* instead.
- `{"op":"replace","id":N,"kind":KIND}` — edit a feature's parameters.
- `{"op":"rename","id":N,"name":"..."}`, `{"op":"set_color","id":N,"color":[r,g,b]}` (linear 0..1, or null).
- `{"op":"suppress","id":N,"suppressed":true}`.
- `{"op":"remove","id":N}` — dependents are rewired to the removed feature's input.
- `{"op":"ungroup","id":N}` — dissolve a Group (transforms baked onto members).
- `{"op":"reorder","id":N,"index":I}` — refused if a reference would point forward.
- `{"op":"duplicate","id":N}` → a copy of the whole subtree feeding N.
- `{"op":"set_rollback","position":P}`, `{"op":"rollback_to_tip"}`, `{"op":"tidy"}`.
- `{"op":"import_step","file":"servo.step","data":BASE64}` (+ optional `"name"`)
  → stores the STEP file in the project and adds an `Import` feature that
  replays it. Prefer `cadette_import`, which reads the file for you.

Ids created by `add` in one batch cannot be referenced later in the same
batch — do two calls (add, read the id, then add the fillet on it).

## Feature kinds (`KIND`)

Primitives (no inputs): `{"Box":{"size":[x,y,z]}}` (corner at the origin),
`{"Cylinder":{"radius":r,"height":h}}` (axis +Z from the origin),
`{"Sphere":{"radius":r}}` (centred at the origin). Position them with a
Translate.

Imported bodies: `{"Import":{"asset":"servo.step"}}` — a STEP file stored in
the project (added with `cadette_import`, never by hand: the asset must
exist). Every solid in the file becomes one body, exact B-rep in the file's
units converted to the document's, placed where the file put it (Translate
it like a primitive). Its faces and edges work like any other body's — read
them with `detail`, then fillet, chamfer, shell, cut or thread. An assembly
file (several solids) is one compound body like a Pattern's (`solids` in
the report says how many): boolean it as one, or import the parts
separately. A file that isn't STEP, is truncated, has syntax errors or holds
no solid is refused at import, so nothing lands. Topology is exact but not
necessarily identical to the exporter's (periodic faces may be split), so
edge anchors from before an export don't carry across. A big file costs its
parse time on every CLI call (the window caches it).
`cadette_export_step` writes the visible bodies back out for other CAD.

Sketch faces: `{"Sketch":{"plane":PLANE,"profile":{"Rectangle":{"width":w,"height":h}}}}`
or `"profile":{"Circle":{"radius":r}}` (centred on the plane origin).
`PLANE` is `"Xy"`, `"Xz"`, `"Yz"`, or
`{"Custom":{"origin":[..],"x_dir":[..],"y_dir":[..]}}` (e.g. a face's plane
from a face's centroid + normal). A `ConstraintSketch` holds a solved point/line
sketch — prefer the simple profiles unless you need free shapes.

Operations (reference earlier ids):
- `{"Extrude":{"source":sketchId,"distance":d}}` — along the plane normal (negative = the other way).
- `{"Translate":{"source":id,"offset":[x,y,z]}}`
- `{"Rotate":{"source":id,"axis":[x,y,z],"angle":radians,"center":[x,y,z]}}`
- `{"Scale":{"source":id,"factors":[sx,sy,sz],"anchor":[x,y,z]}}`
- `{"Mirror":{"source":id,"origin":[x,y,z],"normal":[x,y,z]}}`
- `{"Boolean":{"op":"Union"|"Subtract"|"Intersect","target":id,"tool":id}}`
- `{"FilletAll":{"source":id,"radius":r}}`
- `{"Fillet":{"source":id,"edges":[{"point":[x,y,z]}],"radius":r}}` — each point
  must lie on the edge (within 1 mm): use an edge's `midpoint` from `detail`.
- `{"ChamferAll":{"source":id,"distance":d}}` — a symmetric 45° bevel, `d` back
  along both faces of every edge. Its own feature, not a fillet variant.
- `{"Chamfer":{"source":id,"edges":[{"point":[x,y,z]}],"distance":d}}` — anchors as for Fillet.
  Blends need sharp edges: chamfer a shaft's end BEFORE threading it, and a
  body whose every edge is already filleted has nothing left to chamfer.
- `{"Shell":{"source":id,"thickness":t,"open":[{"point":[x,y,z],"normal":[x,y,z]}]}}`
  — hollow the body to a uniform wall `t` measured INWARD (the outside is
  unchanged); each `open` face (anchored like PushPull) is removed so the cavity
  is reachable. Omit `open` for a closed hollow. Rules: the body must be ONE
  solid (shell the seed before patterning it); `t` under half the body's
  smallest span; a face to open needs at least one sharp edge (a fully
  filleted top can't be opened — shell first, fillet after); anchor a face
  with a hole in it by a point ON the face (its `centroid` may sit in the
  hole). Inner concave corners come out rounded with radius `t`; a wall
  thicker than a fillet it passes under loses that inner rounding (exactly
  equal fails). Keep `t` under half the thinnest neck of a multi-lobed body:
  where the inward offset would pinch off, the kernel's offset is unreliable.
  Failures are regen errors with a reason.
- `{"PushPull":{"source":id,"anchor":{"point":[x,y,z],"normal":[x,y,z]},"distance":d}}`
  — a planar face by a point on it (its `centroid`) and its normal; positive adds material.
- `{"Revolve":{"source":sketchId,"axis":{"point":[x,y,z]},"angle":radians}}` — the axis is one of the profile's own straight edges, named by a point on it.
- `{"Group":{"members":[ids]}}`
- `{"Thread":{"source":id,"helix":{"center":[x,y,z],"axis":[x,y,z],"radius":r,"pitch":p,"length":l,"left_handed":false},"depth":d,"internal":false}}`
  — cut an ISO 60° thread into a cylindrical face: `radius` is that face's
  (a shaft at its major diameter, or with `internal: true` a hole at its minor
  diameter), `center` + `axis` from the face's `axis_point`/`axis` in `detail`.
  The helix runs `length` along the axis from `center`; let it overshoot the
  ends by a pitch (it only cuts air). ISO coarse pitches: M3 0.5, M4 0.7,
  M5 0.8, M6 1.0, M8 1.25, M10 1.5, M12 1.75; `depth` = 0.541 × pitch.
  Optional `"starts":n` repeats the groove `n` times around the axis; the
  helix `pitch` is then the lead (advance per turn) and the grooves sit
  `pitch/n` apart, which sizes them. A **knurl** is two steep multi-start
  threads: e.g. on a 20 mm knob `pitch: 60, starts: 30, depth: 0.5`, once
  right-handed and once with `left_handed: true`.
- `{"Sweep":{"profile":sketchId,"path":{"Helix":{"helix":HELIX}}}}` — a coil
  or spring: the sketch face is carried rigidly round the helix (same `HELIX`
  object as Thread) keeping its orientation relative to the axis, so sketch
  the section on a plane through the axis. The coil's radius is the section's
  own distance from the axis (the helix `radius` field is irrelevant here).
- `{"Pattern":{"source":id,"count":n,"layout":{"Linear":{"step":[x,y,z]}}}}`,
  `"layout":{"Circular":{"center":[x,y,z],"axis":[x,y,z],"angle":radians}}` or
  `"layout":{"Helical":{"center":[..],"axis":[..],"angle":radians,"rise":d}}`
  (each instance turned `angle` AND moved `rise` along the axis — bumps up a
  knob) — `count` instances (the seed in place plus `count-1` copies, each
  moved one more step), bundled as one body like a Group. Optional
  `"skip":[i,..]` leaves instances out (0 = the seed). For a bolt circle use
  `angle = 2π/count`. To repeat a *cut*, `insert_transform` the Pattern on
  the boolean's tool: the Subtract then removes every instance.

Typical part: add a Box → add a Cylinder → insert_transform a Translate on the
cylinder to place it → add a Boolean Subtract (target box, tool cylinder) →
add FilletAll on the result → `cadette_info` to verify volume/bbox → render.
Four holes instead of one: insert_transform a Linear Pattern (count 4) on the
placed cylinder before the Subtract, or on it afterwards — the splice rewires
the Subtract's tool to the pattern either way.
