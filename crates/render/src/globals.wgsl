// Shared per-frame uniforms + selection helpers. This file is prepended to the
// viewport shaders at pipeline creation (see `with_globals` in scene.rs), so
// the `Globals` layout lives in exactly one WGSL place, mirroring the Rust
// `Globals` struct in scene.rs.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    light_dir : vec4<f32>,
    // faces = [selected_face_count, 0, hovered_face, has_hov]
    faces : vec4<u32>,
    // edges = [selected_edge_count, 0, hovered_edge, has_hov]
    edges : vec4<u32>,
    // up to 64 selected edge ids, packed 4 per vec4; edges.x of them valid.
    sel_edges : array<vec4<u32>, 16>,
    // up to 64 selected face ids, packed 4 per vec4; faces.x of them valid.
    sel_faces : array<vec4<u32>, 16>,
    // theme colors (rgb in .xyz): unlit material base + feature-edge color.
    face_base : vec4<f32>,
    edge_base : vec4<f32>,
    // Hover-ghost overlay color: rgb + alpha.
    ghost_color : vec4<f32>,
    // Build-Graph hover body tint: [first_face, end_face, active, 0].
    hover_body : vec4<u32>,
    // [body_tint, 0, 0, 0]: vertex-colour strength (0 = face_base, 1 = full).
    tint : vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

// True when `id` is among the first `count` ids packed 4-per-vec4 in `ids`.
fn contains_id(ids : array<vec4<u32>, 16>, count : u32, id : u32) -> bool {
    for (var i : u32 = 0u; i < count; i = i + 1u) {
        let v = ids[i / 4u];
        var f : u32;
        switch (i % 4u) {
            case 0u: { f = v.x; }
            case 1u: { f = v.y; }
            case 2u: { f = v.z; }
            default: { f = v.w; }
        }
        if (f == id) { return true; }
    }
    return false;
}

fn is_selected_face(id : u32) -> bool {
    return contains_id(globals.sel_faces, globals.faces.x, id);
}

fn is_selected_edge(id : u32) -> bool {
    return contains_id(globals.sel_edges, globals.edges.x, id);
}
