// Lit shader for the viewport. Z-up world space; key directional light + a
// camera "headlight" + ambient. The selected face (by id) is tinted.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    light_dir : vec4<f32>,
    // faces = [selected_face_count, 0, hovered_face, has_hov]
    faces : vec4<u32>,
    edges : vec4<u32>,
    // sel_edges is declared so sel_faces lands at the right offset (we don't use it here).
    sel_edges : array<vec4<u32>, 16>,
    // up to 64 selected face ids, packed 4 per vec4; faces.x of them valid.
    sel_faces : array<vec4<u32>, 16>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

fn is_selected_face(id : u32) -> bool {
    let count = globals.faces.x;
    for (var i : u32 = 0u; i < count; i = i + 1u) {
        let v = globals.sel_faces[i / 4u];
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

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) world_normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
    @location(2) @interpolate(flat) face_id : u32,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) face_id : u32,
) -> VsOut {
    var out : VsOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.world_normal = normal;
    out.world_pos = position;
    out.face_id = face_id;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    // Two-sided: faces seen from behind (e.g. inside a bored hole) still light.
    var n = normalize(in.world_normal);
    let view = normalize(globals.camera_pos.xyz - in.world_pos);
    if (dot(n, view) < 0.0) {
        n = -n;
    }

    let l = normalize(globals.light_dir.xyz);
    let key = max(dot(n, l), 0.0);
    let head = max(dot(n, view), 0.0) * 0.3;
    let ambient = 0.2;

    var base = vec3<f32>(0.62, 0.66, 0.72);
    if (is_selected_face(in.face_id)) {
        base = vec3<f32>(1.0, 0.62, 0.25); // clicked selection (strong)
    } else if (globals.faces.w == 1u && in.face_id == globals.faces.z) {
        base = vec3<f32>(0.78, 0.85, 1.0); // hover pre-highlight (subtle)
    }
    let shade = ambient + key * 0.7 + head;
    return vec4<f32>(base * shade, 1.0);
}
