// Lit shader for the viewport. Z-up world space; key directional light + a
// camera "headlight" + ambient. The selected face (by id) is tinted.
// `Globals`/`is_selected_face` come from globals.wgsl, prepended at build.

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) world_normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
    @location(2) @interpolate(flat) face_id : u32,
    @location(3) @interpolate(flat) color : vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) face_id : u32,
    @location(3) color : vec3<f32>,
) -> VsOut {
    var out : VsOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.world_normal = normal;
    out.world_pos = position;
    out.face_id = face_id;
    out.color = color;
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

    // Per-body colour if set (non-negative), else the theme's default material.
    var base = globals.face_base.xyz;
    if (in.color.r >= 0.0) {
        base = in.color;
    }
    if (is_selected_face(in.face_id)) {
        base = vec3<f32>(1.0, 0.62, 0.25); // clicked selection (strong)
    } else if (globals.faces.w == 1u && in.face_id == globals.faces.z) {
        base = vec3<f32>(0.78, 0.85, 1.0); // hover pre-highlight (subtle)
    }
    let shade = ambient + key * 0.7 + head;
    return vec4<f32>(base * shade, 1.0);
}
