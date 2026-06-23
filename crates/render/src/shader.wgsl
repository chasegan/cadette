// Minimal lit shader for the Milestone B viewport.
// Z-up world space; shading is a key directional light + a camera "headlight"
// + ambient, so the form reads clearly from any orbit angle.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    light_dir : vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) world_normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
) -> VsOut {
    var out : VsOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.world_normal = normal;
    out.world_pos = position;
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

    let base = vec3<f32>(0.62, 0.66, 0.72);
    let shade = ambient + key * 0.7 + head;
    return vec4<f32>(base * shade, 1.0);
}
