// Crisp feature edges drawn as dark lines over the shaded faces.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    light_dir : vec4<f32>,
    selected : vec4<u32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
    var clip = globals.view_proj * vec4<f32>(position, 1.0);
    // Nudge toward the camera so lines sit crisply on the shaded surface
    // (wgpu clip z is [0, 1]; smaller = nearer). Avoids z-fighting without the
    // depth-bias state, which is illegal for line topology.
    clip.z = clip.z - 0.0003 * clip.w;
    return clip;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.13, 0.15, 0.18, 1.0);
}
