// Crisp feature edges drawn as dark lines over the shaded faces. The selected
// edge is tinted strong, the hovered edge subtle.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    light_dir : vec4<f32>,
    faces : vec4<u32>,
    // edges = [selected_edge, has_sel, hovered_edge, has_hov]
    edges : vec4<u32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) @interpolate(flat) edge_id : u32,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) edge_id : u32,
) -> VsOut {
    var out : VsOut;
    var clip = globals.view_proj * vec4<f32>(position, 1.0);
    // Nudge toward the camera so lines sit crisply on the shaded surface
    // (wgpu clip z is [0, 1]; smaller = nearer). Avoids z-fighting without the
    // depth-bias state, which is illegal for line topology.
    clip.z = clip.z - 0.0003 * clip.w;
    out.clip = clip;
    out.edge_id = edge_id;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    if (globals.edges.y == 1u && in.edge_id == globals.edges.x) {
        return vec4<f32>(1.0, 0.62, 0.25, 1.0); // selected edge (strong)
    } else if (globals.edges.w == 1u && in.edge_id == globals.edges.z) {
        return vec4<f32>(0.45, 0.62, 0.95, 1.0); // hovered edge (subtle)
    }
    return vec4<f32>(0.13, 0.15, 0.18, 1.0);
}
