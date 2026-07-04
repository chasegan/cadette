// Crisp feature edges drawn as dark lines over the shaded faces. Selected edges
// (a set, for multi-edge fillet) are tinted strong, the hovered edge subtle.
// `Globals`/`is_selected_edge` come from globals.wgsl, prepended at build.

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
    if (is_selected_edge(in.edge_id)) {
        return vec4<f32>(1.0, 0.62, 0.25, 1.0); // selected edge (strong)
    } else if (globals.edges.w == 1u && in.edge_id == globals.edges.z) {
        return vec4<f32>(0.45, 0.62, 0.95, 1.0); // hovered edge (subtle)
    }
    return vec4<f32>(globals.edge_base.xyz, 1.0);
}
