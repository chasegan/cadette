// Transform gizmo: colored axis lines drawn on top of the model. Depth test is
// disabled in the pipeline so handles stay grabbable even when occluded.
// `Globals` comes from globals.wgsl, prepended at build (only view_proj is used).

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) color : vec4<f32>,
) -> VsOut {
    var out : VsOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
