// Picking pass: render each fragment's source face id to an R32Uint target.
// Output is face_id + 1 so that 0 means "no geometry" (the cleared value).

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    light_dir : vec4<f32>,
    faces : vec4<u32>,
    edges : vec4<u32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) @interpolate(flat) face_id : u32,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(2) face_id : u32,
) -> VsOut {
    var out : VsOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.face_id = face_id;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) u32 {
    return in.face_id + 1u;
}
