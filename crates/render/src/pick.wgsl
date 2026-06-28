// Picking pass: render each fragment's source face id to an R32Uint target
// (face_id + 1, so 0 = no geometry) and its depth to an R32Float target, so the
// host can reject edges that screen-space picking found but are occluded.

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

struct FsOut {
    @location(0) id : u32,
    @location(1) depth : f32,
};

@fragment
fn fs_main(in : VsOut) -> FsOut {
    var out : FsOut;
    out.id = in.face_id + 1u;
    out.depth = in.clip.z; // NDC depth [0,1], post depth-test (front-most)
    return out;
}
