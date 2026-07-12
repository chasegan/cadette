// Translucent x-ray ghost for the Build Graph hover preview: a step's own
// result drawn over the whole scene (depth test off) so consumed geometry —
// a cylinder that became a bore, a body swallowed by a boolean — reads
// through the final part. `Globals` comes from globals.wgsl, prepended at
// pipeline creation; the color + alpha ride in `globals.ghost_color`.

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) world_normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
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
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    // Two-sided view-relative shading: the silhouette matters more than the
    // lighting, so a simple facing-ratio shape read is enough.
    var n = normalize(in.world_normal);
    let view = normalize(globals.camera_pos.xyz - in.world_pos);
    if (dot(n, view) < 0.0) {
        n = -n;
    }
    let shade = 0.55 + 0.45 * max(dot(n, view), 0.0);
    return vec4<f32>(globals.ghost_color.xyz * shade, globals.ghost_color.w);
}
