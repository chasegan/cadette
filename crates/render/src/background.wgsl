// Fullscreen vertical gradient behind the scene. Drawn first, with depth writes
// off, so the grid/model/edges/gizmo composite over it normally. Colors are
// linear RGB; the sRGB target re-encodes on write.

struct Bg {
    top : vec4<f32>,
    bottom : vec4<f32>,
};

@group(0) @binding(0) var<uniform> bg : Bg;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    // 0 at the bottom of the screen, 1 at the top.
    @location(0) t : f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VsOut {
    // A single triangle that covers the whole clip rect.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let xy = corners[vid];
    var out : VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.t = xy.y * 0.5 + 0.5; // NDC y (-1..1) -> 0..1
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let c = mix(bg.bottom.xyz, bg.top.xyz, clamp(in.t, 0.0, 1.0));
    return vec4<f32>(c, 1.0);
}
