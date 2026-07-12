//! The origin-plane construction grids: line geometry for the enabled
//! workplanes plus the persistent vertex-buffer factory.

use glam::Vec3;

use crate::controller::{Axis3, GridSpec};
use crate::math::rgba;
use crate::theme::ViewportTheme;
use crate::GizmoVertex;

/// Drop the grid just below z=0 so it doesn't z-fight a model base sitting there.
const GRID_Z: f32 = -0.05;
/// Don't draw minor lines once they'd be denser than this. The ceiling is the
/// vertex budget: `GRID_MAX_LINES/2` steps per side × 2 directions × 2 verts
/// is ~2400 verts per plane, ~7200 with all three planes — under
/// [`GRID_MAX_VERTS`]. Raising this further needs a bigger buffer.
const GRID_MAX_LINES: f32 = 600.0;
/// Capacity of the persistent grid vertex buffer (sized for all three planes).
pub(crate) const GRID_MAX_VERTS: u64 = 12288;

/// The finest grid spacing that will actually draw at `half_extent` — anything
/// finer hits the line-density cap. Hosts should clamp the user's spacing to
/// this so the setting never silently under-delivers.
pub fn min_grid_spacing(half_extent: f32) -> f32 {
    2.0 * half_extent / GRID_MAX_LINES
}

pub(crate) fn make_grid_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cdt-grid"),
        size: GRID_MAX_VERTS * std::mem::size_of::<GizmoVertex>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Line geometry for the origin-plane construction grids: each enabled plane
/// gets faint minor lines at the grid spacing, brighter major lines every 10,
/// and colored axis lines. The XY ground sits just below z=0 (so it doesn't
/// z-fight a model base on the plane); the vertical guides pass through origin.
pub(crate) fn grid_geometry(spec: GridSpec, theme: &ViewportTheme) -> Vec<GizmoVertex> {
    if spec.spacing <= 0.0 || spec.half_extent <= 0.0 {
        return Vec::new();
    }
    let mut v = Vec::new();
    if spec.visible {
        plane_grid(&mut v, spec, theme, Axis3::X, Axis3::Y, Axis3::Z, GRID_Z);
    }
    if spec.xz {
        plane_grid(&mut v, spec, theme, Axis3::X, Axis3::Z, Axis3::Y, 0.0);
    }
    if spec.yz {
        plane_grid(&mut v, spec, theme, Axis3::Y, Axis3::Z, Axis3::X, 0.0);
    }
    v
}

/// Append one plane's grid to `v`. The plane spans world axes `ua`/`va`, sits at
/// `offset` along its normal `na`; the in-plane axis lines take their axis color.
fn plane_grid(
    v: &mut Vec<GizmoVertex>,
    spec: GridSpec,
    theme: &ViewportTheme,
    ua: Axis3,
    va: Axis3,
    na: Axis3,
    offset: f32,
) {
    let ext = spec.half_extent;
    // Cap the visual density: clamp the line spacing so there are never more than
    // ~GRID_MAX_LINES lines across (keeps a huge extent / tiny spacing readable
    // and the vertex count bounded). Snapping still uses the real spacing.
    let minor = spec.spacing.max(2.0 * ext / GRID_MAX_LINES);
    let draw_minor = spec.spacing >= 2.0 * ext / GRID_MAX_LINES;
    let minor_c = theme.grid_minor;
    let major_c = theme.grid_major;
    let ua_c = rgba(ua.color(), 0.55);
    let va_c = rgba(va.color(), 0.55);

    let normal = na.dir() * offset;
    let world = |a: f32, b: f32| ua.dir() * a + va.dir() * b + normal;
    let mut line = |a: Vec3, b: Vec3, c: [f32; 4]| {
        v.push(GizmoVertex { position: a.to_array(), color: c });
        v.push(GizmoVertex { position: b.to_array(), color: c });
    };
    let n = (ext / minor).floor() as i32;
    for i in -n..=n {
        let on_major = i % 10 == 0;
        if !on_major && !draw_minor {
            continue;
        }
        let p = i as f32 * minor;
        let base = if on_major { major_c } else { minor_c };
        // Constant-`ua` line (runs along `va`); at ua=0 it is the `va` axis.
        line(world(p, -ext), world(p, ext), if i == 0 { va_c } else { base });
        // Constant-`va` line (runs along `ua`); at va=0 it is the `ua` axis.
        line(world(-ext, p), world(ext, p), if i == 0 { ua_c } else { base });
    }
}
