//! Projection helpers handed to the UI so it can draw on the 3D viewport.
//!
//! [`ViewContext`] captures the current camera as a view-projection matrix plus
//! the viewport size (in egui points). It converts between a point on a sketch
//! plane and its screen position, and casts a screen position back onto a plane
//! — everything the interactive sketcher needs to draw an overlay and to place
//! geometry under the cursor. Planes are passed as plain `[f64; 3]` arrays so
//! this stays free of `rmf-core` types.

use egui::{Pos2, Vec2};
use glam::{Mat4, Vec3, Vec4};

/// A snapshot of the camera for one frame's UI.
pub struct ViewContext {
    view_proj: Mat4,
    inv_view_proj: Mat4,
    /// Viewport size in egui points (matches egui pointer coordinates).
    size: Vec2,
}

impl ViewContext {
    pub(crate) fn new(view_proj: Mat4, size: Vec2) -> Self {
        Self {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            size,
        }
    }

    /// Project a world point to a screen position (egui points), or `None` if it
    /// is behind the camera. Used to anchor overlays (e.g. the gizmo readout).
    pub fn project(&self, world: [f64; 3]) -> Option<Pos2> {
        self.project_world(vec3(world))
    }

    /// The viewport size in egui points (for clamping overlays to the screen).
    pub fn size(&self) -> Vec2 {
        self.size
    }

    /// Project a world point to a screen position (egui points), or `None` if
    /// it is behind the camera.
    fn project_world(&self, world: Vec3) -> Option<Pos2> {
        let clip = self.view_proj * world.extend(1.0);
        if clip.w <= 1e-6 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Pos2::new(
            (ndc.x * 0.5 + 0.5) * self.size.x,
            (0.5 - ndc.y * 0.5) * self.size.y,
        ))
    }

    /// Screen position of a sketch-plane point `uv`, on the plane
    /// `origin + u·x_dir + v·y_dir`.
    pub fn project_plane_point(
        &self,
        origin: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        uv: [f64; 2],
    ) -> Option<Pos2> {
        let world = vec3(origin) + vec3(x_dir) * uv[0] as f32 + vec3(y_dir) * uv[1] as f32;
        self.project_world(world)
    }

    /// Cast a screen position onto a plane and return its `[u, v]` coordinates
    /// in the plane's frame, or `None` if the ray misses the plane.
    pub fn cursor_on_plane(
        &self,
        screen: Pos2,
        origin: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        normal: [f64; 3],
    ) -> Option<[f64; 2]> {
        let nx = screen.x / self.size.x * 2.0 - 1.0;
        let ny = 1.0 - screen.y / self.size.y * 2.0;
        let near = self.unproject(nx, ny, 0.0)?;
        let far = self.unproject(nx, ny, 1.0)?;
        let dir = far - near;

        let o = vec3(origin);
        let n = vec3(normal);
        let denom = dir.dot(n);
        if denom.abs() < 1e-6 {
            return None; // ray parallel to the plane
        }
        let t = (o - near).dot(n) / denom;
        if t < 0.0 {
            return None; // plane is behind the camera
        }
        let rel = (near + dir * t) - o;
        Some([rel.dot(vec3(x_dir)) as f64, rel.dot(vec3(y_dir)) as f64])
    }

    fn unproject(&self, nx: f32, ny: f32, nz: f32) -> Option<Vec3> {
        let p = self.inv_view_proj * Vec4::new(nx, ny, nz, 1.0);
        if p.w.abs() < 1e-9 {
            return None;
        }
        Some(p.truncate() / p.w)
    }
}

fn vec3(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;

    #[test]
    fn plane_point_round_trips_through_the_screen() {
        // A point on the XY plane should project to a screen position that casts
        // back onto the same plane coordinates — the invariant the interactive
        // sketcher relies on for both drawing and picking.
        let cam = OrbitCamera::framing(Vec3::ZERO, 50.0);
        let view = ViewContext::new(cam.view_proj(1280.0 / 820.0), Vec2::new(1280.0, 820.0));

        let (origin, x_dir, y_dir, normal) =
            ([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let uv = [12.0, -7.0];

        let screen = view
            .project_plane_point(origin, x_dir, y_dir, uv)
            .expect("point in front of camera");
        let back = view
            .cursor_on_plane(screen, origin, x_dir, y_dir, normal)
            .expect("ray hits the plane");

        assert!((back[0] - uv[0]).abs() < 1e-2, "u: {} vs {}", back[0], uv[0]);
        assert!((back[1] - uv[1]).abs() < 1e-2, "v: {} vs {}", back[1], uv[1]);
    }
}
