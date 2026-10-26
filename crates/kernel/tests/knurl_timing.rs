//! Where the time goes in a multi-start (knurl) thread cut. Run with
//! `cargo test -p cdt-kernel --test knurl_timing -- --nocapture --ignored`.

use cdt_kernel::Solid;
use std::time::Instant;

fn groove(starts: u32, k: u32) -> Solid {
    let lead = 60.0;
    let spacing = lead / starts as f64;
    let elems = cdt_core::thread_profile(spacing, 0.6, 10.0, false);
    let mut pts = Vec::new();
    for e in &elems {
        let [u, v] = e.start();
        pts.push(u);
        pts.push(v);
    }
    let axis = [0.0, 0.0, 1.0];
    let x_dir = [0.0, -1.0, 0.0];
    let center = [0.0, 0.0, -2.0];
    let face = Solid::profile_face(center, x_dir, axis, &pts, &[0.0; 20]).unwrap();
    let helix = Solid::helix_wire(center, axis, x_dir, 10.0, lead, 16.0, false).unwrap();
    let tool = face.sweep(&helix, axis).unwrap();
    let angle = std::f64::consts::TAU * k as f64 / starts as f64;
    tool.rotate(center, axis, angle).unwrap()
}

#[test]
#[ignore]
fn knurl_timing() {
    let knob = Solid::cylinder(10.0, 12.0).unwrap();
    let t = Instant::now();
    let g0 = groove(30, 0);
    println!("one groove sweep: {:?}, faces {}", t.elapsed(), g0.face_count().unwrap());
    let t = Instant::now();
    let cut1 = knob.cut(&g0).unwrap();
    println!("one groove cut: {:?}, faces {}", t.elapsed(), cut1.face_count().unwrap());

    // Sequential cuts.
    let t = Instant::now();
    let mut acc = knob.clone();
    for k in 0..30 {
        acc = acc.cut(&groove(30, k)).unwrap();
        if k % 5 == 4 {
            println!("  sequential after {} grooves: {:?}", k + 1, t.elapsed());
        }
    }
    println!("sequential 30 cuts: {:?}, faces {}, vol {}", t.elapsed(), acc.face_count().unwrap(), acc.volume().unwrap());

    // Compound tool, one cut.
    let t = Instant::now();
    let mut comp = groove(30, 0);
    for k in 1..30 {
        comp = comp.compound(&groove(30, k)).unwrap();
    }
    println!("compound build: {:?}", t.elapsed());
    let t = Instant::now();
    let r = knob.cut(&comp).unwrap();
    println!("compound cut: {:?}, faces {}, vol {}", t.elapsed(), r.face_count().unwrap(), r.volume().unwrap());

    // The same through the tracked boolean (what regen uses).
    let t = Instant::now();
    let tr = knob.boolean_tracked(&comp, cdt_kernel::TrackedBoolOp::Cut).unwrap();
    println!("tracked compound cut (C++): {:?}, face records {}, edge records {}, sections {}",
        t.elapsed(), tr.faces.len(), tr.edges.len(), tr.sections.len());
    let t = Instant::now();
    let _r2 = tr.result.face_count().unwrap();
    println!("face_count: {:?}", t.elapsed());
}

#[test]
#[ignore]
fn knurl_timing_through_backend() {
    use cdt_core::{BooleanOp, DVec3, FeatureId, GeometryBackend, Helix};
    use cdt_kernel::KernelBackend;
    let starts = 12u32;
    let mut be = KernelBackend::default();
    be.begin_feature(FeatureId(1));
    let knob = be.make_cylinder(10.0, 12.0).unwrap();
    be.begin_feature(FeatureId(2));
    let helix = Helix { center: DVec3::new(0.0, 0.0, -2.0), axis: DVec3::Z, radius: 10.0, pitch: 60.0, length: 16.0, left_handed: false };
    let groove = cdt_core::thread_profile(60.0 / starts as f64, 0.6, 10.0, false);
    let t = Instant::now();
    let tool = be.helix_sweep(&groove, helix).unwrap();
    println!("backend helix_sweep: {:?}, faces {}", t.elapsed(), tool.face_count().unwrap());
    let t = Instant::now();
    let mut turned = Vec::new();
    for s in 1..starts {
        let angle = std::f64::consts::TAU * s as f64 / starts as f64;
        turned.push(be.rotate(&tool, helix.center, helix.axis, angle).unwrap());
    }
    let mut members = vec![&tool];
    members.extend(turned.iter());
    let comp = be.compound(&members).unwrap();
    println!("backend rotate+compound: {:?}", t.elapsed());
    let t = Instant::now();
    let r = be.boolean(BooleanOp::Subtract, &knob, &comp).unwrap();
    println!("backend boolean ({starts} starts): {:?}, faces {}", t.elapsed(), r.face_count().unwrap());
}
