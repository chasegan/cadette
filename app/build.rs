//! Embeds the application icon into the Windows `.exe`.
//!
//! The runtime window icon (taskbar + title bar) is loaded from
//! `assets/icon/icon-256.png` at run time in `cdt-render`. This build script
//! covers the *other* Windows icon: the one Explorer shows for the executable
//! file itself, which lives in the binary's resource table and so must be baked
//! in at link time. `winresource` invokes the Windows SDK resource compiler,
//! available on the `windows-latest` CI runner that ships the release build.
//!
//! Gated on the *host* OS via `cfg`: the `winresource` build-dependency is only
//! pulled in on a Windows host (see Cargo.toml), so the reference to it must
//! likewise vanish on macOS/Linux, where `main` is an empty no-op. Native
//! per-platform CI builds have host == target, so the Windows release exe gets
//! the icon and other hosts skip it cleanly.

fn main() {
    #[cfg(target_os = "windows")]
    embed_windows_icon();
}

#[cfg(target_os = "windows")]
fn embed_windows_icon() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let icon = std::path::Path::new(&manifest_dir).join("../assets/icon/icon.ico");

    let mut res = winresource::WindowsResource::new();
    res.set_icon(icon.to_str().expect("icon path is valid UTF-8"));
    if let Err(e) = res.compile() {
        // Warn rather than fail the build: a missing resource compiler should
        // not block a local Windows dev build.
        println!("cargo:warning=failed to embed Windows exe icon: {e}");
    }

    println!("cargo:rerun-if-changed=../assets/icon/icon.ico");
}
