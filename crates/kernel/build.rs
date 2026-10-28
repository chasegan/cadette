//! Compiles the C++ cxx bridge and links it against a system OpenCASCADE.
//!
//! OCCT is located via (in order):
//!   1. `CDT_OCCT_PREFIX` env var, if set — e.g. a vcpkg triplet dir like
//!      `$VCPKG_INSTALLATION_ROOT/installed/x64-windows`,
//!   2. a per-OS default: the Homebrew prefixes on macOS (Apple Silicon, then
//!      Intel), the vcpkg layout under `VCPKG_INSTALLATION_ROOT` on Windows,
//!      `/usr` on Linux (needs OCCT >= 7.8 — TKDESTEP doesn't exist before).
//!
//! Headers live in `<prefix>/include/opencascade`; libraries in `<prefix>/lib`.

use std::path::Path;

/// OCCT toolkit libraries the bridge links against. Names are identical across
/// platforms (`libTKernel.dylib` / `libTKernel.so` / `TKernel.lib`): the
/// `dylib=` link kind resolves each correctly per target.
const OCCT_LIBS: &[&str] = &[
    "TKernel",
    "TKMath",
    "TKG2d",
    "TKG3d",
    "TKGeomBase",
    "TKGeomAlgo",
    "TKBRep",
    "TKTopAlgo",
    "TKPrim",
    "TKBO",       // Boolean operations (BRepAlgoAPI_*)
    "TKBool",
    "TKFillet",   // BRepFilletAPI_MakeFillet
    "TKOffset",   // BRepOffsetAPI_MakePipe (sweep)
    "TKShHealing",
    "TKMesh",     // BRepMesh_IncrementalMesh
    "TKDESTL",    // StlAPI_Writer (moved here in OCCT 7.7+)
    "TKXSBase",   // data-exchange base (XSControl_Reader)
    "TKDE",       // data-exchange framework (OCCT 7.8+)
    "TKDESTEP",   // STEPControl_Reader / STEPControl_Writer (OCCT 7.8+)
];

fn occt_prefix(target_os: &str) -> String {
    if let Ok(prefix) = std::env::var("CDT_OCCT_PREFIX") {
        return prefix;
    }
    match target_os {
        "macos" => {
            // Apple Silicon Homebrew, then Intel Homebrew.
            for candidate in ["/opt/homebrew/opt/opencascade", "/usr/local/opt/opencascade"] {
                if Path::new(candidate).is_dir() {
                    return candidate.to_string();
                }
            }
            "/opt/homebrew/opt/opencascade".to_string()
        }
        "windows" => {
            // GitHub's windows runners predefine VCPKG_INSTALLATION_ROOT.
            let root = std::env::var("VCPKG_INSTALLATION_ROOT")
                .unwrap_or_else(|_| r"C:\vcpkg".to_string());
            format!(r"{root}\installed\x64-windows")
        }
        _ => "/usr".to_string(),
    }
}

fn install_hint(target_os: &str) -> &'static str {
    match target_os {
        "macos" => "Install with `brew install opencascade`",
        "windows" => "Install with `vcpkg install opencascade --triplet x64-windows`",
        _ => "Install OpenCASCADE >= 7.7 via your package manager or vcpkg",
    }
}

fn main() {
    // Build scripts run on the host, so target facts must come from the env
    // cargo sets — not #[cfg], which describes the host.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let prefix = occt_prefix(&target_os);

    let include = format!("{prefix}/include/opencascade");
    let libdir = format!("{prefix}/lib");

    if !Path::new(&include).is_dir() {
        panic!(
            "OpenCASCADE headers not found at {include}.\n\
             {}, or set CDT_OCCT_PREFIX to your OCCT install prefix.",
            install_hint(&target_os)
        );
    }

    let mut bridge = cxx_build::bridge("src/ffi.rs");
    bridge
        .file("src/ffi/bridge.cpp")
        .include(&include)
        .std("c++17")
        .warnings(false);
    if target_os == "windows" {
        // Keep windows.h's min/max macros from mangling OCCT headers.
        bridge.define("NOMINMAX", None);
    }
    bridge.compile("cdt_kernel_bridge");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/ffi/bridge.cpp");
    println!("cargo:rerun-if-changed=src/ffi/bridge.hpp");
    println!("cargo:rerun-if-env-changed=CDT_OCCT_PREFIX");

    println!("cargo:rustc-link-search=native={libdir}");
    // Embed an rpath so the binary finds the OCCT dylibs at runtime. Windows
    // has no rpath (link.exe rejects the flag); DLLs are found via PATH or by
    // sitting next to the exe.
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
    for lib in OCCT_LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
