//! Compiles the C++ cxx bridge and links it against a system OpenCASCADE.
//!
//! OCCT is located via (in order):
//!   1. `CDT_OCCT_PREFIX` env var, if set,
//!   2. `<prefix>/include/opencascade` + `<prefix>/lib` under the Homebrew
//!      default `/opt/homebrew/opt/opencascade` (Apple Silicon).
//!
//! Headers live in `<prefix>/include/opencascade`; libraries in `<prefix>/lib`.

use std::path::Path;

/// OCCT toolkit libraries the bridge links against. macOS two-level namespacing
/// resolves each dylib's own dependencies at load time, so this is the set we
/// reference directly plus their immediate neighbours.
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
    "TKShHealing",
    "TKMesh",     // BRepMesh_IncrementalMesh
    "TKDESTL",    // StlAPI_Writer (moved here in OCCT 7.7+)
];

fn main() {
    let prefix = std::env::var("CDT_OCCT_PREFIX")
        .unwrap_or_else(|_| "/opt/homebrew/opt/opencascade".to_string());

    let include = format!("{prefix}/include/opencascade");
    let libdir = format!("{prefix}/lib");

    if !Path::new(&include).is_dir() {
        panic!(
            "OpenCASCADE headers not found at {include}.\n\
             Install with `brew install opencascade`, or set CDT_OCCT_PREFIX to \
             your OCCT install prefix."
        );
    }

    cxx_build::bridge("src/ffi.rs")
        .file("src/ffi/bridge.cpp")
        .include(&include)
        .std("c++17")
        .warnings(false)
        .compile("cdt_kernel_bridge");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/ffi/bridge.cpp");
    println!("cargo:rerun-if-changed=src/ffi/bridge.hpp");
    println!("cargo:rerun-if-env-changed=CDT_OCCT_PREFIX");

    println!("cargo:rustc-link-search=native={libdir}");
    // Embed an rpath so the resulting binary finds the OCCT dylibs at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    for lib in OCCT_LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
