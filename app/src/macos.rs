//! macOS-only process/bundle identity glue.

/// Give the process a proper macOS identity so the menu bar reads "Cadette"
/// (winit builds "About/Hide/Quit <processName>" from `NSProcessInfo`) and the
/// standard About panel shows the name + version — an unbundled dev binary
/// otherwise shows "cdt-app". The shipped `.app` gets these from its Info.plist;
/// this makes local runs match.
pub fn set_macos_app_identity(name: &str, version: &str) {
    use objc2::msg_send;
    use objc2_foundation::{NSBundle, NSProcessInfo, NSString};
    // Safety: correctly-typed main-thread message sends, run once before the
    // event loop. The main bundle's info dictionary is mutable in-process, so the
    // standard About panel picks up the injected name/version.
    unsafe {
        let ns_name = NSString::from_str(name);
        let proc = NSProcessInfo::processInfo();
        let _: () = msg_send![&proc, setProcessName: &*ns_name];

        if let Some(info) = NSBundle::mainBundle().infoDictionary() {
            let ns_version = NSString::from_str(version);
            let v_key = NSString::from_str("CFBundleShortVersionString");
            let _: () = msg_send![&info, setObject: &*ns_version, forKey: &*v_key];
            let n_key = NSString::from_str("CFBundleName");
            let _: () = msg_send![&info, setObject: &*ns_name, forKey: &*n_key];
        }
    }
}
