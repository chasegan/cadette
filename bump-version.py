#!/usr/bin/env python3
"""Set Cadette's version.

Unlike a multi-file project, Cadette keeps its version in ONE place: the
workspace `Cargo.toml` (`[workspace.package] version`). Every crate inherits it
via `version.workspace = true`, and the app compiles it in as
`env!("CARGO_PKG_VERSION")` (shown in the About window + the macOS menu). The
release `.dmg`'s Info.plist and the git tag are both derived from it, and CI
fails a release whose `vX.Y.Z` tag doesn't equal this version. So there's nothing
to keep "in sync" across files — this just bumps the single line and refreshes
Cargo.lock.

Usage:
    python bump-version.py            # print the current version
    python bump-version.py 0.2.0      # set the version, refresh Cargo.lock

Then cut the release by committing and tagging to MATCH:
    git commit -am "Release v0.2.0" && git tag v0.2.0
"""
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cargo = root / "Cargo.toml"
text = cargo.read_text()

# The workspace version is the first line-anchored `version = "..."`. Dependency
# versions are inline (`serde = { version = "1" }`), never at the start of a line.
match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
if not match:
    sys.exit("Could not find [workspace.package] version in Cargo.toml")
current = match.group(1)

if len(sys.argv) == 1:
    print(current)
    sys.exit(0)
if len(sys.argv) != 2:
    sys.exit("Usage: python bump-version.py [NEW_VERSION]")

new = sys.argv[1]
if not re.fullmatch(r"\d+\.\d+\.\d+", new):
    sys.exit(f"Version must be X.Y.Z, got {new!r}")

cargo.write_text(
    re.sub(r'^(version\s*=\s*")[^"]+(")', rf"\g<1>{new}\g<2>",
           text, count=1, flags=re.MULTILINE)
)

# Refresh Cargo.lock so the recorded workspace-crate versions match (best effort).
subprocess.run(["cargo", "update", "--workspace", "--offline"], cwd=root, check=False)

print(f"Bumped {current} -> {new}")
print(f"Next: git commit -am 'Release v{new}' && git tag v{new}")
