//! Binary files that travel with a document — the STEP file an `Import`
//! feature replays — kept in memory beside the history and stored by the
//! project crate as `assets/<name>` entries of the `.cdt` archive.
//!
//! Assets are not part of `document.json` (`#[serde(skip)]`): the document
//! stays hand-editable and a feature refers to its asset by name. Bytes sit
//! behind an `Arc`, so the document clones the undo stack and the regen
//! cache make stay cheap.

use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
struct Entry {
    bytes: Arc<[u8]>,
    /// A content hash, computed once: what the regen cache keys on, and
    /// what makes dedupe cheap.
    hash: u64,
}

#[derive(Clone, Default)]
pub struct Assets(BTreeMap<String, Entry>);

impl Assets {
    pub fn get(&self, name: &str) -> Option<&Arc<[u8]>> {
        self.0.get(name).map(|e| &e.bytes)
    }

    /// A hash of the asset's bytes — changes when the bytes do, which a
    /// regen cache must notice even though the feature's kind (the name)
    /// is unchanged.
    pub fn fingerprint(&self, name: &str) -> Option<u64> {
        self.0.get(name).map(|e| e.hash)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// Store `bytes` under `name`, or under a numbered variant of it
    /// (`servo 2.step`) when the name is taken by different bytes. Returns
    /// the name used. Identical bytes already stored under the name or one
    /// of its numbered variants are shared, not duplicated — importing the
    /// same file again, in any order, stores it once.
    pub fn insert(&mut self, name: &str, bytes: Vec<u8>) -> String {
        let name = sanitize(name);
        let hash = content_hash(&bytes);
        let (stem, ext) = split_ext(&name);
        let same_family = |candidate: &str| {
            candidate == name
                || candidate
                    .strip_prefix(stem)
                    .and_then(|rest| rest.strip_suffix(ext))
                    .is_some_and(|mid| mid.starts_with(' ') && mid[1..].chars().all(|c| c.is_ascii_digit()))
        };
        if let Some((existing, _)) = self
            .0
            .iter()
            .find(|(n, e)| e.hash == hash && e.bytes.as_ref() == bytes.as_slice() && same_family(n))
        {
            return existing.clone();
        }
        let mut chosen = name.clone();
        let mut n = 2;
        while self.0.contains_key(&chosen) {
            chosen = format!("{stem} {n}{ext}");
            n += 1;
        }
        self.0.insert(chosen.clone(), Entry { bytes: Arc::from(bytes), hash });
        chosen
    }

    /// Store bytes under exactly `name`, replacing any previous entry — for
    /// loading an archive, where the names are already settled.
    pub fn set(&mut self, name: &str, bytes: Vec<u8>) {
        let hash = content_hash(&bytes);
        self.0.insert(name.to_string(), Entry { bytes: Arc::from(bytes), hash });
    }

    pub fn remove(&mut self, name: &str) -> Option<Arc<[u8]>> {
        self.0.remove(name).map(|e| e.bytes)
    }

    /// Keep only the assets `keep` accepts (by name).
    pub fn retain(&mut self, mut keep: impl FnMut(&str) -> bool) {
        self.0.retain(|name, _| keep(name));
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<[u8]>)> {
        self.0.iter().map(|(k, e)| (k.as_str(), &e.bytes))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether `name` is one [`Assets::insert`] could have produced: a bare
    /// file name with no directory part. Archive readers use it to refuse
    /// entries like `assets/../x`.
    pub fn is_clean_name(name: &str) -> bool {
        sanitize(name) == name
    }
}

fn content_hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    // `DefaultHasher::new()` is keyed with constants, so the hash is stable
    // across runs; it only needs to tell two byte strings apart.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// `servo.step` → (`servo`, `.step`); `noext` → (`noext`, ``).
fn split_ext(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(dot) if dot > 0 => (&name[..dot], &name[dot..]),
        _ => (name, ""),
    }
}

/// An asset name is a bare file name: no directories, nothing an archive
/// entry can't hold, never empty.
fn sanitize(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
    let cleaned: String = base.chars().filter(|c| !c.is_control()).collect();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "asset".to_string()
    } else {
        cleaned
    }
}

impl<'a> IntoIterator for &'a Assets {
    type Item = (&'a str, &'a Arc<[u8]>);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl std::fmt::Debug for Assets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.0.iter().map(|(k, e)| (k, format!("{} bytes", e.bytes.len()))))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_share_a_name_and_different_bytes_get_a_numbered_one() {
        let mut assets = Assets::default();
        assert_eq!(assets.insert("servo.step", b"a".to_vec()), "servo.step");
        assert_eq!(assets.insert("servo.step", b"a".to_vec()), "servo.step");
        assert_eq!(assets.insert("servo.step", b"b".to_vec()), "servo 2.step");
        assert_eq!(assets.insert("servo.step", b"c".to_vec()), "servo 3.step");
        assert_eq!(assets.len(), 3);
    }

    #[test]
    fn identical_bytes_are_found_under_a_numbered_variant_too() {
        let mut assets = Assets::default();
        assert_eq!(assets.insert("servo.step", b"a".to_vec()), "servo.step");
        assert_eq!(assets.insert("servo.step", b"b".to_vec()), "servo 2.step");
        assert_eq!(assets.insert("servo.step", b"b".to_vec()), "servo 2.step", "the second file again");
        assert_eq!(assets.insert("servo.step", b"a".to_vec()), "servo.step", "the first file again");
        assert_eq!(assets.len(), 2);
        // Same bytes under an unrelated name are a different asset.
        assert_eq!(assets.insert("bolt.step", b"a".to_vec()), "bolt.step");
        assert_ne!(assets.fingerprint("servo.step"), assets.fingerprint("servo 2.step"));
        assert_eq!(assets.fingerprint("servo.step"), assets.fingerprint("bolt.step"));
    }

    #[test]
    fn clean_names_have_no_directory_part() {
        assert!(Assets::is_clean_name("servo.step"));
        assert!(Assets::is_clean_name("my part ü.STEP"));
        for bad in ["../x.step", "sub/x.step", "/abs.step", "", ".", "..", "a\\b.step", "tab\t.step"] {
            assert!(!Assets::is_clean_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn names_are_bare_file_names() {
        let mut assets = Assets::default();
        assert_eq!(assets.insert("/tmp/parts/servo.step", b"a".to_vec()), "servo.step");
        assert_eq!(assets.insert(r"C:\parts\bolt.stp", b"a".to_vec()), "bolt.stp");
        assert_eq!(assets.insert("", b"a".to_vec()), "asset");
        assert_eq!(assets.insert("..", b"b".to_vec()), "asset 2");
    }
}
