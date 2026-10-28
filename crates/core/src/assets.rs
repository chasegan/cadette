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

#[derive(Clone, Default)]
pub struct Assets(BTreeMap<String, Arc<[u8]>>);

impl Assets {
    pub fn get(&self, name: &str) -> Option<&Arc<[u8]>> {
        self.0.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// Store `bytes` under `name`, or under a numbered variant of it
    /// (`servo 2.step`) when the name is taken by different bytes. Returns
    /// the name used. Identical bytes under the same name are shared, not
    /// duplicated — importing the same file twice stores it once.
    pub fn insert(&mut self, name: &str, bytes: Vec<u8>) -> String {
        let name = sanitize(name);
        if let Some(existing) = self.0.get(&name) {
            if existing.as_ref() == bytes.as_slice() {
                return name;
            }
        }
        let (stem, ext) = match name.rfind('.') {
            Some(dot) if dot > 0 => (&name[..dot], &name[dot..]),
            _ => (name.as_str(), ""),
        };
        let mut chosen = name.clone();
        let mut n = 2;
        while self.0.contains_key(&chosen) {
            chosen = format!("{stem} {n}{ext}");
            n += 1;
        }
        self.0.insert(chosen.clone(), Arc::from(bytes));
        chosen
    }

    /// Store bytes under exactly `name`, replacing any previous entry — for
    /// loading an archive, where the names are already settled.
    pub fn set(&mut self, name: &str, bytes: Vec<u8>) {
        self.0.insert(name.to_string(), Arc::from(bytes));
    }

    pub fn remove(&mut self, name: &str) -> Option<Arc<[u8]>> {
        self.0.remove(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<[u8]>)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
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
            .entries(self.0.iter().map(|(k, v)| (k, format!("{} bytes", v.len()))))
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
    fn names_are_bare_file_names() {
        let mut assets = Assets::default();
        assert_eq!(assets.insert("/tmp/parts/servo.step", b"a".to_vec()), "servo.step");
        assert_eq!(assets.insert(r"C:\parts\bolt.stp", b"a".to_vec()), "bolt.stp");
        assert_eq!(assets.insert("", b"a".to_vec()), "asset");
        assert_eq!(assets.insert("..", b"b".to_vec()), "asset 2");
    }
}
