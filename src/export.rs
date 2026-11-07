use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

fn null_to_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Raw ChatGPT export types. Every field is optional: the format has changed
/// repeatedly and a strict struct would reject otherwise usable exports.
#[derive(Debug, Deserialize, Default)]
pub struct RawAuthor {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct RawThought {
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct RawContent {
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub parts: Vec<Value>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub thoughts: Vec<RawThought>,
}

#[derive(Debug, Deserialize, Default)]
pub struct RawMessage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub author: Option<RawAuthor>,
    #[serde(default)]
    pub create_time: Option<f64>,
    #[serde(default)]
    pub update_time: Option<f64>,
    #[serde(default)]
    pub content: Option<RawContent>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub end_turn: Option<bool>,
    #[serde(default)]
    pub weight: Option<f64>,
    #[serde(default)]
    pub recipient: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize, Default)]
pub struct RawNode {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub message: Option<RawMessage>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub children: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct RawConversation {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub create_time: Option<f64>,
    #[serde(default)]
    pub update_time: Option<f64>,
    #[serde(default)]
    pub current_node: Option<String>,
    #[serde(default)]
    pub default_model_slug: Option<String>,
    #[serde(default)]
    pub gizmo_id: Option<String>,
    #[serde(default)]
    pub conversation_template_id: Option<String>,
    #[serde(default)]
    pub is_archived: Option<bool>,
    #[serde(default)]
    pub is_starred: Option<bool>,
    #[serde(default)]
    pub memory_scope: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub mapping: HashMap<String, RawNode>,
    /// Some 2023 exports used a flat array instead of a mapping.
    #[serde(default, deserialize_with = "null_to_default")]
    pub messages: Vec<RawMessage>,
}

pub enum SourceKind {
    Zip,
    Directory,
}

impl SourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceKind::Zip => "zip",
            SourceKind::Directory => "directory",
        }
    }
}

pub struct ExportArchive {
    kind: SourceKind,
    zip: Option<zip::ZipArchive<BufReader<File>>>,
    dir: Option<PathBuf>,
    names: Vec<String>,
}

impl ExportArchive {
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            let mut names = Vec::new();
            collect_files(path, path, 0, &mut names)?;
            names.sort();
            if names.is_empty() {
                bail!("directory {} contains no files", path.display());
            }
            return Ok(Self {
                kind: SourceKind::Directory,
                zip: None,
                dir: Some(path.to_path_buf()),
                names,
            });
        }

        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut zip = zip::ZipArchive::new(BufReader::new(file))
            .with_context(|| format!("{} is not a readable zip archive", path.display()))?;
        let mut names = Vec::new();
        for i in 0..zip.len() {
            let entry = zip.by_index(i)?;
            if entry.is_dir() {
                continue;
            }
            names.push(entry.name().to_string());
        }
        names.sort();
        if names.is_empty() {
            bail!("zip archive {} contains no files", path.display());
        }
        Ok(Self {
            kind: SourceKind::Zip,
            zip: Some(zip),
            dir: None,
            names,
        })
    }

    pub fn kind(&self) -> &SourceKind {
        &self.kind
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Files that hold conversation arrays, in a deterministic order. Handles
    /// both `conversations.json` and the `conversations-000.json` split form.
    pub fn conversation_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self
            .names
            .iter()
            .filter(|name| is_conversation_file(name))
            .cloned()
            .collect();
        files.sort();
        files
    }

    pub fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        if let Some(zip) = self.zip.as_mut() {
            let mut entry = zip
                .by_name(name)
                .with_context(|| format!("zip entry {name} disappeared"))?;
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            return Ok(buf);
        }
        let root = self.dir.as_ref().context("directory source missing root")?;
        let path = root.join(name);
        std::fs::read(&path).with_context(|| format!("read {}", path.display()))
    }

    /// Content hash of the immutable export. Directory exports hash the
    /// conversation files together with their names.
    pub fn fingerprint(&self, path: &Path) -> Result<String> {
        if matches!(self.kind, SourceKind::Zip) {
            return crate::util::sha256_file(path);
        }
        let root = self.dir.as_ref().context("directory source missing root")?;
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        for name in self.conversation_files() {
            hasher.update(name.as_bytes());
            hasher.update([0u8]);
            let bytes = std::fs::read(root.join(&name))?;
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
        Ok(crate::util::hex(&hasher.finalize()))
    }
}

fn collect_files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) -> Result<()> {
    if depth > 5 {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, depth + 1, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .with_context(|| format!("{} is outside {}", path.display(), root.display()))?;
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

pub fn is_conversation_file(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    if base == "conversations.json" {
        return true;
    }
    let Some(rest) = base.strip_prefix("conversations-") else {
        return false;
    };
    let Some(num) = rest.strip_suffix(".json") else {
        return false;
    };
    !num.is_empty() && num.chars().all(|c| c.is_ascii_digit())
}

/// Split a top-level JSON array into the raw byte slices of its elements.
/// This avoids materialising a 200MB+ export as one serde_json Value tree.
pub fn split_json_array(bytes: &[u8]) -> Result<Vec<&[u8]>> {
    let mut start = 0usize;
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        start = 3;
    }
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    if start >= bytes.len() {
        bail!("conversations file is empty");
    }
    if bytes[start] != b'[' {
        bail!("conversations file does not start with a JSON array");
    }

    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut elem_start: Option<usize> = None;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => {
                if depth == 1 && b == b'{' && elem_start.is_none() {
                    elem_start = Some(i);
                }
                depth += 1;
            }
            b']' | b'}' => {
                depth -= 1;
                if depth < 0 {
                    bail!("unbalanced JSON around byte {i}");
                }
                if depth == 1 && b == b'}' {
                    if let Some(s) = elem_start.take() {
                        out.push(&bytes[s..=i]);
                    }
                }
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        bail!("conversations JSON array is not closed");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_elements_with_strings_and_nesting() {
        let raw = br#"[
          {"id":"a","text":"brackets ] and braces } inside"},
          {"id":"b","nested":{"x":[1,2,3]},"quote":"say \"hi\""},
          {"id":"c"}
        ]"#;
        let parts = split_json_array(raw).unwrap();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].starts_with(b"{"));
        assert!(parts[2].ends_with(b"}"));
    }

    #[test]
    fn recognises_conversation_filenames() {
        assert!(is_conversation_file("conversations.json"));
        assert!(is_conversation_file("export/conversations-000.json"));
        assert!(is_conversation_file("conversations-012.json"));
        assert!(!is_conversation_file("chat.html"));
        assert!(!is_conversation_file("shared_conversations.json"));
        assert!(!is_conversation_file("conversations-abc.json"));
    }
}
