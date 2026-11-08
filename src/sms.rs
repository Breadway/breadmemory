use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::importer::{self, ImportOptions, ImportReport};
use crate::util::{ensure_private_dir, sha256_hex};

#[derive(Debug, Deserialize)]
struct SmsEntry {
    ts: i64,
    side: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    src: Option<String>,
}

pub struct SmsImport {
    pub report: ImportReport,
    pub conversation_id: String,
    pub messages: usize,
    pub first_ts: Option<f64>,
    pub last_ts: Option<f64>,
    pub stored_source: PathBuf,
}

/// Import a merged SMS thread as a normal archive conversation.
///
/// The corrected format is a JSON array of `{ts, side, text, src}` where `ts`
/// is Unix milliseconds and `side` is `you` or `her`. The conversation is
/// linear, so it reuses the normal chunking, FTS, hybrid retrieval and raw
/// storage path. The original file is copied into the private archive so the
/// source remains available after /tank changes.
pub fn import_sms(
    conn: &mut Connection,
    path: &Path,
    conversation_id: Option<&str>,
    title: Option<&str>,
    mut options: ImportOptions,
) -> Result<SmsImport> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut entries: Vec<SmsEntry> = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse SMS JSON {}", path.display()))?;
    if entries.is_empty() {
        bail!("SMS file contains no messages");
    }
    entries.sort_by_key(|entry| entry.ts);

    let source_sha = sha256_hex(&bytes);
    ensure_private_dir(&options.data_dir)?;
    let source_dir = options.data_dir.join("sms-sources");
    ensure_private_dir(&source_dir)?;
    let stored_source = source_dir.join(format!("{source_sha}.json"));
    if !stored_source.exists() {
        std::fs::write(&stored_source, &bytes)
            .with_context(|| format!("preserve {}", stored_source.display()))?;
        crate::util::restrict_permissions(&stored_source, 0o600);
    }

    let conversation_id = conversation_id
        .map(str::to_string)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("thread");
            let safe: String = stem
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() {
                        ch.to_ascii_lowercase()
                    } else {
                        '-'
                    }
                })
                .collect();
            format!("sms-{}", safe.trim_matches('-'))
        });
    let title = title
        .map(str::to_string)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("SMS thread ({} messages)", entries.len()));

    let mut ids = Vec::with_capacity(entries.len());
    let mut duplicate_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for entry in &entries {
        let digest = sha256_hex(
            format!(
                "{}|{}|{}|{}",
                entry.ts,
                entry.side,
                entry.text,
                entry.src.as_deref().unwrap_or("")
            )
            .as_bytes(),
        );
        let base = format!("sms-{}", &digest[..16]);
        let count = duplicate_counts.entry(base.clone()).or_insert(0);
        let id = if *count == 0 {
            base
        } else {
            format!("{base}-{count}")
        };
        *count += 1;
        ids.push(id);
    }

    let first_ts = entries.first().map(|entry| entry.ts as f64 / 1000.0);
    let last_ts = entries.last().map(|entry| entry.ts as f64 / 1000.0);
    let mut mapping = Map::new();
    mapping.insert(
        "root".to_string(),
        json!({
            "id": "root",
            "message": Value::Null,
            "parent": Value::Null,
            "children": ids.first().cloned().into_iter().collect::<Vec<_>>(),
        }),
    );

    for (index, entry) in entries.iter().enumerate() {
        let node_id = &ids[index];
        let parent = if index == 0 {
            "root".to_string()
        } else {
            ids[index - 1].clone()
        };
        let children: Vec<String> = ids.get(index + 1).cloned().into_iter().collect();
        let is_you = entry.side.eq_ignore_ascii_case("you");
        let role = if is_you { "user" } else { "contact" };
        let author_name = if is_you { "You" } else { "Addi" };
        mapping.insert(
            node_id.clone(),
            json!({
                "id": node_id,
                "parent": parent,
                "children": children,
                "message": {
                    "id": node_id,
                    "author": { "role": role, "name": author_name },
                    "create_time": entry.ts as f64 / 1000.0,
                    "update_time": entry.ts as f64 / 1000.0,
                    "content": {
                        "content_type": "text",
                        "parts": [entry.text],
                    },
                    "metadata": {
                        "source": "sms",
                        "phone": entry.src,
                        "side": entry.side,
                        "ts_ms": entry.ts,
                    },
                },
            }),
        );
    }

    let conversation = json!({
        "id": conversation_id,
        "title": title,
        "create_time": first_ts,
        "update_time": last_ts,
        "current_node": ids.last(),
        "conversation_template_id": "sms",
        "is_archived": false,
        "mapping": mapping,
    });

    ensure_private_dir(&options.data_dir)?;
    let temp = tempfile::Builder::new()
        .prefix("breadmemory-sms-")
        .tempdir_in(&options.data_dir)
        .context("create temporary SMS import directory")?;
    let conversations_path = temp.path().join("conversations.json");
    std::fs::write(
        &conversations_path,
        serde_json::to_vec(&json!([conversation]))?,
    )
    .with_context(|| format!("write {}", conversations_path.display()))?;

    options.extract_attachments = false;
    options.source_label = Some(path.display().to_string());
    options.progress = true;
    let report = importer::import(conn, temp.path(), &options)?;

    Ok(SmsImport {
        report,
        conversation_id,
        messages: entries.len(),
        first_ts,
        last_ts,
        stored_source,
    })
}
