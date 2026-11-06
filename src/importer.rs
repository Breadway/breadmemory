use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Transaction};

use crate::chunk::{chunk_text, DEFAULT_MAX_CHARS, DEFAULT_OVERLAP};
use crate::db;
use crate::export::{split_json_array, ExportArchive, RawConversation, RawMessage};
use crate::text::{extract_attachment_refs, extract_text, meta_bool, meta_str};
use crate::util::{ensure_private_dir, normalize_id, now_iso, restrict_permissions, sha256_hex};

#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub force: bool,
    pub extract_attachments: bool,
    pub data_dir: PathBuf,
    pub attachment_dir: PathBuf,
}

impl ImportOptions {
    pub fn new(data_dir: PathBuf) -> Self {
        let attachment_dir = data_dir.join("attachments");
        Self {
            force: false,
            extract_attachments: true,
            data_dir,
            attachment_dir,
        }
    }
}

#[derive(Debug, Default)]
pub struct ImportReport {
    pub source_path: String,
    pub source_kind: String,
    pub source_sha256: String,
    pub conversations_seen: usize,
    pub conversations_new: usize,
    pub conversations_updated: usize,
    pub conversations_unchanged: usize,
    pub conversations_failed: usize,
    pub messages_written: usize,
    pub chunks_written: usize,
    pub attachments_linked: usize,
    pub attachments_extracted: usize,
    pub errors: Vec<String>,
    pub elapsed_ms: u128,
}

impl ImportReport {
    fn note_error(&mut self, message: String) {
        if self.errors.len() < 50 {
            self.errors.push(message);
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "conversations_seen": self.conversations_seen,
            "conversations_new": self.conversations_new,
            "conversations_updated": self.conversations_updated,
            "conversations_unchanged": self.conversations_unchanged,
            "conversations_failed": self.conversations_failed,
            "messages_written": self.messages_written,
            "chunks_written": self.chunks_written,
            "attachments_linked": self.attachments_linked,
            "attachments_extracted": self.attachments_extracted,
            "errors": self.errors,
            "elapsed_ms": self.elapsed_ms,
        })
    }
}

enum Outcome {
    New {
        messages: usize,
        chunks: usize,
        attachments: usize,
    },
    Updated {
        messages: usize,
        chunks: usize,
        attachments: usize,
    },
    Unchanged,
}

#[derive(Debug)]
struct Node {
    parent: Option<String>,
    children: Vec<String>,
    message: Option<RawMessage>,
    create_time: f64,
}

struct FileIndex {
    exact: HashMap<String, String>,
    all: Vec<(String, String)>,
}

impl FileIndex {
    fn build(names: &[String]) -> Self {
        let mut exact = HashMap::new();
        let mut all = Vec::new();
        for name in names {
            let base = name.rsplit('/').next().unwrap_or(name);
            if base == "conversations.json" || base.starts_with("conversations-") {
                continue;
            }
            let full = normalize_id(base);
            let stem = base.rsplit_once('.').map(|(s, _)| s).unwrap_or(base);
            let stem_norm = normalize_id(stem);
            exact.entry(full).or_insert_with(|| name.clone());
            exact
                .entry(stem_norm.clone())
                .or_insert_with(|| name.clone());
            all.push((stem_norm, name.clone()));
        }
        all.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        Self { exact, all }
    }

    fn resolve(&self, file_id: &str) -> Option<String> {
        let needle = normalize_id(file_id);
        if needle.is_empty() {
            return None;
        }
        if let Some(path) = self.exact.get(&needle) {
            return Some(path.clone());
        }
        if needle.len() >= 6 {
            for (stem, path) in &self.all {
                if stem.starts_with(&needle) {
                    return Some(path.clone());
                }
            }
            for (stem, path) in &self.all {
                if stem.contains(&needle) {
                    return Some(path.clone());
                }
            }
        }
        None
    }
}

pub fn import(conn: &mut Connection, path: &Path, options: &ImportOptions) -> Result<ImportReport> {
    let started = Instant::now();
    let mut archive = ExportArchive::open(path)?;
    let conversation_files = archive.conversation_files();
    if conversation_files.is_empty() {
        bail!(
            "{} contains no conversations.json or conversations-NNN.json",
            path.display()
        );
    }
    let fingerprint = archive.fingerprint(path)?;
    let file_index = FileIndex::build(archive.names());
    let source_kind = archive.kind().as_str().to_string();
    let started_at = now_iso();

    conn.execute(
        "INSERT INTO imports(source_path, source_kind, source_sha256, started_at, status)
         VALUES (?1, ?2, ?3, ?4, 'running')",
        params![
            path.display().to_string(),
            source_kind,
            fingerprint,
            started_at
        ],
    )?;
    let import_id = conn.last_insert_rowid();

    let mut report = ImportReport {
        source_path: path.display().to_string(),
        source_kind,
        source_sha256: fingerprint.clone(),
        ..Default::default()
    };

    for file_name in &conversation_files {
        let bytes = archive
            .read(file_name)
            .with_context(|| format!("read conversation file {file_name}"))?;
        let elements = split_json_array(&bytes)
            .with_context(|| format!("parse conversation array in {file_name}"))?;

        let tx = conn.transaction()?;
        for (index, element) in elements.iter().enumerate() {
            report.conversations_seen += 1;
            let conversation: RawConversation = match serde_json::from_slice(element) {
                Ok(value) => value,
                Err(err) => {
                    report.conversations_failed += 1;
                    report.note_error(format!(
                        "{file_name}[{index}]: invalid conversation JSON: {err}"
                    ));
                    continue;
                }
            };
            tx.execute_batch("SAVEPOINT conversation_import")?;
            let outcome = process_conversation(
                &tx,
                conversation,
                element,
                import_id,
                &file_index,
                options.force,
            );
            match outcome {
                Ok(outcome) => {
                    tx.execute_batch("RELEASE conversation_import")?;
                    match outcome {
                        Outcome::New {
                            messages,
                            chunks,
                            attachments,
                        } => {
                            report.conversations_new += 1;
                            report.messages_written += messages;
                            report.chunks_written += chunks;
                            report.attachments_linked += attachments;
                        }
                        Outcome::Updated {
                            messages,
                            chunks,
                            attachments,
                        } => {
                            report.conversations_updated += 1;
                            report.messages_written += messages;
                            report.chunks_written += chunks;
                            report.attachments_linked += attachments;
                        }
                        Outcome::Unchanged => report.conversations_unchanged += 1,
                    }
                }
                Err(err) => {
                    // Roll back this conversation only. A malformed or failed
                    // reimport must not leave a half-replaced conversation.
                    tx.execute_batch(
                        "ROLLBACK TO conversation_import; RELEASE conversation_import",
                    )?;
                    report.conversations_failed += 1;
                    report.note_error(format!("{file_name}[{index}]: {err:#}"));
                }
            }
        }
        tx.commit()?;
    }

    if options.extract_attachments {
        if let Err(err) = extract_pending_attachments(conn, &mut archive, options, &mut report) {
            report.note_error(format!("attachment extraction: {err:#}"));
        }
    }

    report.elapsed_ms = started.elapsed().as_millis();
    conn.execute(
        "UPDATE imports SET finished_at = ?1, status = 'complete',
            conversations_seen = ?2, conversations_new = ?3, conversations_updated = ?4,
            conversations_unchanged = ?5, conversations_failed = ?6,
            messages_written = ?7, chunks_written = ?8, attachments_written = ?9
         WHERE id = ?10",
        params![
            now_iso(),
            report.conversations_seen as i64,
            report.conversations_new as i64,
            report.conversations_updated as i64,
            report.conversations_unchanged as i64,
            report.conversations_failed as i64,
            report.messages_written as i64,
            report.chunks_written as i64,
            report.attachments_extracted as i64,
            import_id
        ],
    )?;
    Ok(report)
}

fn process_conversation(
    tx: &Transaction<'_>,
    conversation: RawConversation,
    raw_bytes: &[u8],
    import_id: i64,
    file_index: &FileIndex,
    force: bool,
) -> Result<Outcome> {
    let conversation_id = conversation
        .id
        .clone()
        .or_else(|| conversation.conversation_id.clone())
        .context("conversation has no id")?;
    let raw_hash = sha256_hex(raw_bytes);

    let existing: Option<(String, Option<i64>)> = tx
        .query_row(
            "SELECT raw_sha256, first_import_id FROM conversations WHERE id = ?1",
            params![conversation_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;

    if let Some((stored_hash, _)) = existing.as_ref() {
        if stored_hash == &raw_hash && !force {
            return Ok(Outcome::Unchanged);
        }
    }

    let first_import_id = existing
        .as_ref()
        .and_then(|(_, first)| *first)
        .unwrap_or(import_id);
    let is_new = existing.is_none();

    if !is_new {
        tx.execute(
            "DELETE FROM conversations WHERE id = ?1",
            params![conversation_id],
        )?;
    }

    let title = conversation
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .unwrap_or("Untitled")
        .to_string();

    let mut create_time = conversation.create_time;
    let mut update_time = conversation.update_time;
    if create_time.is_none() {
        create_time = conversation
            .mapping
            .values()
            .filter_map(|node| node.message.as_ref().and_then(|m| m.create_time))
            .fold(None, |acc: Option<f64>, value| {
                Some(acc.map_or(value, |current| current.min(value)))
            });
    }
    if update_time.is_none() {
        update_time = conversation
            .mapping
            .values()
            .filter_map(|node| node.message.as_ref().and_then(|m| m.create_time))
            .fold(None, |acc: Option<f64>, value| {
                Some(acc.map_or(value, |current| current.max(value)))
            });
    }

    tx.execute(
        "INSERT INTO conversations
            (id, title, create_time, update_time, current_node, default_model_slug, gizmo_id,
             conversation_template_id, is_archived, is_starred, memory_scope, raw_sha256,
             first_import_id, last_import_id, imported_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            conversation_id,
            title,
            create_time,
            update_time,
            conversation.current_node,
            conversation.default_model_slug,
            conversation.gizmo_id,
            conversation.conversation_template_id,
            conversation.is_archived.unwrap_or(false) as i64,
            conversation.is_starred.unwrap_or(false) as i64,
            conversation.memory_scope,
            raw_hash,
            first_import_id,
            import_id,
            now_iso()
        ],
    )?;

    let compressed = zstd::stream::encode_all(raw_bytes, 3).context("compress raw conversation")?;
    tx.execute(
        "INSERT INTO conversation_raw(conversation_id, encoding, import_id, data)
         VALUES (?1, 'zstd', ?2, ?3)",
        params![conversation_id, import_id, compressed],
    )?;

    let mut nodes: HashMap<String, Node> = HashMap::new();
    for (key, node) in conversation.mapping {
        let node_id = node.id.unwrap_or(key);
        let create_time = node
            .message
            .as_ref()
            .and_then(|m| m.create_time)
            .unwrap_or(0.0);
        nodes.insert(
            node_id.clone(),
            Node {
                parent: node.parent,
                children: node.children,
                message: node.message,
                create_time,
            },
        );
    }

    // Legacy exports used a flat messages array with no tree.
    if nodes.is_empty() && !conversation.messages.is_empty() {
        let mut previous: Option<String> = None;
        for (index, message) in conversation.messages.into_iter().enumerate() {
            let node_id = message
                .id
                .clone()
                .unwrap_or_else(|| format!("legacy-{index}"));
            let create_time = message.create_time.unwrap_or(0.0);
            let parent = previous.clone();
            if let Some(parent_id) = parent.as_ref() {
                if let Some(parent_node) = nodes.get_mut(parent_id) {
                    parent_node.children.push(node_id.clone());
                }
            }
            previous = Some(node_id.clone());
            nodes.insert(
                node_id,
                Node {
                    parent,
                    children: Vec::new(),
                    message: Some(message),
                    create_time,
                },
            );
        }
    }

    if nodes.is_empty() {
        bail!("conversation has neither a mapping nor messages");
    }

    let mut ids: Vec<String> = nodes.keys().cloned().collect();
    ids.sort_by(|a, b| {
        nodes[a]
            .create_time
            .partial_cmp(&nodes[b].create_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });

    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    for id in &ids {
        children.entry(id.clone()).or_default();
    }
    for id in &ids {
        for child in &nodes[id].children {
            if nodes.contains_key(child) {
                let list = children.get_mut(id).expect("child map entry");
                if !list.contains(child) {
                    list.push(child.clone());
                }
            }
        }
    }
    for id in &ids {
        if let Some(parent) = nodes[id].parent.as_ref() {
            if nodes.contains_key(parent) {
                let list = children.get_mut(parent).expect("parent entry");
                if !list.contains(id) {
                    list.push(id.clone());
                }
            }
        }
    }

    let roots: Vec<String> = ids
        .iter()
        .filter(|id| match nodes[*id].parent.as_ref() {
            Some(parent) => !nodes.contains_key(parent),
            None => true,
        })
        .cloned()
        .collect();

    let mut order: Vec<(String, i64)> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    for root in &roots {
        walk_tree(root, 0, &children, &mut visited, &mut order);
    }
    for id in &ids {
        if !visited.contains(id) {
            walk_tree(id, 0, &children, &mut visited, &mut order);
        }
    }

    let mut active: HashSet<String> = HashSet::new();
    let mut cursor = conversation
        .current_node
        .clone()
        .filter(|node| nodes.contains_key(node))
        .or_else(|| {
            ids.iter()
                .filter(|id| children.get(*id).map(|c| c.is_empty()).unwrap_or(true))
                .max_by(|a, b| {
                    nodes[*a]
                        .create_time
                        .partial_cmp(&nodes[*b].create_time)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
        });
    while let Some(node_id) = cursor {
        if !active.insert(node_id.clone()) {
            break;
        }
        cursor = nodes.get(&node_id).and_then(|node| node.parent.clone());
    }

    let node_message_ids: HashMap<String, String> = nodes
        .iter()
        .filter_map(|(id, node)| {
            node.message
                .as_ref()
                .map(|message| (id.clone(), message.id.clone().unwrap_or_else(|| id.clone())))
        })
        .collect();

    let mut edge_stmt = tx.prepare(
        "INSERT OR IGNORE INTO message_edges(conversation_id, parent_node_id, child_node_id, child_index)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    for id in &ids {
        for (index, child) in nodes[id].children.iter().enumerate() {
            edge_stmt.execute(params![conversation_id, id, child, index as i64])?;
        }
        let mut index = nodes[id].children.len() as i64;
        for child in &children[id] {
            if !nodes[id].children.contains(child) {
                edge_stmt.execute(params![conversation_id, id, child, index])?;
                index += 1;
            }
        }
    }
    drop(edge_stmt);

    let mut message_stmt = tx.prepare(
        "INSERT INTO messages
            (conversation_id, node_id, message_id, parent_node_id, parent_message_id,
             author_role, author_name, create_time, update_time, status, end_turn, weight,
             recipient, channel, content_type, model_slug, is_hidden, hidden_reason,
             is_on_active_branch, depth, ord, text, metadata_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                 ?17, ?18, ?19, ?20, ?21, ?22, ?23)",
    )?;
    let mut chunk_stmt = tx.prepare(
        "INSERT INTO chunks
            (conversation_id, node_id, message_id, chunk_index, start_offset, end_offset,
             text, title, role, hidden)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;
    let mut attachment_stmt = tx.prepare(
        "INSERT OR IGNORE INTO attachments
            (file_id, conversation_id, node_id, name, mime_type, size_bytes, source,
             asset_pointer, archive_path, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;

    let mut message_count = 0usize;
    let mut active_count = 0usize;
    let mut chunk_count = 0usize;
    let mut attachment_count = 0usize;

    for (ord, (node_id, depth)) in order.iter().enumerate() {
        let node = &nodes[node_id];
        let Some(message) = node.message.as_ref() else {
            continue;
        };
        let role = message
            .author
            .as_ref()
            .and_then(|author| author.role.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let author_name = message
            .author
            .as_ref()
            .and_then(|author| author.name.clone());
        let content_type = message
            .content
            .as_ref()
            .and_then(|content| content.content_type.clone())
            .unwrap_or_else(|| "text".to_string());
        let hidden_reason = hidden_reason(message, &role, &content_type);
        let is_hidden = hidden_reason.is_some();
        let on_active = active.contains(node_id);
        if on_active {
            active_count += 1;
        }
        let message_id = message.id.clone().unwrap_or_else(|| node_id.clone());
        let parent_message_id = node
            .parent
            .as_ref()
            .and_then(|parent| node_message_ids.get(parent))
            .cloned();
        let text = extract_text(message);
        let metadata_json = message.metadata.as_ref().and_then(metadata_json_string);

        message_stmt.execute(params![
            conversation_id,
            node_id,
            message_id,
            node.parent,
            parent_message_id,
            role,
            author_name,
            message.create_time,
            message.update_time,
            message.status,
            message.end_turn.map(|b| b as i64),
            message.weight,
            message.recipient,
            message.channel,
            content_type,
            meta_str(message.metadata.as_ref(), "model_slug"),
            is_hidden as i64,
            hidden_reason,
            on_active as i64,
            depth,
            ord as i64,
            text,
            metadata_json
        ])?;
        message_count += 1;

        for chunk in chunk_text(&text, DEFAULT_MAX_CHARS, DEFAULT_OVERLAP) {
            chunk_stmt.execute(params![
                conversation_id,
                node_id,
                message_id,
                chunk.index as i64,
                chunk.start as i64,
                chunk.end as i64,
                chunk.text,
                title,
                role,
                is_hidden as i64
            ])?;
            chunk_count += 1;
        }

        for reference in extract_attachment_refs(message, node_id) {
            let archive_path = file_index.resolve(&reference.file_id);
            let status = if archive_path.is_some() {
                "referenced"
            } else {
                "missing"
            };
            let inserted = attachment_stmt.execute(params![
                reference.file_id,
                conversation_id,
                reference.node_id,
                reference.name,
                reference.mime_type,
                reference.size_bytes,
                reference.source,
                reference.asset_pointer,
                archive_path,
                status
            ])?;
            attachment_count += inserted;
        }
    }

    drop(message_stmt);
    drop(chunk_stmt);
    drop(attachment_stmt);

    tx.execute(
        "UPDATE conversations SET message_count = ?1, active_message_count = ?2,
            chunk_count = ?3, attachment_count = ?4 WHERE id = ?5",
        params![
            message_count as i64,
            active_count as i64,
            chunk_count as i64,
            attachment_count as i64,
            conversation_id
        ],
    )?;

    Ok(if is_new {
        Outcome::New {
            messages: message_count,
            chunks: chunk_count,
            attachments: attachment_count,
        }
    } else {
        Outcome::Updated {
            messages: message_count,
            chunks: chunk_count,
            attachments: attachment_count,
        }
    })
}

fn walk_tree(
    start: &str,
    depth: i64,
    children: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    order: &mut Vec<(String, i64)>,
) {
    let mut stack = vec![(start.to_string(), depth)];
    while let Some((node_id, depth)) = stack.pop() {
        if !visited.insert(node_id.clone()) {
            continue;
        }
        order.push((node_id.clone(), depth));
        if let Some(kids) = children.get(&node_id) {
            for child in kids.iter().rev() {
                if !visited.contains(child) {
                    stack.push((child.clone(), depth + 1));
                }
            }
        }
    }
}

fn hidden_reason(message: &RawMessage, role: &str, content_type: &str) -> Option<&'static str> {
    if role == "system" {
        return Some("system");
    }
    if message.weight == Some(0.0) {
        return Some("weight_zero");
    }
    if meta_bool(
        message.metadata.as_ref(),
        "is_visually_hidden_from_conversation",
    ) {
        return Some("visually_hidden");
    }
    match content_type {
        "user_editable_context" | "model_editable_context" => Some("editable_context"),
        "thoughts" => Some("reasoning"),
        _ => None,
    }
}

fn metadata_json_string(value: &serde_json::Value) -> Option<String> {
    let text = serde_json::to_string(value).ok()?;
    if text.chars().count() > 32_000 {
        None
    } else {
        Some(text)
    }
}

fn extract_pending_attachments(
    conn: &Connection,
    archive: &mut ExportArchive,
    options: &ImportOptions,
    report: &mut ImportReport,
) -> Result<()> {
    let pending: Vec<(String, Option<String>)> = {
        let mut stmt = conn.prepare(
            "SELECT file_id, MAX(archive_path) FROM attachments
             WHERE status = 'referenced' GROUP BY file_id",
        )?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut values = Vec::new();
        for row in rows {
            values.push(row?);
        }
        values
    };

    for (file_id, archive_path) in pending {
        let Some(archive_path) = archive_path else {
            conn.execute(
                "UPDATE attachments SET status = 'missing' WHERE file_id = ?1 AND status = 'referenced'",
                params![file_id],
            )?;
            continue;
        };
        let bytes = match archive.read(&archive_path) {
            Ok(bytes) => bytes,
            Err(err) => {
                report.note_error(format!("attachment {file_id} at {archive_path}: {err:#}"));
                conn.execute(
                    "UPDATE attachments SET status = 'missing' WHERE file_id = ?1 AND status = 'referenced'",
                    params![file_id],
                )?;
                continue;
            }
        };
        let digest = sha256_hex(&bytes);
        let extension = sanitized_extension(&archive_path);
        let subdir = &digest[..2];
        let dir = options.attachment_dir.join(subdir);
        ensure_private_dir(&dir)?;
        let destination = dir.join(format!("{digest}.{extension}"));
        if !destination.exists() {
            std::fs::write(&destination, &bytes)
                .with_context(|| format!("write {}", destination.display()))?;
            restrict_permissions(&destination, 0o600);
        }
        let relative = destination
            .strip_prefix(&options.data_dir)
            .unwrap_or(&destination)
            .to_string_lossy()
            .replace('\\', "/");
        conn.execute(
            "UPDATE attachments SET status = 'extracted', sha256 = ?1, stored_path = ?2
             WHERE file_id = ?3 AND status = 'referenced'",
            params![digest, relative, file_id],
        )?;
        report.attachments_extracted += 1;
    }
    Ok(())
}

fn sanitized_extension(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            ext.chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .take(12)
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|ext| !ext.is_empty())
        .unwrap_or_else(|| "bin".to_string())
}

pub struct ReindexReport {
    pub messages: u64,
    pub chunks: u64,
}

type ReindexRow = (
    i64,
    String,
    String,
    Option<String>,
    String,
    bool,
    String,
    String,
);

/// Rebuild chunks and the FTS index from the preserved message rows. The
/// current context layer is deliberately untouched.
pub fn reindex(conn: &mut Connection, db_path: &Path) -> Result<ReindexReport> {
    conn.execute("DELETE FROM chunks", [])?;
    let reader = db::open_secondary(db_path)?;
    let mut last_rowid: i64 = 0;
    let mut message_total: u64 = 0;
    let mut chunk_total: u64 = 0;

    loop {
        let batch: Vec<ReindexRow> = {
            let mut stmt = reader.prepare(
                "SELECT m.rowid, m.conversation_id, m.node_id, m.message_id, m.author_role,
                        m.is_hidden, m.text, c.title
                 FROM messages m JOIN conversations c ON c.id = m.conversation_id
                 WHERE m.rowid > ?1 ORDER BY m.rowid LIMIT 500",
            )?;
            let rows = stmt.query_map(params![last_rowid], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get::<_, i64>(5)? != 0,
                    row.get(6)?,
                    row.get(7)?,
                ))
            })?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row?);
            }
            values
        };
        if batch.is_empty() {
            break;
        }

        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks
                    (conversation_id, node_id, message_id, chunk_index, start_offset, end_offset,
                     text, title, role, hidden)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for (rowid, conversation_id, node_id, message_id, role, hidden, text, title) in &batch {
                last_rowid = *rowid;
                message_total += 1;
                for chunk in chunk_text(text, DEFAULT_MAX_CHARS, DEFAULT_OVERLAP) {
                    stmt.execute(params![
                        conversation_id,
                        node_id,
                        message_id,
                        chunk.index as i64,
                        chunk.start as i64,
                        chunk.end as i64,
                        chunk.text,
                        title,
                        role,
                        *hidden as i64
                    ])?;
                    chunk_total += 1;
                }
            }
        }
        tx.commit()?;
    }

    Ok(ReindexReport {
        messages: message_total,
        chunks: chunk_total,
    })
}

pub fn rebuild_fts(conn: &Connection) -> Result<()> {
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])?;
    Ok(())
}
