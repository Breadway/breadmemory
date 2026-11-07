use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::util::{now_iso, now_unix, truncate_chars};

pub const CATEGORIES: &[&str] = &[
    "identity",
    "preference",
    "project",
    "hardware",
    "decision",
    "relationship",
    "routine",
    "general",
];
pub const CONFIDENCES: &[&str] = &["stated", "inferred", "confirmed"];
pub const STATUSES: &[&str] = &["active", "proposed", "superseded", "retracted"];

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "was", "were",
    "are", "but", "not", "you", "your", "his", "her", "their", "our", "its", "into", "onto",
    "about", "over", "under", "than", "then", "there", "here", "when", "where", "what", "which",
    "who", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "just", "also", "now", "use", "uses", "used",
    "using", "get", "got", "make", "made", "like", "want", "need", "one", "two", "way", "thing",
    "does", "did", "doing", "done", "is", "of", "to", "in", "on", "at", "by", "as", "it", "be",
    "or", "an", "a", "i", "me", "my", "we", "us",
];

#[derive(Debug, Clone)]
pub struct ContextFact {
    pub id: i64,
    pub key: String,
    pub value: String,
    pub category: String,
    pub status: String,
    pub confidence: String,
    pub source_kind: String,
    pub source_conversation_id: Option<String>,
    pub source_message_id: Option<String>,
    pub source_quote: Option<String>,
    pub note: Option<String>,
    pub pinned: bool,
    pub external_system: Option<String>,
    pub external_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub verified_at: Option<String>,
    pub source_title: Option<String>,
    pub source_date: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ScoredFact {
    pub fact: ContextFact,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct SetRequest {
    pub key: String,
    pub value: String,
    pub category: String,
    pub confidence: String,
    pub status: String,
    pub source_kind: String,
    pub source_conversation_id: Option<String>,
    pub source_message_id: Option<String>,
    pub source_quote: Option<String>,
    pub note: Option<String>,
    pub pinned: bool,
    pub actor: String,
}

impl SetRequest {
    pub fn manual(key: &str, value: &str, actor: &str) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
            category: "general".to_string(),
            confidence: "stated".to_string(),
            status: "active".to_string(),
            source_kind: "manual".to_string(),
            source_conversation_id: None,
            source_message_id: None,
            source_quote: None,
            note: None,
            pinned: false,
            actor: actor.to_string(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let key = self.key.trim();
        if key.is_empty() {
            bail!("key is required");
        }
        if key.chars().count() > 120 {
            bail!("key is longer than 120 characters");
        }
        if !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':'))
        {
            bail!("key may only contain letters, digits, '.', '-', '_' and ':'");
        }
        if self.value.trim().is_empty() {
            bail!("value is required");
        }
        if self.value.chars().count() > 4000 {
            bail!("value is longer than 4000 characters");
        }
        if !CATEGORIES.contains(&self.category.as_str()) {
            bail!(
                "unknown category '{}'; expected one of {}",
                self.category,
                CATEGORIES.join(", ")
            );
        }
        if !CONFIDENCES.contains(&self.confidence.as_str()) {
            bail!(
                "unknown confidence '{}'; expected one of {}",
                self.confidence,
                CONFIDENCES.join(", ")
            );
        }
        if !matches!(self.status.as_str(), "active" | "proposed") {
            bail!("status must be 'active' or 'proposed'");
        }
        if self.source_conversation_id.is_none() && self.source_quote.is_some() {
            bail!("a source quote needs a source conversation");
        }
        Ok(())
    }
}

pub fn normalize_key(key: &str) -> String {
    key.trim().to_ascii_lowercase()
}

/// Create or update an active fact. Only one active row exists per key; the
/// previous value is recorded in the audit trail.
pub fn set_fact(conn: &mut Connection, request: &SetRequest) -> Result<ContextFact> {
    request.validate()?;
    let key = normalize_key(&request.key);
    let now = now_iso();

    let tx = conn.transaction()?;
    // A proposal must never update the active fact: proposals are inert until
    // the user accepts them, even when the key already has a current value.
    let existing: Option<(i64, String)> = if request.status == "active" {
        tx.query_row(
            "SELECT id, value FROM current_context WHERE key = ?1 AND status = 'active'",
            params![key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
    } else {
        None
    };

    let id = if let Some((id, old_value)) = existing {
        tx.execute(
            "UPDATE current_context SET value = ?1, category = ?2, confidence = ?3,
                    source_kind = ?4, source_conversation_id = ?5, source_message_id = ?6,
                    source_quote = ?7, note = ?8, pinned = ?9, updated_at = ?10,
                    verified_at = CASE WHEN ?3 = 'confirmed' THEN ?10 ELSE verified_at END
             WHERE id = ?11",
            params![
                request.value.trim(),
                request.category,
                request.confidence,
                request.source_kind,
                request.source_conversation_id,
                request.source_message_id,
                request.source_quote,
                request.note,
                request.pinned as i64,
                now,
                id
            ],
        )?;
        record_event(
            &tx,
            id,
            &key,
            "updated",
            Some(&old_value),
            Some(request.value.trim()),
            &request.actor,
            None,
            &now,
        )?;
        id
    } else {
        tx.execute(
            "INSERT INTO current_context
                (key, value, category, status, confidence, source_kind,
                 source_conversation_id, source_message_id, source_quote, note, pinned,
                 created_at, updated_at, verified_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12,
                     CASE WHEN ?5 = 'confirmed' THEN ?12 ELSE NULL END)",
            params![
                key,
                request.value.trim(),
                request.category,
                request.status,
                request.confidence,
                request.source_kind,
                request.source_conversation_id,
                request.source_message_id,
                request.source_quote,
                request.note,
                request.pinned as i64,
                now
            ],
        )?;
        let id = tx.last_insert_rowid();
        record_event(
            &tx,
            id,
            &key,
            "created",
            None,
            Some(request.value.trim()),
            &request.actor,
            None,
            &now,
        )?;
        id
    };
    tx.commit()?;
    fact_by_id(conn, id)?.context("fact disappeared after write")
}

/// Promote the newest proposal for a key into current context. The previous
/// active fact, if any, becomes superseded.
pub fn accept_proposal(conn: &mut Connection, key: &str, actor: &str) -> Result<ContextFact> {
    let key = normalize_key(key);
    let now = now_iso();
    let tx = conn.transaction()?;

    let proposal: Option<(i64, String)> = tx
        .query_row(
            "SELECT id, value FROM current_context
             WHERE key = ?1 AND status = 'proposed' ORDER BY id DESC LIMIT 1",
            params![key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((proposal_id, value)) = proposal else {
        bail!("no proposed fact for key '{key}'");
    };

    let active: Option<(i64, String)> = tx
        .query_row(
            "SELECT id, value FROM current_context WHERE key = ?1 AND status = 'active'",
            params![key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((active_id, old_value)) = active {
        tx.execute(
            "UPDATE current_context SET status = 'superseded', updated_at = ?1 WHERE id = ?2",
            params![now, active_id],
        )?;
        record_event(
            &tx,
            active_id,
            &key,
            "superseded",
            Some(&old_value),
            Some(&value),
            actor,
            Some("replaced by accepted proposal"),
            &now,
        )?;
    }

    tx.execute(
        "UPDATE current_context SET status = 'active', updated_at = ?1, verified_at = ?1 WHERE id = ?2",
        params![now, proposal_id],
    )?;
    tx.execute(
        "UPDATE current_context SET status = 'superseded', updated_at = ?1
         WHERE key = ?2 AND status = 'proposed' AND id != ?3",
        params![now, key, proposal_id],
    )?;
    record_event(
        &tx,
        proposal_id,
        &key,
        "accepted",
        None,
        Some(&value),
        actor,
        None,
        &now,
    )?;
    tx.commit()?;
    fact_by_id(conn, proposal_id)?.context("proposal disappeared after accept")
}

pub fn retract(
    conn: &mut Connection,
    key: &str,
    actor: &str,
    reason: Option<&str>,
) -> Result<ContextFact> {
    let key = normalize_key(key);
    let now = now_iso();
    let tx = conn.transaction()?;
    let active: Option<(i64, String)> = tx
        .query_row(
            "SELECT id, value FROM current_context WHERE key = ?1 AND status = 'active'",
            params![key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((id, old_value)) = active else {
        bail!("no active fact for key '{key}'");
    };
    tx.execute(
        "UPDATE current_context SET status = 'retracted', updated_at = ?1 WHERE id = ?2",
        params![now, id],
    )?;
    record_event(
        &tx,
        id,
        &key,
        "retracted",
        Some(&old_value),
        None,
        actor,
        reason,
        &now,
    )?;
    tx.commit()?;
    fact_by_id(conn, id)?.context("fact disappeared after retract")
}

#[allow(clippy::too_many_arguments)]
fn record_event(
    conn: &Connection,
    context_id: i64,
    key: &str,
    action: &str,
    old_value: Option<&str>,
    new_value: Option<&str>,
    actor: &str,
    reason: Option<&str>,
    at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO current_context_events
            (context_id, key, action, old_value, new_value, actor, reason, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![context_id, key, action, old_value, new_value, actor, reason, at],
    )?;
    Ok(())
}

pub fn get_active(conn: &Connection, key: &str) -> Result<Option<ContextFact>> {
    let key = normalize_key(key);
    let mut facts = query_facts(
        conn,
        "WHERE c.key = ?1 AND c.status = 'active'",
        params![key],
    )?;
    Ok(facts.pop())
}

/// List facts by status. `include_statuses` defaults to active only.
pub fn list(
    conn: &Connection,
    include_statuses: &[&str],
    category: Option<&str>,
    query: Option<&str>,
) -> Result<Vec<ContextFact>> {
    let statuses: Vec<&str> = if include_statuses.is_empty() {
        vec!["active"]
    } else {
        include_statuses.to_vec()
    };
    let placeholders = vec!["?"; statuses.len()].join(", ");
    let sql = format!(
        "SELECT {FACT_COLUMNS} FROM current_context c
         LEFT JOIN conversations cv ON cv.id = c.source_conversation_id
         WHERE c.status IN ({placeholders}) {category_clause} {query_clause}
         ORDER BY c.pinned DESC, c.category, c.key, c.id",
        category_clause = if category.is_some() {
            "AND c.category = ?"
        } else {
            ""
        },
        query_clause = if query.is_some() {
            "AND (c.key LIKE ? OR c.value LIKE ? OR c.note LIKE ?)"
        } else {
            ""
        },
    );

    let mut values: Vec<Box<dyn rusqlite::ToSql>> = statuses
        .iter()
        .map(|s| Box::new(s.to_string()) as Box<dyn rusqlite::ToSql>)
        .collect();
    if let Some(category) = category {
        values.push(Box::new(category.to_string()));
    }
    let like = query.map(|q| format!("%{}%", q.trim()));
    if let Some(like) = like.as_ref() {
        values.push(Box::new(like.clone()));
        values.push(Box::new(like.clone()));
        values.push(Box::new(like.clone()));
    }
    let refs: Vec<&dyn rusqlite::ToSql> = values.iter().map(|v| v.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(refs.as_slice(), map_fact)?;
    let mut facts = Vec::new();
    for row in rows {
        facts.push(row?);
    }
    Ok(facts)
}

/// Rank facts for per-turn recall. Pinned facts are always eligible; other
/// facts must share terms with the query. This mirrors Odysseus' pinned plus
/// relevant-recall behaviour without embeddings.
pub fn recall(
    conn: &Connection,
    query: &str,
    limit: usize,
    include_proposed: bool,
) -> Result<Vec<ScoredFact>> {
    let statuses: &[&str] = if include_proposed {
        &["active", "proposed"]
    } else {
        &["active"]
    };
    let facts = list(conn, statuses, None, None)?;
    let query_terms = match_terms(query);
    let mut scored = Vec::new();

    for fact in facts {
        let fact_terms = match_terms(&format!(
            "{} {} {} {}",
            fact.key,
            fact.value,
            fact.category,
            fact.note.as_deref().unwrap_or("")
        ));
        let overlap = query_terms.intersection(&fact_terms).count() as f64;
        let union = query_terms.union(&fact_terms).count().max(1) as f64;
        let mut score = overlap / union;
        if fact.pinned {
            score += 0.15;
        }
        if score > 0.0 || fact.pinned {
            scored.push(ScoredFact { fact, score });
        }
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.fact.updated_at.cmp(&a.fact.updated_at))
    });
    scored.truncate(limit);
    Ok(scored)
}

fn match_terms(text: &str) -> HashSet<String> {
    let mut terms = HashSet::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            for lower in c.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            let token = std::mem::take(&mut current);
            if token.chars().count() >= 2 && !STOPWORDS.contains(&token.as_str()) {
                terms.insert(token);
            }
        }
    }
    if !current.is_empty() && current.chars().count() >= 2 && !STOPWORDS.contains(&current.as_str())
    {
        terms.insert(current);
    }
    terms
}

pub fn fact_by_id(conn: &Connection, id: i64) -> Result<Option<ContextFact>> {
    let mut facts = query_facts(conn, "WHERE c.id = ?1", params![id])?;
    Ok(facts.pop())
}

const FACT_COLUMNS: &str = "c.id, c.key, c.value, c.category, c.status, c.confidence,
    c.source_kind, c.source_conversation_id, c.source_message_id, c.source_quote,
    c.note, c.pinned, c.external_system, c.external_id,
    c.created_at, c.updated_at, c.verified_at, cv.title, cv.create_time";

fn query_facts(
    conn: &Connection,
    clause: &str,
    params: impl rusqlite::Params,
) -> Result<Vec<ContextFact>> {
    let sql = format!(
        "SELECT {FACT_COLUMNS} FROM current_context c
         LEFT JOIN conversations cv ON cv.id = c.source_conversation_id
         {clause} ORDER BY c.id"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params, map_fact)?;
    let mut facts = Vec::new();
    for row in rows {
        facts.push(row?);
    }
    Ok(facts)
}

fn map_fact(row: &rusqlite::Row<'_>) -> rusqlite::Result<ContextFact> {
    Ok(ContextFact {
        id: row.get(0)?,
        key: row.get(1)?,
        value: row.get(2)?,
        category: row.get(3)?,
        status: row.get(4)?,
        confidence: row.get(5)?,
        source_kind: row.get(6)?,
        source_conversation_id: row.get(7)?,
        source_message_id: row.get(8)?,
        source_quote: row.get(9)?,
        note: row.get(10)?,
        pinned: row.get::<_, i64>(11)? != 0,
        external_system: row.get(12)?,
        external_id: row.get(13)?,
        created_at: row.get(14)?,
        updated_at: row.get(15)?,
        verified_at: row.get(16)?,
        source_title: row.get(17)?,
        source_date: row.get(18)?,
    })
}

pub fn export_json(conn: &Connection) -> Result<Value> {
    let mut stmt = conn.prepare(
        "SELECT key, value, category, status, confidence, source_kind,
                source_conversation_id, source_message_id, source_quote, note, pinned,
                external_system, external_id,
                created_at, updated_at, verified_at
         FROM current_context ORDER BY key, id",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(json!({
            "key": row.get::<_, String>(0)?,
            "value": row.get::<_, String>(1)?,
            "category": row.get::<_, String>(2)?,
            "status": row.get::<_, String>(3)?,
            "confidence": row.get::<_, String>(4)?,
            "source_kind": row.get::<_, String>(5)?,
            "source_conversation_id": row.get::<_, Option<String>>(6)?,
            "source_message_id": row.get::<_, Option<String>>(7)?,
            "source_quote": row.get::<_, Option<String>>(8)?,
            "note": row.get::<_, Option<String>>(9)?,
            "pinned": row.get::<_, i64>(10)? != 0,
            "external_system": row.get::<_, Option<String>>(11)?,
            "external_id": row.get::<_, Option<String>>(12)?,
            "created_at": row.get::<_, String>(13)?,
            "updated_at": row.get::<_, String>(14)?,
            "verified_at": row.get::<_, Option<String>>(15)?,
        }))
    })?;
    let mut facts = Vec::new();
    for row in rows {
        facts.push(row?);
    }
    Ok(json!({
        "format": "breadmemory.current_context.v1",
        "exported_at": now_iso(),
        "facts": facts
    }))
}

pub fn import_json(
    conn: &mut Connection,
    document: &Value,
    actor: &str,
    replace: bool,
) -> Result<usize> {
    let facts = document
        .get("facts")
        .and_then(Value::as_array)
        .context("current context file has no 'facts' array")?;
    let existing: i64 =
        conn.query_row("SELECT COUNT(*) FROM current_context", [], |row| row.get(0))?;
    if existing > 0 && !replace {
        bail!("current context already has {existing} fact(s); pass --replace to restore over it");
    }

    let now = now_iso();
    let tx = conn.transaction()?;
    if replace {
        tx.execute("DELETE FROM current_context", [])?;
    }
    let mut count = 0usize;
    for fact in facts {
        let key = fact.get("key").and_then(Value::as_str).unwrap_or_default();
        let value = fact
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if key.trim().is_empty() || value.trim().is_empty() {
            continue;
        }
        let request = SetRequest {
            key: normalize_key(key),
            value: value.to_string(),
            category: fact
                .get("category")
                .and_then(Value::as_str)
                .unwrap_or("general")
                .to_string(),
            confidence: fact
                .get("confidence")
                .and_then(Value::as_str)
                .unwrap_or("stated")
                .to_string(),
            status: fact
                .get("status")
                .and_then(Value::as_str)
                .filter(|s| matches!(*s, "active" | "proposed"))
                .unwrap_or("active")
                .to_string(),
            source_kind: fact
                .get("source_kind")
                .and_then(Value::as_str)
                .unwrap_or("import")
                .to_string(),
            source_conversation_id: fact
                .get("source_conversation_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            source_message_id: fact
                .get("source_message_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            source_quote: fact
                .get("source_quote")
                .and_then(Value::as_str)
                .map(str::to_string),
            note: fact.get("note").and_then(Value::as_str).map(str::to_string),
            pinned: fact.get("pinned").and_then(Value::as_bool).unwrap_or(false),
            actor: actor.to_string(),
        };
        if request.validate().is_err() {
            continue;
        }
        tx.execute(
            "INSERT INTO current_context
                (key, value, category, status, confidence, source_kind,
                 source_conversation_id, source_message_id, source_quote, note, pinned,
                 external_system, external_id, created_at, updated_at, verified_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?14, ?15)",
            params![
                request.key,
                request.value,
                request.category,
                request.status,
                request.confidence,
                request.source_kind,
                request.source_conversation_id,
                request.source_message_id,
                request.source_quote,
                request.note,
                request.pinned as i64,
                fact.get("external_system").and_then(Value::as_str),
                fact.get("external_id").and_then(Value::as_str),
                fact.get("created_at")
                    .and_then(Value::as_str)
                    .unwrap_or(&now),
                fact.get("verified_at").and_then(Value::as_str),
            ],
        )?;
        let id = tx.last_insert_rowid();
        record_event(
            &tx,
            id,
            &request.key,
            "imported",
            None,
            Some(&request.value),
            actor,
            None,
            &now,
        )?;
        count += 1;
    }
    tx.commit()?;
    Ok(count)
}

#[derive(Debug, Default)]
pub struct OdysseusImportReport {
    pub source: String,
    pub imported: usize,
    pub updated: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

/// Import Odysseus' memory.json (or the legacy array format) into the current
/// context layer. Rows are keyed by Odysseus memory id, so re-running is
/// idempotent. Retracted or superseded breadmemory facts are never resurrected.
pub fn import_odysseus(
    conn: &mut Connection,
    path: &std::path::Path,
    actor: &str,
) -> Result<OdysseusImportReport> {
    let memory_file = if path.is_dir() {
        path.join("memory.json")
    } else {
        path.to_path_buf()
    };
    let bytes =
        std::fs::read(&memory_file).with_context(|| format!("read {}", memory_file.display()))?;
    let document: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", memory_file.display()))?;
    let items: &[Value] = match &document {
        Value::Array(items) => items,
        Value::Object(map) => map
            .get("memories")
            .and_then(Value::as_array)
            .context("memory.json object has no 'memories' array")?,
        _ => bail!("memory.json is not an array or object"),
    };

    let mut report = OdysseusImportReport {
        source: memory_file.display().to_string(),
        ..Default::default()
    };
    let now = now_iso();

    for item in items {
        if item.get("source").and_then(Value::as_str) == Some("breadmemory") {
            // Projection entries written by `sync-odysseus`; importing them
            // back would duplicate breadmemory facts.
            report.skipped += 1;
            continue;
        }
        let text = item
            .get("text")
            .and_then(Value::as_str)
            .or_else(|| item.get("content").and_then(Value::as_str))
            .map(str::trim)
            .unwrap_or_default();
        if text.is_empty() {
            report.skipped += 1;
            continue;
        }
        let source_category = item
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("fact");
        let category = map_odysseus_category(source_category);
        let external_id = item
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| crate::util::sha256_hex(text.as_bytes())[..16].to_string());
        let pinned = item.get("pinned").and_then(Value::as_bool).unwrap_or(false);
        let created_at = item
            .get("timestamp")
            .and_then(timestamp_value)
            .map(|ts| crate::util::ts_to_iso(Some(ts)).unwrap_or_else(|| now.clone()))
            .unwrap_or_else(|| now.clone());
        let note = odysseus_note(item);

        let existing: Option<(i64, String, String, String)> = conn
            .query_row(
                "SELECT id, status, value, key FROM current_context
                 WHERE external_system = 'odysseus' AND external_id = ?1",
                params![external_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;

        if let Some((id, status, old_value, key)) = existing {
            if !matches!(status.as_str(), "active" | "proposed") {
                report.skipped += 1;
                continue;
            }
            conn.execute(
                "UPDATE current_context SET value = ?1, category = ?2, pinned = ?3,
                        note = ?4, updated_at = ?5
                 WHERE id = ?6",
                params![text, category, pinned as i64, note, now, id],
            )?;
            if old_value != text {
                record_event(
                    conn,
                    id,
                    &key,
                    "synced",
                    Some(&old_value),
                    Some(text),
                    actor,
                    Some("odysseus memory import"),
                    &now,
                )?;
            }
            report.updated += 1;
            continue;
        }

        let key = odysseus_key(category, text, &external_id);
        let inserted = conn.execute(
            "INSERT INTO current_context
                (key, value, category, status, confidence, source_kind, note, pinned,
                 external_system, external_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'active', 'stated', 'odysseus', ?4, ?5, 'odysseus', ?6, ?7, ?8)",
            params![
                key,
                text,
                category,
                note,
                pinned as i64,
                external_id,
                created_at,
                now
            ],
        );
        match inserted {
            Ok(_) => {
                let id = conn.last_insert_rowid();
                record_event(
                    conn,
                    id,
                    &key,
                    "imported",
                    None,
                    Some(text),
                    actor,
                    Some("odysseus memory import"),
                    &now,
                )?;
                report.imported += 1;
            }
            Err(err) => {
                report.errors.push(format!("{external_id}: {err}"));
                report.skipped += 1;
            }
        }
    }
    Ok(report)
}

/// Export active facts as an Odysseus-compatible memory list. Uploading this
/// to Odysseus' /api/memory/import yields suggestions with text and category;
/// replacing memory.json preserves ids, pinned flags and timestamps.
pub fn export_odysseus(conn: &Connection) -> Result<Value> {
    let facts = list(conn, &["active"], None, None)?;
    let items: Vec<Value> = facts
        .iter()
        .map(|fact| {
            let id = fact
                .external_id
                .clone()
                .filter(|_| fact.external_system.as_deref() == Some("odysseus"))
                .unwrap_or_else(|| {
                    format!("bm-{}", &crate::util::sha256_hex(fact.key.as_bytes())[..16])
                });
            json!({
                "id": id,
                "text": fact.value,
                "category": odysseus_category(&fact.category),
                "source": "breadmemory",
                "owner": Value::Null,
                "session_id": Value::Null,
                "timestamp": unix_from_iso(&fact.created_at),
                "pinned": fact.pinned,
                "breadmemory": {
                    "key": fact.key,
                    "confidence": fact.confidence,
                    "source_kind": fact.source_kind,
                    "source_conversation_id": fact.source_conversation_id,
                    "source_message_id": fact.source_message_id,
                    "projected_text": fact.value
                }
            })
        })
        .collect();
    Ok(Value::Array(items))
}

#[derive(Debug, Default)]
pub struct OdysseusSyncReport {
    pub source: String,
    pub imported: usize,
    pub updated: usize,
    pub skipped: usize,
    pub native_edits: usize,
    pub projected: usize,
    pub native_kept: usize,
    pub errors: Vec<String>,
}

/// Two-way sync with Odysseus' memory store.
///
/// 1. Edits made to breadmemory projections in Odysseus are detected by
///    comparing the entry text with the `projected_text` breadmemory last
///    wrote, and are imported back into current context.
/// 2. New native memories (source != "breadmemory") are imported, keyed by
///    their Odysseus id.
/// 3. Active current-context facts are projected back into `memory.json` as
///    native-compatible entries, so Odysseus' existing per-turn injection keeps
///    working without upstream code changes.
///
/// Imported native rows keep their original id, so the projection replaces the
/// native copy instead of duplicating it. Retracting a fact removes its
/// projection on the next sync. If both sides change between syncs, the
/// Odysseus-side edit wins and is projected back.
pub fn sync_odysseus(
    conn: &mut Connection,
    path: &std::path::Path,
    owner: &str,
    actor: &str,
) -> Result<OdysseusSyncReport> {
    let memory_file = if path.is_dir() {
        path.join("memory.json")
    } else {
        path.to_path_buf()
    };
    let bytes =
        std::fs::read(&memory_file).with_context(|| format!("read {}", memory_file.display()))?;
    let document: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", memory_file.display()))?;
    let mut entries: Vec<Value> = match document {
        Value::Array(items) => items,
        Value::Object(map) => map
            .get("memories")
            .and_then(Value::as_array)
            .cloned()
            .context("memory.json object has no 'memories' array")?,
        _ => bail!("memory.json is not an array or object"),
    };

    let native_edits = apply_projected_edits(conn, &entries, actor)?;
    let imported = import_odysseus(conn, &memory_file, actor)?;
    let mut report = OdysseusSyncReport {
        source: imported.source,
        imported: imported.imported,
        updated: imported.updated,
        skipped: imported.skipped,
        native_edits,
        errors: imported.errors,
        ..Default::default()
    };

    let facts = list(conn, &["active"], None, None)?;
    let mut by_id: std::collections::HashMap<String, usize> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            entry
                .get("id")
                .and_then(Value::as_str)
                .map(|id| (id.to_string(), index))
        })
        .collect();
    let mut projected_ids = std::collections::HashSet::new();

    for fact in &facts {
        let id = fact
            .external_id
            .clone()
            .filter(|_| fact.external_system.as_deref() == Some("odysseus"))
            .unwrap_or_else(|| {
                format!("bm-{}", &crate::util::sha256_hex(fact.key.as_bytes())[..16])
            });
        let entry = json!({
            "id": id.clone(),
            "text": fact.value,
            "category": odysseus_category(&fact.category),
            "source": "breadmemory",
            "owner": if owner.trim().is_empty() { Value::Null } else { Value::String(owner.to_string()) },
            "session_id": Value::Null,
            "timestamp": unix_from_iso(&fact.updated_at),
            "pinned": fact.pinned,
            "breadmemory": {
                "key": fact.key,
                "confidence": fact.confidence,
                "source_kind": fact.source_kind,
                "source_conversation_id": fact.source_conversation_id,
                "source_message_id": fact.source_message_id,
                "projected_text": fact.value
            }
        });
        projected_ids.insert(id.clone());
        if let Some(index) = by_id.get(&id).copied() {
            entries[index] = entry;
        } else {
            by_id.insert(id, entries.len());
            entries.push(entry);
        }
    }

    entries.retain(|entry| {
        if entry.get("source").and_then(Value::as_str) != Some("breadmemory") {
            return true;
        }
        entry
            .get("id")
            .and_then(Value::as_str)
            .map(|id| projected_ids.contains(id))
            .unwrap_or(false)
    });

    report.projected = facts.len();
    report.native_kept = entries
        .iter()
        .filter(|entry| entry.get("source").and_then(Value::as_str) != Some("breadmemory"))
        .count();

    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&Value::Array(entries))?
    );
    let temp = memory_file.with_extension("json.breadmemory.tmp");
    std::fs::write(&temp, text).with_context(|| format!("write {}", temp.display()))?;
    std::fs::rename(&temp, &memory_file)
        .with_context(|| format!("replace {}", memory_file.display()))?;
    Ok(report)
}

fn apply_projected_edits(conn: &Connection, entries: &[Value], actor: &str) -> Result<usize> {
    let now = now_iso();
    let mut applied = 0usize;
    for entry in entries {
        if entry.get("source").and_then(Value::as_str) != Some("breadmemory") {
            continue;
        }
        let Some(breadmemory) = entry.get("breadmemory") else {
            continue;
        };
        let Some(key) = breadmemory.get("key").and_then(Value::as_str) else {
            continue;
        };
        let Some(projected) = breadmemory.get("projected_text").and_then(Value::as_str) else {
            continue;
        };
        let Some(text) = entry
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
        else {
            continue;
        };
        if text == projected {
            continue;
        }
        let Some(fact) = get_active(conn, key)? else {
            continue;
        };
        if fact.value == text {
            continue;
        }
        let pinned = entry
            .get("pinned")
            .and_then(Value::as_bool)
            .unwrap_or(fact.pinned);
        conn.execute(
            "UPDATE current_context SET value = ?1, pinned = ?2, updated_at = ?3 WHERE id = ?4",
            params![text, pinned as i64, now, fact.id],
        )?;
        record_event(
            conn,
            fact.id,
            &fact.key,
            "synced",
            Some(&fact.value),
            Some(text),
            actor,
            Some("edited in Odysseus memory"),
            &now,
        )?;
        applied += 1;
    }
    Ok(applied)
}

fn odysseus_note(item: &Value) -> Option<String> {
    let mut parts = vec!["odysseus memory".to_string()];
    if let Some(category) = item.get("category").and_then(Value::as_str) {
        parts.push(format!("category={category}"));
    }
    if let Some(source) = item.get("source").and_then(Value::as_str) {
        parts.push(format!("source={source}"));
    }
    if let Some(owner) = item.get("owner").and_then(Value::as_str) {
        parts.push(format!("owner={owner}"));
    }
    if let Some(session) = item.get("session_id").and_then(Value::as_str) {
        parts.push(format!("session={session}"));
    }
    if let Some(uses) = item.get("uses").and_then(Value::as_i64) {
        parts.push(format!("uses={uses}"));
    }
    Some(parts.join("; "))
}

fn timestamp_value(value: &Value) -> Option<f64> {
    let raw = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(text) => {
            if let Ok(number) = text.parse::<f64>() {
                number
            } else {
                return chrono::DateTime::parse_from_rfc3339(text)
                    .ok()
                    .map(|dt| dt.timestamp() as f64);
            }
        }
        _ => return None,
    };
    if raw > 100_000_000_000.0 {
        Some(raw / 1000.0)
    } else {
        Some(raw)
    }
}

fn unix_from_iso(value: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|dt| dt.timestamp())
        .unwrap_or_else(|_| now_unix() as i64)
}

fn map_odysseus_category(category: &str) -> &'static str {
    match category.to_ascii_lowercase().as_str() {
        "identity" | "personal" => "identity",
        "preference" | "preferences" => "preference",
        "project" | "projects" | "goal" | "goals" => "project",
        "hardware" | "device" | "devices" => "hardware",
        "decision" | "decisions" => "decision",
        "contact" | "contacts" | "relationship" | "relationships" | "people" | "person" => {
            "relationship"
        }
        "routine" | "routines" | "habit" | "habits" => "routine",
        _ => "general",
    }
}

fn odysseus_category(category: &str) -> &'static str {
    match category {
        "identity" => "identity",
        "preference" => "preference",
        "project" => "project",
        "relationship" => "contact",
        _ => "fact",
    }
}

fn odysseus_key(category: &str, text: &str, external_id: &str) -> String {
    let mut words: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if cleaned.len() >= 2 && !STOPWORDS.contains(&cleaned.as_str()) {
            words.push(cleaned);
        }
        if words.len() == 5 {
            break;
        }
    }
    let mut slug = words.join("_");
    if slug.is_empty() {
        slug = "memory".to_string();
    }
    if slug.len() > 40 {
        slug.truncate(40);
    }
    let suffix = &crate::util::sha256_hex(external_id.as_bytes())[..6];
    format!("odysseus.{category}.{slug}.{suffix}")
}

pub fn fact_to_json(fact: &ContextFact) -> Value {
    let (quote, quote_truncated) = match fact.source_quote.as_deref() {
        Some(q) => {
            let (text, truncated) = truncate_chars(q, 600);
            (Some(text), truncated)
        }
        None => (None, false),
    };
    json!({
        "id": fact.id,
        "key": fact.key,
        "value": fact.value,
        "category": fact.category,
        "confidence": fact.confidence,
        "status": fact.status,
        "pinned": fact.pinned,
        "updated_at": fact.updated_at,
        "created_at": fact.created_at,
        "verified_at": fact.verified_at,
        "source": {
            "kind": fact.source_kind,
            "conversation_id": fact.source_conversation_id,
            "conversation_title": fact.source_title,
            "conversation_created_at": crate::util::ts_to_iso(fact.source_date),
            "message_id": fact.source_message_id,
            "quote": quote,
            "quote_truncated": quote_truncated,
        },
        "external": {
            "system": fact.external_system,
            "id": fact.external_id,
        },
        "note": fact.note
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn fresh() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let conn = db::open(&dir.path().join("archive.db")).unwrap();
        (dir, conn)
    }

    #[test]
    fn set_update_and_supersede() {
        let (_dir, mut conn) = fresh();
        let mut request = SetRequest::manual("homelab.zfs.arc_max", "16GB", "tester");
        request.category = "hardware".into();
        let fact = set_fact(&mut conn, &request).unwrap();
        assert_eq!(fact.status, "active");

        request.value = "24GB".into();
        let updated = set_fact(&mut conn, &request).unwrap();
        assert_eq!(updated.value, "24GB");
        assert_eq!(updated.id, fact.id);

        let events: i64 = conn
            .query_row("SELECT COUNT(*) FROM current_context_events", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(events, 2);
    }

    #[test]
    fn proposals_need_acceptance() {
        let (_dir, mut conn) = fresh();
        let mut active = SetRequest::manual("project.foo", "current value", "tester");
        active.status = "active".into();
        set_fact(&mut conn, &active).unwrap();

        let mut proposed = SetRequest::manual("project.foo", "candidate value", "odysseus");
        proposed.status = "proposed".into();
        proposed.source_kind = "assistant".into();
        set_fact(&mut conn, &proposed).unwrap();

        // The active fact is unchanged while the proposal is pending.
        let current = get_active(&conn, "project.foo").unwrap().unwrap();
        assert_eq!(current.value, "current value");
        assert_eq!(list(&conn, &["proposed"], None, None).unwrap().len(), 1);

        let accepted = accept_proposal(&mut conn, "project.foo", "tester").unwrap();
        assert_eq!(accepted.status, "active");
        assert_eq!(accepted.value, "candidate value");
        assert_eq!(list(&conn, &[], None, None).unwrap().len(), 1);
    }

    #[test]
    fn provenance_survives_conversation_delete() {
        let (_dir, mut conn) = fresh();
        conn.execute(
            "INSERT INTO conversations(id, title, create_time, raw_sha256, imported_at)
             VALUES('conv-1', 'ZFS tuning', 1700000000.0, 'hash', 'now')",
            [],
        )
        .unwrap();
        let mut request = SetRequest::manual("homelab.zfs.arc_max", "16GB", "tester");
        request.source_conversation_id = Some("conv-1".into());
        request.source_kind = "conversation".into();
        let fact = set_fact(&mut conn, &request).unwrap();
        assert_eq!(fact.source_title.as_deref(), Some("ZFS tuning"));

        conn.execute("DELETE FROM conversations WHERE id = 'conv-1'", [])
            .unwrap();
        let fact = get_active(&conn, "homelab.zfs.arc_max").unwrap().unwrap();
        assert_eq!(fact.source_conversation_id.as_deref(), Some("conv-1"));
    }

    #[test]
    fn export_import_round_trip() {
        let (_dir, mut conn) = fresh();
        let mut request = SetRequest::manual("identity.name", "Riley", "tester");
        request.category = "identity".into();
        set_fact(&mut conn, &request).unwrap();
        let payload = export_json(&conn).unwrap();

        let dir2 = tempfile::tempdir().unwrap();
        let mut conn2 = db::open(&dir2.path().join("archive.db")).unwrap();
        let n = import_json(&mut conn2, &payload, "tester", true).unwrap();
        assert_eq!(n, 1);
        let fact = get_active(&conn2, "identity.name").unwrap().unwrap();
        assert_eq!(fact.value, "Riley");
    }

    #[test]
    fn odysseus_import_is_idempotent_and_respects_retraction() {
        let (dir, mut conn) = fresh();
        let memory_file = dir.path().join("memory.json");
        std::fs::write(
            &memory_file,
            r#"[
              {"id":"m1","text":"Riley lives in Sydney","category":"identity","timestamp":1700000000,"pinned":true},
              {"id":"m2","text":"Uses ZFS on the homelab","category":"fact","timestamp":1700000100}
            ]"#,
        )
        .unwrap();

        let report = import_odysseus(&mut conn, &memory_file, "tester").unwrap();
        assert_eq!(report.imported, 2);
        let report = import_odysseus(&mut conn, &memory_file, "tester").unwrap();
        assert_eq!(report.imported, 0);
        assert_eq!(report.updated, 2);

        let facts = list(&conn, &[], None, None).unwrap();
        assert_eq!(facts.len(), 2);
        assert!(facts.iter().any(|fact| fact.pinned));
        assert!(facts
            .iter()
            .all(|fact| fact.external_system.as_deref() == Some("odysseus")));

        let key = facts
            .iter()
            .find(|fact| fact.value.contains("Sydney"))
            .unwrap()
            .key
            .clone();
        retract(&mut conn, &key, "tester", Some("test")).unwrap();
        let report = import_odysseus(&mut conn, &memory_file, "tester").unwrap();
        assert_eq!(report.skipped, 1);
        assert_eq!(report.updated, 1);
    }

    #[test]
    fn recall_prioritises_pinned_and_keyword_matches() {
        let (_dir, mut conn) = fresh();
        let mut pinned = SetRequest::manual("identity.name", "Riley", "tester");
        pinned.category = "identity".into();
        pinned.pinned = true;
        set_fact(&mut conn, &pinned).unwrap();
        let mut other = SetRequest::manual("homelab.zfs.arc_max", "ARC max is 24GB", "tester");
        other.category = "hardware".into();
        set_fact(&mut conn, &other).unwrap();

        let results = recall(&conn, "zfs arc", 5, false).unwrap();
        assert_eq!(results[0].fact.key, "homelab.zfs.arc_max");
        assert!(results.iter().any(|result| result.fact.pinned));

        let exported = export_odysseus(&conn).unwrap();
        assert_eq!(exported.as_array().unwrap().len(), 2);
    }

    #[test]
    fn odysseus_sync_imports_and_projects_without_duplicates() {
        let (dir, mut conn) = fresh();
        let memory_file = dir.path().join("memory.json");
        std::fs::write(
            &memory_file,
            r#"[{"id":"native-1","text":"Riley lives in Sydney","category":"identity","owner":"breadway","timestamp":1700000000}]"#,
        )
        .unwrap();

        let report = sync_odysseus(&mut conn, &memory_file, "breadway", "tester").unwrap();
        assert_eq!(report.imported, 1);
        assert_eq!(report.projected, 1);
        assert_eq!(report.native_kept, 0);
        let entries: Vec<Value> =
            serde_json::from_slice(&std::fs::read(&memory_file).unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["id"], "native-1");
        assert_eq!(entries[0]["source"], "breadmemory");
        assert_eq!(entries[0]["owner"], "breadway");

        // Editing the projection in Odysseus is imported back.
        let imported_key = list(&conn, &[], None, None)
            .unwrap()
            .into_iter()
            .find(|fact| fact.value.contains("Sydney"))
            .unwrap()
            .key;
        let mut edited = entries.clone();
        edited[0]["text"] = serde_json::json!("Riley lives in Sydney, Australia");
        edited[0]["pinned"] = serde_json::json!(true);
        std::fs::write(&memory_file, serde_json::to_vec(&edited).unwrap()).unwrap();
        let report = sync_odysseus(&mut conn, &memory_file, "breadway", "tester").unwrap();
        assert_eq!(report.native_edits, 1);
        let fact = get_active(&conn, &imported_key).unwrap().unwrap();
        assert_eq!(fact.value, "Riley lives in Sydney, Australia");
        assert!(fact.pinned);

        // A breadmemory edit is projected outward and is not treated as a
        // native edit on the following pass.
        let mut update = SetRequest::manual(&imported_key, "Riley lives in Sydney", "tester");
        update.category = fact.category.clone();
        update.pinned = true;
        set_fact(&mut conn, &update).unwrap();
        let report = sync_odysseus(&mut conn, &memory_file, "breadway", "tester").unwrap();
        assert_eq!(report.native_edits, 0);
        let entries: Vec<Value> =
            serde_json::from_slice(&std::fs::read(&memory_file).unwrap()).unwrap();
        assert_eq!(entries[0]["text"], "Riley lives in Sydney");

        let mut request = SetRequest::manual("project.breadmemory", "Uses breadmemory", "tester");
        request.category = "project".to_string();
        request.pinned = true;
        set_fact(&mut conn, &request).unwrap();

        let report = sync_odysseus(&mut conn, &memory_file, "breadway", "tester").unwrap();
        assert_eq!(report.imported, 0);
        assert_eq!(report.projected, 2);
        let entries: Vec<Value> =
            serde_json::from_slice(&std::fs::read(&memory_file).unwrap()).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries
            .iter()
            .any(|entry| entry["text"] == "Uses breadmemory" && entry["pinned"] == true));

        // A new native memory is imported and projected with its original id.
        let mut native: Vec<Value> =
            serde_json::from_slice(&std::fs::read(&memory_file).unwrap()).unwrap();
        native.push(serde_json::json!({
            "id": "native-2", "text": "Prefers Rust", "category": "preference",
            "owner": "breadway", "timestamp": 1700000200
        }));
        std::fs::write(&memory_file, serde_json::to_vec(&native).unwrap()).unwrap();
        let report = sync_odysseus(&mut conn, &memory_file, "breadway", "tester").unwrap();
        assert_eq!(report.imported, 1);
        assert_eq!(report.projected, 3);
        let entries: Vec<Value> =
            serde_json::from_slice(&std::fs::read(&memory_file).unwrap()).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry["id"] == "native-2")
                .count(),
            1
        );

        // Retracting a fact removes its projection on the next sync.
        retract(&mut conn, "project.breadmemory", "tester", Some("test")).unwrap();
        let report = sync_odysseus(&mut conn, &memory_file, "breadway", "tester").unwrap();
        assert_eq!(report.projected, 2);
        let entries: Vec<Value> =
            serde_json::from_slice(&std::fs::read(&memory_file).unwrap()).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(!entries
            .iter()
            .any(|entry| entry["breadmemory"]["key"] == "project.breadmemory"));
    }
}
