use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use rusqlite::{Connection, OptionalExtension};

use crate::util::{ensure_private_dir, now_iso, restrict_permissions};

pub const SCHEMA_VERSION: i64 = 2;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path            TEXT NOT NULL,
    source_kind            TEXT NOT NULL,
    source_sha256          TEXT NOT NULL,
    started_at             TEXT NOT NULL,
    finished_at            TEXT,
    status                 TEXT NOT NULL,
    conversations_seen     INTEGER NOT NULL DEFAULT 0,
    conversations_new      INTEGER NOT NULL DEFAULT 0,
    conversations_updated  INTEGER NOT NULL DEFAULT 0,
    conversations_unchanged INTEGER NOT NULL DEFAULT 0,
    conversations_failed   INTEGER NOT NULL DEFAULT 0,
    messages_written       INTEGER NOT NULL DEFAULT 0,
    chunks_written         INTEGER NOT NULL DEFAULT 0,
    attachments_written    INTEGER NOT NULL DEFAULT 0,
    error                  TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id                      TEXT PRIMARY KEY,
    title                   TEXT NOT NULL DEFAULT '',
    create_time             REAL,
    update_time             REAL,
    current_node            TEXT,
    default_model_slug      TEXT,
    gizmo_id                TEXT,
    conversation_template_id TEXT,
    is_archived             INTEGER NOT NULL DEFAULT 0,
    is_starred              INTEGER NOT NULL DEFAULT 0,
    memory_scope            TEXT,
    raw_sha256              TEXT NOT NULL,
    first_import_id         INTEGER REFERENCES imports(id),
    last_import_id          INTEGER REFERENCES imports(id),
    imported_at             TEXT NOT NULL,
    message_count           INTEGER NOT NULL DEFAULT 0,
    active_message_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count             INTEGER NOT NULL DEFAULT 0,
    attachment_count        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS conversations_create_time ON conversations(create_time);
CREATE INDEX IF NOT EXISTS conversations_update_time ON conversations(update_time);

-- Verbatim conversation JSON, compressed, so the archive can be verified and
-- re-derived without the original zip.
CREATE TABLE IF NOT EXISTS conversation_raw (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    encoding        TEXT NOT NULL,
    import_id       INTEGER REFERENCES imports(id),
    data            BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    conversation_id   TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    node_id           TEXT NOT NULL,
    message_id        TEXT,
    parent_node_id    TEXT,
    parent_message_id TEXT,
    author_role       TEXT NOT NULL,
    author_name       TEXT,
    create_time       REAL,
    update_time       REAL,
    status            TEXT,
    end_turn          INTEGER,
    weight            REAL,
    recipient         TEXT,
    channel           TEXT,
    content_type      TEXT NOT NULL DEFAULT 'text',
    model_slug        TEXT,
    is_hidden         INTEGER NOT NULL DEFAULT 0,
    hidden_reason     TEXT,
    is_on_active_branch INTEGER NOT NULL DEFAULT 0,
    depth             INTEGER,
    ord               INTEGER NOT NULL DEFAULT 0,
    text              TEXT NOT NULL DEFAULT '',
    metadata_json     TEXT,
    PRIMARY KEY (conversation_id, node_id)
);
CREATE INDEX IF NOT EXISTS messages_message_id ON messages(message_id);
CREATE INDEX IF NOT EXISTS messages_conv_ord ON messages(conversation_id, ord);
CREATE INDEX IF NOT EXISTS messages_conv_active ON messages(conversation_id, is_on_active_branch, depth);

-- Original children arrays, including placeholder nodes whose message is null.
CREATE TABLE IF NOT EXISTS message_edges (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_node_id  TEXT NOT NULL,
    child_node_id   TEXT NOT NULL,
    child_index     INTEGER NOT NULL,
    PRIMARY KEY (conversation_id, parent_node_id, child_node_id)
);
CREATE INDEX IF NOT EXISTS message_edges_parent ON message_edges(conversation_id, parent_node_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    node_id         TEXT NOT NULL,
    message_id      TEXT,
    chunk_index     INTEGER NOT NULL,
    start_offset    INTEGER NOT NULL,
    end_offset      INTEGER NOT NULL,
    text            TEXT NOT NULL,
    title           TEXT NOT NULL,
    role            TEXT NOT NULL,
    hidden          INTEGER NOT NULL DEFAULT 0,
    UNIQUE (conversation_id, node_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_conversation ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS chunks_node ON chunks(conversation_id, node_id);
CREATE INDEX IF NOT EXISTS chunks_message_id ON chunks(message_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    title,
    content='chunks',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES ('delete', old.id, old.text, old.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES ('delete', old.id, old.text, old.title);
    INSERT INTO chunks_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;

CREATE TABLE IF NOT EXISTS attachments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    node_id         TEXT NOT NULL DEFAULT '',
    name            TEXT,
    mime_type       TEXT,
    size_bytes      INTEGER,
    source          TEXT NOT NULL,
    asset_pointer   TEXT,
    archive_path    TEXT,
    sha256          TEXT,
    stored_path     TEXT,
    status          TEXT NOT NULL,
    UNIQUE (conversation_id, file_id, node_id, source)
);
CREATE INDEX IF NOT EXISTS attachments_file_id ON attachments(file_id);
CREATE INDEX IF NOT EXISTS attachments_conversation ON attachments(conversation_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dims        INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    embedded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_embeddings_model ON chunk_embeddings(model);

-- Current context: explicit, human-managed facts. This is NOT derived from the
-- archive and must never be wiped by a reindex or reimport. Provenance points
-- back to conversation/message ids without a foreign key so reimports do not
-- orphan the citations.
CREATE TABLE IF NOT EXISTS current_context (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    key                    TEXT NOT NULL,
    value                  TEXT NOT NULL,
    category               TEXT NOT NULL DEFAULT 'general',
    status                 TEXT NOT NULL DEFAULT 'active',
    confidence             TEXT NOT NULL DEFAULT 'stated',
    source_kind            TEXT NOT NULL DEFAULT 'manual',
    source_conversation_id TEXT,
    source_message_id      TEXT,
    source_quote           TEXT,
    note                   TEXT,
    pinned                 INTEGER NOT NULL DEFAULT 0,
    external_system        TEXT,
    external_id            TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    verified_at            TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS current_context_active_key
    ON current_context(key) WHERE status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS current_context_external
    ON current_context(external_system, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS current_context_status
    ON current_context(status, category);

CREATE TABLE IF NOT EXISTS current_context_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id INTEGER NOT NULL,
    key        TEXT NOT NULL,
    action     TEXT NOT NULL,
    old_value  TEXT,
    new_value  TEXT,
    actor      TEXT NOT NULL,
    reason     TEXT,
    at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS current_context_events_key ON current_context_events(key, at);
"#;

const SCHEMA_V2: &str = r#"
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dims        INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    embedded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_embeddings_model ON chunk_embeddings(model);
"#;

pub fn open(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        ensure_private_dir(parent)?;
    }
    let conn =
        Connection::open(path).with_context(|| format!("open database {}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(10))?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA foreign_keys=ON;
         PRAGMA temp_store=MEMORY;",
    )?;
    migrate(&conn)?;
    secure_database_files(path);
    Ok(conn)
}

/// A second connection for reading while writing, used by reindex.
pub fn open_secondary(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(10))?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;
    let current: i64 = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    if current > SCHEMA_VERSION {
        bail!(
            "database schema version {current} is newer than this binary supports ({SCHEMA_VERSION})"
        );
    }
    if current == 0 {
        conn.execute_batch("BEGIN IMMEDIATE;")?;
        match conn.execute_batch(SCHEMA) {
            Ok(()) => {
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?1)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [SCHEMA_VERSION.to_string()],
                )?;
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?1)",
                    [now_iso()],
                )?;
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES('archive_id', ?1)",
                    [crate::util::sha256_hex(now_iso().as_bytes())[..32].to_string()],
                )?;
                conn.execute_batch("COMMIT;")?;
            }
            Err(err) => {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(err).context("apply schema");
            }
        }
    } else if current < SCHEMA_VERSION {
        conn.execute_batch("BEGIN IMMEDIATE;")?;
        let result = (|| -> Result<()> {
            if current < 2 {
                conn.execute_batch(SCHEMA_V2)?;
            }
            conn.execute(
                "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
                [SCHEMA_VERSION.to_string()],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => conn.execute_batch("COMMIT;")?,
            Err(err) => {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(err).context("migrate schema");
            }
        }
    }
    Ok(())
}

fn secure_database_files(path: &Path) {
    restrict_permissions(path, 0o600);
    let base = path.to_string_lossy();
    for suffix in ["-wal", "-shm"] {
        let candidate = format!("{base}{suffix}");
        if Path::new(&candidate).exists() {
            restrict_permissions(Path::new(&candidate), 0o600);
        }
    }
}

pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| {
            row.get(0)
        })
        .optional()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_schema_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.db");
        let conn = open(&path).unwrap();
        let version = get_meta(&conn, "schema_version").unwrap();
        assert_eq!(version.as_deref(), Some("2"));
        drop(conn);

        let conn = open(&path).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count >= 10);
    }

    #[test]
    fn fts_table_answers() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open(&dir.path().join("archive.db")).unwrap();
        conn.execute(
            "INSERT INTO conversations(id, title, raw_sha256, imported_at) VALUES('c1', 'ZFS', 'x', 'now')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chunks(conversation_id, node_id, chunk_index, start_offset, end_offset, text, title, role)
             VALUES('c1', 'n1', 0, 0, 10, 'zfs arc_max tuning', 'ZFS', 'user')",
            [],
        )
        .unwrap();
        let hits: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'arc*'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(hits, 1);
    }
}
