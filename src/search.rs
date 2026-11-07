use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use rusqlite::{named_params, params, Connection, OptionalExtension};

use crate::util::{truncate_chars, ts_to_iso};

#[derive(Debug, Clone)]
pub struct Hit {
    pub chunk_id: i64,
    pub conversation_id: String,
    pub conversation_title: String,
    pub conversation_created_at: Option<f64>,
    pub node_id: String,
    pub message_id: Option<String>,
    pub role: String,
    pub message_created_at: Option<f64>,
    pub active_branch: bool,
    pub chunk_index: i64,
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub limit: usize,
    pub role: Option<String>,
    pub after: Option<f64>,
    pub before: Option<f64>,
    pub include_hidden: bool,
    pub conversation_id: Option<String>,
    pub exclude_conversation_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub create_time: Option<f64>,
    pub update_time: Option<f64>,
    pub message_count: i64,
    pub active_message_count: i64,
    pub chunk_count: i64,
    pub is_archived: bool,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MessageRow {
    pub node_id: String,
    pub message_id: Option<String>,
    pub parent_node_id: Option<String>,
    pub role: String,
    pub create_time: Option<f64>,
    pub text: String,
    pub hidden: bool,
    pub active: bool,
    pub depth: i64,
}

#[derive(Debug, Clone)]
pub struct ConversationDetail {
    pub summary: ConversationSummary,
    pub messages: Vec<MessageRow>,
    pub branch_points: i64,
    pub all_branches: bool,
    pub truncated: bool,
    pub omitted: usize,
}

#[derive(Debug, Clone)]
pub struct ContextResult {
    pub conversation: ConversationSummary,
    pub messages: Vec<MessageRow>,
    pub target_index: usize,
    pub alternatives: usize,
    pub branch_note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RelatedGroup {
    pub conversation_id: String,
    pub title: String,
    pub create_time: Option<f64>,
    pub score: f64,
    pub matches: Vec<Hit>,
}

#[derive(Debug, Clone)]
pub struct LastImport {
    pub source_path: String,
    pub finished_at: Option<String>,
    pub status: String,
    pub conversations_new: i64,
    pub conversations_updated: i64,
    pub conversations_unchanged: i64,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ArchiveStatus {
    pub schema_version: String,
    pub archive_id: String,
    pub db_path: String,
    pub db_bytes: u64,
    pub conversations: i64,
    pub messages: i64,
    pub chunks: i64,
    pub attachments_extracted: i64,
    pub attachments_missing: i64,
    pub text_chars: i64,
    pub context_active: i64,
    pub context_proposed: i64,
    pub last_import: Option<LastImport>,
}

const STOPWORDS: &[&str] = &[
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "your",
    "yours",
    "with",
    "this",
    "that",
    "these",
    "those",
    "from",
    "have",
    "has",
    "had",
    "was",
    "were",
    "will",
    "would",
    "can",
    "could",
    "should",
    "shall",
    "may",
    "might",
    "must",
    "about",
    "into",
    "onto",
    "over",
    "under",
    "then",
    "than",
    "there",
    "here",
    "when",
    "where",
    "what",
    "which",
    "who",
    "whom",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "too",
    "very",
    "just",
    "also",
    "now",
    "use",
    "using",
    "used",
    "get",
    "got",
    "make",
    "made",
    "like",
    "want",
    "need",
    "one",
    "two",
    "way",
    "thing",
    "things",
    "does",
    "did",
    "doing",
    "done",
    "its",
    "it's",
    "i'm",
    "ive",
    "i've",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "cant",
    "can't",
    "wont",
    "won't",
    "couldnt",
    "couldn't",
    "shouldnt",
    "shouldn't",
    "wouldnt",
    "wouldn't",
    "let",
    "lets",
    "let's",
    "yes",
    "yeah",
    "okay",
    "ok",
    "sure",
    "thanks",
    "thank",
    "please",
    "still",
    "even",
    "much",
    "many",
    "lot",
    "really",
    "actually",
    "basically",
    "maybe",
    "perhaps",
    "however",
    "because",
    "while",
    "after",
    "before",
    "again",
    "ever",
    "never",
    "always",
    "often",
    "something",
    "anything",
    "nothing",
    "everything",
    "someone",
    "anyone",
    "everyone",
    "think",
    "know",
    "see",
    "look",
    "say",
    "said",
    "tell",
    "told",
    "give",
    "gave",
    "take",
    "took",
    "come",
    "came",
    "went",
    "go",
    "goes",
    "going",
    "put",
    "keep",
    "kept",
    "try",
    "tried",
];

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in input.chars() {
        if c.is_alphanumeric() || c == '_' {
            for lower in c.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Build a safe FTS5 query: quoted phrases are kept, other terms become
/// prefix matches joined with OR. Returns None when nothing searchable is left.
pub fn build_fts_query(input: &str) -> Option<String> {
    let mut terms: Vec<String> = Vec::new();
    for (index, segment) in input.split('"').enumerate() {
        if index % 2 == 1 {
            let phrase = segment.trim();
            if !phrase.is_empty() {
                terms.push(format!("\"{}\"", phrase.replace('"', "")));
            }
            continue;
        }
        for token in tokenize(segment) {
            if token.chars().count() < 2 {
                continue;
            }
            if matches!(token.as_str(), "and" | "or" | "not" | "near") {
                continue;
            }
            if token.chars().count() >= 3 {
                terms.push(format!("{token}*"));
            } else {
                terms.push(token);
            }
        }
    }
    let mut seen = HashSet::new();
    terms.retain(|term| seen.insert(term.clone()));
    terms.truncate(32);
    if terms.is_empty() {
        return None;
    }
    Some(terms.join(" OR "))
}

/// Distinctive terms for relatedness queries.
pub fn keywords(text: &str, max: usize) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for token in tokenize(text) {
        let len = token.chars().count();
        if !(3..=48).contains(&len) {
            continue;
        }
        if STOPWORDS.contains(&token.as_str()) {
            continue;
        }
        if token.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        *counts.entry(token).or_insert(0) += 1;
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| {
        let score_a = a.1 * a.0.chars().count().min(24);
        let score_b = b.1 * b.0.chars().count().min(24);
        score_b.cmp(&score_a).then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .take(max)
        .map(|(token, _)| token)
        .collect()
}

pub fn search_keyword(conn: &Connection, query: &str, options: &SearchOptions) -> Result<Vec<Hit>> {
    let Some(fts_query) = build_fts_query(query) else {
        return Ok(Vec::new());
    };
    let limit = options.limit.clamp(1, 50);
    let fetch = (limit * 6).clamp(limit, 300);

    let mut stmt = conn.prepare(
        "SELECT c.id, c.conversation_id, cv.title, cv.create_time, c.node_id, c.message_id,
                COALESCE(m.author_role, 'unknown'), m.create_time, m.is_on_active_branch,
                c.chunk_index, c.text,
                bm25(chunks_fts, 1.0, 1.2) AS rank
         FROM chunks_fts
         JOIN chunks c ON c.id = chunks_fts.rowid
         JOIN conversations cv ON cv.id = c.conversation_id
         LEFT JOIN messages m
                ON m.conversation_id = c.conversation_id AND m.node_id = c.node_id
         WHERE chunks_fts MATCH :q
           AND (:include_hidden = 1 OR c.hidden = 0)
           AND (:role IS NULL OR m.author_role = :role)
           AND (:after IS NULL OR cv.create_time >= :after)
           AND (:before IS NULL OR cv.create_time <= :before)
           AND (:conv IS NULL OR c.conversation_id = :conv)
           AND (:exclude_conv IS NULL OR c.conversation_id != :exclude_conv)
         ORDER BY rank
         LIMIT :fetch",
    )?;

    let rows = stmt.query_map(
        named_params! {
            ":q": fts_query,
            ":include_hidden": options.include_hidden as i64,
            ":role": options.role,
            ":after": options.after,
            ":before": options.before,
            ":conv": options.conversation_id,
            ":exclude_conv": options.exclude_conversation_id,
            ":fetch": fetch as i64,
        },
        |row| {
            Ok(Hit {
                chunk_id: row.get(0)?,
                conversation_id: row.get(1)?,
                conversation_title: row.get(2)?,
                conversation_created_at: row.get(3)?,
                node_id: row.get(4)?,
                message_id: row.get(5)?,
                role: row.get(6)?,
                message_created_at: row.get(7)?,
                active_branch: row.get::<_, i64>(8)? != 0,
                chunk_index: row.get(9)?,
                text: row.get(10)?,
                score: -row.get::<_, f64>(11)?,
            })
        },
    )?;

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut hits = Vec::new();
    for row in rows {
        let hit = row?;
        if seen.insert((hit.conversation_id.clone(), hit.node_id.clone())) {
            hits.push(hit);
            if hits.len() >= limit {
                break;
            }
        }
    }
    // FTS5 bm25 values are tiny and negative. Normalise the displayed score
    // to the best hit in this result set; ordering is unchanged.
    if let Some(best) = hits.iter().map(|hit| hit.score).reduce(f64::max) {
        if best > 0.0 {
            for hit in &mut hits {
                hit.score = hit.score / best * 100.0;
            }
        }
    }
    Ok(hits)
}

pub fn resolve_conversation_id(conn: &Connection, needle: &str) -> Result<String> {
    let needle = needle.trim();
    if needle.is_empty() {
        bail!("conversation id is empty");
    }
    let exact: Option<String> = conn
        .query_row(
            "SELECT id FROM conversations WHERE id = ?1",
            params![needle],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(id) = exact {
        return Ok(id);
    }

    let mut stmt = conn.prepare(
        "SELECT id, title FROM conversations
         WHERE id LIKE ?1 OR title LIKE ?2
         ORDER BY COALESCE(update_time, create_time) DESC LIMIT 10",
    )?;
    let rows = stmt.query_map(
        params![format!("{needle}%"), format!("%{needle}%")],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    )?;
    let mut matches = Vec::new();
    for row in rows {
        matches.push(row?);
    }
    match matches.len() {
        0 => bail!("no conversation matches '{needle}'"),
        1 => Ok(matches.remove(0).0),
        _ => {
            let list = matches
                .iter()
                .take(6)
                .map(|(id, title)| format!("{id} ({title})"))
                .collect::<Vec<_>>()
                .join(", ");
            bail!("conversation reference '{needle}' is ambiguous: {list}")
        }
    }
}

pub fn conversation_summary(conn: &Connection, id: &str) -> Result<ConversationSummary> {
    conn.query_row(
        "SELECT id, title, create_time, update_time, message_count, active_message_count,
                chunk_count, is_archived, default_model_slug
         FROM conversations WHERE id = ?1",
        params![id],
        |row| {
            Ok(ConversationSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                create_time: row.get(2)?,
                update_time: row.get(3)?,
                message_count: row.get(4)?,
                active_message_count: row.get(5)?,
                chunk_count: row.get(6)?,
                is_archived: row.get::<_, i64>(7)? != 0,
                model: row.get(8)?,
            })
        },
    )
    .with_context(|| format!("conversation {id} not found"))
}

pub fn get_conversation(
    conn: &Connection,
    reference: &str,
    include_hidden: bool,
    all_branches: bool,
    max_chars: usize,
) -> Result<ConversationDetail> {
    let id = resolve_conversation_id(conn, reference)?;
    let summary = conversation_summary(conn, &id)?;
    let active_messages: i64 = conn.query_row(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1 AND is_on_active_branch = 1",
        params![id],
        |row| row.get(0),
    )?;
    let use_active = !all_branches && active_messages > 0;

    let sql = format!(
        "SELECT node_id, message_id, parent_node_id, author_role, create_time, text,
                is_hidden, is_on_active_branch, COALESCE(depth, 0)
         FROM messages
         WHERE conversation_id = ?1
           {hidden_clause}
           {branch_clause}
         ORDER BY {order}",
        hidden_clause = if include_hidden {
            ""
        } else {
            "AND is_hidden = 0"
        },
        branch_clause = if use_active {
            "AND is_on_active_branch = 1"
        } else {
            ""
        },
        order = if use_active {
            "COALESCE(depth, 0), ord"
        } else {
            "ord"
        },
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![id], map_message)?;

    let mut messages = Vec::new();
    let mut used = 0usize;
    let mut truncated = false;
    let mut omitted = 0usize;
    for row in rows {
        let mut message = row?;
        if truncated {
            omitted += 1;
            continue;
        }
        let length = message.text.chars().count();
        if used + length <= max_chars {
            used += length;
            messages.push(message);
        } else {
            let remaining = max_chars.saturating_sub(used);
            let (text, _) = truncate_chars(&message.text, remaining);
            message.text = text;
            used = max_chars;
            truncated = true;
            messages.push(message);
        }
    }

    let branch_points: i64 = conn.query_row(
        "SELECT COUNT(*) FROM (
            SELECT parent_node_id FROM message_edges WHERE conversation_id = ?1
            GROUP BY parent_node_id HAVING COUNT(*) > 1
         )",
        params![id],
        |row| row.get(0),
    )?;

    Ok(ConversationDetail {
        summary,
        messages,
        branch_points,
        all_branches,
        truncated,
        omitted,
    })
}

fn map_message(row: &rusqlite::Row<'_>) -> rusqlite::Result<MessageRow> {
    Ok(MessageRow {
        node_id: row.get(0)?,
        message_id: row.get(1)?,
        parent_node_id: row.get(2)?,
        role: row.get(3)?,
        create_time: row.get(4)?,
        text: row.get(5)?,
        hidden: row.get::<_, i64>(6)? != 0,
        active: row.get::<_, i64>(7)? != 0,
        depth: row.get(8)?,
    })
}

fn resolve_message(
    conn: &Connection,
    reference: &str,
    conversation_hint: Option<&str>,
) -> Result<(String, String)> {
    let reference = reference.trim();
    if reference.is_empty() {
        bail!("message id is empty");
    }
    let hint = match conversation_hint {
        Some(hint) => Some(resolve_conversation_id(conn, hint)?),
        None => None,
    };

    let exact = query_message_refs(conn, reference, false, hint.as_deref())?;
    if exact.len() == 1 {
        return Ok(exact.into_iter().next().unwrap());
    }
    if exact.len() > 1 {
        bail!(
            "message '{reference}' is ambiguous; pass a conversation id: {}",
            exact
                .iter()
                .take(6)
                .map(|(conversation, node)| format!("{conversation}/{node}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let prefixed = query_message_refs(conn, reference, true, hint.as_deref())?;
    match prefixed.len() {
        1 => Ok(prefixed.into_iter().next().unwrap()),
        0 => bail!("no message matches '{reference}'"),
        _ => bail!(
            "message reference '{reference}' is ambiguous; pass a conversation id: {}",
            prefixed
                .iter()
                .take(6)
                .map(|(conversation, node)| format!("{conversation}/{node}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn query_message_refs(
    conn: &Connection,
    reference: &str,
    prefix: bool,
    conversation: Option<&str>,
) -> Result<Vec<(String, String)>> {
    let (clause, pattern) = if prefix {
        (
            "(message_id LIKE ?1 OR node_id LIKE ?1)",
            format!("{reference}%"),
        )
    } else {
        ("(message_id = ?1 OR node_id = ?1)", reference.to_string())
    };
    let sql = format!(
        "SELECT DISTINCT conversation_id, node_id FROM messages
         WHERE {clause} AND (?2 IS NULL OR conversation_id = ?2)
         ORDER BY conversation_id, ord LIMIT 10"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![pattern, conversation], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?;
    let mut values = Vec::new();
    for row in rows {
        values.push(row?);
    }
    Ok(values)
}

pub fn message_text(
    conn: &Connection,
    reference: &str,
    hint: Option<&str>,
) -> Result<(String, String, String)> {
    let (conversation_id, node_id) = resolve_message(conn, reference, hint)?;
    let (message_id, text): (Option<String>, String) = conn.query_row(
        "SELECT message_id, text FROM messages WHERE conversation_id = ?1 AND node_id = ?2",
        params![conversation_id, node_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((
        conversation_id,
        message_id.unwrap_or_else(|| node_id.clone()),
        text,
    ))
}

pub fn get_context(
    conn: &Connection,
    reference: &str,
    conversation_hint: Option<&str>,
    before: usize,
    after: usize,
    include_hidden: bool,
) -> Result<ContextResult> {
    let (conversation_id, target_node) = resolve_message(conn, reference, conversation_hint)?;
    let summary = conversation_summary(conn, &conversation_id)?;

    let mut stmt = conn.prepare(
        "SELECT node_id, message_id, parent_node_id, author_role, create_time, text,
                is_hidden, is_on_active_branch, COALESCE(depth, 0)
         FROM messages WHERE conversation_id = ?1 ORDER BY ord",
    )?;
    let rows = stmt.query_map(params![conversation_id], map_message)?;
    let mut all = Vec::new();
    for row in rows {
        all.push(row?);
    }
    if all.is_empty() {
        bail!("conversation {conversation_id} has no messages");
    }

    let index_by_node: HashMap<String, usize> = all
        .iter()
        .enumerate()
        .map(|(index, message)| (message.node_id.clone(), index))
        .collect();
    let _target_index = *index_by_node.get(&target_node).with_context(|| {
        format!("message {target_node} is not in conversation {conversation_id}")
    })?;

    let parent_of: HashMap<String, Option<String>> = all
        .iter()
        .map(|message| (message.node_id.clone(), message.parent_node_id.clone()))
        .collect();
    let mut children_of: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, message) in all.iter().enumerate() {
        if let Some(parent) = message.parent_node_id.as_ref() {
            children_of.entry(parent.clone()).or_default().push(index);
        }
    }
    for indexes in children_of.values_mut() {
        indexes.sort_by_key(|index| {
            (
                all[*index].depth,
                all[*index].create_time.unwrap_or(0.0) as i64,
            )
        });
    }

    let mut chain: Vec<usize> = Vec::new();
    let mut cursor = Some(target_node.clone());
    let mut guard = 0usize;
    while let Some(node) = cursor {
        guard += 1;
        if guard > 10_000 {
            break;
        }
        let Some(index) = index_by_node.get(&node).copied() else {
            break;
        };
        chain.push(index);
        cursor = parent_of.get(&node).cloned().flatten();
    }
    chain.reverse();

    let mut cursor = target_node.clone();
    let mut guard = 0usize;
    while guard < 10_000 {
        guard += 1;
        let Some(indexes) = children_of.get(&cursor) else {
            break;
        };
        if indexes.is_empty() {
            break;
        }
        let target = &all[*index_by_node.get(&cursor).expect("cursor in map")];
        let candidate = indexes
            .iter()
            .copied()
            .filter(|index| all[*index].active == target.active)
            .max_by(|a, b| {
                let left = &all[*a];
                let right = &all[*b];
                left.create_time
                    .unwrap_or(0.0)
                    .partial_cmp(&right.create_time.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.depth.cmp(&right.depth))
            })
            .or_else(|| indexes.last().copied());
        let Some(next) = candidate else { break };
        chain.push(next);
        cursor = all[next].node_id.clone();
    }

    let position = chain
        .iter()
        .position(|index| all[*index].node_id == target_node)
        .unwrap_or(0);
    let start = position.saturating_sub(before);
    let end = (position + after + 1).min(chain.len());

    let mut messages = Vec::new();
    for index in &chain[start..end] {
        let message = &all[*index];
        if message.hidden && !include_hidden && message.node_id != target_node {
            continue;
        }
        messages.push(message.clone());
    }

    let alternatives = children_of
        .get(&target_node)
        .map(|children| children.len().saturating_sub(1))
        .unwrap_or(0);
    let branch_note = if alternatives > 0 {
        Some(format!(
            "This message has {alternatives} alternative continuation(s); other branches were not included. Use get_conversation with include_branches=true to see them."
        ))
    } else {
        None
    };

    let adjusted_target = messages
        .iter()
        .position(|message| message.node_id == target_node)
        .unwrap_or(0);

    Ok(ContextResult {
        conversation: summary,
        messages,
        target_index: adjusted_target,
        alternatives,
        branch_note,
    })
}

pub fn find_related(
    conn: &Connection,
    text: &str,
    exclude_conversation: Option<&str>,
    limit: usize,
    include_hidden: bool,
) -> Result<Vec<RelatedGroup>> {
    let terms = keywords(text, 16);
    if terms.is_empty() {
        bail!("not enough distinctive terms to find related discussions");
    }
    let query = terms.join(" ");
    let options = SearchOptions {
        limit: (limit * 6).clamp(12, 200),
        include_hidden,
        exclude_conversation_id: exclude_conversation.map(str::to_string),
        ..Default::default()
    };
    let hits = search_keyword(conn, &query, &options)?;

    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, RelatedGroup> = HashMap::new();
    for hit in hits {
        let group = groups
            .entry(hit.conversation_id.clone())
            .or_insert_with(|| {
                order.push(hit.conversation_id.clone());
                RelatedGroup {
                    conversation_id: hit.conversation_id.clone(),
                    title: hit.conversation_title.clone(),
                    create_time: hit.conversation_created_at,
                    score: 0.0,
                    matches: Vec::new(),
                }
            });
        group.score += hit.score.max(0.0);
        if group.matches.len() < 2 {
            group.matches.push(hit);
        }
    }
    let mut results: Vec<RelatedGroup> = order
        .into_iter()
        .filter_map(|id| groups.remove(&id))
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    Ok(results)
}

pub fn list_conversations(
    conn: &Connection,
    limit: usize,
    offset: usize,
) -> Result<Vec<ConversationSummary>> {
    let mut stmt = conn.prepare(
        "SELECT id, title, create_time, update_time, message_count, active_message_count,
                chunk_count, is_archived, default_model_slug
         FROM conversations
         ORDER BY COALESCE(update_time, create_time) DESC, id
         LIMIT ?1 OFFSET ?2",
    )?;
    let rows = stmt.query_map(params![limit as i64, offset as i64], |row| {
        Ok(ConversationSummary {
            id: row.get(0)?,
            title: row.get(1)?,
            create_time: row.get(2)?,
            update_time: row.get(3)?,
            message_count: row.get(4)?,
            active_message_count: row.get(5)?,
            chunk_count: row.get(6)?,
            is_archived: row.get::<_, i64>(7)? != 0,
            model: row.get(8)?,
        })
    })?;
    let mut values = Vec::new();
    for row in rows {
        values.push(row?);
    }
    Ok(values)
}

pub fn archive_status(conn: &Connection, db_path: &str) -> Result<ArchiveStatus> {
    let schema_version = crate::db::get_meta(conn, "schema_version")?.unwrap_or_default();
    let archive_id = crate::db::get_meta(conn, "archive_id")?.unwrap_or_default();
    let conversations = count(conn, "SELECT COUNT(*) FROM conversations")?;
    let messages = count(conn, "SELECT COUNT(*) FROM messages")?;
    let chunks = count(conn, "SELECT COUNT(*) FROM chunks")?;
    let attachments_extracted = count(
        conn,
        "SELECT COUNT(*) FROM attachments WHERE status = 'extracted'",
    )?;
    let attachments_missing = count(
        conn,
        "SELECT COUNT(*) FROM attachments WHERE status = 'missing'",
    )?;
    let text_chars = count(conn, "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM messages")?;
    let context_active = count(
        conn,
        "SELECT COUNT(*) FROM current_context WHERE status = 'active'",
    )?;
    let context_proposed = count(
        conn,
        "SELECT COUNT(*) FROM current_context WHERE status = 'proposed'",
    )?;

    let last_import = conn
        .query_row(
            "SELECT source_path, finished_at, status, conversations_new,
                    conversations_updated, conversations_unchanged, error
             FROM imports ORDER BY id DESC LIMIT 1",
            [],
            |row| {
                Ok(LastImport {
                    source_path: row.get(0)?,
                    finished_at: row.get(1)?,
                    status: row.get(2)?,
                    conversations_new: row.get(3)?,
                    conversations_updated: row.get(4)?,
                    conversations_unchanged: row.get(5)?,
                    error: row.get(6)?,
                })
            },
        )
        .optional()?;

    let db_bytes = std::fs::metadata(db_path)
        .map(|meta| meta.len())
        .unwrap_or(0);

    Ok(ArchiveStatus {
        schema_version,
        archive_id,
        db_path: db_path.to_string(),
        db_bytes,
        conversations,
        messages,
        chunks,
        attachments_extracted,
        attachments_missing,
        text_chars,
        context_active,
        context_proposed,
        last_import,
    })
}

fn count(conn: &Connection, sql: &str) -> Result<i64> {
    Ok(conn.query_row(sql, [], |row| row.get(0))?)
}

pub fn hit_to_json(hit: &Hit) -> serde_json::Value {
    serde_json::json!({
        "chunk_id": hit.chunk_id,
        "conversation_id": hit.conversation_id,
        "conversation_title": hit.conversation_title,
        "conversation_created_at": ts_to_iso(hit.conversation_created_at),
        "conversation_created_unix": hit.conversation_created_at,
        "message_id": hit.message_id,
        "node_id": hit.node_id,
        "role": hit.role,
        "message_created_at": ts_to_iso(hit.message_created_at),
        "message_created_unix": hit.message_created_at,
        "on_active_branch": hit.active_branch,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "text": hit.text,
    })
}

pub fn message_to_json(message: &MessageRow, is_target: bool) -> serde_json::Value {
    serde_json::json!({
        "message_id": message.message_id,
        "node_id": message.node_id,
        "parent_node_id": message.parent_node_id,
        "role": message.role,
        "created_at": ts_to_iso(message.create_time),
        "created_unix": message.create_time,
        "on_active_branch": message.active,
        "hidden": message.hidden,
        "text": message.text,
        "is_target": is_target,
    })
}

pub fn conversation_summary_json(summary: &ConversationSummary) -> serde_json::Value {
    serde_json::json!({
        "conversation_id": summary.id,
        "title": summary.title,
        "created_at": ts_to_iso(summary.create_time),
        "created_unix": summary.create_time,
        "updated_at": ts_to_iso(summary.update_time),
        "updated_unix": summary.update_time,
        "message_count": summary.message_count,
        "active_message_count": summary.active_message_count,
        "chunk_count": summary.chunk_count,
        "archived": summary.is_archived,
        "model": summary.model,
    })
}

pub fn conversation_detail_json(detail: &ConversationDetail) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = detail
        .messages
        .iter()
        .map(|message| {
            let mut value = message_to_json(message, false);
            value["conversation_id"] = serde_json::json!(detail.summary.id);
            value
        })
        .collect();
    serde_json::json!({
        "conversation": conversation_summary_json(&detail.summary),
        "branch_points": detail.branch_points,
        "view": if detail.all_branches { "all_branches" } else { "active_branch" },
        "truncated": detail.truncated,
        "omitted_messages": detail.omitted,
        "messages": messages,
    })
}

pub fn context_result_json(result: &ContextResult) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = result
        .messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let mut value = message_to_json(message, index == result.target_index);
            value["conversation_id"] = serde_json::json!(result.conversation.id);
            value
        })
        .collect();
    serde_json::json!({
        "conversation": conversation_summary_json(&result.conversation),
        "target_index": result.target_index,
        "alternative_continuations_not_shown": result.alternatives,
        "branch_note": result.branch_note,
        "messages": messages,
    })
}

pub fn related_json(groups: &[RelatedGroup]) -> serde_json::Value {
    let values: Vec<serde_json::Value> = groups
        .iter()
        .map(|group| {
            serde_json::json!({
                "conversation_id": group.conversation_id,
                "conversation_title": group.title,
                "conversation_created_at": ts_to_iso(group.create_time),
                "conversation_created_unix": group.create_time,
                "score": group.score,
                "matches": group.matches.iter().map(hit_to_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::json!(values)
}

pub fn status_json(status: &ArchiveStatus) -> serde_json::Value {
    serde_json::json!({
        "schema_version": status.schema_version,
        "archive_id": status.archive_id,
        "database_path": status.db_path,
        "database_bytes": status.db_bytes,
        "conversations": status.conversations,
        "messages": status.messages,
        "chunks": status.chunks,
        "attachments_extracted": status.attachments_extracted,
        "attachments_missing": status.attachments_missing,
        "text_characters": status.text_chars,
        "current_context": {
            "active": status.context_active,
            "proposed": status.context_proposed,
        },
        "last_import": status.last_import.as_ref().map(|last| serde_json::json!({
            "source_path": last.source_path,
            "finished_at": last.finished_at,
            "status": last.status,
            "conversations_new": last.conversations_new,
            "conversations_updated": last.conversations_updated,
            "conversations_unchanged": last.conversations_unchanged,
            "error": last.error,
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_safe_fts_queries() {
        assert_eq!(
            build_fts_query("zfs arc_max"),
            Some("zfs* OR arc_max*".to_string())
        );
        assert_eq!(
            build_fts_query("\"exact phrase\" other"),
            Some("\"exact phrase\" OR other*".to_string())
        );
        assert_eq!(build_fts_query("a"), None);
        // FTS operators are stripped by tokenisation.
        assert_eq!(
            build_fts_query("foo AND (bar OR baz)"),
            Some("foo* OR bar* OR baz*".to_string())
        );
    }

    #[test]
    fn keywords_skip_stopwords_and_short_tokens() {
        let terms = keywords("The ZFS ARC max was set to 16GB for the tank pool", 10);
        assert!(terms.contains(&"zfs".to_string()));
        assert!(terms.contains(&"tank".to_string()));
        assert!(!terms.contains(&"the".to_string()));
    }
}
