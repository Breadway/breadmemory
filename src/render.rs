use crate::context::ContextFact;
use crate::importer::ImportReport;
use crate::search::{
    ArchiveStatus, ContextResult, ConversationDetail, ConversationSummary, Hit, RelatedGroup,
};
use crate::util::{human_ts, truncate_chars};

pub fn render_import(report: &ImportReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Imported {}\n", report.source_path));
    out.push_str(&format!(
        "  source:      {} (sha256 {})\n",
        report.source_kind,
        &report.source_sha256[..report.source_sha256.len().min(16)]
    ));
    out.push_str(&format!(
        "  conversations: {} seen, {} new, {} updated, {} unchanged, {} failed\n",
        report.conversations_seen,
        report.conversations_new,
        report.conversations_updated,
        report.conversations_unchanged,
        report.conversations_failed
    ));
    out.push_str(&format!(
        "  rows:        {} messages, {} chunks, {} attachment links, {} files extracted\n",
        report.messages_written,
        report.chunks_written,
        report.attachments_linked,
        report.attachments_extracted
    ));
    out.push_str(&format!("  elapsed:     {} ms\n", report.elapsed_ms));
    if !report.errors.is_empty() {
        out.push_str("  warnings:\n");
        for error in &report.errors {
            out.push_str(&format!("    - {error}\n"));
        }
    }
    out
}

pub fn render_status(status: &ArchiveStatus) -> String {
    let mut out = String::new();
    out.push_str("breadmemory archive\n");
    out.push_str(&format!("  database:      {}\n", status.db_path));
    out.push_str(&format!(
        "  size:          {:.1} MiB\n",
        status.db_bytes as f64 / (1024.0 * 1024.0)
    ));
    out.push_str(&format!(
        "  schema:        v{} (archive {})\n",
        status.schema_version, status.archive_id
    ));
    out.push_str(&format!(
        "  conversations: {}\n  messages:      {}\n  chunks:        {}\n",
        status.conversations, status.messages, status.chunks
    ));
    out.push_str(&format!(
        "  text:          {} characters\n",
        status.text_chars
    ));
    out.push_str(&format!(
        "  attachments:   {} extracted, {} missing\n",
        status.attachments_extracted, status.attachments_missing
    ));
    out.push_str(&format!(
        "  context:       {} active fact(s), {} proposed\n",
        status.context_active, status.context_proposed
    ));
    match &status.last_import {
        Some(last) => {
            out.push_str(&format!(
                "  last import:   {} at {}, {} new / {} updated / {} unchanged\n",
                last.source_path,
                last.finished_at.as_deref().unwrap_or("unknown"),
                last.conversations_new,
                last.conversations_updated,
                last.conversations_unchanged
            ));
            if let Some(error) = &last.error {
                out.push_str(&format!("                 error: {error}\n"));
            }
        }
        None => out.push_str("  last import:   none\n"),
    }
    out
}

pub fn render_search(hits: &[Hit], query: &str) -> String {
    if hits.is_empty() {
        return format!("No results for '{query}'.\n");
    }
    let mut out = format!(
        "{} result(s) for '{query}' (keyword search)\n\n",
        hits.len()
    );
    for (index, hit) in hits.iter().enumerate() {
        out.push_str(&format!(
            "[{}] {} ({})\n",
            index + 1,
            hit.conversation_title,
            human_ts(hit.conversation_created_at)
        ));
        out.push_str(&format!(
            "    conv {} | message {} | {}{} | score {:.2}\n",
            hit.conversation_id,
            hit.message_id.as_deref().unwrap_or(&hit.node_id),
            hit.role,
            if hit.active_branch {
                ""
            } else {
                " (inactive branch)"
            },
            hit.score
        ));
        let snippet = hit.text.trim();
        let (snippet, _) = truncate_chars(snippet, 420);
        out.push_str(&format!("    {}\n\n", snippet));
    }
    out
}

pub fn render_conversations(items: &[ConversationSummary]) -> String {
    if items.is_empty() {
        return "No conversations in the archive.\n".to_string();
    }
    let mut out = String::new();
    for item in items {
        out.push_str(&format!(
            "{}  {}  [{} msgs, {} active, {} chunks]\n",
            item.id,
            human_ts(item.create_time),
            item.message_count,
            item.active_message_count,
            item.chunk_count
        ));
        out.push_str(&format!("  {}\n", item.title));
    }
    out
}

pub fn render_conversation(detail: &ConversationDetail) -> String {
    let summary = &detail.summary;
    let mut out = String::new();
    out.push_str(&format!("{}\n", summary.title));
    out.push_str(&format!("  id:       {}\n", summary.id));
    out.push_str(&format!(
        "  created:  {}    updated: {}\n",
        human_ts(summary.create_time),
        human_ts(summary.update_time)
    ));
    out.push_str(&format!(
        "  messages: {} total, {} active    chunks: {}    branch points: {}\n",
        summary.message_count,
        summary.active_message_count,
        summary.chunk_count,
        detail.branch_points
    ));
    if let Some(model) = &summary.model {
        out.push_str(&format!("  model:    {model}\n"));
    }
    if detail.all_branches {
        out.push_str("  view:     all branches\n");
    } else {
        out.push_str("  view:     active branch\n");
    }
    out.push('\n');
    for message in &detail.messages {
        let marker = if message.active { "" } else { " [branch]" };
        let hidden = if message.hidden { " [hidden]" } else { "" };
        out.push_str(&format!(
            "--- {} @ {}{}{}  ({})\n",
            message.role,
            human_ts(message.create_time),
            marker,
            hidden,
            message.message_id.as_deref().unwrap_or(&message.node_id)
        ));
        out.push_str(&format!("{}\n\n", message.text.trim_end()));
    }
    if detail.truncated {
        out.push_str(&format!(
            "Output truncated at the configured character limit; {} further message(s) omitted.\n",
            detail.omitted
        ));
    }
    out
}

pub fn render_context(result: &ContextResult) -> String {
    let mut out = format!(
        "{} ({})\n",
        result.conversation.title, result.conversation.id
    );
    out.push_str(&format!(
        "{} message(s) around the target",
        result.messages.len()
    ));
    if result.alternatives > 0 {
        out.push_str(&format!(
            ", {} alternative continuation(s) not shown",
            result.alternatives
        ));
    }
    out.push_str("\n\n");
    for (index, message) in result.messages.iter().enumerate() {
        let target = index == result.target_index;
        let marker = if target { ">>>" } else { "   " };
        let hidden = if message.hidden { " [hidden]" } else { "" };
        out.push_str(&format!(
            "{marker} {} @ {}{}  ({})\n",
            message.role,
            human_ts(message.create_time),
            hidden,
            message.message_id.as_deref().unwrap_or(&message.node_id)
        ));
        if target {
            out.push_str(&format!("{}\n\n", message.text.trim_end()));
        } else {
            let (snippet, _) = truncate_chars(&collapse(&message.text), 500);
            out.push_str(&format!("    {snippet}\n\n"));
        }
    }
    if let Some(note) = &result.branch_note {
        out.push_str(note);
        out.push('\n');
    }
    out
}

pub fn render_related(groups: &[RelatedGroup]) -> String {
    if groups.is_empty() {
        return "No related conversations found.\n".to_string();
    }
    let mut out = format!(
        "{} related conversation(s) (keyword similarity)\n\n",
        groups.len()
    );
    for (index, group) in groups.iter().enumerate() {
        out.push_str(&format!(
            "[{}] {} ({})\n",
            index + 1,
            group.title,
            human_ts(group.create_time)
        ));
        out.push_str(&format!(
            "    conv {} | score {:.2}\n",
            group.conversation_id, group.score
        ));
        for hit in &group.matches {
            let (snippet, _) = truncate_chars(&collapse(&hit.text), 300);
            out.push_str(&format!("    - {}: {}\n", hit.role, snippet));
        }
        out.push('\n');
    }
    out
}

pub fn render_facts(facts: &[ContextFact], heading: &str) -> String {
    if facts.is_empty() {
        return format!("{heading}: no facts.\n");
    }
    let mut out = format!("{heading}: {} fact(s)\n\n", facts.len());
    let mut category = String::new();
    for fact in facts {
        if fact.category != category {
            category = fact.category.clone();
            out.push_str(&format!("[{}]\n", category));
        }
        out.push_str(&format!("  {} = {}\n", fact.key, fact.value));
        out.push_str(&format!(
            "      status {}, confidence {}, source {}{}{}\n",
            fact.status,
            fact.confidence,
            fact.source_kind,
            fact.source_conversation_id
                .as_deref()
                .map(|id| format!(" ({id})"))
                .unwrap_or_default(),
            if fact.pinned { ", pinned" } else { "" }
        ));
        out.push_str(&format!("      updated {}\n", fact.updated_at));
        if let Some(note) = &fact.note {
            out.push_str(&format!("      note: {note}\n"));
        }
    }
    out
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
