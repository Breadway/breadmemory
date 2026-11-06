use std::collections::HashSet;

use serde_json::Value;

use crate::export::{RawContent, RawMessage};
use crate::util::file_id_from_pointer;

/// Extract the human-readable text of a message.
///
/// The export has changed shape several times, so this handles every known
/// content type plus the fallbacks (`text`, `result`, `thoughts`) and keeps
/// fenced code blocks intact.
pub fn extract_text(msg: &RawMessage) -> String {
    let Some(content) = msg.content.as_ref() else {
        return String::new();
    };
    let content_type = content.content_type.as_deref().unwrap_or("text");

    let mut text = if content_type == "thoughts" {
        thoughts_text(content)
    } else {
        parts_text(content, content_type)
    };

    if text.trim().is_empty() {
        if let Some(t) = content.text.as_deref() {
            text = t.to_string();
        }
    }
    if text.trim().is_empty() {
        if let Some(result) = content.result.as_ref() {
            if let Some(t) = value_text(result) {
                text = t;
            }
        }
    }
    if text.trim().is_empty() {
        text = thoughts_text(content);
    }

    strip_citation_markers(&text).trim().to_string()
}

fn parts_text(content: &RawContent, content_type: &str) -> String {
    let mut pieces: Vec<String> = Vec::new();
    for part in &content.parts {
        collect_part_text(part, &mut pieces);
    }
    let mut text = pieces.join("\n");
    if content_type == "code" {
        let body = text.trim_end();
        if !body.is_empty() && !body.trim_start().starts_with("```") {
            text = format!("```\n{body}\n```");
        }
    }
    text
}

fn collect_part_text(part: &Value, out: &mut Vec<String>) {
    match part {
        Value::String(s) => {
            if !s.is_empty() {
                out.push(s.clone());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_part_text(item, out);
            }
        }
        Value::Object(map) => {
            let part_type = map.get("content_type").and_then(Value::as_str);
            match part_type {
                Some("image_asset_pointer") => {
                    out.push(asset_marker("image", map));
                }
                Some("audio_asset_pointer") | Some("real_time_user_audio_video_asset_pointer") => {
                    out.push(asset_marker("audio", map));
                }
                Some("video_asset_pointer") => {
                    out.push(asset_marker("video", map));
                }
                Some("audio_transcription") => {
                    if let Some(text) = map.get("text").and_then(Value::as_str) {
                        out.push(text.to_string());
                    }
                }
                _ => {
                    if let Some(text) = map.get("text").and_then(Value::as_str) {
                        out.push(text.to_string());
                    } else if let Some(text) = map.get("content").and_then(Value::as_str) {
                        out.push(text.to_string());
                    }
                }
            }
        }
        _ => {}
    }
}

fn asset_marker(kind: &str, map: &serde_json::Map<String, Value>) -> String {
    match map.get("asset_pointer").and_then(Value::as_str) {
        Some(pointer) if !pointer.is_empty() => {
            format!("[{kind}: {}]", file_id_from_pointer(pointer))
        }
        _ => format!("[{kind}]"),
    }
}

fn thoughts_text(content: &RawContent) -> String {
    let mut pieces = Vec::new();
    for thought in &content.thoughts {
        if let Some(summary) = thought.summary.as_deref() {
            if !summary.trim().is_empty() {
                pieces.push(summary.trim().to_string());
            }
        }
        if let Some(body) = thought.content.as_deref() {
            if !body.trim().is_empty() {
                pieces.push(body.trim().to_string());
            }
        }
    }
    pieces.join("\n\n")
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                if let Some(text) = value_text(item) {
                    parts.push(text);
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        Value::Object(map) => map
            .get("text")
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .or_else(|| {
                map.get("content")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
            }),
        _ => None,
    }
}

/// Remove ChatGPT's citation spans. They are opaque markers such as
/// `\u{e200}cite\u{e202}turn0file0\u{e201}` or `【cite】【turn0file0】`;
/// user-written CJK brackets are left alone.
pub fn strip_citation_markers(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{e200}' {
            for n in chars.by_ref() {
                if n == '\u{e201}' {
                    break;
                }
            }
            continue;
        }
        if c == '【' {
            let probe: String = chars.clone().take(64).collect();
            let inner = probe.split('】').next().unwrap_or(&probe);
            let lower = inner.to_ascii_lowercase();
            if lower.starts_with("cite")
                || lower.starts_with("turn")
                || lower.starts_with("filecite")
            {
                for n in chars.by_ref() {
                    if n == '】' {
                        break;
                    }
                }
                continue;
            }
        }
        out.push(c);
    }
    out
}

pub fn meta_bool(meta: Option<&Value>, key: &str) -> bool {
    meta.and_then(|m| m.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn meta_str<'a>(meta: Option<&'a Value>, key: &str) -> Option<&'a str> {
    meta.and_then(|m| m.get(key)).and_then(Value::as_str)
}

#[derive(Debug, Clone)]
pub struct AttachmentRef {
    pub file_id: String,
    pub node_id: String,
    pub name: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<i64>,
    pub source: String,
    pub asset_pointer: Option<String>,
}

/// Collect every file reference in a message: uploads listed in metadata,
/// citations, and asset pointers embedded in multimodal parts.
pub fn extract_attachment_refs(msg: &RawMessage, node_id: &str) -> Vec<AttachmentRef> {
    let mut refs = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    if let Some(meta) = msg.metadata.as_ref() {
        if let Some(items) = meta.get("attachments").and_then(Value::as_array) {
            for item in items {
                let id = item
                    .get("id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("file_id").and_then(Value::as_str))
                    .or_else(|| {
                        item.get("asset_pointer")
                            .and_then(Value::as_str)
                            .map(|_| "")
                    });
                let Some(raw_id) = id else { continue };
                let pointer = item.get("asset_pointer").and_then(Value::as_str);
                let file_id = if raw_id.is_empty() {
                    pointer.map(file_id_from_pointer).unwrap_or_default()
                } else {
                    raw_id.to_string()
                };
                if file_id.is_empty() {
                    continue;
                }
                push_ref(
                    &mut refs,
                    &mut seen,
                    AttachmentRef {
                        file_id,
                        node_id: node_id.to_string(),
                        name: item.get("name").and_then(Value::as_str).map(str::to_string),
                        mime_type: item
                            .get("mime_type")
                            .and_then(Value::as_str)
                            .or_else(|| item.get("mime").and_then(Value::as_str))
                            .map(str::to_string),
                        size_bytes: item
                            .get("size")
                            .and_then(Value::as_i64)
                            .or_else(|| item.get("size_bytes").and_then(Value::as_i64)),
                        source: "metadata".to_string(),
                        asset_pointer: pointer.map(str::to_string),
                    },
                );
            }
        }

        if let Some(items) = meta.get("citations").and_then(Value::as_array) {
            for item in items {
                let id = item
                    .pointer("/metadata/file_id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("file_id").and_then(Value::as_str));
                let Some(id) = id else { continue };
                push_ref(
                    &mut refs,
                    &mut seen,
                    AttachmentRef {
                        file_id: id.to_string(),
                        node_id: node_id.to_string(),
                        name: item
                            .pointer("/metadata/title")
                            .and_then(Value::as_str)
                            .or_else(|| item.get("title").and_then(Value::as_str))
                            .map(str::to_string),
                        mime_type: None,
                        size_bytes: None,
                        source: "citation".to_string(),
                        asset_pointer: None,
                    },
                );
            }
        }
    }

    if let Some(content) = msg.content.as_ref() {
        for part in &content.parts {
            collect_part_refs(part, node_id, &mut refs, &mut seen);
        }
    }
    refs
}

fn collect_part_refs(
    part: &Value,
    node_id: &str,
    refs: &mut Vec<AttachmentRef>,
    seen: &mut HashSet<String>,
) {
    match part {
        Value::Array(items) => {
            for item in items {
                collect_part_refs(item, node_id, refs, seen);
            }
        }
        Value::Object(map) => {
            if let Some(pointer) = map.get("asset_pointer").and_then(Value::as_str) {
                let file_id = file_id_from_pointer(pointer);
                if !file_id.is_empty() {
                    push_ref(
                        refs,
                        seen,
                        AttachmentRef {
                            file_id,
                            node_id: node_id.to_string(),
                            name: None,
                            mime_type: None,
                            size_bytes: map.get("size_bytes").and_then(Value::as_i64),
                            source: "asset_pointer".to_string(),
                            asset_pointer: Some(pointer.to_string()),
                        },
                    );
                }
            }
        }
        _ => {}
    }
}

fn push_ref(refs: &mut Vec<AttachmentRef>, seen: &mut HashSet<String>, reference: AttachmentRef) {
    let key = format!("{}|{}", reference.source, reference.file_id);
    if seen.insert(key) {
        refs.push(reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(value: Value) -> RawMessage {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn multimodal_keeps_all_text_and_marks_images() {
        let msg = message(json!({
            "author": {"role": "user"},
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    "before",
                    {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-abc"},
                    "after"
                ]
            }
        }));
        let text = extract_text(&msg);
        assert!(text.contains("before"));
        assert!(text.contains("after"));
        assert!(text.contains("[image: file-abc]"));
    }

    #[test]
    fn code_content_is_fenced() {
        let msg = message(json!({
            "content": {"content_type": "code", "parts": ["fn main() {}\n"]}
        }));
        assert_eq!(extract_text(&msg), "```\nfn main() {}\n```");
    }

    #[test]
    fn thoughts_and_fallback_result() {
        let msg = message(json!({
            "content": {"content_type": "thoughts", "thoughts": [{"summary": "plan", "content": "do it"}]}
        }));
        assert!(extract_text(&msg).contains("plan"));

        let msg = message(json!({
            "content": {"content_type": "tether_browsing_display", "result": "browsed text"}
        }));
        assert_eq!(extract_text(&msg), "browsed text");
    }

    #[test]
    fn strips_only_citation_brackets() {
        let raw = "a\u{e200}cite\u{e202}turn0file0\u{e201}b 【cite】【turn1view2】c 【注】";
        assert_eq!(strip_citation_markers(raw), "ab c 【注】");
    }

    #[test]
    fn collects_attachment_references() {
        let msg = message(json!({
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    {"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_123", "size_bytes": 10}
                ]
            },
            "metadata": {
                "attachments": [{"id": "file-abc", "name": "notes.pdf", "mime_type": "application/pdf", "size": 42}]
            }
        }));
        let refs = extract_attachment_refs(&msg, "node-1");
        assert_eq!(refs.len(), 2);
        assert!(refs
            .iter()
            .any(|r| r.file_id == "file-abc" && r.name.as_deref() == Some("notes.pdf")));
        assert!(refs
            .iter()
            .any(|r| r.file_id == "file_123" && r.size_bytes == Some(10)));
    }
}
