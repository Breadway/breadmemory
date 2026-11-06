use std::collections::HashSet;

pub const DEFAULT_MAX_CHARS: usize = 1000;
pub const DEFAULT_OVERLAP: usize = 120;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    start: usize,
    end: usize,
    code: bool,
}

/// Split text into overlapping chunks while keeping fenced code blocks whole
/// where possible. Offsets are byte offsets into the original text.
pub fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<Chunk> {
    let max_chars = max_chars.max(200);
    let overlap = overlap.min(max_chars / 2);

    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;

    for segment in segments(text) {
        let segment_chars = text[segment.start..segment.end].chars().count();
        if segment_chars > max_chars {
            if let Some(pair) = current.take() {
                pieces.push(pair);
            }
            pieces.extend(split_long(
                text,
                segment.start,
                segment.end,
                max_chars,
                overlap,
                segment.code,
            ));
            continue;
        }
        match current {
            None => current = Some((segment.start, segment.end)),
            Some((start, end)) => {
                let joined = text[start..end].chars().count() + 1 + segment_chars;
                if joined <= max_chars {
                    current = Some((start, segment.end));
                } else {
                    pieces.push((start, end));
                    current = Some((segment.start, segment.end));
                }
            }
        }
    }
    if let Some(pair) = current {
        pieces.push(pair);
    }

    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (start, end) in pieces {
        let (start, end) = trim_span(text, start, end);
        if start >= end {
            continue;
        }
        let piece = &text[start..end];
        if piece.chars().count() < 3 {
            continue;
        }
        if !seen.insert(piece.to_string()) {
            continue;
        }
        out.push(Chunk {
            index: out.len(),
            start,
            end,
            text: piece.to_string(),
        });
    }
    out
}

fn segments(text: &str) -> Vec<Segment> {
    let mut result = Vec::new();
    let mut prose_start: Option<usize> = None;
    let mut prose_end = 0usize;
    let mut fence_start: Option<usize> = None;

    for (line_start, line_end) in line_spans(text) {
        let line = &text[line_start..line_end];
        let trimmed = line.trim_start();
        let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");

        if let Some(start) = fence_start {
            if is_fence {
                result.push(Segment {
                    start,
                    end: line_end,
                    code: true,
                });
                fence_start = None;
            }
            continue;
        }

        if is_fence {
            if let Some(start) = prose_start.take() {
                if prose_end > start {
                    result.push(Segment {
                        start,
                        end: prose_end,
                        code: false,
                    });
                }
            }
            fence_start = Some(line_start);
            continue;
        }

        if line.trim().is_empty() {
            if let Some(start) = prose_start.take() {
                if prose_end > start {
                    result.push(Segment {
                        start,
                        end: prose_end,
                        code: false,
                    });
                }
            }
            continue;
        }

        if prose_start.is_none() {
            prose_start = Some(line_start);
        }
        prose_end = line_end;
    }

    if let Some(start) = fence_start {
        result.push(Segment {
            start,
            end: text.len(),
            code: true,
        });
    }
    if let Some(start) = prose_start {
        if text.len() > start {
            result.push(Segment {
                start,
                end: text.len(),
                code: false,
            });
        }
    }

    result.sort_by_key(|segment| segment.start);
    result
}

fn line_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    for (i, c) in text.char_indices() {
        if c == '\n' {
            spans.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < text.len() {
        spans.push((start, text.len()));
    }
    spans
}

fn split_long(
    text: &str,
    start: usize,
    end: usize,
    max_chars: usize,
    overlap: usize,
    code: bool,
) -> Vec<(usize, usize)> {
    let indices: Vec<usize> = text[start..end]
        .char_indices()
        .map(|(i, _)| start + i)
        .collect();
    let n = indices.len();
    if n <= max_chars {
        return vec![(start, end)];
    }

    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < n {
        let mut stop = (cursor + max_chars).min(n);
        if stop < n {
            let floor = cursor + (max_chars * 7 / 10);
            let mut best = None;
            for k in (floor..stop).rev() {
                let b = text.as_bytes()[indices[k]];
                let acceptable = if code {
                    b == b'\n'
                } else {
                    b == b' ' || b == b'\n' || b == b'\t'
                };
                if acceptable {
                    best = Some(k);
                    break;
                }
            }
            if let Some(k) = best {
                if k > cursor {
                    stop = k;
                }
            }
        }
        let end_byte = if stop < n { indices[stop] } else { end };
        out.push((indices[cursor], end_byte));
        if stop >= n {
            break;
        }
        let mut next = stop.saturating_sub(overlap);
        if next <= cursor {
            next = stop;
        }
        cursor = next;
    }
    out
}

fn trim_span(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    while start < end {
        let c = text[start..].chars().next().unwrap();
        if !c.is_whitespace() {
            break;
        }
        start += c.len_utf8();
    }
    while end > start {
        let c = text[..end].chars().next_back().unwrap();
        if !c.is_whitespace() {
            break;
        }
        end -= c.len_utf8();
    }
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_code_blocks_whole() {
        let intro = "Intro paragraph. ".repeat(10);
        let text =
            format!("{intro}\n\n```rust\nfn main() {{\n    println!(\"hi\");\n}}\n```\n\nOutro.");
        let chunks = chunk_text(&text, 200, 40);
        let code_chunk = chunks
            .iter()
            .find(|c| c.text.contains("fn main"))
            .expect("code chunk");
        assert!(code_chunk.text.contains("```rust"));
        assert!(code_chunk.text.contains("```"));
        assert!(!code_chunk.text.contains("Intro"));
        assert!(code_chunk.text.starts_with("```"));
    }

    #[test]
    fn long_prose_overlaps_and_advances() {
        let text = (0..200)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_text(&text, 200, 40);
        assert!(chunks.len() > 3);
        for chunk in &chunks {
            assert!(chunk.text.chars().count() <= 200);
        }
        assert!(chunks[0].end > chunks[1].start);
        assert_eq!(chunks[0].start, 0);
    }

    #[test]
    fn offsets_map_back_to_text() {
        let text = "alpha\n\nbeta\n\n```sh\necho hi\n```";
        for chunk in chunk_text(text, 300, 30) {
            assert_eq!(&text[chunk.start..chunk.end], chunk.text);
        }
    }

    #[test]
    fn unicode_is_not_split() {
        let text = "日本語のテキストを分割するテスト。".repeat(40);
        let chunks = chunk_text(&text, 100, 20);
        assert!(!chunks.is_empty());
        for chunk in chunks {
            assert_eq!(&text[chunk.start..chunk.end], chunk.text);
        }
    }
}
