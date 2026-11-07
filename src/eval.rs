use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::search::Hit;

#[derive(Debug, Clone, Deserialize)]
pub struct EvalCase {
    pub query: String,
    #[serde(default)]
    pub expected_conversation_ids: Vec<String>,
    #[serde(default)]
    pub expected_message_ids: Vec<String>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EvalSummary {
    pub label: String,
    pub cases: usize,
    pub k: usize,
    pub hits: usize,
    pub recall_at_k: f64,
    pub mrr: f64,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct CaseOutcome {
    pub query: String,
    pub rank: Option<usize>,
    pub latency_ms: f64,
    pub top_title: Option<String>,
}

pub fn load_cases(path: &Path) -> Result<Vec<EvalCase>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read eval cases {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse eval cases {}", path.display()))?;
    let cases = match value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(map) => map
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .context("eval file object has no 'cases' array")?,
        _ => bail!("eval file must be an array or an object with 'cases'"),
    };
    if cases.is_empty() {
        bail!("eval file contains no cases");
    }
    let parsed: Vec<EvalCase> = cases
        .into_iter()
        .map(serde_json::from_value)
        .collect::<std::result::Result<_, _>>()
        .context("parse eval case")?;
    Ok(parsed)
}

/// 1-based rank of the first expected conversation or message in the hits.
pub fn first_match(hits: &[Hit], case: &EvalCase, k: usize) -> Option<usize> {
    hits.iter()
        .take(k)
        .position(|hit| {
            case.expected_conversation_ids
                .iter()
                .any(|id| id == &hit.conversation_id)
                || hit
                    .message_id
                    .as_ref()
                    .map(|id| {
                        case.expected_message_ids
                            .iter()
                            .any(|expected| expected == id)
                    })
                    .unwrap_or(false)
                || case
                    .expected_message_ids
                    .iter()
                    .any(|expected| expected == &hit.node_id)
        })
        .map(|index| index + 1)
}

pub fn summarize(
    label: &str,
    k: usize,
    ranks: &[Option<usize>],
    latencies_ms: &[f64],
) -> EvalSummary {
    let cases = ranks.len();
    let hits = ranks.iter().filter(|rank| rank.is_some()).count();
    let recall_at_k = if cases == 0 {
        0.0
    } else {
        hits as f64 / cases as f64
    };
    let mrr = if cases == 0 {
        0.0
    } else {
        ranks
            .iter()
            .map(|rank| rank.map(|rank| 1.0 / rank as f64).unwrap_or(0.0))
            .sum::<f64>()
            / cases as f64
    };
    let avg_latency_ms = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64
    };
    EvalSummary {
        label: label.to_string(),
        cases,
        k,
        hits,
        recall_at_k,
        mrr,
        avg_latency_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_correct() {
        let summary = summarize(
            "test",
            5,
            &[Some(1), Some(3), None, Some(5)],
            &[10.0, 20.0, 30.0, 40.0],
        );
        assert_eq!(summary.cases, 4);
        assert_eq!(summary.hits, 3);
        assert!((summary.recall_at_k - 0.75).abs() < 1e-9);
        let expected_mrr = (1.0 + 1.0 / 3.0 + 0.0 + 1.0 / 5.0) / 4.0;
        assert!((summary.mrr - expected_mrr).abs() < 1e-9);
        assert!((summary.avg_latency_ms - 25.0).abs() < 1e-9);
    }

    #[test]
    fn empty_metrics_are_zero() {
        let summary = summarize("empty", 10, &[], &[]);
        assert_eq!(summary.cases, 0);
        assert_eq!(summary.recall_at_k, 0.0);
        assert_eq!(summary.mrr, 0.0);
    }
}
