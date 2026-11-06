use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, SecondsFormat, TimeZone, Utc};
use sha2::{Digest, Sha256};

pub fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn now_iso() -> String {
    ts_to_iso(Some(now_unix())).unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string())
}

pub fn ts_to_iso(ts: Option<f64>) -> Option<String> {
    let ts = ts?;
    if !ts.is_finite() {
        return None;
    }
    let secs = ts.trunc() as i64;
    let nanos = ((ts.fract().abs()) * 1_000_000_000.0) as u32;
    Utc.timestamp_opt(secs, nanos)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
}

pub fn human_ts(ts: Option<f64>) -> String {
    let ts = match ts {
        Some(v) if v.is_finite() => v,
        _ => return "unknown".to_string(),
    };
    let secs = ts.trunc() as i64;
    match Utc.timestamp_opt(secs, 0).single() {
        Some(dt) => dt.format("%Y-%m-%d %H:%M UTC").to_string(),
        None => "unknown".to_string(),
    }
}

pub fn parse_time_bound(raw: &str, end_of_day: bool) -> Result<f64> {
    if let Ok(v) = raw.parse::<f64>() {
        return Ok(v);
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Ok(dt.timestamp() as f64);
    }
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"] {
        if let Ok(date) = NaiveDate::parse_from_str(raw, fmt) {
            let dt = date
                .and_hms_opt(0, 0, 0)
                .context("invalid date components")?
                .and_utc();
            let mut ts = dt.timestamp() as f64;
            if end_of_day {
                ts += 86_399.0;
            }
            return Ok(ts);
        }
    }
    bail!("cannot parse time '{raw}': use YYYY-MM-DD, RFC 3339, or unix seconds");
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

pub fn truncate_chars(s: &str, max: usize) -> (String, bool) {
    if s.chars().count() <= max {
        return (s.to_string(), false);
    }
    let cut: String = s.chars().take(max).collect();
    (format!("{cut}..."), true)
}

pub fn normalize_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn file_id_from_pointer(pointer: &str) -> String {
    if let Some((_, rest)) = pointer.split_once("://") {
        rest.trim_matches('/').to_string()
    } else {
        pointer.trim_matches('/').to_string()
    }
}

#[cfg(unix)]
pub fn restrict_permissions(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        let mut perms = meta.permissions();
        perms.set_mode(mode);
        let _ = std::fs::set_permissions(path, perms);
    }
}

#[cfg(not(unix))]
pub fn restrict_permissions(_path: &Path, _mode: u32) {}

pub fn ensure_private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
    restrict_permissions(path, 0o700);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_date_bounds() {
        let start = parse_time_bound("2025-01-02", false).unwrap();
        let end = parse_time_bound("2025-01-02", true).unwrap();
        assert_eq!(end - start, 86_399.0);
        assert!(parse_time_bound("nonsense", false).is_err());
    }

    #[test]
    fn pointer_ids() {
        assert_eq!(
            file_id_from_pointer("file-service://file-abc123"),
            "file-abc123"
        );
        assert_eq!(file_id_from_pointer("sediment://file_xyz"), "file_xyz");
    }
}
