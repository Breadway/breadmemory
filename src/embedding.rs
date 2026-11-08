use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use rusqlite::{params, Connection};
use tokenizers::utils::padding::{PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::TruncationParams;
use tokenizers::Tokenizer;

use crate::util::now_iso;

pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const MODEL_DIMS: usize = 384;
pub const MAX_TOKENS: usize = 128;

enum Backend {
    Onnx {
        session: Session,
        tokenizer: Tokenizer,
        with_token_type_ids: bool,
        sentence_embedding: bool,
    },
    Http {
        url: String,
        model: String,
    },
}

pub struct Embedder {
    backend: Backend,
}

impl Embedder {
    pub fn load(model_dir: &Path) -> Result<Self> {
        if let Ok(url) = std::env::var("BREADMEMORY_EMBEDDING_URL") {
            let url = url.trim().to_string();
            if !url.is_empty() {
                let model = std::env::var("BREADMEMORY_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| MODEL_ID.to_string());
                eprintln!("breadmemory: using HTTP embedding service {url} (model {model})");
                return Ok(Self {
                    backend: Backend::Http { url, model },
                });
            }
        }

        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !model_path.exists() || !tokenizer_path.exists() {
            bail!(
                "embedding model not found in {}: expected model.onnx and tokenizer.json.\n\
                 Copy or download {MODEL_ID} into that directory, or set BREADMEMORY_MODEL_DIR.",
                model_dir.display()
            );
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("load tokenizer {}: {err}", tokenizer_path.display()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_TOKENS),
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|err| anyhow!("configure tokenizer truncation: {err}"))?;

        let mut builder = Session::builder().context("create ONNX Runtime session builder")?;
        builder = builder
            .with_intra_threads(4)
            .map_err(|err| anyhow!("set ONNX Runtime intra-op threads: {err}"))?;
        let session = builder
            .commit_from_file(&model_path)
            .with_context(|| format!("load ONNX model {}", model_path.display()))?;
        let with_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        let sentence_embedding = session
            .outputs()
            .iter()
            .any(|output| output.name() == "sentence_embedding");

        Ok(Self {
            backend: Backend::Onnx {
                session,
                tokenizer,
                with_token_type_ids,
                sentence_embedding,
            },
        })
    }

    /// Embed a batch of texts. Vectors are mean-pooled and L2-normalised.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &mut self.backend {
            Backend::Onnx {
                session,
                tokenizer,
                with_token_type_ids,
                sentence_embedding,
            } => embed_onnx(
                session,
                tokenizer,
                *with_token_type_ids,
                *sentence_embedding,
                texts,
            ),
            Backend::Http { url, model } => embed_http(url, model, texts),
        }
    }
}

fn embed_onnx(
    session: &mut Session,
    tokenizer: &Tokenizer,
    with_token_type_ids: bool,
    sentence_embedding: bool,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let owned: Vec<String> = texts.iter().map(|text| (*text).to_string()).collect();
    let encodings = tokenizer
        .encode_batch(owned, true)
        .map_err(|err| anyhow!("tokenize batch: {err}"))?;
    let batch = encodings.len();
    let seq_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(0);
    if batch == 0 || seq_len == 0 {
        bail!("tokenizer produced an empty batch");
    }

    let mut ids = vec![0i64; batch * seq_len];
    let mut mask = vec![0i64; batch * seq_len];
    let mut types = vec![0i64; batch * seq_len];
    for (row, encoding) in encodings.iter().enumerate() {
        for (col, id) in encoding.get_ids().iter().enumerate() {
            ids[row * seq_len + col] = *id as i64;
        }
        for (col, value) in encoding.get_attention_mask().iter().enumerate() {
            mask[row * seq_len + col] = *value as i64;
        }
        for (col, value) in encoding.get_type_ids().iter().enumerate() {
            types[row * seq_len + col] = *value as i64;
        }
    }

    let shape = vec![batch as i64, seq_len as i64];
    let ids = Tensor::from_array((shape.clone(), ids)).context("build input_ids tensor")?;
    let mask_tensor = Tensor::from_array((shape.clone(), mask.clone()))
        .context("build attention_mask tensor")?;

    let mut inputs: Vec<(&str, Tensor<i64>)> =
        vec![("input_ids", ids), ("attention_mask", mask_tensor)];
    if with_token_type_ids {
        inputs.push((
            "token_type_ids",
            Tensor::from_array((shape, types)).context("build token_type_ids tensor")?,
        ));
    }
    let outputs = session.run(inputs).context("run embedding model")?;

    if sentence_embedding {
        let (out_shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .context("extract sentence_embedding")?;
        let dims = *out_shape.last().unwrap_or(&0) as usize;
        if dims != MODEL_DIMS {
            bail!("model returned {dims} dimensions, expected {MODEL_DIMS}");
        }
        let mut vectors = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = row * dims;
            let mut vector = data[start..start + dims].to_vec();
            normalize(&mut vector);
            vectors.push(vector);
        }
        return Ok(vectors);
    }

    let (out_shape, data) = outputs["last_hidden_state"]
        .try_extract_tensor::<f32>()
        .context("extract last_hidden_state")?;
    if out_shape.len() != 3 {
        bail!("unexpected last_hidden_state rank: {}", out_shape.len());
    }
    let out_seq = out_shape[1] as usize;
    let dims = out_shape[2] as usize;
    if dims != MODEL_DIMS {
        bail!("model returned {dims} dimensions, expected {MODEL_DIMS}");
    }

    let mut vectors = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut vector = vec![0f32; dims];
        let mut count = 0f32;
        for token in 0..out_seq {
            if mask[row * seq_len + token] == 0 {
                continue;
            }
            count += 1.0;
            let offset = (row * out_seq + token) * dims;
            for (dim, value) in data[offset..offset + dims].iter().enumerate() {
                vector[dim] += *value;
            }
        }
        let divisor = count.max(1.0);
        for value in &mut vector {
            *value /= divisor;
        }
        normalize(&mut vector);
        vectors.push(vector);
    }
    Ok(vectors)
}

fn embed_http(url: &str, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let response = ureq::post(url)
        .timeout(std::time::Duration::from_secs(30))
        .send_json(serde_json::json!({ "input": texts, "model": model }))
        .map_err(|err| anyhow!("embedding service {url}: {err}"))?;
    let value: serde_json::Value = response
        .into_json()
        .context("parse embedding response")?;
    let data = value
        .get("data")
        .and_then(|value| value.as_array())
        .ok_or_else(|| anyhow!("embedding response has no data array"))?;
    if data.len() != texts.len() {
        bail!(
            "embedding service returned {} vectors for {} texts",
            data.len(),
            texts.len()
        );
    }
    let mut vectors = Vec::with_capacity(data.len());
    for item in data {
        let embedding = item
            .get("embedding")
            .and_then(|value| value.as_array())
            .ok_or_else(|| anyhow!("embedding response item has no embedding array"))?;
        if embedding.len() != MODEL_DIMS {
            bail!(
                "embedding service returned {} dimensions, expected {MODEL_DIMS}",
                embedding.len()
            );
        }
        let mut vector = Vec::with_capacity(MODEL_DIMS);
        for value in embedding {
            vector.push(
                value
                    .as_f64()
                    .ok_or_else(|| anyhow!("embedding value is not a number"))?
                    as f32,
            );
        }
        normalize(&mut vector);
        vectors.push(vector);
    }
    Ok(vectors)
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

pub fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

pub fn default_model_dir(data_dir: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("BREADMEMORY_MODEL_DIR") {
        return PathBuf::from(path);
    }
    data_dir.join("models").join("all-MiniLM-L6-v2")
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddingStatus {
    pub model: Option<String>,
    pub dims: Option<usize>,
    pub embedded_chunks: i64,
    pub total_chunks: i64,
    pub ready: bool,
}

impl EmbeddingStatus {
    pub fn coverage(&self) -> f64 {
        if self.total_chunks == 0 {
            0.0
        } else {
            self.embedded_chunks as f64 / self.total_chunks as f64
        }
    }
}

pub fn status(conn: &Connection) -> Result<EmbeddingStatus> {
    let model = crate::db::get_meta(conn, "embedding_model")?;
    let dims = crate::db::get_meta(conn, "embedding_dims")?.and_then(|value| value.parse().ok());
    let embedded_chunks = conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| {
        row.get(0)
    })?;
    let total_chunks = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    Ok(EmbeddingStatus {
        model,
        dims,
        embedded_chunks,
        total_chunks,
        ready: embedded_chunks > 0,
    })
}

/// Embed chunks that do not have a vector yet. Returns the number embedded.
pub fn embed_pending(
    conn: &mut Connection,
    embedder: &mut Embedder,
    model: &str,
    batch_size: usize,
    limit: Option<usize>,
    progress: bool,
) -> Result<usize> {
    let batch_size = batch_size.clamp(1, 128);
    let mut done = 0usize;

    loop {
        let remaining = limit
            .map(|limit| limit.saturating_sub(done))
            .unwrap_or(usize::MAX);
        if remaining == 0 {
            break;
        }
        let take = remaining.min(batch_size);
        let rows: Vec<(i64, String)> = {
            let mut stmt = conn.prepare(
                "SELECT c.id, c.text FROM chunks c
                 LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
                 WHERE e.chunk_id IS NULL
                 ORDER BY c.id LIMIT ?1",
            )?;
            let rows =
                stmt.query_map(params![take as i64], |row| Ok((row.get(0)?, row.get(1)?)))?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row?);
            }
            values
        };
        if rows.is_empty() {
            break;
        }

        let texts: Vec<&str> = rows.iter().map(|(_, text)| text.as_str()).collect();
        let vectors = embedder.embed(&texts)?;
        if vectors.len() != rows.len() {
            bail!(
                "embedder returned {} vectors for {} chunks",
                vectors.len(),
                rows.len()
            );
        }

        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO chunk_embeddings
                    (chunk_id, model, dims, vector, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let embedded_at = now_iso();
            for ((chunk_id, _), vector) in rows.iter().zip(&vectors) {
                stmt.execute(params![
                    chunk_id,
                    model,
                    vector.len() as i64,
                    encode_vector(vector),
                    embedded_at
                ])?;
            }
        }
        tx.commit()?;
        done += rows.len();
        if progress && done.is_multiple_of(batch_size.max(1)) {
            eprintln!("breadmemory: embedded {done} chunks");
        }
    }

    if done > 0 {
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedding_model', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![model],
        )?;
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedding_dims', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![MODEL_DIMS.to_string()],
        )?;
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedded_at', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![now_iso()],
        )?;
    }
    Ok(done)
}

pub fn clear(conn: &Connection) -> Result<()> {
    conn.execute("DELETE FROM chunk_embeddings", [])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_round_trip() {
        let vector = vec![0.25f32, -0.5, 1.0];
        let bytes = encode_vector(&vector);
        assert_eq!(decode_vector(&bytes), vector);
        assert_eq!(dot(&vector, &vector), 1.3125);
    }

    #[test]
    fn normalization_is_unit_length() {
        let mut vector = vec![3.0, 4.0];
        normalize(&mut vector);
        assert!((dot(&vector, &vector) - 1.0).abs() < 1e-6);
    }
}
