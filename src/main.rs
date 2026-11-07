use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};

use breadmemory::context::SetRequest;
use breadmemory::importer::ImportOptions;
use breadmemory::util::{parse_time_bound, truncate_chars};
use breadmemory::{context, db, embedding, eval, importer, mcp, render, search};

#[derive(Parser)]
#[command(
    name = "breadmemory",
    version,
    about = "Local-first archive of ChatGPT conversations with a current-context memory layer",
    long_about = "breadmemory imports the official ChatGPT data export into a private SQLite archive, \
indexes it with FTS5, and exposes it to assistants over MCP. A separate, human-managed current \
context layer stores durable facts with provenance."
)]
struct Cli {
    /// Archive database path (default: <home>/archive.db)
    #[arg(long, global = true, env = "BREADMEMORY_DB", value_name = "PATH")]
    db: Option<PathBuf>,

    /// Data directory for the archive and extracted attachments
    #[arg(long, global = true, env = "BREADMEMORY_HOME", value_name = "DIR")]
    home: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create the database and private data directory
    Init,

    /// Import a ChatGPT export zip or an extracted export directory
    Import {
        path: PathBuf,
        #[arg(
            long,
            help = "Re-import conversations even when their content hash is unchanged"
        )]
        force: bool,
        #[arg(long, help = "Record attachment references without extracting files")]
        no_attachments: bool,
        #[arg(long)]
        json: bool,
    },

    /// Show archive and import status
    Status {
        #[arg(long)]
        json: bool,
    },

    /// List conversations, newest first
    Conversations {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long)]
        json: bool,
    },

    /// Keyword or hybrid search the archive
    Search {
        #[arg(required = true, num_args = 1..)]
        query: Vec<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long)]
        role: Option<String>,
        #[arg(long, help = "YYYY-MM-DD, RFC 3339, or unix seconds")]
        after: Option<String>,
        #[arg(long, help = "YYYY-MM-DD, RFC 3339, or unix seconds")]
        before: Option<String>,
        #[arg(long)]
        include_hidden: bool,
        #[arg(long, help = "Skip the semantic lane even when embeddings exist")]
        keyword_only: bool,
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f64,
        #[arg(long)]
        json: bool,
    },

    /// Print a conversation
    Show {
        conversation: String,
        #[arg(long)]
        all_branches: bool,
        #[arg(long)]
        include_hidden: bool,
        #[arg(long, default_value_t = 40_000)]
        max_chars: usize,
        #[arg(long)]
        json: bool,
    },

    /// Retrieve a message with surrounding context
    ContextWindow {
        message: String,
        #[arg(long)]
        conversation: Option<String>,
        #[arg(long, default_value_t = 3)]
        before: usize,
        #[arg(long, default_value_t = 3)]
        after: usize,
        #[arg(long)]
        include_hidden: bool,
        #[arg(long)]
        json: bool,
    },

    /// Find related archived discussions
    Related {
        #[arg(num_args = 0..)]
        query: Vec<String>,
        #[arg(long)]
        message: Option<String>,
        #[arg(long)]
        conversation: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },

    /// Current-context memory layer commands
    Context {
        #[command(subcommand)]
        command: ContextCommand,
    },

    /// Print the preserved original conversation JSON
    Raw { conversation: String },

    /// Rebuild chunks and the FTS index from stored messages
    Reindex,

    /// Rebuild only the FTS index from stored chunks
    RebuildFts,

    /// Generate local embeddings for chunks that do not have them
    Embed {
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        #[arg(long, help = "Embed at most this many chunks")]
        limit: Option<usize>,
        #[arg(long, help = "Delete existing vectors and embed every chunk again")]
        all: bool,
        #[arg(long)]
        json: bool,
    },

    /// Run the retrieval evaluation harness
    Eval {
        cases: PathBuf,
        #[arg(long, default_value_t = 5)]
        k: usize,
        #[arg(long, help = "Also evaluate hybrid retrieval when embeddings exist")]
        compare: bool,
        #[arg(long)]
        json: bool,
    },

    /// Run the MCP server over stdio
    Serve,
}

#[derive(Subcommand)]
enum ContextCommand {
    /// List current facts
    List {
        #[arg(long)]
        category: Option<String>,
        #[arg(long, help = "Include proposed, superseded and retracted facts")]
        all: bool,
        #[arg(long, help = "Only proposed facts")]
        proposed: bool,
        #[arg(long)]
        json: bool,
    },

    /// Show one active fact by key
    Get {
        key: String,
        #[arg(long)]
        json: bool,
    },

    /// Set an active fact, with optional provenance
    Set {
        key: String,
        value: String,
        #[arg(long, default_value = "general")]
        category: String,
        #[arg(long, default_value = "stated")]
        confidence: String,
        #[arg(long, default_value = "active", value_parser = ["active", "proposed"])]
        status: String,
        #[arg(long, help = "Always inject this fact during recall")]
        pinned: bool,
        #[arg(long)]
        conversation: Option<String>,
        #[arg(long)]
        message: Option<String>,
        #[arg(long)]
        quote: Option<String>,
        #[arg(long)]
        note: Option<String>,
        #[arg(long, env = "BREADMEMORY_ACTOR", default_value = "cli")]
        actor: String,
        #[arg(long)]
        json: bool,
    },

    /// Accept the newest proposal for a key
    Accept {
        key: String,
        #[arg(long, env = "BREADMEMORY_ACTOR", default_value = "cli")]
        actor: String,
        #[arg(long)]
        json: bool,
    },

    /// Retract an active fact
    Retract {
        key: String,
        #[arg(long)]
        reason: Option<String>,
        #[arg(long, env = "BREADMEMORY_ACTOR", default_value = "cli")]
        actor: String,
        #[arg(long)]
        json: bool,
    },

    /// Rank facts for a query (pinned facts plus keyword matches)
    Recall {
        #[arg(required = true, num_args = 1..)]
        query: Vec<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long)]
        include_proposed: bool,
        #[arg(long)]
        json: bool,
    },

    /// Export the current context layer as JSON
    Export {
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Restore the current context layer from a breadmemory JSON export
    Import {
        file: PathBuf,
        #[arg(long, help = "Delete existing facts before restoring")]
        replace: bool,
        #[arg(long, env = "BREADMEMORY_ACTOR", default_value = "cli")]
        actor: String,
    },

    /// Import Odysseus memory.json into the current context layer
    #[command(name = "import-odysseus")]
    ImportOdysseus {
        path: PathBuf,
        #[arg(long, env = "BREADMEMORY_ACTOR", default_value = "cli")]
        actor: String,
        #[arg(long)]
        json: bool,
    },

    /// Export active facts in Odysseus memory.json format
    #[command(name = "export-odysseus")]
    ExportOdysseus {
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Import new Odysseus memories, then project active facts back into memory.json
    #[command(name = "sync-odysseus")]
    SyncOdysseus {
        path: PathBuf,
        #[arg(long, env = "BREADMEMORY_ODYSSEUS_OWNER")]
        owner: Option<String>,
        #[arg(long, env = "BREADMEMORY_ACTOR", default_value = "cli")]
        actor: String,
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("breadmemory: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let home = cli.home.unwrap_or_else(default_home);
    let db_path = cli.db.unwrap_or_else(|| home.join("archive.db"));
    let db_label = db_path.display().to_string();

    match cli.command {
        Command::Init => {
            let conn = db::open(&db_path)?;
            let _ = conn;
            println!("Archive ready at {db_label}");
            println!(
                "Attachments directory: {}",
                home.join("attachments").display()
            );
            Ok(())
        }

        Command::Import {
            path,
            force,
            no_attachments,
            json,
        } => {
            let mut conn = db::open(&db_path)?;
            let mut options = ImportOptions::new(home.clone());
            options.force = force;
            options.extract_attachments = !no_attachments;
            options.progress = true;
            let report = importer::import(&mut conn, &path, &options)?;
            if json {
                print_json(&report.to_json())?;
            } else {
                print!("{}", render::render_import(&report));
                for error in &report.errors {
                    eprintln!("warning: {error}");
                }
            }
            Ok(())
        }

        Command::Status { json } => {
            let conn = db::open(&db_path)?;
            let status = search::archive_status(&conn, &db_label)?;
            if json {
                print_json(&search::status_json(&status))?;
            } else {
                print!("{}", render::render_status(&status));
            }
            Ok(())
        }

        Command::Conversations {
            limit,
            offset,
            json,
        } => {
            let conn = db::open(&db_path)?;
            let items = search::list_conversations(&conn, limit.clamp(1, 500), offset)?;
            if json {
                let values: Vec<Value> = items
                    .iter()
                    .map(search::conversation_summary_json)
                    .collect();
                print_json(&Value::Array(values))?;
            } else {
                print!("{}", render::render_conversations(&items));
            }
            Ok(())
        }

        Command::Search {
            query,
            limit,
            role,
            after,
            before,
            include_hidden,
            keyword_only,
            rrf_k,
            json,
        } => {
            let conn = db::open(&db_path)?;
            let query = query.join(" ");
            let options = search::SearchOptions {
                limit,
                role,
                after: after
                    .as_deref()
                    .map(|value| parse_time_bound(value, false))
                    .transpose()?,
                before: before
                    .as_deref()
                    .map(|value| parse_time_bound(value, true))
                    .transpose()?,
                include_hidden,
                ..Default::default()
            };
            let embedding_status = embedding::status(&conn)?;
            let mut mode = "keyword";
            let hits = if keyword_only || !embedding_status.ready {
                search::search_keyword(&conn, &query, &options)?
            } else {
                match embedding::Embedder::load(&embedding::default_model_dir(&home)) {
                    Ok(mut embedder) => {
                        let (hits, selected_mode) = search::search_hybrid(
                            &conn,
                            &mut embedder,
                            &query,
                            &options,
                            false,
                            false,
                            rrf_k,
                        )?;
                        mode = selected_mode;
                        hits
                    }
                    Err(err) => {
                        eprintln!(
                            "breadmemory: semantic search unavailable ({err:#}); using keyword search"
                        );
                        search::search_keyword(&conn, &query, &options)?
                    }
                }
            };
            if json {
                let values: Vec<Value> = hits.iter().map(search::hit_to_json).collect();
                print_json(&json!({ "mode": mode, "results": values }))?;
            } else {
                print!("{}", render::render_search(&hits, &query, mode));
            }
            Ok(())
        }

        Command::Show {
            conversation,
            all_branches,
            include_hidden,
            max_chars,
            json,
        } => {
            let conn = db::open(&db_path)?;
            let detail = search::get_conversation(
                &conn,
                &conversation,
                include_hidden,
                all_branches,
                max_chars,
            )?;
            if json {
                print_json(&search::conversation_detail_json(&detail))?;
            } else {
                print!("{}", render::render_conversation(&detail));
            }
            Ok(())
        }

        Command::ContextWindow {
            message,
            conversation,
            before,
            after,
            include_hidden,
            json,
        } => {
            let conn = db::open(&db_path)?;
            let result = search::get_context(
                &conn,
                &message,
                conversation.as_deref(),
                before,
                after,
                include_hidden,
            )?;
            if json {
                print_json(&search::context_result_json(&result))?;
            } else {
                print!("{}", render::render_context(&result));
            }
            Ok(())
        }

        Command::Related {
            query,
            message,
            conversation,
            limit,
            json,
        } => {
            let conn = db::open(&db_path)?;
            let (text, exclude) = match message.as_deref() {
                Some(reference) => {
                    let (conversation_id, _message_id, text) =
                        search::message_text(&conn, reference, conversation.as_deref())?;
                    (text, Some(conversation_id))
                }
                None => {
                    if query.is_empty() {
                        bail!("pass a query or --message");
                    }
                    (query.join(" "), None)
                }
            };
            let embedding_status = embedding::status(&conn)?;
            let mut embedder = if embedding_status.ready {
                match embedding::Embedder::load(&embedding::default_model_dir(&home)) {
                    Ok(loaded) => Some(loaded),
                    Err(err) => {
                        eprintln!(
                            "breadmemory: semantic relatedness unavailable ({err:#}); using keyword"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let (groups, mode) = search::find_related(
                &conn,
                embedder.as_mut(),
                &text,
                exclude.as_deref(),
                limit,
                false,
            )?;
            if json {
                print_json(&json!({
                    "mode": mode,
                    "related": search::related_json(&groups),
                }))?;
            } else {
                print!("{}", render::render_related(&groups, mode));
            }
            Ok(())
        }

        Command::Context { command } => context_command(&db_path, command),

        Command::Raw { conversation } => {
            let conn = db::open(&db_path)?;
            let id = search::resolve_conversation_id(&conn, &conversation)?;
            let compressed: Vec<u8> = conn
                .query_row(
                    "SELECT data FROM conversation_raw WHERE conversation_id = ?1",
                    [&id],
                    |row| row.get(0),
                )
                .with_context(|| format!("no preserved raw JSON for conversation {id}"))?;
            let decoded = zstd::stream::decode_all(&compressed[..])
                .context("decompress preserved conversation JSON")?;
            std::io::stdout().write_all(&decoded)?;
            println!();
            Ok(())
        }

        Command::Reindex => {
            let mut conn = db::open(&db_path)?;
            let report = importer::reindex(&mut conn, &db_path)?;
            println!(
                "Reindexed {} messages into {} chunks; FTS index rebuilt from chunks.",
                report.messages, report.chunks
            );
            Ok(())
        }

        Command::RebuildFts => {
            let conn = db::open(&db_path)?;
            importer::rebuild_fts(&conn)?;
            println!("FTS index rebuilt from stored chunks.");
            Ok(())
        }

        Command::Embed {
            batch_size,
            limit,
            all,
            json,
        } => {
            let mut conn = db::open(&db_path)?;
            if all {
                embedding::clear(&conn)?;
            }
            let model_dir = embedding::default_model_dir(&home);
            let mut embedder = embedding::Embedder::load(&model_dir)?;
            let started = std::time::Instant::now();
            let count = embedding::embed_pending(
                &mut conn,
                &mut embedder,
                embedding::MODEL_ID,
                batch_size,
                limit,
                true,
            )?;
            let status = embedding::status(&conn)?;
            if json {
                print_json(&json!({
                    "embedded": count,
                    "elapsed_ms": started.elapsed().as_millis(),
                    "model": status.model,
                    "dims": status.dims,
                    "embedded_chunks": status.embedded_chunks,
                    "total_chunks": status.total_chunks,
                    "coverage": status.coverage(),
                }))?;
            } else {
                println!(
                    "Embedded {count} chunks in {:.1}s ({}/{} chunks, {:.1}% coverage).",
                    started.elapsed().as_secs_f64(),
                    status.embedded_chunks,
                    status.total_chunks,
                    status.coverage() * 100.0
                );
            }
            Ok(())
        }

        Command::Eval {
            cases,
            k,
            compare,
            json,
        } => {
            let conn = db::open(&db_path)?;
            let cases = eval::load_cases(&cases)?;
            let k = k.clamp(1, 50);
            let options = search::SearchOptions {
                limit: k,
                ..Default::default()
            };
            let (keyword_outcomes, keyword_summary) = evaluate_mode(
                &cases,
                k,
                |query| search::search_keyword(&conn, query, &options),
                "keyword",
            )?;
            let embedding_status = embedding::status(&conn)?;
            let mut hybrid_result = None;
            if compare && embedding_status.ready {
                let model_dir = embedding::default_model_dir(&home);
                let mut embedder = embedding::Embedder::load(&model_dir)?;
                hybrid_result = Some(evaluate_mode(
                    &cases,
                    k,
                    |query| {
                        search::search_hybrid(
                            &conn,
                            &mut embedder,
                            query,
                            &options,
                            false,
                            false,
                            60.0,
                        )
                        .map(|(hits, _)| hits)
                    },
                    "hybrid",
                )?);
            }

            if json {
                let mut document = json!({
                    "cases": cases.len(),
                    "k": k,
                    "keyword": summary_json(&keyword_summary),
                    "keyword_details": keyword_outcomes.iter().map(outcome_json).collect::<Vec<_>>(),
                });
                if let Some((outcomes, summary)) = &hybrid_result {
                    document["hybrid"] = summary_json(summary);
                    document["hybrid_details"] =
                        Value::Array(outcomes.iter().map(outcome_json).collect());
                }
                print_json(&document)?;
            } else {
                print_eval_summary(&keyword_outcomes, &keyword_summary);
                if let Some((outcomes, summary)) = &hybrid_result {
                    println!();
                    print_eval_summary(outcomes, summary);
                } else if compare {
                    println!("\nHybrid evaluation skipped: embeddings are not ready.");
                }
            }
            Ok(())
        }

        Command::Serve => mcp::serve(&db_path),
    }
}

fn context_command(db_path: &std::path::Path, command: ContextCommand) -> Result<()> {
    let mut conn = db::open(db_path)?;
    match command {
        ContextCommand::List {
            category,
            all,
            proposed,
            json,
        } => {
            let statuses: Vec<&str> = if all {
                context::STATUSES.to_vec()
            } else if proposed {
                vec!["proposed"]
            } else {
                vec!["active"]
            };
            let facts = context::list(&conn, &statuses, category.as_deref(), None)?;
            if json {
                let values: Vec<Value> = facts.iter().map(context::fact_to_json).collect();
                print_json(&Value::Array(values))?;
            } else {
                print!("{}", render::render_facts(&facts, "Current context"));
            }
            Ok(())
        }

        ContextCommand::Get { key, json } => {
            let fact = context::get_active(&conn, &key)?
                .with_context(|| format!("no active fact for key '{key}'"))?;
            if json {
                print_json(&context::fact_to_json(&fact))?;
            } else {
                print!("{}", render::render_facts(&[fact], "Current context"));
            }
            Ok(())
        }

        ContextCommand::Set {
            key,
            value,
            category,
            confidence,
            status,
            pinned,
            conversation,
            message,
            quote,
            note,
            actor,
            json,
        } => {
            let mut request = SetRequest::manual(&key, &value, &actor);
            request.category = category;
            request.confidence = confidence;
            request.status = status;
            request.pinned = pinned;
            request.note = note;
            let mut quote = quote;
            let mut message = message;
            let mut conversation_id = match conversation.as_deref() {
                Some(reference) => Some(search::resolve_conversation_id(&conn, reference)?),
                None => None,
            };
            if let Some(reference) = message.as_deref() {
                let (found_conversation, found_message, text) =
                    search::message_text(&conn, reference, conversation_id.as_deref())?;
                if let Some(existing) = conversation_id.as_deref() {
                    if existing != found_conversation {
                        bail!(
                            "message belongs to conversation {found_conversation}, not {existing}"
                        );
                    }
                }
                conversation_id = Some(found_conversation);
                message = Some(found_message);
                if quote.is_none() {
                    let (snippet, _) = truncate_chars(&text, 400);
                    quote = Some(snippet);
                }
            }
            if conversation_id.is_some() {
                request.source_kind = "conversation".to_string();
            }
            request.source_conversation_id = conversation_id;
            request.source_message_id = message;
            request.source_quote = quote;
            let fact = context::set_fact(&mut conn, &request)?;
            if json {
                print_json(&context::fact_to_json(&fact))?;
            } else {
                println!(
                    "{} = {} ({} {}, {})",
                    fact.key, fact.value, fact.status, fact.category, fact.confidence
                );
            }
            Ok(())
        }

        ContextCommand::Accept { key, actor, json } => {
            let fact = context::accept_proposal(&mut conn, &key, &actor)?;
            if json {
                print_json(&context::fact_to_json(&fact))?;
            } else {
                println!("Accepted {} = {}", fact.key, fact.value);
            }
            Ok(())
        }

        ContextCommand::Retract {
            key,
            reason,
            actor,
            json,
        } => {
            let fact = context::retract(&mut conn, &key, &actor, reason.as_deref())?;
            if json {
                print_json(&context::fact_to_json(&fact))?;
            } else {
                println!("Retracted {}", fact.key);
            }
            Ok(())
        }

        ContextCommand::Recall {
            query,
            limit,
            include_proposed,
            json,
        } => {
            let query = query.join(" ");
            let results = context::recall(&conn, &query, limit, include_proposed)?;
            if json {
                let values: Vec<Value> = results
                    .iter()
                    .map(|result| {
                        let mut value = context::fact_to_json(&result.fact);
                        value["relevance"] = json!(result.score);
                        value
                    })
                    .collect();
                print_json(&Value::Array(values))?;
            } else if results.is_empty() {
                println!("No matching current facts.");
            } else {
                for result in &results {
                    println!(
                        "{:.2}  {} = {}{}",
                        result.score,
                        result.fact.key,
                        result.fact.value,
                        if result.fact.pinned { " [pinned]" } else { "" }
                    );
                }
            }
            Ok(())
        }

        ContextCommand::Export { out } => {
            let document = context::export_json(&conn)?;
            write_json(&document, out.as_deref())
        }

        ContextCommand::Import {
            file,
            replace,
            actor,
        } => {
            let document: Value = serde_json::from_slice(
                &std::fs::read(&file).with_context(|| format!("read {}", file.display()))?,
            )
            .with_context(|| format!("parse {}", file.display()))?;
            let count = context::import_json(&mut conn, &document, &actor, replace)?;
            println!("Restored {count} current-context fact(s).");
            Ok(())
        }

        ContextCommand::ImportOdysseus { path, actor, json } => {
            let report = context::import_odysseus(&mut conn, &path, &actor)?;
            if json {
                print_json(&json!({
                    "source": report.source,
                    "imported": report.imported,
                    "updated": report.updated,
                    "skipped": report.skipped,
                    "errors": report.errors,
                }))?;
            } else {
                println!("Imported from {}", report.source);
                println!(
                    "  {} new, {} updated, {} skipped",
                    report.imported, report.updated, report.skipped
                );
                for error in &report.errors {
                    eprintln!("warning: {error}");
                }
            }
            Ok(())
        }

        ContextCommand::ExportOdysseus { out } => {
            let document = context::export_odysseus(&conn)?;
            write_json(&document, out.as_deref())
        }

        ContextCommand::SyncOdysseus {
            path,
            owner,
            actor,
            json,
        } => {
            let owner = owner.unwrap_or_default();
            let report = context::sync_odysseus(&mut conn, &path, &owner, &actor)?;
            if json {
                print_json(&json!({
                    "source": report.source,
                    "imported": report.imported,
                    "updated": report.updated,
                    "skipped": report.skipped,
                    "native_edits": report.native_edits,
                    "projected": report.projected,
                    "native_kept": report.native_kept,
                    "errors": report.errors,
                }))?;
            } else {
                println!("Synced {}", report.source);
                println!(
                    "  imported {} new, {} updated, {} skipped, {} native edit(s), {} projected, {} native kept",
                    report.imported,
                    report.updated,
                    report.skipped,
                    report.native_edits,
                    report.projected,
                    report.native_kept
                );
                for error in &report.errors {
                    eprintln!("warning: {error}");
                }
            }
            Ok(())
        }
    }
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn write_json(value: &Value, out: Option<&std::path::Path>) -> Result<()> {
    let text = format!("{}\n", serde_json::to_string_pretty(value)?);
    match out {
        Some(path) => {
            std::fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
        }
        None => print!("{text}"),
    }
    Ok(())
}

fn default_home() -> PathBuf {
    if let Ok(home) = std::env::var("BREADMEMORY_HOME") {
        return PathBuf::from(home);
    }
    if let Ok(data) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(data).join("breadmemory");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".local/share/breadmemory")
}

fn evaluate_mode(
    cases: &[eval::EvalCase],
    k: usize,
    mut run: impl FnMut(&str) -> Result<Vec<search::Hit>>,
    label: &str,
) -> Result<(Vec<eval::CaseOutcome>, eval::EvalSummary)> {
    let mut outcomes = Vec::with_capacity(cases.len());
    let mut ranks = Vec::with_capacity(cases.len());
    let mut latencies = Vec::with_capacity(cases.len());
    for case in cases {
        let started = std::time::Instant::now();
        let hits = run(&case.query)?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let rank = eval::first_match(&hits, case, k);
        ranks.push(rank);
        latencies.push(latency_ms);
        outcomes.push(eval::CaseOutcome {
            query: case.query.clone(),
            rank,
            latency_ms,
            top_title: hits.first().map(|hit| hit.conversation_title.clone()),
        });
    }
    let summary = eval::summarize(label, k, &ranks, &latencies);
    Ok((outcomes, summary))
}

fn summary_json(summary: &eval::EvalSummary) -> Value {
    json!({
        "label": summary.label,
        "cases": summary.cases,
        "k": summary.k,
        "hits": summary.hits,
        "recall_at_k": summary.recall_at_k,
        "mrr": summary.mrr,
        "avg_latency_ms": summary.avg_latency_ms,
    })
}

fn outcome_json(outcome: &eval::CaseOutcome) -> Value {
    json!({
        "query": outcome.query,
        "rank": outcome.rank,
        "latency_ms": outcome.latency_ms,
        "top_title": outcome.top_title,
    })
}

fn print_eval_summary(outcomes: &[eval::CaseOutcome], summary: &eval::EvalSummary) {
    for outcome in outcomes {
        match outcome.rank {
            Some(rank) => println!(
                "  hit  #{rank}  {:.0} ms  {}",
                outcome.latency_ms, outcome.query
            ),
            None => println!(
                "  MISS      {:.0} ms  {}  (top: {})",
                outcome.latency_ms,
                outcome.query,
                outcome.top_title.as_deref().unwrap_or("none")
            ),
        }
    }
    println!(
        "{}: {}/{} hits, recall@{} {:.3}, MRR {:.3}, {:.0} ms/query",
        summary.label,
        summary.hits,
        summary.cases,
        summary.k,
        summary.recall_at_k,
        summary.mrr,
        summary.avg_latency_ms
    );
}
