# Odysseus integration

Two integration levels are supported.

## 1. Live: two-way sync bridge (recommended)

`breadmemory context sync-odysseus <memory.json> --owner <user>` does both
directions in one pass:

- New and edited native Odysseus memories are imported into `current_context`.
- Active `current_context` facts are projected into `memory.json` with
  `source: "breadmemory"` and a `projected_text` marker.
- Edits made to a projected entry in Odysseus are detected by comparing `text`
  with `projected_text` and imported back on the next pass.
- Retracting a fact in breadmemory removes its projection.

Because Odysseus reads `memory.json` on every chat turn, this gives automatic
ChatGPT-style memory injection without patching Odysseus.

On Hestia a systemd user timer runs the sync every 30 seconds:

```
~/.config/systemd/user/breadmemory-sync.service
~/.config/systemd/user/breadmemory-sync.timer
```

It runs the breadmemory binary inside the Odysseus container with
`BREADMEMORY_DB=/app/data/breadmemory/archive.db`.

Policy: breadmemory is the source of truth. If both sides change between syncs,
the Odysseus-side edit wins and is projected back. Deleting a projected fact in
Odysseus does not delete it in breadmemory; retract it in breadmemory instead
(or via the MCP propose/accept flow).

## 2. Optional: MemoryProvider adapter

`breadmemory_provider.py` implements Odysseus' `MemoryProvider` interface and
shells out to the breadmemory CLI. Use it when you want breadmemory exposed via
the provider registry in addition to the native memory projection.

It requires the breadmemory binary on PATH or `BREADMEMORY_BIN`, and
`BREADMEMORY_DB` pointing at the archive. `remember()` writes a proposal, never
an active fact.

Copy it into the Odysseus source tree and register it in
`src/app_initializer.py`:

```python
from src.memory_provider_breadmemory import BreadmemoryProvider

memory_provider_registry = MemoryProviderRegistry([
    NativeMemoryProvider(memory_manager, memory_vector),
    BreadmemoryProvider(),
])
```

Do not register the same facts through both the provider and the native
projection if you care about avoiding duplicate listings. Either use the sync
bridge (default) or the provider, not both.

## 3. Normal chat retrieval bridge

Normal chat does not run the agent tool loop. `breadmemory_bridge.py` gives it
the useful read-only part automatically: when a message explicitly mentions
breadmemory, ChatGPT archive/history, previous chats, "what did we discuss", or
"do you remember", Odysseus runs `breadmemory search` and appends the top
results as an untrusted context block. No function schemas and no write tools.

Deployment on Hestia mounts `src/breadmemory_bridge.py` and a patched
`routes/chat_helpers.py` into the container, then sets:

```
BREADMEMORY_BRIDGE_ENABLED=1
BREADMEMORY_BIN=/opt/breadmemory/breadmemory
BREADMEMORY_DB=/app/data/breadmemory/archive.db
BREADMEMORY_MODEL_DIR=/app/data/breadmemory/models/all-MiniLM-L6-v2
ORT_DYLIB_PATH=/opt/breadmemory/libonnxruntime.so
```

The call is skipped in agent mode and incognito chats.

## 4. Agent mode tool selection

Agent mode calls tools through function schemas, and its turn-contract
families are an authorisation boundary. Breadmemory needs an explicit family:

- `src/turn_contract.py`: add a `breadmemory` family containing the six
  `mcp__8a8c0ff2e107__*` schemas, and route archive/memory phrases to it before
  generic web/search heuristics. The family is gated by the request classifier,
  not offered on every turn.
- `src/tool_index.py`: add `get_current_context` to `ALWAYS_AVAILABLE`, and map
  archive/memory phrases to the bare breadmemory tool names in
  `_KEYWORD_HINTS`.
- `src/agent_loop.py`: add `breadmemory`, `chatgpt archive`, `past chats`, and
  `chat history` to `_MCP_KEYWORDS` for local models, and match MCP schemas by
  bare tool name as well as the qualified server-ID name.

With those, an agent-mode message such as "search for Addi in breadmemory"
selects `mcp__breadmemory__search_conversations` and the other breadmemory
tools instead of reporting that no such tool exists. Verified with
`requested_capabilities()` and `resolve_turn_contract()` inside the running
container: the query resolves to family `breadmemory` and offers all six tools.
