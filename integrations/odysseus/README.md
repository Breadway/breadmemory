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
