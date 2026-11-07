"""Optional Odysseus MemoryProvider backed by breadmemory's current context.

The default live integration is the two-way `breadmemory context sync-odysseus`
bridge, which projects facts into Odyssey's native memory.json so per-turn
injection works without patching Odysseus. This adapter is for deployments that
want breadmemory exposed through Odysseus' MemoryProvider registry as well.

It shells out to the breadmemory binary and uses only the Python standard
library. Writes are proposals and must be accepted with:

    breadmemory context accept <key>
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from src.memory_provider import MemoryProvider, MemoryRecord, MemorySearchHit

CATEGORY_MAP = {
    "identity": "identity",
    "personal": "identity",
    "preference": "preference",
    "preferences": "preference",
    "project": "project",
    "projects": "project",
    "goal": "project",
    "hardware": "hardware",
    "device": "hardware",
    "decision": "decision",
    "contact": "relationship",
    "relationship": "relationship",
    "routine": "routine",
    "habit": "routine",
    "fact": "general",
}


def _key_for(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    slug = "_".join(words)[:40] or "memory"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return f"odysseus.provider.{slug}.{digest}"


class BreadmemoryProvider(MemoryProvider):
    provider_id = "breadmemory"
    display_name = "breadmemory current context"

    def __init__(
        self,
        binary: Optional[str] = None,
        db_path: Optional[str] = None,
        actor: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.binary = binary or os.environ.get("BREADMEMORY_BIN", "breadmemory")
        self.db_path = db_path or os.environ.get("BREADMEMORY_DB")
        self.actor = actor or os.environ.get("BREADMEMORY_ACTOR", "odysseus-provider")
        self.timeout = timeout

    def _run(self, args: List[str]) -> str:
        env = os.environ.copy()
        if self.db_path:
            env["BREADMEMORY_DB"] = self.db_path
        result = subprocess.run(
            [self.binary, *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"breadmemory {args[0]} failed: {message}")
        return result.stdout

    def _json(self, args: List[str]) -> Any:
        output = self._run([*args, "--json"]).strip()
        return json.loads(output) if output else None

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def _record(self, fact: Dict[str, Any]) -> MemoryRecord:
        source = fact.get("source") or {}
        key = fact.get("key", "")
        updated = fact.get("updated_at") or ""
        try:
            timestamp = int(
                time.mktime(time.strptime(updated[:19], "%Y-%m-%dT%H:%M:%S"))
            )
        except (ValueError, OverflowError):
            timestamp = int(time.time())
        return MemoryRecord(
            id=key,
            text=fact.get("value", ""),
            timestamp=timestamp,
            category=fact.get("category", "general"),
            source=source.get("kind", "breadmemory"),
            owner=None,
            session_id=None,
            metadata={
                "breadmemory_key": key,
                "confidence": fact.get("confidence"),
                "pinned": fact.get("pinned", False),
                "provenance": {
                    "conversation_id": source.get("conversation_id"),
                    "message_id": source.get("message_id"),
                    "quote": source.get("quote"),
                },
            },
        )

    async def remember(
        self,
        text: str,
        *,
        owner: Optional[str] = None,
        session_id: Optional[str] = None,
        category: str = "fact",
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        text = (text or "").strip()
        if not text:
            raise ValueError("memory text cannot be empty")
        key = _key_for(text)
        breadmemory_category = CATEGORY_MAP.get((category or "fact").lower(), "general")
        self._run(
            [
                "context",
                "set",
                key,
                text,
                "--category",
                breadmemory_category,
                "--status",
                "proposed",
                "--actor",
                self.actor,
            ]
        )
        return MemoryRecord(
            id=key,
            text=text,
            timestamp=int(time.time()),
            category=breadmemory_category,
            source=source,
            owner=owner,
            session_id=session_id,
            metadata={
                **(metadata or {}),
                "breadmemory_key": key,
                "status": "proposed",
            },
        )

    async def recall(
        self,
        query: str,
        *,
        owner: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemorySearchHit]:
        facts = self._json(["context", "recall", query, "--limit", str(top_k)]) or []
        return [
            MemorySearchHit(
                memory=self._record(fact),
                provider_id=self.provider_id,
                score=fact.get("relevance"),
            )
            for fact in facts
        ]

    async def list_memories(
        self,
        *,
        owner: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        facts = self._json(["context", "list"]) or []
        return [self._record(fact) for fact in facts[:limit]]

    async def delete(self, memory_id: str, *, owner: Optional[str] = None) -> bool:
        try:
            self._run(
                [
                    "context",
                    "retract",
                    memory_id,
                    "--reason",
                    "retracted through Odysseus",
                    "--actor",
                    self.actor,
                ]
            )
        except RuntimeError:
            return False
        return True
