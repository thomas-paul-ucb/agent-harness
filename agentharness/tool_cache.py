"""Exact-match cache for tool calls.

Different scope from SemanticCache (whole tasks) and TokenBudget (context
within one run): this one avoids redoing the *same* tool call across any
run, ever. It's plain memoization, not semantic matching — tool arguments
are structured data (a search string, a calculator expression), so a call
either matches exactly or it doesn't; there's no "paraphrase" concept to
account for the way there is with natural-language tasks.

Backed by SQLite so the cache persists across process runs, not just
within one Python session.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ToolCache:
    """Persistent exact-match cache for (tool_name, arguments) -> result."""

    def __init__(self, db_path: str = ".cache/tool_cache.sqlite3") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_cache (
                cache_key TEXT PRIMARY KEY,
                result TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        # sort_keys ensures {"a": 1, "b": 2} and {"b": 2, "a": 1} hit the same entry
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"

    def lookup(self, tool_name: str, arguments: dict[str, Any]) -> Any | None:
        """Return the cached result for this exact call, or None if uncached."""
        key = self._key(tool_name, arguments)
        row = self._conn.execute(
            "SELECT result FROM tool_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def store(self, tool_name: str, arguments: dict[str, Any], result: Any) -> None:
        """Store the result of a tool call for future exact-match lookups."""
        key = self._key(tool_name, arguments)
        self._conn.execute(
            "INSERT OR REPLACE INTO tool_cache (cache_key, result) VALUES (?, ?)",
            (key, json.dumps(result)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()