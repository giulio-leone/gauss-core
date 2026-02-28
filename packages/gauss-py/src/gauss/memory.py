"""
Memory — In-memory session persistence via native Rust.
"""

from __future__ import annotations

import json
from typing import Any


class Memory:
    """
    In-memory session memory backed by Rust.

    Example:
        >>> memory = Memory()
        >>> memory.store("session_1", "User likes pizza")
        >>> results = memory.recall("session_1", "food preferences")
    """

    def __init__(self) -> None:
        from gauss._native import create_memory

        self._handle = create_memory()

    def store(self, session_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a memory entry."""
        from gauss._native import memory_store

        meta_json = json.dumps(metadata) if metadata else "{}"
        memory_store(self._handle, session_id, content, meta_json)

    def recall(self, session_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Recall memories matching a query."""
        from gauss._native import memory_recall

        result_json = memory_recall(self._handle, session_id, query, limit)
        return json.loads(result_json) if result_json else []

    def clear(self, session_id: str) -> None:
        """Clear all memories for a session."""
        from gauss._native import memory_clear

        memory_clear(self._handle, session_id)

    def destroy(self) -> None:
        """Release the underlying Rust memory."""
        from gauss._native import destroy_memory

        destroy_memory(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
