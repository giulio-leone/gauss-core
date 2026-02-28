"""
Memory — In-memory session persistence via native Rust.
Rust signature: memory_store(handle, entry_json), memory_recall(handle, options_json), memory_clear(handle, session_id?)
"""

from __future__ import annotations

import json
from typing import Any, Optional


class Memory:
    """
    In-memory session memory backed by Rust.

    Example:
        >>> memory = Memory()
        >>> memory.store({"session_id": "s1", "content": "Hello", "metadata": {}})
        >>> results = memory.recall(session_id="s1")
    """

    def __init__(self) -> None:
        from gauss._native import create_memory

        self._handle = create_memory()

    async def store(self, entry: dict[str, Any]) -> None:
        """Store a memory entry. Entry must have 'session_id', 'content', optional 'metadata'."""
        from gauss._native import memory_store

        await memory_store(self._handle, json.dumps(entry))

    async def recall(
        self,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Recall memories matching options."""
        from gauss._native import memory_recall

        options: dict[str, Any] = {}
        if session_id:
            options["session_id"] = session_id
        if query:
            options["query"] = query
        options["limit"] = limit

        result_json = await memory_recall(self._handle, json.dumps(options))
        return json.loads(result_json) if result_json else []

    async def clear(self, session_id: Optional[str] = None) -> None:
        """Clear memories, optionally filtering by session."""
        from gauss._native import memory_clear

        await memory_clear(self._handle, session_id)

    def destroy(self) -> None:
        """Release the underlying Rust memory."""
        from gauss._native import destroy_memory

        destroy_memory(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
