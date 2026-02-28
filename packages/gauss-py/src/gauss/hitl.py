"""
HITL — Human-in-the-loop: Approval manager and checkpoint store via Rust.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class ApprovalManager:
    """
    Human-in-the-loop approval manager backed by Rust.

    Example:
        >>> approvals = ApprovalManager()
        >>> req_id = approvals.request("dangerous_tool", {"action": "delete"}, "session_1")
        >>> approvals.approve(req_id)
    """

    def __init__(self) -> None:
        from gauss._native import create_approval_manager

        self._handle = create_approval_manager()

    def request(self, tool_name: str, args: dict[str, Any], session_id: str) -> str:
        """Request approval for a tool call. Returns the request ID."""
        from gauss._native import approval_request

        return approval_request(self._handle, tool_name, json.dumps(args), session_id)

    def approve(self, request_id: str, modified_args: Optional[dict[str, Any]] = None) -> None:
        """Approve a pending request, optionally with modified arguments."""
        from gauss._native import approval_approve

        approval_approve(
            self._handle,
            request_id,
            json.dumps(modified_args) if modified_args else None,
        )

    def deny(self, request_id: str, reason: Optional[str] = None) -> None:
        """Deny a pending request."""
        from gauss._native import approval_deny

        approval_deny(self._handle, request_id, reason)

    def list_pending(self) -> list[dict[str, Any]]:
        """List all pending approval requests."""
        from gauss._native import approval_list_pending

        result = approval_list_pending(self._handle)
        return json.loads(result) if isinstance(result, str) else result

    def destroy(self) -> None:
        from gauss._native import destroy_approval_manager

        destroy_approval_manager(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


class CheckpointStore:
    """
    Checkpoint store for saving/loading agent state via Rust.

    Example:
        >>> store = CheckpointStore()
        >>> await store.save({"id": "cp-1", "agent_name": "researcher", "state": {...}})
        >>> cp = await store.load("cp-1")
    """

    def __init__(self) -> None:
        from gauss._native import create_checkpoint_store

        self._handle = create_checkpoint_store()

    async def save(self, checkpoint: dict[str, Any]) -> None:
        """Save a checkpoint."""
        from gauss._native import checkpoint_save

        await checkpoint_save(self._handle, json.dumps(checkpoint))

    async def load(self, checkpoint_id: str) -> dict[str, Any]:
        """Load a checkpoint by ID."""
        from gauss._native import checkpoint_load

        result = await checkpoint_load(self._handle, checkpoint_id)
        return json.loads(result) if isinstance(result, str) else result

    def destroy(self) -> None:
        from gauss._native import destroy_checkpoint_store

        destroy_checkpoint_store(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
