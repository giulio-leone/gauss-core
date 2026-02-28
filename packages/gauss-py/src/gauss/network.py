"""
Network — Multi-agent networks with supervisor delegation via Rust.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class Network:
    """
    Multi-agent network backed by Rust.

    Example:
        >>> net = Network()
        >>> net.add_agent("researcher", provider_handle=1)
        >>> net.add_agent("writer", provider_handle=2)
        >>> net.set_supervisor("researcher")
        >>> result = await net.delegate("researcher", "Write an article")
    """

    def __init__(self) -> None:
        from gauss._native import create_network

        self._handle = create_network()

    def add_agent(
        self,
        name: str,
        provider_handle: int,
        card: Optional[dict[str, Any]] = None,
        connections: Optional[list[str]] = None,
    ) -> "Network":
        """Add an agent to the network."""
        from gauss._native import network_add_agent

        network_add_agent(
            self._handle,
            name,
            provider_handle,
            json.dumps(card) if card else None,
            connections,
        )
        return self

    def set_supervisor(self, name: str) -> "Network":
        """Set the supervisor agent for the network."""
        from gauss._native import network_set_supervisor

        network_set_supervisor(self._handle, name)
        return self

    async def delegate(self, agent_name: str, prompt: str) -> dict[str, Any]:
        """Delegate a task to a specific agent in the network."""
        from gauss._native import network_delegate

        messages = [{"role": "user", "content": prompt}]
        result_json = await network_delegate(
            self._handle, agent_name, json.dumps(messages)
        )
        return json.loads(result_json) if isinstance(result_json, str) else result_json

    def destroy(self) -> None:
        """Release the underlying Rust network."""
        from gauss._native import destroy_network

        destroy_network(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
