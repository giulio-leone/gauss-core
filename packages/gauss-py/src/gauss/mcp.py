"""
MCP — Model Context Protocol server via Rust.
"""

from __future__ import annotations

import json
from typing import Any


class McpServer:
    """
    MCP (Model Context Protocol) server backed by Rust.

    Example:
        >>> server = McpServer("my-server", "1.0.0")
        >>> server.add_tool({"name": "search", "description": "Search the web", "inputSchema": {...}})
        >>> response = await server.handle({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
    """

    def __init__(self, name: str, version: str) -> None:
        from gauss._native import create_mcp_server

        self._handle = create_mcp_server(name, version)

    def add_tool(self, tool: dict[str, Any]) -> "McpServer":
        """Add a tool definition to the MCP server."""
        from gauss._native import mcp_server_add_tool

        mcp_server_add_tool(self._handle, json.dumps(tool))
        return self

    async def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC message and return the response."""
        from gauss._native import mcp_server_handle

        result_json = await mcp_server_handle(self._handle, json.dumps(message))
        return json.loads(result_json) if isinstance(result_json, str) else result_json

    def destroy(self) -> None:
        """Release the underlying Rust MCP server."""
        from gauss._native import destroy_mcp_server

        destroy_mcp_server(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
