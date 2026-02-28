"""
Plugin — Plugin registry for telemetry and memory plugins via Rust.
"""

from __future__ import annotations

import json
from typing import Any


class PluginRegistry:
    """
    Plugin registry backed by Rust.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.add_telemetry()
        >>> registry.add_memory()
        >>> registry.list()
        ['TelemetryPlugin', 'MemoryPlugin']
    """

    def __init__(self) -> None:
        from gauss._native import create_plugin_registry

        self._handle = create_plugin_registry()

    def add_telemetry(self) -> "PluginRegistry":
        """Register the telemetry plugin."""
        from gauss._native import plugin_registry_add_telemetry

        plugin_registry_add_telemetry(self._handle)
        return self

    def add_memory(self) -> "PluginRegistry":
        """Register the memory plugin."""
        from gauss._native import plugin_registry_add_memory

        plugin_registry_add_memory(self._handle)
        return self

    def list(self) -> list[str]:
        """List all registered plugins."""
        from gauss._native import plugin_registry_list

        return plugin_registry_list(self._handle)

    def emit(self, event: dict[str, Any]) -> None:
        """Emit an event to all registered plugins."""
        from gauss._native import plugin_registry_emit

        plugin_registry_emit(self._handle, json.dumps(event))

    def destroy(self) -> None:
        """Release the underlying Rust plugin registry."""
        from gauss._native import destroy_plugin_registry

        destroy_plugin_registry(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
