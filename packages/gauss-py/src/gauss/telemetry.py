"""
Telemetry — Span tracking and metrics collection via Rust.
Rust returns JSON strings, not native dicts.
"""

from __future__ import annotations

import json
from typing import Any


class Telemetry:
    """
    Telemetry collector backed by Rust.

    Example:
        >>> telemetry = Telemetry()
        >>> telemetry.record_span({"name": "agent.run", "duration_ms": 150})
        >>> spans = telemetry.export_spans()
    """

    def __init__(self) -> None:
        from gauss._native import create_telemetry

        self._handle = create_telemetry()

    def record_span(self, span: dict[str, Any]) -> None:
        """Record a telemetry span."""
        from gauss._native import telemetry_record_span

        telemetry_record_span(self._handle, json.dumps(span))

    def export_spans(self) -> list[dict[str, Any]]:
        """Export all recorded spans."""
        from gauss._native import telemetry_export_spans

        result = telemetry_export_spans(self._handle)
        return json.loads(result) if isinstance(result, str) else (result or [])

    def export_metrics(self) -> dict[str, Any]:
        """Export aggregated metrics."""
        from gauss._native import telemetry_export_metrics

        result = telemetry_export_metrics(self._handle)
        return json.loads(result) if isinstance(result, str) else (result or {})

    def clear(self) -> None:
        """Clear all telemetry data."""
        from gauss._native import telemetry_clear

        telemetry_clear(self._handle)

    def destroy(self) -> None:
        """Release the underlying Rust telemetry collector."""
        from gauss._native import destroy_telemetry

        destroy_telemetry(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
