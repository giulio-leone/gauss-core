"""
Stream — Partial JSON parsing utility via Rust.
"""

from __future__ import annotations

from typing import Optional


def parse_partial_json(text: str) -> Optional[str]:
    """Parse potentially incomplete JSON from streaming, returns completed JSON or None."""
    from gauss._native import py_parse_partial_json

    return py_parse_partial_json(text)
