"""Shared test fixtures and markers."""

import pytest


def has_native() -> bool:
    """Check if the native (_native) module is available."""
    try:
        import gauss._native
        return True
    except (ImportError, ModuleNotFoundError):
        return False


requires_native = pytest.mark.skipif(
    not has_native(),
    reason="Native binary not built (requires maturin build)",
)
