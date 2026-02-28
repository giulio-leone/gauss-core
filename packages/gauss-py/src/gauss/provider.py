"""
Provider — Native Rust-backed LLM provider factory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Provider:
    """Handle to a native Rust LLM provider."""

    handle: int
    provider_type: str
    model: str

    def destroy(self) -> None:
        """Release the underlying Rust provider."""
        from gauss._native import destroy_provider

        destroy_provider(self.handle)


# API key environment variable mapping
_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_AI_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": "",
    "deepseek": "DEEPSEEK_API_KEY",
}


def gauss(
    provider_type: str,
    model: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> Provider:
    """
    Create a native Rust-backed LLM provider.

    Args:
        provider_type: One of "openai", "anthropic", "google", "groq", "ollama", "deepseek"
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        api_key: API key (auto-detected from environment if not provided)
        base_url: Custom base URL
        max_retries: Max retry attempts

    Returns:
        Provider handle for use with Agent()

    Example:
        >>> model = gauss("openai", "gpt-4o")
        >>> model = gauss("anthropic", "claude-sonnet-4-20250514", api_key="sk-...")
    """
    from gauss._native import create_provider

    if api_key is None:
        env_var = _API_KEY_ENV.get(provider_type, "")
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(
                    f"Missing API key for {provider_type}. "
                    f"Set {env_var} environment variable or pass api_key."
                )
        else:
            api_key = ""

    handle = create_provider(
        provider_type, model, api_key, base_url=base_url, max_retries=max_retries
    )

    return Provider(handle=handle, provider_type=provider_type, model=model)
