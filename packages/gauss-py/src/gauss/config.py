"""
Config — Agent configuration utilities via Rust.
"""

from __future__ import annotations


def config_from_json(json_str: str) -> str:
    """Parse and validate an agent config from JSON. Returns normalized JSON."""
    from gauss._native import agent_config_from_json

    return agent_config_from_json(json_str)


def resolve_env(value: str) -> str:
    """Resolve environment variable references in a config value (e.g. '${OPENAI_API_KEY}')."""
    from gauss._native import agent_config_resolve_env

    return agent_config_resolve_env(value)
