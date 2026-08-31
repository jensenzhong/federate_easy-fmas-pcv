"""Frozen provider settings shared by strict execution and provenance gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_TEMPERATURE = 0.8
DEEPSEEK_TIMEOUT_SECONDS = 60


def deepseek_protocol_config() -> dict[str, Any]:
    return {
        "model": DEEPSEEK_MODEL,
        "base_url": DEEPSEEK_BASE_URL,
        "temperature": DEEPSEEK_TEMPERATURE,
        "timeout_seconds": DEEPSEEK_TIMEOUT_SECONDS,
    }


def deepseek_provenance(*, enabled: bool) -> dict[str, Any]:
    if type(enabled) is not bool:
        raise TypeError("enabled must be an exact boolean")
    if enabled:
        return {"enabled": True, **deepseek_protocol_config()}
    return {
        "enabled": False,
        "model": None,
        "base_url": None,
        "temperature": None,
        "timeout_seconds": None,
    }


def deepseek_client_settings_from_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if type(provenance) is not dict:
        raise ValueError("provenance must be an exact dictionary")
    deepseek = provenance.get("deepseek")
    if type(deepseek) is not dict or deepseek != deepseek_provenance(enabled=True):
        raise ValueError("provenance DeepSeek settings do not match the frozen protocol")
    return {
        "model": deepseek["model"],
        "base_url": deepseek["base_url"],
        "timeout_seconds": deepseek["timeout_seconds"],
    }


__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TEMPERATURE",
    "DEEPSEEK_TIMEOUT_SECONDS",
    "deepseek_client_settings_from_provenance",
    "deepseek_protocol_config",
    "deepseek_provenance",
]
