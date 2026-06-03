"""Shared helpers for comparison result persistence."""

from __future__ import annotations

from typing import Any


def yaml_safe(value):
    """Recursively convert values into YAML-friendly Python primitives."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe(v) for v in value]
    return repr(value)


def safe_file_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return stem.strip("_") or "results"
