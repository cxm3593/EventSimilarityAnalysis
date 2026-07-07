"""Shared helpers for comparison result persistence."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import yaml  # pyright: ignore[reportMissingModuleSource]


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


def prepare_run_dir(
    output_root: str | Path,
    metric_name: str,
    config: dict[str, Any],
    *,
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped run directory and persist the config used for the run.

    The directory name includes the metric so runs are easy to tell apart in
    ``output/``. It is created immediately (before comparisons start) and a
    copy of the config is written as ``run_config.yaml``.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{metric_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return run_dir
