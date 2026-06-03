"""Distance / similarity metric dispatch for event comparisons."""

from __future__ import annotations

from typing import Any


SUPPORTED_METRICS = frozenset({"mmd"})


def compare_events(events_a, events_b, metric: str = "mmd", **kwargs) -> dict[str, Any]:
    """Compare two event subsets with the requested metric."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric: {metric!r}. Supported metrics: {sorted(SUPPORTED_METRICS)}"
        )

    if metric == "mmd":
        from .mmd import mmd_analysis

        return mmd_analysis(events_a, events_b, **kwargs)

    raise ValueError(f"Unsupported metric: {metric!r}")
