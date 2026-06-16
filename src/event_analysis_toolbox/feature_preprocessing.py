"""Shared feature preprocessing for event comparisons.

This is the single place where structured event data is turned into the plain
``(N, D)`` float matrices that metrics operate on. It handles:

* dropping polarity fields,
* normalizing the time axis so each window starts at ``t = 0``,
* applying per-feature scaling.

Metrics receive ready-to-use feature arrays and stay free of any field/scaling
logic.
"""

from __future__ import annotations

import numpy as np


POLARITY_FIELDS = {"p", "polarity", "pol"}


def resolve_feature_names(dtype, feature_names=None) -> tuple[str, ...] | None:
    """Return the feature field names to compare.

    When ``feature_names`` is given it is used as-is (polarity fields rejected).
    Otherwise the structured-array fields are used with polarity dropped. Returns
    ``None`` for unstructured arrays.
    """
    if feature_names is not None:
        lowered = {name.lower() for name in feature_names}
        if lowered & POLARITY_FIELDS:
            raise ValueError("feature_names must not include polarity fields.")
        return tuple(feature_names)

    if dtype.names is None:
        return None

    return tuple(name for name in dtype.names if name.lower() not in POLARITY_FIELDS)


def resolve_feature_scales(feature_scales, feature_names, default=1.0):
    """Normalize ``feature_scales`` into a sequence aligned with ``feature_names``.

    Accepts either a sequence (returned unchanged) or a ``{name: scale}`` mapping
    (reordered to match ``feature_names``; missing names default to ``default``).
    Returns ``None`` when ``feature_scales`` is ``None``.
    """
    if feature_scales is None:
        return None

    if isinstance(feature_scales, dict):
        if feature_names is None:
            raise ValueError("feature_scales as a mapping requires named feature_names.")
        return tuple(float(feature_scales.get(name, default)) for name in feature_names)

    return feature_scales


def window_features(
    events,
    *,
    feature_names=None,
    feature_scales=None,
    time_field: str = "t",
    time_origin=None,
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Convert one window's events into a metric-ready ``(N, D)`` float matrix.

    Args:
        events: Structured event array (or plain array-like) for a single window.
        feature_names: Optional explicit fields to compare. Polarity fields are
            rejected. Defaults to all non-polarity structured fields.
        feature_scales: Optional per-feature divisors (sequence or
            ``{name: scale}`` mapping) applied after time normalization.
        time_field: Name of the time field to normalize.
        time_origin: Value subtracted from the time column so the window starts at
            ``t = 0`` (typically the window's start time). Skipped when ``None``.

    Returns:
        A tuple of the ``(N, D)`` float64 feature matrix and the feature names
        (``None`` for unstructured input).
    """
    events = np.asarray(events)
    names = resolve_feature_names(events.dtype, feature_names)

    if names is not None:
        features = np.column_stack([events[name] for name in names]).astype(np.float64)
    else:
        features = np.asarray(events, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

    if time_origin is not None and names is not None and time_field in names:
        time_index = names.index(time_field)
        features[:, time_index] -= float(time_origin)

    scales = resolve_feature_scales(feature_scales, names)
    if scales is not None:
        features = features / np.asarray(scales, dtype=np.float64)

    return features, names
