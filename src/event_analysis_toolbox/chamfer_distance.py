"""Chamfer distance between event point clouds via Open3D."""

from __future__ import annotations

from typing import Any

import numpy as np

from .mmd import _resolve_feature_scales


_POLARITY_FIELDS = {"p", "polarity", "pol"}


def _event_feature_names(dtype):
    if dtype.names is None:
        return None
    return tuple(name for name in dtype.names if name.lower() not in _POLARITY_FIELDS)


def _events_to_features(
    events,
    *,
    feature_names=None,
    feature_scales=None,
    max_events=None,
) -> tuple[np.ndarray, tuple[str, ...] | None, int]:
    """Convert structured event data to an ``(N, D)`` feature matrix."""
    events = np.asarray(events)
    n_events = len(events)
    if max_events is not None:
        n_events = min(n_events, max_events)
        events = events[:n_events]

    if events.dtype.names is not None:
        if feature_names is None:
            feature_names = _event_feature_names(events.dtype)
        features = np.column_stack([events[name] for name in feature_names])
    else:
        features = np.asarray(events, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        feature_names = None

    resolved_scales = _resolve_feature_scales(feature_scales, feature_names)
    if resolved_scales is not None:
        features = features / np.asarray(resolved_scales, dtype=np.float64)

    return np.asarray(features, dtype=np.float64), feature_names, n_events


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for Chamfer distance. Install with: uv sync --extra chamfer"
        ) from exc
    return o3d


def _resolve_dtype(o3d, dtype: str):
    dtype_key = dtype.lower()
    if not hasattr(o3d.core, dtype_key):
        raise ValueError(f"Unsupported chamfer dtype: {dtype!r}. Use 'float32' or 'float64'.")
    return getattr(o3d.core, dtype_key)


def _resolve_device(o3d, device: str):
    device_name = device.strip()
    if not device_name:
        raise ValueError("chamfer device must be a non-empty string such as 'CUDA:0' or 'CPU:0'.")

    o3d_device = o3d.core.Device(device_name)
    if o3d_device.get_type() == o3d.core.Device.DeviceType.CUDA:
        if not o3d.core.cuda.is_available():
            raise RuntimeError(
                f"chamfer device is {device_name!r} but Open3D CUDA is unavailable. "
                "Install an Open3D build with CUDA support "
                "(https://www.open3d.org/docs/release/compilation.html) "
                "or set chamfer.device to 'CPU:0'."
            )
    return o3d_device


def _features_to_point_cloud(o3d, features: np.ndarray, device, dtype):
    point_cloud = o3d.t.geometry.PointCloud(device)
    point_cloud.point.positions = o3d.core.Tensor(
        np.asarray(features, dtype=np.float32 if dtype == o3d.core.float32 else np.float64),
        dtype,
        device,
    )
    return point_cloud


def _validate_inputs(
    events_a,
    events_b,
    *,
    max_events,
    device,
    dtype,
):
    if len(events_a) == 0 or len(events_b) == 0:
        raise ValueError("Both inputs must contain at least one event.")

    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive.")

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")

    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'.")


def chamfer_analysis(
    events_a,
    events_b,
    *,
    feature_names=None,
    feature_scales=None,
    max_events,
    device,
    dtype,
) -> dict[str, Any]:
    """Compute Chamfer distance between two event subsets using Open3D.

    Event polarity fields (``p``, ``polarity``, ``pol``) are ignored automatically
    for structured event arrays unless explicitly listed in ``feature_names``.

    Open3D defines Chamfer distance as the sum of mean nearest-neighbor
    distances in both directions between the two point clouds.
    """
    _validate_inputs(
        events_a,
        events_b,
        max_events=max_events,
        device=device,
        dtype=dtype,
    )

    o3d = _require_open3d()
    o3d_device = _resolve_device(o3d, device)
    o3d_dtype = _resolve_dtype(o3d, dtype)

    features_a, names_a, n_a = _events_to_features(
        events_a,
        feature_names=feature_names,
        feature_scales=feature_scales,
        max_events=max_events,
    )
    features_b, names_b, n_b = _events_to_features(
        events_b,
        feature_names=feature_names or names_a,
        feature_scales=feature_scales,
        max_events=max_events,
    )

    if names_a != names_b:
        raise ValueError("events_a and events_b must have the same non-polarity fields.")

    point_cloud_a = _features_to_point_cloud(o3d, features_a, o3d_device, o3d_dtype)
    point_cloud_b = _features_to_point_cloud(o3d, features_b, o3d_device, o3d_dtype)

    metric_params = o3d.t.geometry.MetricParameters()
    metrics = point_cloud_a.compute_metrics(
        point_cloud_b,
        (o3d.t.geometry.Metric.ChamferDistance,),
        metric_params,
    )

    value = float(metrics.cpu().numpy()[0])
    return {
        "chamfer_distance": value,
        "distance": value,
        "events_a": n_a,
        "events_b": n_b,
        "features": names_a,
        "feature_scales": _resolve_feature_scales(feature_scales, names_a),
        "device": device,
        "dtype": dtype,
        "max_events": max_events,
        "open3d_cuda_available": bool(o3d.core.cuda.is_available()),
    }
