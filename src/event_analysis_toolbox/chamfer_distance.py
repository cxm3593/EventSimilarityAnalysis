"""Chamfer distance between event point clouds via Open3D.

Operates on plain ``(N, D)`` feature matrices. Feature extraction, time
normalization, and scaling are handled by
:mod:`event_analysis_toolbox.feature_preprocessing` before this call.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
    np_dtype = np.float32 if dtype == o3d.core.float32 else np.float64
    point_cloud = o3d.t.geometry.PointCloud(device)
    point_cloud.point.positions = o3d.core.Tensor(
        np.ascontiguousarray(features, dtype=np_dtype),
        dtype,
        device,
    )
    return point_cloud


def _validate_inputs(features_a, features_b, device, dtype):
    if len(features_a) == 0 or len(features_b) == 0:
        raise ValueError("Both inputs must contain at least one event.")

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")

    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'.")


def chamfer_analysis(features_a, features_b, *, device, dtype) -> dict[str, Any]:
    """Compute Chamfer distance between two feature matrices using Open3D.

    Open3D defines Chamfer distance as the sum of mean nearest-neighbor
    distances in both directions between the two point clouds.

    Args:
        features_a: ``(N, D)`` float array of features for the first window.
        features_b: ``(M, D)`` float array of features for the second window.
        device: Open3D device string, e.g. ``"CUDA:0"`` or ``"CPU:0"``.
        dtype: Tensor dtype, ``"float32"`` or ``"float64"``.
    """
    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)

    _validate_inputs(features_a, features_b, device, dtype)

    o3d = _require_open3d()
    o3d_device = _resolve_device(o3d, device)
    o3d_dtype = _resolve_dtype(o3d, dtype)

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
        "events_a": len(features_a),
        "events_b": len(features_b),
        "device": device,
        "dtype": dtype,
        "open3d_cuda_available": bool(o3d.core.cuda.is_available()),
    }
