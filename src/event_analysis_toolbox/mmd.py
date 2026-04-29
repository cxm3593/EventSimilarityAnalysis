import importlib.util
import math
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]


_POLARITY_FIELDS = {"p", "polarity", "pol"}
_NVIDIA_CUDA_PACKAGES = ("nvidia.cuda_nvrtc", "nvidia.cublas")
_CUDA_PATH_CONFIGURED = False


class _NumpyBackend:
    name = "numpy"
    xp = np

    def asarray(self, values):
        return self.xp.asarray(values, dtype=self.xp.float64)

    def to_float(self, value):
        return float(value)

    def after_chunk_pair(self):
        pass


def _configure_cuda_path_from_python_packages():
    # Idempotent: registering the same DLL directory many times on Windows
    # eventually yields WinError 206 ("filename or extension too long"), so we
    # only run this configuration once per process.
    global _CUDA_PATH_CONFIGURED
    if _CUDA_PATH_CONFIGURED:
        return
    _CUDA_PATH_CONFIGURED = True

    cuda_roots = []

    for package_name in _NVIDIA_CUDA_PACKAGES:
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.submodule_search_locations is None:
            continue
        cuda_roots.append(Path(next(iter(spec.submodule_search_locations))))

    if not cuda_roots:
        return

    if not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = str(cuda_roots[0])

    if os.name == "nt":
        for cuda_root in cuda_roots:
            cuda_bin = cuda_root / "bin"
            if cuda_bin.exists():
                os.add_dll_directory(str(cuda_bin))


class _CupyBackend:
    name = "cupy"

    def __init__(self):
        _configure_cuda_path_from_python_packages()

        try:
            cp = importlib.import_module("cupy")
        except ImportError as exc:
            raise ImportError(
                "CuPy is required for backend='cupy'. Install a CUDA-matched CuPy "
                "package, for example: uv add cupy-cuda12x"
            ) from exc

        self.xp = cp

    def asarray(self, values):
        return self.xp.asarray(values, dtype=self.xp.float64)

    def to_float(self, value):
        return float(self.xp.asnumpy(value))

    def after_chunk_pair(self):
        self.xp.get_default_memory_pool().free_all_blocks()


def _resolve_feature_scales(feature_scales, feature_names, default=1.0):
    """Normalize ``feature_scales`` into a sequence aligned with ``feature_names``.

    Accepts either a sequence (returned unchanged) or a ``{name: scale}`` mapping
    (which is reordered to match ``feature_names`` and missing names get
    ``default``). Returns ``None`` when ``feature_scales`` is ``None``.
    """
    if feature_scales is None:
        return None

    if isinstance(feature_scales, dict):
        if feature_names is None:
            raise ValueError("feature_scales as a mapping requires named feature_names.")
        return tuple(float(feature_scales.get(name, default)) for name in feature_names)

    return feature_scales


def _resolve_backend(backend):
    if backend in (None, "numpy"):
        return _NumpyBackend()

    if backend == "cupy":
        return _CupyBackend()

    raise ValueError(f"Unsupported backend: {backend}")


def _event_feature_names(dtype):
    if dtype.names is None:
        return None

    return tuple(name for name in dtype.names if name.lower() not in _POLARITY_FIELDS)


def _data_feature_names(data):
    dtype = getattr(data, "dtype", None)
    if dtype is None:
        return None

    return _event_feature_names(dtype)


def _event_chunk_to_features(events, feature_names=None, feature_scales=None, backend=None):
    backend = backend or _NumpyBackend()
    events = np.asarray(events)

    if events.dtype.names is not None:
        if feature_names is None:
            feature_names = _event_feature_names(events.dtype)
        features = np.column_stack([events[name] for name in feature_names])
    else:
        features = events
        if features.ndim == 1:
            features = features.reshape(-1, 1)

    features = backend.asarray(features)

    if feature_scales is not None:
        features = features / backend.asarray(feature_scales)

    return features


def _iter_event_chunks(
    data,
    chunk_size,
    max_events=None,
    feature_names=None,
    feature_scales=None,
    backend=None,
):
    n_events = len(data)
    if max_events is not None:
        n_events = min(n_events, max_events)

    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        yield _event_chunk_to_features(data[start:stop], feature_names, feature_scales, backend)


def _chunk_count(n_events, chunk_size):
    return math.ceil(n_events / chunk_size)


def _rbf_kernel_sum(left, right, gamma, backend):
    if left.size == 0 or right.size == 0:
        return 0.0

    xp = backend.xp
    left_norm = xp.sum(left * left, axis=1)[:, None]
    right_norm = xp.sum(right * right, axis=1)[None, :]
    squared_distance = xp.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return backend.to_float(xp.exp(-gamma * squared_distance).sum())


def _kernel_sum(
    data_a,
    data_b,
    chunk_size,
    max_events,
    feature_names,
    feature_scales,
    gamma,
    backend,
    progress,
    description,
):
    n_a = min(len(data_a), max_events) if max_events is not None else len(data_a)
    n_b = min(len(data_b), max_events) if max_events is not None else len(data_b)
    total_chunk_pairs = _chunk_count(n_a, chunk_size) * _chunk_count(n_b, chunk_size)
    total = 0.0

    with tqdm(
        total=total_chunk_pairs,
        desc=description,
        unit="chunk-pair",
        disable=not progress,
    ) as progress_bar:
        for chunk_a in _iter_event_chunks(
            data_a, chunk_size, max_events, feature_names, feature_scales, backend
        ):
            for chunk_b in _iter_event_chunks(
                data_b, chunk_size, max_events, feature_names, feature_scales, backend
            ):
                total += _rbf_kernel_sum(chunk_a, chunk_b, gamma, backend)
                backend.after_chunk_pair()
                progress_bar.update(1)

    return total


def _self_kernel_sum(
    data,
    chunk_size,
    max_events,
    feature_names,
    feature_scales,
    gamma,
    biased,
    backend,
    progress,
    description,
):
    n_events = min(len(data), max_events) if max_events is not None else len(data)
    total = _kernel_sum(
        data,
        data,
        chunk_size,
        max_events,
        feature_names,
        feature_scales,
        gamma,
        backend,
        progress,
        description,
    )

    if not biased:
        # The RBF value of every event with itself is 1.0, so remove the diagonal.
        total -= n_events

    return total


def _validate_inputs(real_data, v2e_data, chunk_size, sigma, gamma, feature_names, max_events):
    if len(real_data) == 0 or len(v2e_data) == 0:
        raise ValueError("Both inputs must contain at least one event.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if sigma is None and gamma is None:
        raise ValueError("Pass either sigma or gamma.")

    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive.")

    if gamma is not None and gamma <= 0:
        raise ValueError("gamma must be positive.")

    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive.")

    if feature_names is not None:
        lowered = {name.lower() for name in feature_names}
        if lowered & _POLARITY_FIELDS:
            raise ValueError("feature_names must not include polarity fields.")


def mmd_analysis(
    real_data,
    v2e_data,
    *,
    chunk_size=50_000,
    sigma=1.0,
    gamma=None,
    feature_names=None,
    feature_scales=None,
    max_events=None,
    biased=False,
    backend="numpy",
    progress=True,
):
    """
    Performs a chunked RBF-kernel MMD analysis between real and v2e event data.

    The event polarity field is ignored automatically for structured event arrays
    with a field named "p", "polarity", or "pol".

    Args:
        real_data: Real event data. Supports HDF5 datasets, structured arrays, or
            plain array-like data.
        v2e_data: V2E event data. Uses the same format as real_data.
        chunk_size: Number of events to load per chunk while computing kernel sums.
        sigma: RBF kernel bandwidth. Ignored when gamma is provided.
        gamma: RBF kernel coefficient. If None, uses 1 / (2 * sigma ** 2).
        feature_names: Optional structured-array fields to compare. Polarity fields
            are rejected if explicitly provided.
        feature_scales: Optional per-feature divisors applied before the kernel
            so axes with different units (e.g. pixels vs. microseconds) become
            comparable. May be either a sequence aligned with ``feature_names``
            or a mapping ``{feature_name: scale}`` (missing names default to 1).
        max_events: Optional cap on the number of events read from each input.
        biased: If True, use the biased MMD estimator. Otherwise use the unbiased
            estimator with self-kernel diagonals removed.
        backend: Array backend to use for kernel math. Supports "numpy" and "cupy".
        progress: If True, show tqdm progress bars for each chunked kernel pass.

    Returns:
        A dictionary containing the MMD value, squared MMD value, kernel sums, and
        algorithm settings used for the comparison.
    """
    _validate_inputs(real_data, v2e_data, chunk_size, sigma, gamma, feature_names, max_events)

    if gamma is None:
        gamma = 1.0 / (2.0 * sigma**2)
        sigma_used = sigma
    else:
        sigma_used = None

    array_backend = _resolve_backend(backend)

    real_features = feature_names or _data_feature_names(real_data)
    v2e_features = feature_names or _data_feature_names(v2e_data)

    if real_features != v2e_features:
        raise ValueError("real_data and v2e_data must have the same non-polarity fields.")

    resolved_feature_scales = _resolve_feature_scales(feature_scales, real_features)

    n_real = min(len(real_data), max_events) if max_events is not None else len(real_data)
    n_v2e = min(len(v2e_data), max_events) if max_events is not None else len(v2e_data)

    if not biased and (n_real < 2 or n_v2e < 2):
        raise ValueError("The unbiased MMD estimator requires at least two events per input.")

    real_kernel_sum = _self_kernel_sum(
        real_data,
        chunk_size,
        max_events,
        real_features,
        resolved_feature_scales,
        gamma,
        biased,
        array_backend,
        progress,
        "MMD real-real",
    )
    v2e_kernel_sum = _self_kernel_sum(
        v2e_data,
        chunk_size,
        max_events,
        v2e_features,
        resolved_feature_scales,
        gamma,
        biased,
        array_backend,
        progress,
        "MMD v2e-v2e",
    )
    cross_kernel_sum = _kernel_sum(
        real_data,
        v2e_data,
        chunk_size,
        max_events,
        real_features,
        resolved_feature_scales,
        gamma,
        array_backend,
        progress,
        "MMD real-v2e",
    )

    if biased:
        real_denominator = n_real * n_real
        v2e_denominator = n_v2e * n_v2e
    else:
        real_denominator = n_real * (n_real - 1)
        v2e_denominator = n_v2e * (n_v2e - 1)

    mmd_squared = (
        real_kernel_sum / real_denominator
        + v2e_kernel_sum / v2e_denominator
        - 2.0 * cross_kernel_sum / (n_real * n_v2e)
    )

    return {
        "mmd": math.sqrt(max(mmd_squared, 0.0)),
        "mmd_squared": mmd_squared,
        "real_events": n_real,
        "v2e_events": n_v2e,
        "features": real_features,
        "feature_scales": resolved_feature_scales,
        "gamma": gamma,
        "sigma": sigma_used,
        "chunk_size": chunk_size,
        "biased": biased,
        "backend": array_backend.name,
        "kernel_sums": {
            "real": real_kernel_sum,
            "v2e": v2e_kernel_sum,
            "cross": cross_kernel_sum,
        },
    }