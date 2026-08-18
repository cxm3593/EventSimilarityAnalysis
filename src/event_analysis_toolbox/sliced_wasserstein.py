"""Sliced Wasserstein distance between event point clouds via POT.

Operates on plain ``(N, D)`` feature matrices. Feature extraction, time
normalization, and scaling are handled by
:mod:`event_analysis_toolbox.feature_preprocessing` before this call.

POT dispatches on the array type of its inputs, so the ``backend`` option simply
controls whether the feature matrices are handed to POT as NumPy (CPU) or CuPy
(CUDA) arrays.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .mmd import _configure_cuda_path_from_python_packages


def _require_pot():
    # POT imports cupy at its own import time to register the GPU backend, and
    # cupy caches its CUDA-root detection on first import. Configure the CUDA DLL
    # path before importing ot so cupy finds the pip-installed CUDA libraries;
    # this is a no-op when the CUDA packages are not installed (numpy-only use).
    _configure_cuda_path_from_python_packages()
    try:
        import ot
    except ImportError as exc:
        raise ImportError(
            "POT (Python Optimal Transport) is required for Sliced Wasserstein "
            "distance. Install with: uv sync --extra sliced"
        ) from exc
    return ot


def _require_cupy():
    _configure_cuda_path_from_python_packages()
    try:
        cp = importlib.import_module("cupy")
    except ImportError as exc:
        raise ImportError(
            "CuPy is required for backend='cupy'. Install a CUDA-matched CuPy "
            "package, for example: uv sync --extra gpu"
        ) from exc
    return cp


def _resolve_array_module(backend):
    if backend in (None, "numpy"):
        return "numpy", np
    if backend == "cupy":
        return "cupy", _require_cupy()
    raise ValueError(
        f"Unsupported backend: {backend!r}. Supported backends: ['numpy', 'cupy']."
    )


def _validate_inputs(features_a, features_b, n_projections, p):
    if len(features_a) == 0 or len(features_b) == 0:
        raise ValueError("Both inputs must contain at least one event.")

    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError(
            "Both inputs must share the same feature dimension: "
            f"got {features_a.shape[1]} and {features_b.shape[1]}."
        )

    if n_projections <= 0:
        raise ValueError("n_projections must be positive.")

    if p < 1:
        raise ValueError("p must be >= 1.")


def sliced_wasserstein_analysis(
    features_a,
    features_b,
    *,
    n_projections: int = 100,
    p: float = 2.0,
    seed: int | None = 0,
    backend: str = "numpy",
) -> dict[str, Any]:
    """Compute the Sliced Wasserstein distance between two feature matrices.

    The Sliced Wasserstein distance averages the 1-D Wasserstein distance over
    many random linear projections of the two empirical (uniform-weight) point
    clouds. It is estimated with :func:`ot.sliced_wasserstein_distance`.

    Args:
        features_a: ``(N, D)`` float array of features for the first window.
        features_b: ``(M, D)`` float array of features for the second window.
            Feature extraction, time normalization, and scaling are handled by
            :mod:`event_analysis_toolbox.feature_preprocessing` before this call.
        n_projections: Number of random 1-D projections used for the estimate.
            Higher values give a smoother, lower-variance estimate at higher cost.
        p: Order of the Wasserstein distance (``p >= 1``). ``p = 2`` is typical.
        seed: Seed for the random projection directions. A fixed seed keeps the
            estimate reproducible across pairwise comparisons.
        backend: Array backend handed to POT: ``"numpy"`` (CPU) or ``"cupy"``
            (CUDA). POT runs the projections and 1-D sorts on the device of the
            arrays it receives.

    Returns:
        A dictionary with the Sliced Wasserstein value and the algorithm
        settings used for the comparison.
    """
    backend_name, xp = _resolve_array_module(backend)

    features_a = xp.ascontiguousarray(xp.asarray(features_a, dtype=xp.float64))
    features_b = xp.ascontiguousarray(xp.asarray(features_b, dtype=xp.float64))

    _validate_inputs(features_a, features_b, n_projections, p)

    # Captured before the arrays are released below, so the result still reports
    # the sizes actually compared.
    n_a, n_b = len(features_a), len(features_b)

    ot = _require_pot()

    try:
        raw_value = ot.sliced_wasserstein_distance(
            features_a,
            features_b,
            n_projections=n_projections,
            p=p,
            seed=seed,
        )
        # POT returns a scalar in the same backend as its inputs; pull it back to a
        # host float regardless of whether it is a NumPy or CuPy value.
        value = float(np.asarray(raw_value if backend_name == "numpy" else xp.asnumpy(raw_value)))
    finally:
        if backend_name == "cupy":
            # POT projects both clouds onto `n_projections` directions, so a single
            # call transiently allocates on the order of
            # (n_a + n_b) * n_projections * 8 bytes plus sort workspaces -- roughly
            # 6.4 GB for a 450k-event window at 100 projections. CuPy's pool caches
            # freed blocks keyed by size, and because every window has a different
            # event count no cached block is ever reusable: the pool grows until the
            # device is exhausted, after which every allocation falls into CuPy's
            # slow out-of-memory retry path. Returning the blocks here keeps device
            # usage flat for ~5% per-call overhead. `_CupyBackend.after_chunk_pair`
            # in mmd.py does the same job for the chunked MMD kernel sums.
            # Only the device pool is released. The pinned (host) pool stages
            # host-device transfers and costs no VRAM, and freeing it forces an
            # expensive re-allocation on the next call (~12% per call measured).
            del features_a, features_b
            xp.get_default_memory_pool().free_all_blocks()

    return {
        "sliced_wasserstein": value,
        "distance": value,
        "events_a": n_a,
        "events_b": n_b,
        "n_projections": n_projections,
        "p": p,
        "seed": seed,
        "backend": backend_name,
    }
