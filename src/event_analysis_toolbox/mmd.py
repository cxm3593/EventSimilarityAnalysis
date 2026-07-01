import importlib.util
import math
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]


_NVIDIA_CUDA_PACKAGES = ("nvidia.cuda_nvrtc", "nvidia.cublas")
_CUDA_PATH_CONFIGURED = False
_DEFAULT_RBF_TARGET_SIMILARITY = 0.01


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


def _resolve_backend(backend):
    if backend in (None, "numpy"):
        return _NumpyBackend()

    if backend == "cupy":
        return _CupyBackend()

    raise ValueError(f"Unsupported backend: {backend}")


def rbf_kernel_params_from_max_distance(
    max_distance,
    target_similarity=_DEFAULT_RBF_TARGET_SIMILARITY,
):
    """Compute RBF parameters for a target similarity at a max distance.

    The RBF kernel is ``exp(-gamma * distance ** 2)`` and
    ``gamma = 1 / (2 * sigma ** 2)``.
    """
    if max_distance <= 0:
        raise ValueError("rbf_kernel_max_distance must be positive.")
    if not 0 < target_similarity < 1:
        raise ValueError("rbf_kernel_target_similarity must be between 0 and 1.")

    log_drop = -math.log(target_similarity)
    gamma = log_drop / (max_distance**2)
    sigma = max_distance / math.sqrt(2.0 * log_drop)

    return {
        "gamma": gamma,
        "sigma": sigma,
        "max_distance": max_distance,
        "target_similarity": target_similarity,
    }


def _resolve_rbf_kernel_params(
    sigma,
    gamma,
    rbf_kernel_max_distance,
    rbf_kernel_target_similarity,
):
    if rbf_kernel_max_distance is not None:
        params = rbf_kernel_params_from_max_distance(
            rbf_kernel_max_distance,
            rbf_kernel_target_similarity,
        )
        return params["sigma"], params["gamma"], params

    if gamma is None:
        gamma = 1.0 / (2.0 * sigma**2)
        sigma_used = sigma
    else:
        sigma_used = None

    return sigma_used, gamma, None


def _resolve_mmd_kernels(
    kernels,
    sigma,
    gamma,
    rbf_kernel_max_distance,
    rbf_kernel_target_similarity,
):
    if kernels is None:
        if rbf_kernel_max_distance is None:
            sigma_used, gamma_used, rbf_kernel_params = _resolve_rbf_kernel_params(
                sigma,
                gamma,
                None,
                rbf_kernel_target_similarity,
            )
            return [{
                "index": 0,
                "gamma": gamma_used,
                "sigma": sigma_used,
                "rbf_kernel_params": rbf_kernel_params,
                "rbf_kernel_max_distance": None,
                "rbf_kernel_target_similarity": None,
            }]
        kernels = [{
            "rbf_kernel_max_distance": rbf_kernel_max_distance,
            "rbf_kernel_target_similarity": rbf_kernel_target_similarity,
        }]

    if not kernels:
        raise ValueError("MMD requires at least one RBF kernel.")

    resolved = []
    for index, kernel in enumerate(kernels):
        max_distance = kernel.get("rbf_kernel_max_distance")
        target_similarity = kernel.get(
            "rbf_kernel_target_similarity",
            _DEFAULT_RBF_TARGET_SIMILARITY,
        )
        if max_distance is None:
            raise ValueError(
                "Each MMD kernel must define rbf_kernel_max_distance."
            )

        sigma_used, gamma_used, rbf_kernel_params = _resolve_rbf_kernel_params(
            None,
            None,
            max_distance,
            target_similarity,
        )
        resolved.append({
            "index": index,
            "gamma": gamma_used,
            "sigma": sigma_used,
            "rbf_kernel_params": rbf_kernel_params,
            "rbf_kernel_max_distance": max_distance,
            "rbf_kernel_target_similarity": target_similarity,
        })

    return resolved


def _resolve_kernel_weights(kernels, weight_method):
    if weight_method != "uniform":
        raise ValueError(
            f"Unsupported MMD weight_method: {weight_method!r}. "
            "Supported methods: ['uniform']"
        )

    weight = 1.0 / len(kernels)
    return [
        {
            **kernel,
            "raw_weight": 1.0,
            "weight": weight,
            "weight_method": weight_method,
        }
        for kernel in kernels
    ]


def _iter_feature_chunks(features, chunk_size, backend):
    for start in range(0, len(features), chunk_size):
        yield backend.asarray(features[start:start + chunk_size])


def _chunk_count(n_events, chunk_size):
    return math.ceil(n_events / chunk_size)


def _rbf_kernel_sums(left, right, kernels, backend):
    if left.size == 0 or right.size == 0:
        return [0.0 for _ in kernels]

    xp = backend.xp
    left_norm = xp.sum(left * left, axis=1)[:, None]
    right_norm = xp.sum(right * right, axis=1)[None, :]
    squared_distance = xp.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return [
        backend.to_float(xp.exp(-kernel["gamma"] * squared_distance).sum())
        for kernel in kernels
    ]


def _kernel_sums(data_a, data_b, chunk_size, kernels, backend, progress, description):
    total_chunk_pairs = _chunk_count(len(data_a), chunk_size) * _chunk_count(len(data_b), chunk_size)
    totals = [0.0 for _ in kernels]

    with tqdm(
        total=total_chunk_pairs,
        desc=description,
        unit="chunk-pair",
        disable=not progress,
    ) as progress_bar:
        for chunk_a in _iter_feature_chunks(data_a, chunk_size, backend):
            for chunk_b in _iter_feature_chunks(data_b, chunk_size, backend):
                chunk_totals = _rbf_kernel_sums(chunk_a, chunk_b, kernels, backend)
                totals = [
                    total + chunk_total
                    for total, chunk_total in zip(totals, chunk_totals)
                ]
                backend.after_chunk_pair()
                progress_bar.update(1)

    return totals


def _self_kernel_sums(data, chunk_size, kernels, biased, backend, progress, description):
    totals = _kernel_sums(data, data, chunk_size, kernels, backend, progress, description)

    if not biased:
        # The RBF value of every event with itself is 1.0, so remove the diagonal.
        totals = [total - len(data) for total in totals]

    return totals


def _validate_inputs(
    features_a,
    features_b,
    chunk_size,
    sigma,
    gamma,
    rbf_kernel_max_distance,
    rbf_kernel_target_similarity,
):
    if len(features_a) == 0 or len(features_b) == 0:
        raise ValueError("Both inputs must contain at least one event.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if sigma is None and gamma is None and rbf_kernel_max_distance is None:
        raise ValueError("Pass sigma, gamma, or rbf_kernel_max_distance.")

    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive.")

    if gamma is not None and gamma <= 0:
        raise ValueError("gamma must be positive.")

    if rbf_kernel_max_distance is not None and rbf_kernel_max_distance <= 0:
        raise ValueError("rbf_kernel_max_distance must be positive.")

    if not 0 < rbf_kernel_target_similarity < 1:
        raise ValueError("rbf_kernel_target_similarity must be between 0 and 1.")


def mmd_analysis(
    features_a,
    features_b,
    *,
    chunk_size=50_000,
    sigma=1.0,
    gamma=None,
    biased=False,
    backend="numpy",
    progress=True,
    rbf_kernel_max_distance=None,
    rbf_kernel_target_similarity=_DEFAULT_RBF_TARGET_SIMILARITY,
    kernels=None,
    weight_method="uniform",
):
    """Perform a chunked RBF-kernel MMD analysis between two feature matrices.

    Args:
        features_a: ``(N, D)`` float array of features for the first window.
        features_b: ``(M, D)`` float array of features for the second window.
            Feature extraction, time normalization, and scaling are handled by
            :mod:`event_analysis_toolbox.feature_preprocessing` before this call.
        chunk_size: Number of events per chunk while computing kernel sums.
        sigma: RBF kernel bandwidth. Ignored when gamma is provided.
        gamma: RBF kernel coefficient. If None, uses 1 / (2 * sigma ** 2).
        biased: If True, use the biased MMD estimator. Otherwise use the unbiased
            estimator with self-kernel diagonals removed.
        backend: Array backend for kernel math. Supports "numpy" and "cupy".
        progress: If True, show tqdm progress bars for each chunked kernel pass.
        rbf_kernel_max_distance: Optional distance threshold used to derive the
            RBF kernel bandwidth so similarity is ``rbf_kernel_target_similarity``
            at this distance and smaller beyond it.
        rbf_kernel_target_similarity: Target similarity at
            ``rbf_kernel_max_distance``. Defaults to 0.01.
        kernels: Optional list of RBF kernel settings. Each kernel must define
            ``rbf_kernel_max_distance`` and may define
            ``rbf_kernel_target_similarity``.
        weight_method: Strategy used to combine per-kernel MMD values. Currently
            supports ``"uniform"``.

    Returns:
        A dictionary with the MMD value, squared MMD value, kernel sums, and the
        algorithm settings used for the comparison.
    """
    features_a = np.ascontiguousarray(features_a, dtype=np.float64)
    features_b = np.ascontiguousarray(features_b, dtype=np.float64)

    _validate_inputs(
        features_a,
        features_b,
        chunk_size,
        sigma,
        gamma,
        rbf_kernel_max_distance,
        rbf_kernel_target_similarity,
    )

    resolved_kernels = _resolve_mmd_kernels(
        kernels,
        sigma,
        gamma,
        rbf_kernel_max_distance,
        rbf_kernel_target_similarity,
    )
    resolved_kernels = _resolve_kernel_weights(resolved_kernels, weight_method)

    array_backend = _resolve_backend(backend)

    n_a = len(features_a)
    n_b = len(features_b)

    if not biased and (n_a < 2 or n_b < 2):
        raise ValueError("The unbiased MMD estimator requires at least two events per input.")

    a_kernel_sums = _self_kernel_sums(
        features_a, chunk_size, resolved_kernels, biased, array_backend, progress, "MMD a-a"
    )
    b_kernel_sums = _self_kernel_sums(
        features_b, chunk_size, resolved_kernels, biased, array_backend, progress, "MMD b-b"
    )
    cross_kernel_sums = _kernel_sums(
        features_a, features_b, chunk_size, resolved_kernels, array_backend, progress, "MMD a-b"
    )

    if biased:
        a_denominator = n_a * n_a
        b_denominator = n_b * n_b
    else:
        a_denominator = n_a * (n_a - 1)
        b_denominator = n_b * (n_b - 1)

    kernel_results = []
    for kernel, a_kernel_sum, b_kernel_sum, cross_kernel_sum in zip(
        resolved_kernels,
        a_kernel_sums,
        b_kernel_sums,
        cross_kernel_sums,
    ):
        kernel_mmd_squared = (
            a_kernel_sum / a_denominator
            + b_kernel_sum / b_denominator
            - 2.0 * cross_kernel_sum / (n_a * n_b)
        )
        kernel_results.append({
            **kernel,
            "mmd": math.sqrt(max(kernel_mmd_squared, 0.0)),
            "mmd_squared": kernel_mmd_squared,
            "kernel_sums": {
                "a": a_kernel_sum,
                "b": b_kernel_sum,
                "cross": cross_kernel_sum,
            },
        })

    mmd_squared = sum(
        kernel["weight"] * kernel["mmd_squared"]
        for kernel in kernel_results
    )
    first_kernel = kernel_results[0]

    return {
        "mmd": math.sqrt(max(mmd_squared, 0.0)),
        "mmd_squared": mmd_squared,
        "events_a": n_a,
        "events_b": n_b,
        "gamma": first_kernel["gamma"],
        "sigma": first_kernel["sigma"],
        "rbf_kernel_params": first_kernel["rbf_kernel_params"],
        "chunk_size": chunk_size,
        "biased": biased,
        "weight_method": weight_method,
        "backend": array_backend.name,
        "kernels": kernel_results,
        "kernel_sums": {
            "a": sum(
                kernel["weight"] * kernel["kernel_sums"]["a"]
                for kernel in kernel_results
            ),
            "b": sum(
                kernel["weight"] * kernel["kernel_sums"]["b"]
                for kernel in kernel_results
            ),
            "cross": sum(
                kernel["weight"] * kernel["kernel_sums"]["cross"]
                for kernel in kernel_results
            ),
        },
    }
