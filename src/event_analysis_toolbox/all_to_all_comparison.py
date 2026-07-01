"""All-to-all window comparison strategy with optional visualization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # pyright: ignore[reportMissingModuleSource]
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

from event_data_toolbox.event_windows_management import EventWindowsManager

from .comparison_common import safe_file_stem, yaml_safe
from .feature_preprocessing import window_features
from .metrics import BaseMetric, get_metric
from .visualization_mds import classical_mds, plot_mds_embedding


__all__ = [
    "all_to_all_comparison",
    "plot_all_to_all_comparison",
    "save_all_to_all_comparison_results",
]

SUPPORTED_VISUALIZERS = frozenset({"mds", None})


def _distance_from_result(result) -> float:
    return float(result.value)


def all_to_all_comparison(
    real_data,
    v2e_data,
    *,
    baseline_start: int,
    baseline_end: int,
    n_real_windows: int = 9,
    n_v2e_windows: int = 10,
    stride: int | None = None,
    real_window_start: int | None = None,
    v2e_window_start: int | None = None,
    metric: str | BaseMetric = "mmd",
    metric_kwargs: dict | None = None,
    feature_names=None,
    feature_scales=None,
    visualizer: str | None = "mds",
    visualizer_kwargs: dict | None = None,
    name: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute pairwise distances between all windows.

    Args:
        metric: Registered distance / similarity metric name or instance.
        metric_kwargs: Algorithm-specific settings forwarded to the metric.
        feature_names: Optional event fields to compare (polarity always dropped).
        feature_scales: Optional per-feature divisors applied during preprocessing.
        visualizer: Layout algorithm for the distance matrix (``"mds"`` or ``None``).
        visualizer_kwargs: Settings for the visualizer (e.g. ``n_components`` for MDS).
    """
    metric_impl = get_metric(metric)
    metric_name = metric_impl.name
    if visualizer not in SUPPORTED_VISUALIZERS:
        raise ValueError(
            f"Unsupported visualizer: {visualizer!r}. "
            f"Supported visualizers: {sorted(v for v in SUPPORTED_VISUALIZERS if v is not None)}"
        )

    window_manager = EventWindowsManager(real_data, v2e_data)
    generated = window_manager.generate(
        baseline_start,
        baseline_end,
        n_real_windows,
        n_v2e_windows,
        stride,
        real_window_start=real_window_start,
        v2e_window_start=v2e_window_start,
    )
    window_settings = generated["settings"]

    all_windows = [generated["baseline"], *generated["real"], *generated["v2e"]]
    windows = [window for window in all_windows if window.n_events >= 2]
    skipped_windows = [
        window.to_dict(include_events=False) for window in all_windows if window.n_events < 2
    ]
    if len(windows) < 2:
        raise ValueError("At least two non-empty windows are required for all-to-all comparison.")

    inner_kwargs = dict(metric_kwargs or {})
    if metric_impl.supports_inner_progress:
        inner_kwargs["progress"] = False
    viz_kwargs = dict(visualizer_kwargs or {})

    window_feature_arrays = [
        window_features(
            window.events,
            feature_names=feature_names,
            feature_scales=feature_scales,
            time_origin=window.start,
        )[0]
        for window in windows
    ]

    n_windows = len(windows)
    distance_matrix = np.zeros((n_windows, n_windows), dtype=np.float64)
    kernel_distance_matrices: list[np.ndarray] = []
    kernel_distance_squared_matrices: list[np.ndarray] = []
    kernel_metadata: list[dict[str, Any]] = []
    total_pairs = n_windows * (n_windows - 1) // 2
    desc = f"All-to-all [{metric_name}]"
    if name:
        desc = f"{desc} {name}"

    with tqdm(total=total_pairs, desc=desc, unit="pair", disable=not progress) as bar:
        for i in range(n_windows):
            for j in range(i + 1, n_windows):
                result = metric_impl.compute(
                    window_feature_arrays[i],
                    window_feature_arrays[j],
                    **inner_kwargs,
                )
                distance = _distance_from_result(result)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                kernels = result.metadata.get("kernels", [])
                if kernels and not kernel_distance_matrices:
                    kernel_distance_matrices = [
                        np.zeros((n_windows, n_windows), dtype=np.float64)
                        for _ in kernels
                    ]
                    kernel_distance_squared_matrices = [
                        np.zeros((n_windows, n_windows), dtype=np.float64)
                        for _ in kernels
                    ]
                    kernel_metadata = [
                        {
                            key: value
                            for key, value in kernel.items()
                            if key not in ("mmd", "mmd_squared", "kernel_sums")
                        }
                        for kernel in kernels
                    ]
                for kernel_index, kernel in enumerate(kernels):
                    kernel_distance = float(kernel.get("mmd", np.nan))
                    kernel_distance_squared = float(kernel.get("mmd_squared", np.nan))
                    kernel_distance_matrices[kernel_index][i, j] = kernel_distance
                    kernel_distance_matrices[kernel_index][j, i] = kernel_distance
                    squared_matrix = kernel_distance_squared_matrices[kernel_index]
                    squared_matrix[i, j] = kernel_distance_squared
                    squared_matrix[j, i] = kernel_distance_squared
                bar.update(1)

    embedding = None
    if visualizer == "mds":
        n_components = int(viz_kwargs.get("n_components", 2))
        embedding = classical_mds(distance_matrix, n_components=n_components)

    window_metadata = []
    for i, window in enumerate(windows):
        entry = window.to_dict(include_events=False)
        if embedding is not None:
            entry["embedding"] = [float(value) for value in embedding[i]]
        window_metadata.append(entry)

    return {
        "name": name,
        "strategy": "all_to_all_comparison",
        "metric": metric_name,
        "visualizer": visualizer,
        "windows": window_metadata,
        "skipped_windows": skipped_windows,
        "distance_matrix": distance_matrix,
        "kernel_distance_matrices": kernel_distance_matrices,
        "kernel_distance_squared_matrices": kernel_distance_squared_matrices,
        "kernel_metadata": kernel_metadata,
        "embedding": embedding,
        "settings": {
            **window_settings,
            "baseline_start": int(baseline_start),
            "baseline_end": int(baseline_end),
            "n_real_windows": int(n_real_windows),
            "n_v2e_windows": int(n_v2e_windows),
            "metric": metric_name,
            "visualizer": visualizer,
            "metric_kwargs": {str(k): yaml_safe(v) for k, v in inner_kwargs.items()},
            "visualizer_kwargs": {str(k): yaml_safe(v) for k, v in viz_kwargs.items()},
        },
    }


def plot_all_to_all_comparison(
    results: dict[str, Any],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    annotate: bool = False,
):
    """Plot all-to-all comparison results using the configured visualizer."""
    visualizer = results.get("visualizer")
    if visualizer == "mds":
        if results.get("embedding") is None:
            raise ValueError("Results do not contain an MDS embedding.")
        return plot_mds_embedding(
            results,
            save_path=str(save_path) if save_path is not None else None,
            show=show,
            title=title,
            figsize=figsize,
            annotate=annotate,
        )
    raise ValueError(
        f"No plot available for visualizer={visualizer!r}. "
        "Use visualizer='mds' when running all_to_all_comparison."
    )


def save_all_to_all_comparison_results(
    results: dict[str, Any],
    output_dir: str | Path,
    *,
    run_name: str | None = "",
    file_prefix: str | None = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> dict[str, str]:
    """Persist all-to-all distances, optional embedding, metadata, and plot."""
    output_dir = Path(output_dir)
    run_dir = output_dir if run_name == "" else output_dir / (run_name or "all_to_all")
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_file_stem(file_prefix or results.get("name") or "all_to_all")

    windows_path = run_dir / f"{stem}_windows.csv"
    embedding = results.get("embedding")
    has_embedding = embedding is not None
    with windows_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["index", "id", "source", "role", "start", "end", "n_events"]
        if has_embedding:
            header.extend(["embed_1", "embed_2"])
        writer.writerow(header)
        for i, window in enumerate(results["windows"]):
            row = [
                i,
                window["id"],
                window["source"],
                window["role"],
                window["start"],
                window["end"],
                window["n_events"],
            ]
            if has_embedding:
                coords = window.get("embedding", [])
                row.extend([
                    coords[0] if coords else "",
                    coords[1] if len(coords) > 1 else "",
                ])
            writer.writerow(row)

    matrix_path = run_dir / f"{stem}_distance_matrix.csv"
    labels = [window["id"] for window in results["windows"]]
    with matrix_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", *labels])
        for label, row in zip(labels, np.asarray(results["distance_matrix"])):
            writer.writerow([label, *[float(value) for value in row]])

    kernel_matrix_paths = []
    for kernel_index, matrix in enumerate(results.get("kernel_distance_matrices", [])):
        kernel_matrix_path = run_dir / f"{stem}_kernel_{kernel_index}_distance_matrix.csv"
        with kernel_matrix_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", *labels])
            for label, row in zip(labels, np.asarray(matrix)):
                writer.writerow([label, *[float(value) for value in row]])
        kernel_matrix_paths.append(str(kernel_matrix_path))

    kernel_squared_matrix_paths = []
    for kernel_index, matrix in enumerate(
        results.get("kernel_distance_squared_matrices", [])
    ):
        kernel_matrix_path = (
            run_dir / f"{stem}_kernel_{kernel_index}_distance_squared_matrix.csv"
        )
        with kernel_matrix_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", *labels])
            for label, row in zip(labels, np.asarray(matrix)):
                writer.writerow([label, *[float(value) for value in row]])
        kernel_squared_matrix_paths.append(str(kernel_matrix_path))

    metadata_path = run_dir / f"{stem}_metadata.yaml"
    with metadata_path.open("w") as f:
        yaml.safe_dump(
            {
                "name": results.get("name"),
                "strategy": results.get("strategy"),
                "metric": results.get("metric"),
                "visualizer": results.get("visualizer"),
                "settings": results["settings"],
                "kernels": yaml_safe(results.get("kernel_metadata", [])),
                "skipped_windows": results.get("skipped_windows", []),
            },
            f,
            sort_keys=False,
        )

    paths = {
        "dir": str(run_dir),
        "windows": str(windows_path),
        "distance_matrix": str(matrix_path),
        "metadata": str(metadata_path),
    }
    if kernel_matrix_paths:
        paths["kernel_distance_matrices"] = kernel_matrix_paths
    if kernel_squared_matrix_paths:
        paths["kernel_distance_squared_matrices"] = kernel_squared_matrix_paths

    if save_plot and results.get("visualizer") == "mds":
        plot_path = run_dir / f"{stem}_plot.png"
        plot_all_to_all_comparison(results, save_path=plot_path, show=show_plot)
        paths["plot"] = str(plot_path)

    return paths
