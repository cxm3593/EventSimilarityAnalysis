"""All-to-all window MMD analysis with classical MDS visualization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
import yaml  # pyright: ignore[reportMissingModuleSource]
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

from .mmd import mmd_analysis


__all__ = [
    "mds_mmd_analysis",
    "plot_mds_mmd",
    "save_mds_mmd_results",
]


def _yaml_safe(value):
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return repr(value)


def _safe_file_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return stem.strip("_") or "results"


class _SortedTimeIndexer:
    """Cache event timestamps once and slice windows via binary search."""

    def __init__(self, data):
        self.data = data
        self.t = np.asarray(data["t"])

    def select(self, start: int, end: int):
        lo = int(np.searchsorted(self.t, start, side="left"))
        hi = int(np.searchsorted(self.t, end, side="left"))
        return self.data[lo:hi]


def _make_windows(start: int, width: int, count: int, stride: int) -> list[tuple[int, int]]:
    return [(start + i * stride, start + i * stride + width) for i in range(count)]


def _classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Embed a distance matrix with metric/classical MDS."""
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if distance_matrix.shape[0] < 2:
        raise ValueError("At least two windows are required for MDS.")

    squared = distance_matrix**2
    n = squared.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = np.maximum(eigenvalues[:n_components], 0.0)
    coords = eigenvectors[:, :n_components] * np.sqrt(positive)
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords


def _window_entry(source, role, start, end, events):
    label = f"{source}_{role}_{start}_{end}" if role == "baseline" else f"{source}_{start}_{end}"
    return {
        "id": label,
        "label": label,
        "source": source,
        "role": role,
        "start": int(start),
        "end": int(end),
        "n_events": int(len(events)),
        "events": events,
    }


def _collect_windows(
    real_data,
    v2e_data,
    *,
    baseline_start: int,
    baseline_end: int,
    n_real_windows: int,
    n_v2e_windows: int,
    stride: int | None,
    real_window_start: int | None,
    v2e_window_start: int | None,
):
    width = baseline_end - baseline_start
    if width <= 0:
        raise ValueError("baseline_end must be greater than baseline_start.")

    stride_value = width if stride is None else int(stride)
    if stride_value <= 0:
        raise ValueError("stride must be positive.")

    real_window_start = (
        baseline_start + stride_value if real_window_start is None else int(real_window_start)
    )
    v2e_window_start = baseline_start if v2e_window_start is None else int(v2e_window_start)

    real_indexer = _SortedTimeIndexer(real_data)
    v2e_indexer = _SortedTimeIndexer(v2e_data)

    windows = [
        _window_entry(
            "real",
            "baseline",
            baseline_start,
            baseline_end,
            real_indexer.select(baseline_start, baseline_end),
        )
    ]

    for start, end in _make_windows(real_window_start, width, n_real_windows, stride_value):
        windows.append(_window_entry("real", "comparison", start, end, real_indexer.select(start, end)))

    for start, end in _make_windows(v2e_window_start, width, n_v2e_windows, stride_value):
        windows.append(_window_entry("v2e", "comparison", start, end, v2e_indexer.select(start, end)))

    valid = [window for window in windows if window["n_events"] >= 2]
    skipped = [
        {k: v for k, v in window.items() if k != "events"}
        for window in windows
        if window["n_events"] < 2
    ]

    if len(valid) < 2:
        raise ValueError("At least two non-empty windows are required for MDS.")

    return valid, skipped, {
        "width": int(width),
        "stride": int(stride_value),
        "real_window_start": int(real_window_start),
        "v2e_window_start": int(v2e_window_start),
    }


def mds_mmd_analysis(
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
    mmd_kwargs: dict | None = None,
    name: str | None = None,
    n_components: int = 2,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute all-to-all window MMD distances and a classical-MDS embedding."""
    windows, skipped_windows, window_settings = _collect_windows(
        real_data,
        v2e_data,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        n_real_windows=n_real_windows,
        n_v2e_windows=n_v2e_windows,
        stride=stride,
        real_window_start=real_window_start,
        v2e_window_start=v2e_window_start,
    )

    inner_kwargs = dict(mmd_kwargs or {})
    inner_kwargs["progress"] = False

    n_windows = len(windows)
    distance_matrix = np.zeros((n_windows, n_windows), dtype=np.float64)
    total_pairs = n_windows * (n_windows - 1) // 2
    desc = f"MDS all-to-all MMD [{name}]" if name else "MDS all-to-all MMD"

    with tqdm(total=total_pairs, desc=desc, unit="pair", disable=not progress) as bar:
        for i in range(n_windows):
            for j in range(i + 1, n_windows):
                result = mmd_analysis(windows[i]["events"], windows[j]["events"], **inner_kwargs)
                distance = float(result["mmd"])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                bar.update(1)

    embedding = _classical_mds(distance_matrix, n_components=n_components)

    window_metadata = []
    for i, window in enumerate(windows):
        entry = {k: v for k, v in window.items() if k != "events"}
        entry["mds"] = [float(value) for value in embedding[i]]
        window_metadata.append(entry)

    return {
        "name": name,
        "windows": window_metadata,
        "skipped_windows": skipped_windows,
        "distance_matrix": distance_matrix,
        "embedding": embedding,
        "settings": {
            **window_settings,
            "baseline_start": int(baseline_start),
            "baseline_end": int(baseline_end),
            "n_real_windows": int(n_real_windows),
            "n_v2e_windows": int(n_v2e_windows),
            "n_components": int(n_components),
            "mmd_kwargs": {str(k): _yaml_safe(v) for k, v in inner_kwargs.items()},
        },
    }


def plot_mds_mmd(
    results: dict[str, Any],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    annotate: bool = False,
):
    """Plot the all-to-all MMD distance matrix as a 2D MDS embedding."""
    embedding = np.asarray(results["embedding"])
    windows = results["windows"]

    fig, ax = plt.subplots(figsize=figsize)

    groups = [
        ("real", "baseline", "tab:green", "*", "real baseline"),
        ("real", "comparison", "tab:blue", "o", "real windows"),
        ("v2e", "comparison", "tab:orange", "s", "v2e windows"),
    ]
    for source, role, color, marker, label in groups:
        indices = [
            i for i, window in enumerate(windows)
            if window["source"] == source and window["role"] == role
        ]
        if not indices:
            continue
        ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            c=color,
            marker=marker,
            label=label,
            edgecolors="black",
            linewidths=0.4,
            s=120 if role == "baseline" else 50,
            alpha=0.85,
        )

    if annotate:
        for i, window in enumerate(windows):
            ax.annotate(str(window["start"]), (embedding[i, 0], embedding[i, 1]), fontsize=8)

    name = results.get("name")
    default_title = "MDS of all-to-all window MMD distances"
    if name:
        default_title = f"[{name}] {default_title}"
    stride = results.get("settings", {}).get("stride")
    if stride is not None:
        default_title += f" (stride={stride} μs)"

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title or default_title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def save_mds_mmd_results(
    results: dict[str, Any],
    output_dir: str | Path,
    *,
    run_name: str | None = "",
    file_prefix: str | None = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> dict[str, str]:
    """Persist all-to-all MMD distances, MDS coordinates, metadata, and plot."""
    output_dir = Path(output_dir)
    run_dir = output_dir if run_name == "" else output_dir / (run_name or "mds_mmd")
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_file_stem(file_prefix or results.get("name") or "mds")

    windows_path = run_dir / f"{stem}_mds_windows.csv"
    with windows_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "id", "source", "role", "start", "end", "n_events", "mds_1", "mds_2"])
        for i, window in enumerate(results["windows"]):
            coords = window["mds"]
            writer.writerow([
                i,
                window["id"],
                window["source"],
                window["role"],
                window["start"],
                window["end"],
                window["n_events"],
                coords[0],
                coords[1] if len(coords) > 1 else 0.0,
            ])

    matrix_path = run_dir / f"{stem}_mds_distance_matrix.csv"
    labels = [window["id"] for window in results["windows"]]
    with matrix_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", *labels])
        for label, row in zip(labels, np.asarray(results["distance_matrix"])):
            writer.writerow([label, *[float(value) for value in row]])

    metadata_path = run_dir / f"{stem}_mds_metadata.yaml"
    with metadata_path.open("w") as f:
        yaml.safe_dump(
            {
                "name": results.get("name"),
                "settings": results["settings"],
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

    if save_plot:
        plot_path = run_dir / f"{stem}_mds_plot.png"
        plot_mds_mmd(results, save_path=plot_path, show=show_plot)
        paths["plot"] = str(plot_path)

    return paths
