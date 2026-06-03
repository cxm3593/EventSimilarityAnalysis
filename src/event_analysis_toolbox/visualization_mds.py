"""Classical MDS embedding and plotting for distance matrices."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Embed a distance matrix with metric/classical MDS."""
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if distance_matrix.shape[0] < 2:
        raise ValueError("At least two items are required for MDS.")

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


def plot_mds_embedding(
    results: dict[str, Any],
    *,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    annotate: bool = False,
):
    """Plot an all-to-all comparison result that includes an MDS embedding."""
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

    metric = results.get("settings", {}).get("metric", "distance")
    name = results.get("name")
    default_title = f"MDS of all-to-all {metric} distances"
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
