"""3D visualization utilities for event-based empirical probabilities.

The MMD analysis used elsewhere in this package is a (kernel) Integral
Probability Metric: it compares two distributions through expectations of a
function class. The most basic empirical estimate of the underlying
distributions is the normalized histogram, i.e. the empirical probability mass
in each spatial-temporal voxel.

This module provides a side-by-side 3D visualization of those empirical
probabilities for two event datasets (typically real vs. synthetic / V2E)
defined over the same (x, y, t) volume, plus the signed difference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pyright: ignore[reportMissingImports]


__all__ = ["plot_empirical_probability_3d"]


def _events_to_array(data, feature_names: Sequence[str]) -> np.ndarray:
    """Extract `feature_names` columns from a structured/HDF5 event array."""
    if getattr(data, "dtype", None) is not None and data.dtype.names is not None:
        return np.column_stack([np.asarray(data[name], dtype=np.float64) for name in feature_names])

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _shared_ranges(real_arr: np.ndarray, v2e_arr: np.ndarray) -> list[tuple[float, float]]:
    """Compute per-axis ranges that contain both datasets so bins are aligned."""
    combined = np.vstack([real_arr, v2e_arr])
    ranges: list[tuple[float, float]] = []
    for i in range(combined.shape[1]):
        lo = float(combined[:, i].min())
        hi = float(combined[:, i].max())
        if hi <= lo:
            hi = lo + 1.0
        ranges.append((lo, hi))
    return ranges


def _empirical_probability(
    events: np.ndarray, bins, ranges
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return a normalized 3D histogram (empirical probability) and bin edges."""
    hist, edges = np.histogramdd(events, bins=bins, range=ranges)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist, list(edges)


def _bin_centers(edges: list[np.ndarray]) -> list[np.ndarray]:
    return [0.5 * (e[:-1] + e[1:]) for e in edges]


def _voxel_centers_grid(centers: list[np.ndarray]):
    cx, cy, ct = centers
    X, Y, T = np.meshgrid(cx, cy, ct, indexing="ij")
    return X, Y, T


def _plot_probability_voxels(
    ax,
    prob: np.ndarray,
    centers: list[np.ndarray],
    *,
    cmap: str,
    title: str,
    feature_names: Sequence[str],
    threshold: float,
    symmetric: bool,
    vmin: float | None = None,
    vmax: float | None = None,
):
    X, Y, T = _voxel_centers_grid(centers)
    flat_prob = prob.ravel()
    mask = (np.abs(flat_prob) > threshold) if symmetric else (flat_prob > threshold)

    xs = X.ravel()[mask]
    ys = Y.ravel()[mask]
    ts = T.ravel()[mask]
    cs = flat_prob[mask]

    if symmetric:
        if vmax is None:
            absmax = float(np.abs(cs).max()) if cs.size else 1.0
            absmax = absmax if absmax > 0 else 1.0
            vmin, vmax = -absmax, absmax
    else:
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = float(cs.max()) if cs.size else 1.0
            vmax = vmax if vmax > 0 else 1.0

    if cs.size:
        size_scale = float(np.abs(cs).max())
        size_scale = size_scale if size_scale > 0 else 1.0
        sizes = 8.0 + 80.0 * (np.abs(cs) / size_scale)
    else:
        sizes = 8.0

    sc = ax.scatter(xs, ys, ts, c=cs, cmap=cmap, vmin=vmin, vmax=vmax, s=sizes, alpha=0.75)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title(title)
    return sc


def plot_empirical_probability_3d(
    real_data,
    v2e_data,
    *,
    feature_names: Sequence[str] = ("x", "y", "t"),
    bins: int | Sequence[int] = (32, 32, 32),
    threshold: float = 0.0,
    cmap_data: str = "viridis",
    cmap_diff: str = "seismic",
    figsize: tuple[float, float] = (18.0, 6.0),
    titles: tuple[str, str, str] = (
        "Real – empirical probability",
        "V2E – empirical probability",
        "Real − V2E",
    ),
    show: bool = True,
    save_path: str | None = None,
):
    """Plot 3 side-by-side 3D views of empirical probability distributions.

    Computes the empirical probability mass (normalized 3D histogram) of
    `real_data` and `v2e_data` over the shared `(feature_names)` volume, then
    draws three 3D scatter plots:

        1. Real empirical probability
        2. V2E empirical probability
        3. Their signed difference (real − v2e)

    Args:
        real_data: Real event data. HDF5 dataset, structured array, or 2D array
            with columns matching `feature_names`.
        v2e_data: Synthetic / V2E event data, same format as `real_data`.
        feature_names: Three field names used as the (x, y, t)-like axes.
        bins: Number of bins per axis (int or 3-tuple).
        threshold: Voxels whose probability magnitude is ≤ `threshold` are
            hidden, which keeps the 3D scatter readable.
        cmap_data: Colormap for the per-dataset probability plots.
        cmap_diff: Diverging colormap for the difference plot.
        figsize: Matplotlib figure size in inches.
        titles: Titles for the three subplots.
        show: If True, calls `plt.show()` after rendering.
        save_path: If given, saves the figure to this path.

    Returns:
        A tuple `(fig, info)` where `info` contains the empirical probability
        arrays, bin edges, and the per-axis ranges used.
    """
    if len(feature_names) != 3:
        raise ValueError("feature_names must have exactly 3 entries for 3D visualization.")

    real_arr = _events_to_array(real_data, feature_names)
    v2e_arr = _events_to_array(v2e_data, feature_names)

    if real_arr.shape[1] != 3 or v2e_arr.shape[1] != 3:
        raise ValueError("Both inputs must yield 3 columns once feature_names is applied.")

    ranges = _shared_ranges(real_arr, v2e_arr)
    real_prob, edges = _empirical_probability(real_arr, bins, ranges)
    v2e_prob, _ = _empirical_probability(v2e_arr, bins, ranges)
    diff = real_prob - v2e_prob

    centers = _bin_centers(edges)

    fig = plt.figure(figsize=figsize)
    ax_real = fig.add_subplot(1, 3, 1, projection="3d")
    ax_v2e = fig.add_subplot(1, 3, 2, projection="3d")
    ax_diff = fig.add_subplot(1, 3, 3, projection="3d")

    shared_vmax = float(max(real_prob.max(), v2e_prob.max()))
    shared_vmax = shared_vmax if shared_vmax > 0 else 1.0

    sc_real = _plot_probability_voxels(
        ax_real,
        real_prob,
        centers,
        cmap=cmap_data,
        title=titles[0],
        feature_names=feature_names,
        threshold=threshold,
        symmetric=False,
        vmin=0.0,
        vmax=shared_vmax,
    )
    fig.colorbar(sc_real, ax=ax_real, shrink=0.6, pad=0.1, label="probability")

    sc_v2e = _plot_probability_voxels(
        ax_v2e,
        v2e_prob,
        centers,
        cmap=cmap_data,
        title=titles[1],
        feature_names=feature_names,
        threshold=threshold,
        symmetric=False,
        vmin=0.0,
        vmax=shared_vmax,
    )
    fig.colorbar(sc_v2e, ax=ax_v2e, shrink=0.6, pad=0.1, label="probability")

    sc_diff = _plot_probability_voxels(
        ax_diff,
        diff,
        centers,
        cmap=cmap_diff,
        title=titles[2],
        feature_names=feature_names,
        threshold=threshold,
        symmetric=True,
    )
    fig.colorbar(sc_diff, ax=ax_diff, shrink=0.6, pad=0.1, label="Δ probability")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    info = {
        "real_probability": real_prob,
        "v2e_probability": v2e_prob,
        "difference": diff,
        "edges": edges,
        "ranges": ranges,
        "feature_names": tuple(feature_names),
        "bins": bins,
    }
    return fig, info
