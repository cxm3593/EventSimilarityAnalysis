"""Systematic windowed MMD analysis.

Compares one *baseline* time window of real event data against multiple
successor windows of the same width — drawn both from the real stream and
from the synthetic / V2E stream — and produces:

    * a structured result dictionary (per-window MMD values + metadata)
    * a `MMD vs window-start-time` plot
    * persisted CSV + YAML metadata + PNG plot under an output folder
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml  # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

from .mmd import mmd_analysis


__all__ = [
    "windowed_mmd_analysis",
    "plot_windowed_mmd",
    "save_windowed_mmd_results",
]


def _yaml_safe(value):
    """Recursively convert values into YAML-friendly Python primitives."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return repr(value)


class _SortedTimeIndexer:
    """Cache event timestamps once and slice windows via binary search.

    Assumes the events are sorted by their `t` field (true for typical
    event-camera HDF5 dumps).
    """

    def __init__(self, data):
        self.data = data
        self.t = np.asarray(data["t"])

    def select(self, start: int, end: int):
        lo = int(np.searchsorted(self.t, start, side="left"))
        hi = int(np.searchsorted(self.t, end, side="left"))
        return self.data[lo:hi]


def _make_windows(start: int, width: int, count: int, stride: int) -> list[tuple[int, int]]:
    """Build ``count`` windows of ``width`` starting at ``start``, stepping by ``stride``.

    ``stride == width`` produces consecutive non-overlapping windows. A larger
    ``stride`` (e.g. a phase period) produces same-phase samples spaced apart.
    """
    return [(start + i * stride, start + i * stride + width) for i in range(count)]


def _comparison_entry(start: int, end: int, chunk, baseline, mmd_kwargs):
    entry: dict[str, Any] = {
        "start": int(start),
        "end": int(end),
        "n_events": int(len(chunk)),
    }
    if len(chunk) < 2:
        entry["mmd"] = float("nan")
        entry["mmd_squared"] = float("nan")
        entry["status"] = "empty_or_too_small"
        return entry

    result = mmd_analysis(baseline, chunk, **mmd_kwargs)
    entry["mmd"] = float(result["mmd"])
    entry["mmd_squared"] = float(result["mmd_squared"])
    entry["status"] = "ok"
    return entry


def windowed_mmd_analysis(
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
    progress: bool = True,
) -> dict[str, Any]:
    """Compare a baseline real window against multiple real and v2e windows.

    Args:
        real_data: Real event dataset (h5py / structured array).
        v2e_data: Synthetic / V2E event dataset, same schema as ``real_data``.
        baseline_start, baseline_end: Time bounds of the baseline window
            (microseconds), drawn from ``real_data``.
        n_real_windows: Number of real comparison windows.
        n_v2e_windows: Number of v2e comparison windows.
        stride: Distance (μs) between successive window starts. Defaults to the
            baseline width, which produces consecutive non-overlapping windows.
            Set to a phase period to sample the same phase across many cycles
            (e.g. ``stride = 315_000`` selects one window per ~3.17 Hz cycle).
        real_window_start: Start time of the first real comparison window.
            Defaults to ``baseline_start + stride`` so the first comparison
            window is one stride away from the baseline.
        v2e_window_start: Start time of the first v2e comparison window.
            Defaults to ``baseline_start`` so v2e windows include the baseline
            phase as their first sample.
        mmd_kwargs: Forwarded to :func:`event_analysis_toolbox.mmd.mmd_analysis`
            (e.g. ``chunk_size``, ``sigma``, ``backend``, ``feature_scales``).
            The inner progress bar is force-disabled here so the outer
            per-window bar stays readable.
        name: Optional label for this analysis (e.g. ``"consecutive"`` /
            ``"periodic"``). Stored in the result and used in plot titles /
            output folder names.
        progress: Show a tqdm bar over the windows.

    Returns:
        A dict with keys ``name``, ``baseline``, ``real_windows``,
        ``v2e_windows`` and ``settings``.
    """
    width = baseline_end - baseline_start
    if width <= 0:
        raise ValueError("baseline_end must be greater than baseline_start.")
    if n_real_windows < 0 or n_v2e_windows < 0:
        raise ValueError("Window counts must be non-negative.")

    stride_value = width if stride is None else int(stride)
    if stride_value <= 0:
        raise ValueError("stride must be positive.")

    real_window_start = (
        baseline_start + stride_value if real_window_start is None else int(real_window_start)
    )
    v2e_window_start = (
        baseline_start if v2e_window_start is None else int(v2e_window_start)
    )

    real_indexer = _SortedTimeIndexer(real_data)
    v2e_indexer = _SortedTimeIndexer(v2e_data)

    baseline = real_indexer.select(baseline_start, baseline_end)
    if len(baseline) < 2:
        raise ValueError(
            f"Baseline window [{baseline_start}, {baseline_end}) μs has too few events: "
            f"{len(baseline)}"
        )

    real_specs = _make_windows(real_window_start, width, n_real_windows, stride_value)
    v2e_specs = _make_windows(v2e_window_start, width, n_v2e_windows, stride_value)

    inner_kwargs = dict(mmd_kwargs or {})
    inner_kwargs["progress"] = False

    real_results: list[dict[str, Any]] = []
    v2e_results: list[dict[str, Any]] = []

    desc = f"Windowed MMD [{name}]" if name else "Windowed MMD"
    total = n_real_windows + n_v2e_windows
    with tqdm(total=total, desc=desc, unit="window", disable=not progress) as bar:
        for win_start, win_end in real_specs:
            chunk = real_indexer.select(win_start, win_end)
            real_results.append(_comparison_entry(win_start, win_end, chunk, baseline, inner_kwargs))
            bar.update(1)

        for win_start, win_end in v2e_specs:
            chunk = v2e_indexer.select(win_start, win_end)
            v2e_results.append(_comparison_entry(win_start, win_end, chunk, baseline, inner_kwargs))
            bar.update(1)

    safe_kwargs = {str(k): _yaml_safe(v) for k, v in inner_kwargs.items()}

    return {
        "name": name,
        "baseline": {
            "start": int(baseline_start),
            "end": int(baseline_end),
            "n_events": int(len(baseline)),
            "source": "real",
        },
        "real_windows": real_results,
        "v2e_windows": v2e_results,
        "settings": {
            "width": int(width),
            "stride": int(stride_value),
            "n_real_windows": int(n_real_windows),
            "n_v2e_windows": int(n_v2e_windows),
            "real_window_start": int(real_window_start),
            "v2e_window_start": int(v2e_window_start),
            "mmd_kwargs": safe_kwargs,
        },
    }


def plot_windowed_mmd(
    results: dict[str, Any],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
    time_unit: str = "μs",
):
    """Plot MMD vs window start time for the real-vs-baseline and v2e-vs-baseline comparisons."""
    fig, ax = plt.subplots(figsize=figsize)

    real = results.get("real_windows", [])
    v2e = results.get("v2e_windows", [])

    if real:
        xs = [w["start"] for w in real]
        ys = [w["mmd"] for w in real]
        ax.plot(xs, ys, marker="o", linestyle="-", label="real vs real-baseline", color="tab:blue")

    if v2e:
        xs = [w["start"] for w in v2e]
        ys = [w["mmd"] for w in v2e]
        ax.plot(xs, ys, marker="s", linestyle="-", label="v2e vs real-baseline", color="tab:orange")

    baseline = results["baseline"]
    ax.axvspan(
        baseline["start"], baseline["end"],
        color="gray", alpha=0.2,
        label=f"baseline real [{baseline['start']}, {baseline['end']}] {time_unit}",
    )

    settings = results.get("settings", {})
    name = results.get("name")
    stride = settings.get("stride")
    width = settings.get("width")

    default_title = f"MMD vs baseline real window [{baseline['start']}, {baseline['end']}] {time_unit}"
    if name:
        default_title = f"[{name}] " + default_title
    if stride is not None and width is not None and stride != width:
        default_title += f"  (stride={stride} {time_unit})"

    ax.set_xlabel(f"Window start time ({time_unit})")
    ax.set_ylabel("MMD")
    ax.set_title(title or default_title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def save_windowed_mmd_results(
    results: dict[str, Any],
    output_dir: str | Path,
    *,
    run_name: str | None = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> dict[str, str]:
    """Persist a windowed MMD run as CSV + YAML metadata + (optionally) a PNG plot.

    Files are written to ``<output_dir>/<run_name>/``. If ``run_name`` is None,
    a timestamped folder ``windowed_mmd_YYYYMMDD_HHMMSS`` is used.
    """
    output_dir = Path(output_dir)
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scheme_name = results.get("name")
        if scheme_name:
            run_name = f"windowed_mmd_{scheme_name}_{timestamp}"
        else:
            run_name = f"windowed_mmd_{timestamp}"

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comparison_set",
            "window_start",
            "window_end",
            "n_events",
            "mmd",
            "mmd_squared",
            "status",
        ])
        for w in results.get("real_windows", []):
            writer.writerow([
                "real_vs_real",
                w["start"], w["end"], w["n_events"],
                w["mmd"], w["mmd_squared"], w.get("status", ""),
            ])
        for w in results.get("v2e_windows", []):
            writer.writerow([
                "v2e_vs_real",
                w["start"], w["end"], w["n_events"],
                w["mmd"], w["mmd_squared"], w.get("status", ""),
            ])

    yaml_path = run_dir / "metadata.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(
            {
                "name": results.get("name"),
                "baseline": results["baseline"],
                "settings": results["settings"],
            },
            f,
            sort_keys=False,
        )

    paths: dict[str, str] = {
        "dir": str(run_dir),
        "csv": str(csv_path),
        "metadata": str(yaml_path),
    }

    if save_plot:
        plot_path = run_dir / "plot.png"
        plot_windowed_mmd(results, save_path=plot_path, show=show_plot)
        paths["plot"] = str(plot_path)

    return paths
