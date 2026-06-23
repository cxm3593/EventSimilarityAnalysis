"""Baseline-vs-window comparison strategy."""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
import yaml  # pyright: ignore[reportMissingModuleSource]
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

from event_data_toolbox.event_windows_management import EventWindowsManager

from .comparison_common import safe_file_stem, yaml_safe
from .event_modifiers import ModifierContext, ModifierPipeline
from .feature_preprocessing import window_features
from .metrics import BaseMetric, get_metric


__all__ = [
    "baseline_comparison",
    "plot_baseline_comparison",
    "save_baseline_comparison_results",
]


def _distance_entry(
    start: int,
    end: int,
    events,
    baseline_features,
    metric_impl: BaseMetric,
    metric_kwargs,
    *,
    feature_names,
    feature_scales,
):
    """Distance between the baseline and one (possibly modified) window."""
    entry: dict[str, Any] = {
        "start": int(start),
        "end": int(end),
        "n_events": int(len(events)),
    }
    if len(events) < 2:
        entry["distance"] = float("nan")
        entry["distance_squared"] = float("nan")
        entry["status"] = "empty_or_too_small"
        return entry

    features, _ = window_features(
        events,
        feature_names=feature_names,
        feature_scales=feature_scales,
        time_origin=start,
    )
    result = metric_impl.compute(baseline_features, features, **metric_kwargs)
    entry["distance"] = float(result.value)
    secondary = metric_impl.secondary_value(result)
    if secondary is not None:
        entry["distance_squared"] = float(secondary)
    entry["status"] = "ok"
    return entry


def baseline_comparison(
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
    modifier_pipelines: list[ModifierPipeline] | None = None,
    sensor: dict | None = None,
    rng=None,
    seed: int | None = None,
    name: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Compare a baseline real window against multiple real and v2e windows.

    Args:
        metric: Registered distance / similarity metric name or instance.
        metric_kwargs: Algorithm-specific settings forwarded to the metric.
        feature_names: Optional event fields to compare (polarity always dropped).
        feature_scales: Optional per-feature divisors applied during preprocessing.
        modifier_pipelines: Optional named modifier pipelines applied to each real
            comparison window, each yielding an extra distance-to-baseline series.
        sensor: Sensor dimensions (e.g. ``{"width": .., "height": ..}``) used by
            modifiers such as ``add_noise``.
        rng: Optional ``numpy.random.Generator`` for reproducible modifiers.
        seed: Seed recorded in result settings (used to build ``rng`` if not given).
    """
    metric_impl = get_metric(metric)
    metric_name = metric_impl.name
    pipelines = list(modifier_pipelines or [])
    if rng is None:
        rng = np.random.default_rng(seed)

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
    baseline_window = generated["baseline"]
    window_settings = generated["settings"]

    if baseline_window.n_events < 2:
        raise ValueError(
            f"Baseline window [{baseline_start}, {baseline_end}) μs has too few events: "
            f"{baseline_window.n_events}"
        )

    inner_kwargs = dict(metric_kwargs or {})
    if metric_impl.supports_inner_progress:
        inner_kwargs["progress"] = False

    baseline_features, _ = window_features(
        baseline_window.events,
        feature_names=feature_names,
        feature_scales=feature_scales,
        time_origin=baseline_window.start,
    )

    real_results: list[dict[str, Any]] = []
    v2e_results: list[dict[str, Any]] = []
    modified_real_results: dict[str, list[dict[str, Any]]] = {p.name: [] for p in pipelines}

    desc = f"Baseline comparison [{metric_name}]"
    if name:
        desc = f"{desc} {name}"
    total = len(generated["real"]) + len(generated["v2e"])
    with tqdm(total=total, desc=desc, unit="window", disable=not progress) as bar:
        for window in generated["real"]:
            real_results.append(
                _distance_entry(
                    window.start,
                    window.end,
                    window.events,
                    baseline_features,
                    metric_impl,
                    inner_kwargs,
                    feature_names=feature_names,
                    feature_scales=feature_scales,
                )
            )
            for pipeline in pipelines:
                context = ModifierContext(
                    rng=rng,
                    window_start=window.start,
                    window_end=window.end,
                    sensor=sensor,
                )
                modified_events = pipeline.apply(window.events, context)
                modified_real_results[pipeline.name].append(
                    _distance_entry(
                        window.start,
                        window.end,
                        modified_events,
                        baseline_features,
                        metric_impl,
                        inner_kwargs,
                        feature_names=feature_names,
                        feature_scales=feature_scales,
                    )
                )
            bar.update(1)

        for window in generated["v2e"]:
            v2e_results.append(
                _distance_entry(
                    window.start,
                    window.end,
                    window.events,
                    baseline_features,
                    metric_impl,
                    inner_kwargs,
                    feature_names=feature_names,
                    feature_scales=feature_scales,
                )
            )
            bar.update(1)

    return {
        "name": name,
        "strategy": "baseline_comparison",
        "metric": metric_name,
        "baseline": {
            "start": baseline_window.start,
            "end": baseline_window.end,
            "n_events": baseline_window.n_events,
            "source": "real",
        },
        "real_windows": real_results,
        "v2e_windows": v2e_results,
        "modified_real_windows": modified_real_results,
        "modifiers": {p.name: p.describe() for p in pipelines},
        "settings": {
            **window_settings,
            "metric": metric_name,
            "metric_kwargs": {str(k): yaml_safe(v) for k, v in inner_kwargs.items()},
            "seed": seed,
        },
    }


def plot_baseline_comparison(
    results: dict[str, Any],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
    time_unit: str = "μs",
):
    """Plot distance vs window start time for baseline comparison results."""
    fig, ax = plt.subplots(figsize=figsize)
    metric = results.get("metric", "distance")
    metric_label = metric.upper() if metric == "mmd" else metric

    real = results.get("real_windows", [])
    v2e = results.get("v2e_windows", [])

    if real:
        xs = [w["start"] for w in real]
        ys = [w["distance"] for w in real]
        ax.plot(xs, ys, marker="o", linestyle="-", label="real vs real-baseline", color="tab:blue")

    if v2e:
        xs = [w["start"] for w in v2e]
        ys = [w["distance"] for w in v2e]
        ax.plot(xs, ys, marker="s", linestyle="-", label="v2e vs real-baseline", color="tab:orange")

    modified_real = results.get("modified_real_windows", {})
    variant_colors = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
    for index, (variant, entries) in enumerate(modified_real.items()):
        if not entries:
            continue
        xs = [w["start"] for w in entries]
        ys = [w["distance"] for w in entries]
        ax.plot(
            xs,
            ys,
            marker="x",
            linestyle="--",
            label=f"real [{variant}] vs real-baseline",
            color=variant_colors[index % len(variant_colors)],
        )

    baseline = results["baseline"]
    ax.axvspan(
        baseline["start"],
        baseline["end"],
        color="gray",
        alpha=0.2,
        label=f"baseline real [{baseline['start']}, {baseline['end']}] {time_unit}",
    )

    settings = results.get("settings", {})
    name = results.get("name")
    stride = settings.get("stride")
    width = settings.get("width")

    default_title = (
        f"{metric_label} vs baseline real window "
        f"[{baseline['start']}, {baseline['end']}] {time_unit}"
    )
    if name:
        default_title = f"[{name}] " + default_title
    if stride is not None and width is not None and stride != width:
        default_title += f"  (stride={stride} {time_unit})"

    ax.set_xlabel(f"Window start time ({time_unit})")
    ax.set_ylabel(metric_label)
    ax.set_title(title or default_title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def save_baseline_comparison_results(
    results: dict[str, Any],
    output_dir: str | Path,
    *,
    run_name: str | None = None,
    file_prefix: str | None = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> dict[str, str]:
    """Persist baseline comparison results as CSV, YAML metadata, and optional plot."""
    output_dir = Path(output_dir)
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scheme_name = results.get("name")
        if scheme_name:
            run_name = f"run_{scheme_name}_{timestamp}"
        else:
            run_name = f"run_{timestamp}"

    run_dir = output_dir if run_name == "" else output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_file_stem(file_prefix) if file_prefix else ""
    csv_path = run_dir / (f"{stem}_baseline_results.csv" if stem else "baseline_results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comparison_set",
            "variant",
            "window_start",
            "window_end",
            "n_events",
            "distance",
            "distance_squared",
            "status",
        ])

        def _write_rows(comparison_set, variant, entries):
            for w in entries:
                writer.writerow([
                    comparison_set,
                    variant,
                    w["start"],
                    w["end"],
                    w["n_events"],
                    w["distance"],
                    w.get("distance_squared", ""),
                    w.get("status", ""),
                ])

        _write_rows("real_vs_real", "original", results.get("real_windows", []))
        for variant, entries in results.get("modified_real_windows", {}).items():
            _write_rows("real_vs_real", variant, entries)
        _write_rows("v2e_vs_real", "original", results.get("v2e_windows", []))

    yaml_path = run_dir / (f"{stem}_baseline_metadata.yaml" if stem else "baseline_metadata.yaml")
    with yaml_path.open("w") as f:
        yaml.safe_dump(
            {
                "name": results.get("name"),
                "strategy": results.get("strategy"),
                "metric": results.get("metric"),
                "baseline": results["baseline"],
                "modifiers": yaml_safe(results.get("modifiers", {})),
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
        plot_path = run_dir / (f"{stem}_baseline_plot.png" if stem else "baseline_plot.png")
        plot_baseline_comparison(results, save_path=plot_path, show=show_plot)
        paths["plot"] = str(plot_path)

    return paths
