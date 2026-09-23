"""Shared windowing, measurement, output and plotting for perturbation studies.

Metric calculations retain all events. Only the HTML point-cloud examples are
subsampled. Each helper handles one stage; perturbation definitions live separately.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
import yaml

if __package__:
    from .common import METRIC_KEYS, MIN_EVENTS, ResultWriter, Stream, Trial, measure, to_points
else:
    from common import METRIC_KEYS, MIN_EVENTS, ResultWriter, Stream, Trial, measure, to_points


# --- Data loading and window preparation ---

@dataclass
class ReferenceWindow:
    """One prepared window and the physical metadata needed by its perturbations.

    points contains (x/sx, y/sy, (t-start_us)/st), without polarity. real remains
    open so a temporal phase offset can read another window from the recording.
    """
    points: np.ndarray
    trial: Trial
    real: Stream
    start_us: int
    end_us: int
    period_index: int
    window_index: int
    scales: dict
    sensor: dict


def load_data(trials_dir: str | Path, name: str) -> tuple[Trial, Stream, Stream]:
    """Return metadata and open real/v2e readers; the caller closes both readers."""
    trial = Trial.load(trials_dir, name)
    real = Stream(trial.real_path, "real")
    try:
        v2e = Stream(trial.v2e_path, "v2e")
    except Exception:
        real.close()
        raise
    return trial, real, v2e


def period_windows(trial: Trial, real: Stream, period: int, window_us: int) -> pd.DataFrame:
    """Return every complete non-overlapping window, excluding a rotation's tail."""
    begin, end = trial.period_bounds(period, real.t_start)
    if end > real.t_end:
        raise ValueError(f"{trial.name}: rotation {period} is not fully covered by the recording")
    starts = np.arange(begin, end - window_us + 1, window_us, dtype=np.int64)
    counts = np.searchsorted(real.t, starts + window_us) - np.searchsorted(real.t, starts)
    return pd.DataFrame({"period_index": period, "window_index": np.arange(len(starts)),
                         "window_start_us": starts, "window_end_us": starts + window_us,
                         "n_events": counts, "eligible": counts >= MIN_EVENTS})


def plan_windows(trial: Trial, real: Stream, args) -> pd.DataFrame:
    """Plan all requested rotations; never silently truncate requested coverage."""
    periods = range(args.first_period, args.first_period + args.periods)
    windows = pd.concat([period_windows(trial, real, period, args.temporal_window)
                         for period in periods], ignore_index=True)
    if windows.empty:
        raise ValueError(f"{trial.name}: no complete windows; reduce temporal_window")
    return windows


def read_points(real: Stream, start_us: int, end_us: int, scales: dict) -> np.ndarray:
    """Read all events in [start,end), explicitly order x/y/t, and re-zero time."""
    events = real.slice(start_us, end_us)
    return to_points(events[["x", "y", "p", "t"]], start_us, scales)


def prepare_window(row, trial: Trial, real: Stream, config: dict) -> ReferenceWindow:
    """Prepare one window without retaining other windows' event arrays."""
    scales = config["feature_scales"]
    return ReferenceWindow(read_points(real, row.window_start_us, row.window_end_us, scales),
                           trial, real, row.window_start_us, row.window_end_us,
                           row.period_index, row.window_index, scales, config["sensor"])


def window_rng(seed: int, method_name: str, window: ReferenceWindow) -> np.random.Generator:
    """Reproducible draws, shared across magnitudes within the same window.

    Reinitializing this generator at each magnitude makes jitter directions and
    subsampling/noise prefixes consistent across the sweep. It is not a new repeat.
    """
    label = f"{seed}|{method_name}|{window.trial.name}|{window.period_index}|{window.window_index}"
    return np.random.default_rng(zlib.crc32(label.encode()) & 0xFFFFFFFF)


# --- Measurement and aggregation ---

def measure_pair(reference: np.ndarray, modified: np.ndarray, metrics: dict, info: dict) -> list[dict]:
    """Measure one pair; retain counts, invalid status and signed MMD estimates."""
    rows = []
    for key, (metric, settings) in metrics.items():
        value, squared = measure(metric, settings, reference, modified)
        status = "ok" if np.isfinite(value) else "nonfinite_distance"
        if min(len(reference), len(modified)) < MIN_EVENTS:
            status = "insufficient_events"
        rows.append({**info, "metric": key, "n_events_a": len(reference),
                     "n_events_b": len(modified), "distance": value,
                     "distance_squared": squared, "status": status})
    return rows


def evaluate_window(method, window: ReferenceWindow, sweep: list[float], seed: int,
                    metrics: dict, writer: ResultWriter, progress):
    """Apply the sweep to one reference window and flush its results incrementally."""
    base = {"trial": window.trial.name, "perturbation": method.name, "seed": seed,
            "period_index": window.period_index, "window_index": window.window_index,
            "window_start_us": window.start_us, "window_end_us": window.end_us}
    for magnitude in sweep:
        modified = method.apply(window, magnitude, window_rng(seed, method.name, window))
        info = {**base, "magnitude": magnitude, method.parameter: magnitude,
                **method.details(window, magnitude, modified)}
        writer.extend(measure_pair(window.points, modified, metrics, info))
        writer.flush()
        progress.update(len(metrics))


def summarize_distances(results: pd.DataFrame) -> pd.DataFrame:
    """Average valid comparisons equally; quantiles are descriptive, not CIs."""
    values = results.copy()
    values.loc[values.status != "ok", ["distance", "distance_squared"]] = np.nan
    summary = (values.groupby(["trial", "perturbation", "magnitude", "metric"], sort=False)
               .agg(mean_distance=("distance", "mean"), median_distance=("distance", "median"),
                    q05_distance=("distance", lambda s: s.quantile(0.05)),
                    q95_distance=("distance", lambda s: s.quantile(0.95)),
                    mean_distance_squared=("distance_squared", "mean"),
                    n_comparisons=("distance", "size"), n_valid=("distance", "count"),
                    mean_n_events_a=("n_events_a", "mean"), mean_n_events_b=("n_events_b", "mean"))
               .reset_index())
    summary["n_excluded"] = summary.n_comparisons - summary.n_valid
    return summary


def evaluate_perturbation(method, trial: Trial, real: Stream, args, config: dict,
                          metrics: dict, output_dir: Path) -> pd.DataFrame:
    """Run one perturbation on one trial, saving configuration before results.

    Every magnitude uses the same reference windows. Temporal coverage is checked
    for the entire phase sweep before any comparison, rather than dropping different
    windows at different offsets. Low-count comparisons remain in CSV with a status.
    """
    windows = plan_windows(trial, real, args)
    sweep = getattr(args, method.sweep_argument)
    method.validate_coverage(trial, real, windows, sweep)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_run_config(output_dir, args, config, metrics, method, trial, windows)
    windows.to_csv(output_dir / "windows.csv", index=False)
    eligible = windows[windows.eligible]
    example_index = eligible.n_events.idxmax() if not eligible.empty else None
    total = len(windows) * len(sweep) * len(metrics)
    with ResultWriter(output_dir / "results.csv") as writer, tqdm(
            total=total, desc=f"{trial.name} / {method.name}", unit="metric", mininterval=10) as progress:
        for row in windows.itertuples():
            window = prepare_window(row, trial, real, config)
            evaluate_window(method, window, sweep, args.seed, metrics, writer, progress)
            if row.Index == example_index:
                magnitude = method.example_magnitude(sweep)
                modified = method.apply(window, magnitude, window_rng(args.seed, method.name, window))
                plot_point_cloud(window, modified, method, magnitude, output_dir / "point_cloud.html")
    summary = summarize_distances(writer.frame())
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_comparison(summary, method, output_dir / f"{method.name}.html")
    return summary


# --- Reproducible output ---

def write_run_config(directory: Path, args, config: dict, metrics: dict,
                     method=None, trial: Trial | None = None, windows: pd.DataFrame | None = None):
    """Record actual metric settings, physical conventions and window selection."""
    record = {"experiment": method.name if method else "perturbation_suite",
              "created_at": datetime.now().isoformat(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "feature_scales": config["feature_scales"], "sensor": config["sensor"],
              "metric_settings": {k: settings for k, (_, settings) in metrics.items()},
              "sampling": "all events retained, except the intentional subsampling perturbation",
              "window_selection": "all complete non-overlapping windows in requested rotations",
              "minimum_events_per_side": MIN_EVENTS,
              "averaging": "equal weight per valid window; descriptive window quantiles, not CIs",
              "randomness": "one seeded realization per window; common draws across magnitudes",
              "mmd_squared": "signed unbiased estimate; distance is sqrt(max(estimate,0))",
              "boundary_policy": "no clipping/removal after spatial transforms; no temporal wraparound",
              "plotting": "highest-count eligible reference window, strongest perturbation; at most 2000 points per cloud"}
    if method is not None:
        record["perturbation_definition"] = method.description
        record["parameter"] = method.parameter
    if trial is not None:
        record["trial"] = {"name": trial.name, "real_path": str(trial.real_path),
                           "v2e_path": str(trial.v2e_path), "ellipse_centre": list(trial.ellipse_centre),
                           "rotation_period_us": int(trial.rotation_period_us),
                           "rotation_period_source": trial.rotation_period_source}
    if windows is not None:
        record["windows"] = {"total": len(windows), "eligible_reference": int(windows.eligible.sum()),
                             "excluded_reference": int((~windows.eligible).sum())}
    with (directory / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)


# --- Visualization (subsampling here never affects measurements) ---

def plot_points(points: np.ndarray) -> np.ndarray:
    """Select a deterministic display subset; leave metric inputs unchanged."""
    indices = np.random.default_rng(0).choice(len(points), min(len(points), 2000), replace=False)
    return points[indices]


def plot_point_cloud(window: ReferenceWindow, modified: np.ndarray, method, magnitude: float, path: Path):
    """Overlay original and modified clouds in the feature space used by metrics.

    For equal-size transforms the display indices correspond. Subsampling, added
    noise and temporal offset can change cardinality, so their display subsets need
    not correspond point-for-point. Each legend gives its original event count.
    """
    reference, changed = plot_points(window.points), plot_points(modified)
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=reference[:, 0], y=reference[:, 1], z=reference[:, 2],
                                 mode="markers", name=f"Reference ({len(window.points)} events)",
                                 marker=dict(size=2, color="#2874a6")))
    figure.add_trace(go.Scatter3d(x=changed[:, 0], y=changed[:, 1], z=changed[:, 2],
                                 mode="markers", name=f"Modified ({len(modified)} events)",
                                 marker=dict(size=2, color="#e67e22")))
    figure.update_layout(title=f"{window.trial.name}: {method.title}, {method.parameter}={magnitude:g}"
                         f"<br><sup>Period {window.period_index}, window {window.window_index}; "
                         "display subsample only (up to 2000 points per cloud)</sup>",
                         scene=dict(xaxis_title=f"x / {window.scales.get('x', 1):g}",
                                    yaxis_title=f"y / {window.scales.get('y', 1):g}",
                                    zaxis_title=f"relative t (µs) / {window.scales.get('t', 1):g}",
                                    aspectmode="data"), template="plotly_white", height=750)
    figure.write_html(path, include_plotlyjs=True)


def add_metric_curves(figure, summary: pd.DataFrame, metric: str, row: int):
    """Add one curve per recording to a metric's subplot."""
    for name, values in summary[summary.metric == metric].groupby("trial", sort=False):
        values = values.sort_values("magnitude")
        figure.add_trace(go.Scatter(x=values.magnitude, y=values.mean_distance, mode="lines+markers",
                                    name=name, legendgroup=name, showlegend=row == 1,
                                    customdata=values[["n_valid", "n_excluded"]].to_numpy(),
                                    hovertemplate="Magnitude: %{x}<br>Mean: %{y}<br>Valid: %{customdata[0]}"
                                                  "<br>Excluded: %{customdata[1]}<extra>%{fullData.name}</extra>"), row=row, col=1)
    figure.update_yaxes(title_text="Mean distance", row=row, col=1)


def plot_comparison(summary: pd.DataFrame, method, path: Path):
    """Five metric subplots with stable recording colours and physical x labels."""
    figure = make_subplots(rows=len(METRIC_KEYS), cols=1, shared_xaxes=True, subplot_titles=list(METRIC_KEYS))
    for row, key in enumerate(METRIC_KEYS, 1):
        add_metric_curves(figure, summary, key, row)
    figure.update_xaxes(title_text=method.x_label, row=len(METRIC_KEYS), col=1)
    palette = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]
    colours = {name: palette[index % len(palette)] for index, name in enumerate(summary.trial.unique())}
    figure.for_each_trace(lambda trace: trace.update(line_color=colours[trace.name]))
    figure.update_layout(title=f"{method.title}: average window distance", template="plotly_white",
                         height=1150, legend=dict(orientation="h", y=1.08))
    figure.write_html(path, include_plotlyjs=True)
