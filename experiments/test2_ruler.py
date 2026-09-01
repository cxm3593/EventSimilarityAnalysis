"""Test 2 - ruler: what does a known amount of difference read as?

Takes a reference span, compares it against the span starting delta later in the same
recording, and sweeps delta. The span is one whole rotation by default.

A span is not handed to a metric whole - a rotation holds nearly three million events
and MMD is quadratic in that. The span is cut into windows of --window-us, window i of
the reference is compared against window i of the shifted span, and the metric is
averaged over all the windows in the span. So one point on the ruler curve is the mean
of a few hundred to a few thousand window comparisons.

Every window has its time axis re-zeroed to its own start, so at delta = 0 each pair is
the identical point set and every metric must read exactly zero. That anchors the
scale; the rise away from zero is the ruler.

The sweep is two-phase on purpose. Nearly all of the change happens below 10 ms, so a
uniform step would step straight over it: fine geometric steps out to 50 ms, then a
fixed coarse step out to one or two whole rotations.

    python experiments/test2_ruler.py --smoke         # a couple of minutes
    python experiments/test2_ruler.py                 # the full run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from common import (METRIC_KEYS, ResultWriter, RunOutput, Trial, build_metrics,
                    cut_windows, events_per_comparison, load_config, load_streams,
                    measure, select_indices, subsample, to_points)

TRIALS_DIR = r"C:/Users/cxm3593/Academic/Workspace/EventCamCalib/output/trials"

MUTED = "#52514e"
METRIC_COLOURS = {"mmd_rbf03": "#2a78d6", "mmd_rbf15": "#eb6834", "mmd_rbf75": "#1baf7a",
                  "swd": "#eda100", "chamfer": "#4a3aa7"}
NEAR_STEPS_US = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
PLOT_MAX_POINTS = 8000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="optical_chopper_data_f1")
    parser.add_argument("--trials-dir", default=TRIALS_DIR)
    parser.add_argument("--baseline-start-us", type=int, default=None,
                        help="where the first reference window begins; default segment start")
    parser.add_argument("--n-baseline-windows", type=int, default=20,
                        help="reference positions to average over")
    parser.add_argument("--baseline-spacing-us", type=int, default=None,
                        help="gap between reference positions; default is one rotation "
                             "divided by their number, or one whole rotation when the "
                             "comparison length is itself a whole rotation")
    parser.add_argument("--span-us", default="period",
                        help="length of each side of a comparison: a number of "
                             "microseconds, or 'period' for one whole rotation")
    parser.add_argument("--window-us", type=int, default=5000,
                        help="the unit the metric actually runs on, inside the span")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap the windows compared per span, spread evenly over it")
    parser.add_argument("--near-steps-us", type=int, nargs="+", default=NEAR_STEPS_US)
    parser.add_argument("--far-step-us", type=int, default=20000)
    parser.add_argument("--max-shift-periods", type=float, default=2.0)
    parser.add_argument("--fixed-step-only", action="store_true",
                        help="uniform sweep instead of the two-phase one")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--floor-from", default=None,
                        help="a test1_null run folder, for the chance band on the figures")
    parser.add_argument("--events-per-comparison", default="none",
                        help="events each side is cut to: 'none' (no cap, default), "
                             "'auto' (the old median-based table), or an integer")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None)
    args = parser.parse_args()
    chosen = args.metrics
    if args.smoke:
        args.n_baseline_windows = 1
        args.far_step_us, args.max_shift_periods = 400000, 0.25
        args.max_windows = args.max_windows or 20
        args.metrics = chosen or ["mmd_rbf15", "chamfer"]
    else:
        args.metrics = chosen or list(METRIC_KEYS)
    return args


def shift_grid(args, rotation_us: int) -> np.ndarray:
    """The delta values to sweep, in microseconds.

    Zero is always present: it is the self-comparison that anchors the scale.
    """
    limit = int(round(args.max_shift_periods * rotation_us))
    if args.fixed_step_only:
        return np.arange(0, limit + 1, args.far_step_us, dtype=np.int64)
    near = np.asarray(sorted(args.near_steps_us), dtype=np.int64)
    near = near[near <= limit]
    far = np.arange(args.far_step_us, limit + 1, args.far_step_us, dtype=np.int64)
    far = far[far > (near.max() if len(near) else 0)]
    return np.unique(np.concatenate([[0], near, far]))


def load_floor(folder, window_us: int) -> dict:
    """Chance level per metric from a test 1 run, as (median, q95)."""
    if not folder:
        return {}
    path = Path(folder) / "summary.csv"
    if not path.exists():
        print(f"floor: {path} not found, figures drawn without the chance band")
        return {}
    table = pd.read_csv(path)
    table = table[table.window_length_us == window_us]
    if table.empty:
        print(f"floor: no {window_us} us rows in {path}, band omitted")
        return {}
    grouped = table.groupby("metric")[["median_distance", "q95_distance"]].mean()
    return {k: (float(v["median_distance"]), float(v["q95_distance"]))
            for k, v in grouped.iterrows()}


def span_windows(real, start_us, span_us, window_us, feature_scales, count, keep,
                 seed):
    """Cut one span into windows and return them as (N, 3) point arrays.

    Each window is re-zeroed to its own start, so window i of a span and window i of
    the same span shifted by delta are directly comparable.
    """
    events = real.slice(start_us, start_us + span_us)
    windows, _ = cut_windows(events, start_us, start_us + span_us, window_us,
                             feature_scales)
    windows = [windows[i] for i in keep]
    if count is None:
        return windows, len(events)
    return ([subsample(w, count, np.random.default_rng(seed + i))
             for i, w in enumerate(windows)], len(events))


def main():
    args = parse_args()
    config = load_config()
    feature_scales = config["feature_scales"]
    trial = Trial.load(args.trials_dir, args.trial)
    real = load_streams(trial, sources=("real",))["real"]
    metrics = build_metrics(args.metrics, config)

    # "period" means each side of a comparison is one whole rotation.
    whole_period = str(args.span_us).lower() == "period"
    span_us = trial.rotation_period_us if whole_period else int(args.span_us)
    args.span_us = span_us
    count = events_per_comparison(args.window_us, args.events_per_comparison)

    baseline_start = args.baseline_start_us
    if baseline_start is None:
        baseline_start = real.t_start
    shifts = shift_grid(args, trial.rotation_period_us)

    # Which windows inside a span get compared. All of them unless capped, and a cap
    # spreads them over the span rather than taking a prefix.
    # Only complete windows are comparable. A final short remainder has a different
    # sample size and temporal support, so including it biases the per-shift average.
    n_windows_total = int(span_us // args.window_us)
    keep = select_indices(n_windows_total, args.max_windows)

    # A whole-rotation span already covers every phase of the chopper, so reference
    # positions are stepped a full rotation apart, which also keeps them from
    # overlapping. Shorter spans are spread across one rotation instead, since the
    # chopper's structure varies with phase and adjacent spans would sample one phase.
    spacing = args.baseline_spacing_us
    if spacing is None:
        spacing = (trial.rotation_period_us if whole_period
                   else trial.rotation_period_us // max(1, args.n_baseline_windows))
    baselines = [int(baseline_start + i * spacing) for i in range(args.n_baseline_windows)]
    keep_baselines = [b for b in baselines
                      if b + int(shifts[-1]) + span_us <= real.t_end]
    if len(keep_baselines) < len(baselines):
        print(f"trimmed baselines {len(baselines)} -> {len(keep_baselines)}: the "
              f"furthest shift would run past the end of the recording")
        baselines = keep_baselines
    if not baselines:
        raise SystemExit(
            f"no reference position fits: one comparison needs "
            f"{(int(shifts[-1]) + span_us) / 1e6:.2f} s but the recording is only "
            f"{(real.t_end - real.t_start) / 1e6:.2f} s. Lower --max-shift-periods.")

    output = RunOutput("test2_ruler", trial,
                       tag=("smoke" if args.smoke
                            else f"period_w{args.window_us}us" if whole_period
                            else f"span{span_us}_w{args.window_us}us"),
                       existing=args.resume)
    parameters = dict(vars(args))
    parameters.update(baseline_starts_us=baselines, shifts_us=[int(s) for s in shifts],
                      events_per_comparison=count,
                      span_length_us=int(span_us),
                      span_is_whole_period=bool(whole_period),
                      windows_per_span=int(len(keep)),
                      windows_per_span_available=int(n_windows_total),
                      baseline_spacing_us=int(spacing),
                      rotation_period_us=trial.rotation_period_us)
    output.write_config(parameters, args.metrics, config)

    print(f"trial {trial.name}   rotation {trial.rotation_period_us:,} us "
          f"({trial.rotation_period_source})")
    length = "one whole rotation" if whole_period else f"{span_us:,} us"
    print(f"span: {length} ({span_us:,} us), compared as {len(keep):,} of "
          f"{n_windows_total:,} windows of {args.window_us:,} us "
          f"({'no cap' if count is None else f'{count:,} events'} per window)")
    print(f"{len(baselines)} reference positions from {baseline_start:,} us, "
          f"spacing {spacing:,} us")
    print(f"{len(shifts)} shifts, 0 to {shifts[-1]:,} us "
          f"({shifts[-1] / trial.rotation_period_us:.2f} rotations)")
    print(f"metrics: {', '.join(args.metrics)}")
    print(f"{len(baselines) * len(shifts) * len(keep) * len(metrics):,} window "
          f"comparisons in total")
    print(f"output: {output.directory}\n")

    # One row per (reference position, shift, metric): the metric averaged over every
    # window in the span. The per-window values are not kept - at a few hundred windows
    # per span and a couple of hundred shifts that is millions of rows per trial.
    writer = ResultWriter(output.csv("results_raw.csv"),
                          key_fields=("baseline_start_us", "shift_us", "metric"),
                          resume=bool(args.resume))
    if args.resume:
        print(f"resuming: {writer.n_rows:,} rows already on disk\n")

    progress = tqdm(total=len(baselines) * len(shifts), unit="shift",
                    dynamic_ncols=True, smoothing=0.05,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} shifts "
                               "[{elapsed}<{remaining}, {rate_fmt}]")

    for baseline in baselines:
        seed = int(baseline % (2 ** 31))
        progress.set_description(f"reference {baseline:,} us")
        reference, n_events_a = span_windows(
            real, baseline, span_us, args.window_us, feature_scales, count, keep, seed)

        for index, shift in enumerate(shifts):
            shift = int(shift)
            pending = [k for k in metrics if not writer.already_done(baseline, shift, k)]
            if not pending:
                progress.update(1)
                continue
            # The same seed as the reference draw, so at shift = 0 each pair is the
            # identical point set and every metric reads exactly zero.
            other, n_events_b = span_windows(
                real, baseline + shift, span_us, args.window_us, feature_scales, count,
                keep, seed)

            values = {k: [] for k in pending}
            squares = {k: [] for k in pending}
            for a, b in zip(reference, other):
                if len(a) == 0 or len(b) == 0:
                    continue
                for key in pending:
                    metric, settings = metrics[key]
                    value, square = measure(metric, settings, a, b)
                    values[key].append(value)
                    squares[key].append(square)

            for key in pending:
                if not values[key]:
                    continue
                column = np.asarray(values[key], dtype=float)
                squared = np.asarray(squares[key], dtype=float)
                writer.append({
                    "trial": trial.name, "baseline_start_us": baseline,
                    "span_length_us": span_us, "window_length_us": args.window_us,
                    "metric": key, "shift_us": shift,
                    "shift_deg": shift / trial.rotation_period_us * 360.0,
                    "n_windows": int(len(column)),
                    "n_events_a": int(n_events_a), "n_events_b": int(n_events_b),
                    "distance": float(column.mean()),
                    "sum_distance": float(column.sum()),
                    "sd_within_span": float(column.std(ddof=1)) if len(column) > 1 else 0.0,
                    "distance_squared": float(squared.mean()),
                })
            writer.flush()
            progress.set_postfix_str(f"shift {shift:,} us")
            progress.update(1)

    progress.close()
    raw = writer.frame()
    if raw.empty:
        print("no rows produced")
        return

    # results.csv carries the spec's schema: averaged across the reference positions.
    results = (raw.groupby(["trial", "span_length_us", "window_length_us", "metric",
                            "shift_us", "shift_deg"])
               .agg(mean_distance=("distance", "mean"),
                    sd_distance=("distance", "std"),
                    mean_sum_distance=("sum_distance", "mean"),
                    mean_sd_within_span=("sd_within_span", "mean"),
                    mean_distance_squared=("distance_squared", "mean"),
                    n_windows=("n_windows", "sum"),
                    n_baseline_windows=("distance", "count"))
               .reset_index())
    results.insert(1, "baseline_start_us", baseline_start)
    results["sd_distance"] = results.sd_distance.fillna(0.0)
    results = results.sort_values(["metric", "shift_us"])
    results.to_csv(output.csv("results.csv"), index=False)

    summary = (results.groupby(["metric", "span_length_us", "window_length_us"])
               .agg(n_shifts=("shift_us", "count"),
                    distance_at_zero=("mean_distance", lambda s: float(s.iloc[0])),
                    min_distance=("mean_distance", "min"),
                    max_distance=("mean_distance", "max"),
                    mean_distance=("mean_distance", "mean"),
                    sd_distance=("mean_distance", "std"))
               .reset_index())
    summary.to_csv(output.csv("summary.csv"), index=False)

    print(f"\nwrote results_raw.csv ({len(raw):,} rows), results.csv "
          f"({len(results):,} rows) and summary.csv")
    print("\ndistance at shift zero (must be essentially zero):")
    zero = results[results.shift_us == 0][["metric", "mean_distance", "sd_distance"]]
    print(zero.to_string(index=False))

    floor = load_floor(args.floor_from, args.window_us)
    make_figures(results, args, trial, output, floor)
    draw_window_overlays(real, baselines[0], shifts, args, trial, feature_scales, output)
    real.close()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _curve_figure(results, args, title, x_column, x_title, log_x, floor, path,
                  x_max=None):
    """One row per metric, sharing the x axis.

    The five metrics differ by three orders of magnitude, so a single shared y axis
    would flatten MMD into the baseline. Each keeps its own scale; nothing is
    rescaled or normalised.
    """
    keys = [k for k in args.metrics if not results[results.metric == k].empty]
    if not keys:
        return
    figure = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                           vertical_spacing=0.035,
                           subplot_titles=keys)
    for row, key in enumerate(keys, start=1):
        block = results[results.metric == key].sort_values(x_column)
        if x_max is not None:
            block = block[block[x_column] <= x_max]
        if log_x:
            block = block[block[x_column] > 0]
        figure.add_trace(go.Scatter(
            x=block[x_column], y=block.mean_distance, mode="lines+markers",
            name=key, legendgroup=key, showlegend=False,
            line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2),
            marker=dict(size=5),
            error_y=dict(type="data", array=block.sd_distance, visible=True,
                         thickness=0.8, width=0)), row=row, col=1)
        if key in floor:
            low, high = floor[key]
            figure.add_hrect(y0=low, y1=high, row=row, col=1,
                             fillcolor="#9aa0a6", opacity=0.25, line_width=0,
                             annotation_text="test 1 chance band" if row == 1 else None,
                             annotation_position="top left")
        figure.update_yaxes(title_text="distance", row=row, col=1)
    figure.update_xaxes(title_text=x_title, row=len(keys), col=1)
    if log_x:
        figure.update_xaxes(type="log")
    figure.update_layout(title=title, template="plotly_white",
                         height=240 * len(keys), showlegend=False,
                         margin=dict(l=70, r=30, t=80, b=60))
    figure.write_html(str(path), include_plotlyjs="cdn")


def make_figures(results, args, trial, output, floor):
    _curve_figure(results, args,
                  f"Ruler, near field &mdash; {args.trial}, {args.window_us} us windows "
                  f"within a {args.span_us:,} us span",
                  "shift_us", "shift delta (us, log)", True, floor,
                  output.figure("ruler_near.html"), x_max=50000)
    _curve_figure(results, args,
                  f"Ruler, full sweep &mdash; {args.trial}, {args.window_us} us windows "
                  f"within a {args.span_us:,} us span",
                  "shift_us", "shift delta (us)", False, floor,
                  output.figure("ruler_full.html"))
    _curve_figure(results, args,
                  f"Ruler in degrees of rotation &mdash; {args.trial}",
                  "shift_deg", "shift (degrees of wheel rotation)", False, floor,
                  output.figure("ruler_deg.html"))
    print(f"figures in {output.figures}")


def draw_window_overlays(real, baseline, shifts, args, trial, feature_scales, output):
    """Reference and shifted window overlaid, so the geometry behind a value shows."""
    quarter = trial.rotation_period_us // 4
    wanted = [0, 1000, 10000, 100000, quarter, trial.rotation_period_us // 2]
    for target in wanted:
        shift = int(min(shifts, key=lambda s: abs(int(s) - target)))
        if shift + args.window_us + baseline > real.t_end:
            continue
        ref = real.slice(baseline, baseline + args.window_us)
        oth = real.slice(baseline + shift, baseline + shift + args.window_us)
        if len(ref) == 0 or len(oth) == 0:
            continue
        figure = go.Figure()
        for events, origin, name, colour in (
                (ref, baseline, "reference", "#2a78d6"),
                (oth, baseline + shift, f"shifted +{shift} us", "#eb6834")):
            step = max(1, len(events) // PLOT_MAX_POINTS)
            shown = events[::step]
            figure.add_trace(go.Scatter3d(
                x=shown["x"], y=shown["y"], z=(shown["t"] - origin) / 1000.0,
                mode="markers", name=f"{name} ({len(events):,} events)",
                marker=dict(size=1.6, color=colour, opacity=0.6), hoverinfo="skip"))
        figure.update_layout(
            title=(f"Reference against +{shift:,} us "
                   f"({shift / trial.rotation_period_us * 360:.2f} deg) &mdash; "
                   f"{args.window_us} us windows"),
            scene=dict(xaxis_title="x (px)", yaxis_title="y (px)",
                       zaxis_title="t within window (ms)"),
            template="plotly_white", height=620, margin=dict(l=0, r=0, t=50, b=0))
        figure.write_html(str(output.figure(f"windows_3d_shift{shift}us.html")),
                          include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
