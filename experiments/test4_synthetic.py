"""Test 4 - synthetic data: where do simulator streams sit against real data?

4a compares every rotation against a baseline real rotation, window by window, for
real, v2e, optional additional simulators, and optionally the modified-real sources.

4b converts the v2e gap into physical equivalents by reading it back off the Test 2
ruler and the Test 3a sweeps. That is interpolation on curves this run did not produce;
if a curve does not reach the v2e value the equivalent is left as not-a-number rather
than extrapolated.

4c is the all-to-all matrix over every real and v2e rotation, laid out with MDS.

    python experiments/test4_synthetic.py --smoke
    python experiments/test4_synthetic.py --ruler-from <test2 run> --sweeps-from <test3a run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

from common import (METRIC_KEYS, ResultWriter, RunOutput, Stream, Trial, build_metrics,
                    events_per_comparison, load_config, load_streams, measure,
                    select_indices, subsample, to_points)
from test3_modifier import apply_generic, stable_seed

TRIALS_DIR = r"C:/Users/cxm3593/Academic/Workspace/EventCamCalib/output/trials"

MUTED = "#52514e"
METRIC_COLOURS = {"mmd_rbf03": "#2a78d6", "mmd_rbf15": "#eb6834", "mmd_rbf75": "#1baf7a",
                  "swd": "#eda100", "chamfer": "#4a3aa7"}
SOURCE_COLOURS = {"real": "#555555", "v2e": "#d1352b", "v2ce": "#2a78d6"}

# One representative magnitude per Test 3a modifier, so each becomes one curve here.
MODIFIED_SOURCES = {
    "real_spatial_offset_xy_2px": ("spatial_offset_xy", 2.0),
    "real_scaling_1.02": ("scaling", 1.02),
    "real_subsample_0.50": ("subsample", 0.50),
    "real_uniform_noise_0.10": ("uniform_noise", 0.10),
    "real_temporal_clump_161us": ("temporal_clump_uniform", 161.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="optical_chopper_data_f1")
    parser.add_argument("--trials-dir", default=TRIALS_DIR)
    parser.add_argument("--periods", type=int, default=None,
                        help="rotations to use; default is every whole one available")
    parser.add_argument("--baseline-period", type=int, default=0)
    parser.add_argument("--window-us", type=int, default=5000)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--no-all-to-all", dest="all_to_all", action="store_false",
                        help="skip 4c, which is the slow part")
    parser.add_argument("--ruler-from", default=None, help="a test2_ruler run folder")
    parser.add_argument("--sweeps-from", default=None,
                        help="a test3_modifier 3a run folder")
    parser.add_argument("--floor-from", default=None, help="a test1_null run folder")
    parser.add_argument("--v2ce-path", type=Path, default=None,
                        help="converted final_masked_v2ce.h5 to include as a source")
    parser.add_argument("--no-modified", dest="modified", action="store_false",
                        help="compare real and simulators only")
    parser.add_argument("--polarity", choices=["drop", "split"], default="drop",
                        help="'split' compares ON and OFF separately, as a check")
    parser.add_argument("--events-per-comparison", default="none",
                        help="events each side is cut to: 'none' (no cap, default), "
                             "'auto' (the old median-based table), or an integer")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None)
    args = parser.parse_args()
    chosen = args.metrics
    if args.smoke:
        args.periods, args.max_windows = 3, 4
        args.metrics = chosen or ["mmd_rbf15", "chamfer"]
    else:
        args.metrics = chosen or list(METRIC_KEYS)
    return args


def whole_periods(trial, streams, requested):
    """How many complete rotations both recordings cover."""
    end = min(s.t_end for s in streams)
    start = streams[0].t_start
    available = int((end - start) // trial.rotation_period_us)
    return available if requested is None else min(requested, available)


def chamfer_directions(a: np.ndarray, b: np.ndarray) -> tuple:
    """Mean nearest-neighbour distance each way.

    Open3D returns only the symmetric sum, and the toolbox is not ours to change here,
    so the two directions are computed alongside it. a_to_b large means points in a
    have no near counterpart in b.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    a_to_b = float(cKDTree(b).query(a, k=1)[0].mean())
    b_to_a = float(cKDTree(a).query(b, k=1)[0].mean())
    return a_to_b, b_to_a


# ---------------------------------------------------------------------------
# 4a - every source against the baseline rotation
# ---------------------------------------------------------------------------

def window_points(stream, start, window_us, feature_scales, polarity):
    events = stream.slice(start, start + window_us)
    if polarity == "drop":
        return {"": to_points(events, start, feature_scales)}
    out = {}
    for label, mask in (("on", events["p"] > 0), ("off", events["p"] <= 0)):
        out[label] = to_points(events[mask], start, feature_scales)
    return out


def run_4a(args, trial, real, simulator_streams, metrics, ctx, count, n_periods, writer):
    feature_scales = ctx["feature_scales"]
    base_begin, _ = trial.period_bounds(args.baseline_period, real.t_start)
    starts_rel = np.arange(0, trial.rotation_period_us, args.window_us)
    chosen = select_indices(len(starts_rel), args.max_windows)
    print(f"4a: {len(chosen)} of {len(starts_rel)} windows per rotation, "
          f"{n_periods} rotations\n")

    sources = ["real"] + list(simulator_streams)
    if args.modified:
        sources += list(MODIFIED_SOURCES)
    for source in sources:
        for period in range(n_periods):
            begin, _ = trial.period_bounds(period, real.t_start)
            for index in chosen:
                index = int(index)
                offset = int(starts_rel[index])
                base_start = base_begin + offset
                start = begin + offset
                base_seed = stable_seed("t4", args.seed, index)

                baseline = window_points(real, base_start, args.window_us,
                                         feature_scales, args.polarity)
                if source in simulator_streams:
                    other = window_points(simulator_streams[source], start, args.window_us,
                                          feature_scales, args.polarity)
                else:
                    other = window_points(real, start, args.window_us,
                                          feature_scales, args.polarity)

                for channel in baseline:
                    # The stored label, not the raw key: drop mode uses "" internally
                    # and writes "all", and the resume check must match what is on disk.
                    label = channel or "all"
                    a = subsample(baseline[channel], count,
                                  np.random.default_rng(base_seed))
                    raw_b = other.get(channel, np.empty((0, 3)))
                    target = count
                    if source in MODIFIED_SOURCES:
                        name, magnitude = MODIFIED_SOURCES[source]
                        raw_b = apply_generic(name, magnitude, raw_b, ctx,
                                              np.random.default_rng(
                                                  stable_seed(source, args.seed, index)))
                        if name == "subsample":
                            reference = count if count is not None else len(raw_b)
                            target = int(round(reference * magnitude))
                    b = subsample(raw_b, target, np.random.default_rng(base_seed + 1))

                    for key, (metric, settings) in metrics.items():
                        if writer.already_done(source, period, index, label, key):
                            continue
                        value, squared = measure(metric, settings, a, b)
                        row = {
                            "trial": trial.name, "source": source,
                            "period_index": period, "metric": key,
                            "window_index": index, "window_start_us": start,
                            "polarity_channel": label,
                            "n_events_a": len(a), "n_events_b": len(b),
                            "distance": value, "distance_squared": squared,
                        }
                        if key == "chamfer":
                            fwd, rev = chamfer_directions(a, b)
                            row["chamfer_a_to_b"], row["chamfer_b_to_a"] = fwd, rev
                        else:
                            row["chamfer_a_to_b"] = float("nan")
                            row["chamfer_b_to_a"] = float("nan")
                        writer.append(row)
            writer.flush()
        print(f"   source {source}: done")


# ---------------------------------------------------------------------------
# 4c - all to all
# ---------------------------------------------------------------------------

def run_4c(args, trial, real, v2e, metrics, ctx, count, n_periods, output):
    feature_scales = ctx["feature_scales"]
    starts_rel = np.arange(0, trial.rotation_period_us, args.window_us)
    chosen = select_indices(len(starts_rel), args.max_windows)
    labels = ([f"real_p{p}" for p in range(n_periods)] +
              [f"v2e_p{p}" for p in range(n_periods)])
    streams = {"real": real, "v2e": v2e}

    print(f"\n4c: {len(labels)} segments, {len(labels) * (len(labels) - 1) // 2} pairs, "
          f"{len(chosen)} windows each")

    cache = {}
    for label in labels:
        source, period = label.rsplit("_p", 1)
        begin, _ = trial.period_bounds(int(period), real.t_start)
        cloud = []
        for index in chosen:
            start = begin + int(starts_rel[int(index)])
            pts = to_points(streams[source].slice(start, start + args.window_us),
                            start, feature_scales)
            cloud.append(subsample(pts, count,
                                   np.random.default_rng(stable_seed("t4c", int(index)))))
        cache[label] = cloud

    matrices = {key: np.zeros((len(labels), len(labels))) for key in metrics}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            per_metric = {key: [] for key in metrics}
            for a, b in zip(cache[labels[i]], cache[labels[j]]):
                for key, (metric, settings) in metrics.items():
                    per_metric[key].append(measure(metric, settings, a, b)[0])
            for key in metrics:
                value = float(np.nanmean(per_metric[key]))
                matrices[key][i, j] = matrices[key][j, i] = value
        print(f"   row {labels[i]} done")

    for key, matrix in matrices.items():
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
        frame.to_csv(output.csv(f"all_to_all_{key}.csv"))
        heatmap_figure(frame, key, args, output)
        mds_figure(frame, key, args, output)
    return matrices, labels


# ---------------------------------------------------------------------------
# 4b - placement
# ---------------------------------------------------------------------------

def invert_curve(x_values, y_values, target):
    """Smallest x whose curve value reaches `target`, by linear interpolation.

    Returns not-a-number when the curve never reaches it: extrapolating past the end
    of a measured sweep would invent a number.
    """
    order = np.argsort(x_values)
    xs, ys = np.asarray(x_values)[order], np.asarray(y_values)[order]
    if not np.isfinite(target) or len(xs) < 2 or target > np.nanmax(ys):
        return float("nan")
    for i in range(1, len(xs)):
        lo, hi = ys[i - 1], ys[i]
        if (lo <= target <= hi) and hi > lo:
            span = (target - lo) / (hi - lo)
            return float(xs[i - 1] + span * (xs[i] - xs[i - 1]))
    return float("nan")


def build_placement(results, args, output, floor):
    ruler = _read(args.ruler_from, "results.csv")
    sweeps = _read(args.sweeps_from, "summary.csv")
    # With --polarity split there is no "all" channel, so group by whatever channels
    # the run actually produced rather than assuming the polarity-dropped one.
    simulator_names = [name for name in SOURCE_COLOURS
                       if name != "real" and name in set(results.source)]
    simulator_mean = (results[results.source.isin(simulator_names)]
                      .groupby(["source", "metric", "polarity_channel"])
                      ["distance"].mean())

    rows = []
    for (source, key, channel), value in simulator_mean.items():
        row = {"source": source, "metric": key, "polarity_channel": channel,
               "mean_distance": float(value),
               "floor": floor.get(key, (float("nan"),))[0], "ceiling": float("nan"),
               "percentage_of_ruler_max": float("nan"),
               "equivalent_shift_us": float("nan"),
               "equivalent_shift_deg": float("nan"),
               "equivalent_spatial_offset_px": float("nan"),
               "equivalent_noise_fraction": float("nan")}
        if ruler is not None:
            block = ruler[ruler.metric == key].sort_values("shift_us")
            if not block.empty:
                row["ceiling"] = float(block.mean_distance.max())
                if row["ceiling"] != 0:
                    row["percentage_of_ruler_max"] = 100.0 * float(value) / row["ceiling"]
                row["equivalent_shift_us"] = invert_curve(
                    block.shift_us.values, block.mean_distance.values, value)
                row["equivalent_shift_deg"] = invert_curve(
                    block.shift_deg.values, block.mean_distance.values, value)
        if sweeps is not None:
            for modifier, column in (("spatial_offset_xy", "equivalent_spatial_offset_px"),
                                     ("uniform_noise", "equivalent_noise_fraction")):
                block = sweeps[(sweeps.metric == key) &
                               (sweeps.modifier == modifier)].sort_values("magnitude")
                if not block.empty:
                    row[column] = invert_curve(block.magnitude.values,
                                               block.mean_distance.values, value)
        rows.append(row)

    placement = pd.DataFrame(rows)
    placement.to_csv(output.csv("placement.csv"), index=False)
    print(f"\nwrote placement.csv ({len(placement)} rows)")
    if ruler is None:
        print("   note: --ruler-from not given, shift equivalents are not-a-number")
    if sweeps is None:
        print("   note: --sweeps-from not given, modifier equivalents are not-a-number")
    return placement, ruler


def _read(folder, name):
    if not folder:
        return None
    path = Path(folder) / name
    if not path.exists():
        print(f"   {path} not found")
        return None
    return pd.read_csv(path)


def load_floor(folder, window_us):
    if not folder:
        return {}
    path = Path(folder) / "summary.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    table = table[table.window_length_us == window_us]
    if table.empty:
        return {}
    grouped = table.groupby("metric")[["median_distance", "q95_distance"]].mean()
    return {k: (float(v["median_distance"]), float(v["q95_distance"]))
            for k, v in grouped.iterrows()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def baseline_figure(results, args, output, floor):
    keys = list(args.metrics)
    channels = sorted(results.polarity_channel.unique())
    dashes = {"all": None, "on": None, "off": "dash"}
    figure = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                           vertical_spacing=0.035, subplot_titles=keys)
    for row, key in enumerate(keys, start=1):
        source_order = [name for name in SOURCE_COLOURS if name in set(results.source)]
        source_order += [name for name in MODIFIED_SOURCES if name in set(results.source)]
        for source in source_order:
            for channel in channels:
                block = results[results.polarity_channel == channel]
                line = (block[(block.metric == key) & (block.source == source)]
                        .groupby("period_index")["distance"].agg(["mean", "std"])
                        .reset_index())
                if line.empty:
                    continue
                label = source if channel == "all" else f"{source} {channel}"
                figure.add_trace(go.Scatter(
                    x=line.period_index, y=line["mean"], mode="lines+markers",
                    name=label, legendgroup=label, showlegend=(row == 1),
                    line=dict(color=SOURCE_COLOURS.get(source, MUTED),
                              width=2.5 if source in SOURCE_COLOURS else 1.4,
                              dash=(dashes.get(channel)
                                    if source in SOURCE_COLOURS else "dot")),
                    marker=dict(size=6)), row=row, col=1)
        if key in floor:
            low, high = floor[key]
            figure.add_hrect(y0=low, y1=high, row=row, col=1, fillcolor="#9aa0a6",
                             opacity=0.25, line_width=0)
        figure.update_yaxes(title_text="distance", row=row, col=1)
    figure.update_xaxes(title_text="rotation index", row=len(keys), col=1)
    figure.update_layout(title=f"Against the baseline rotation &mdash; {args.trial}, "
                               f"{args.window_us} us windows",
                         template="plotly_white", height=250 * len(keys),
                         margin=dict(l=70, r=30, t=90, b=60),
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    figure.write_html(str(output.figure("baseline_comparison.html")),
                      include_plotlyjs="cdn")


def phase_figure(results, args, trial, output):
    keys = list(args.metrics)
    channels = sorted(results.polarity_channel.unique())
    dashes = {"all": None, "on": None, "off": "dash"}
    source_order = [name for name in SOURCE_COLOURS if name in set(results.source)]
    base = results[results.source.isin(source_order)]
    figure = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                           vertical_spacing=0.035, subplot_titles=keys)
    for row, key in enumerate(keys, start=1):
        for source in source_order:
            for channel in channels:
                block = base[base.polarity_channel == channel]
                line = (block[(block.metric == key) & (block.source == source)]
                        .groupby("window_index")["distance"].mean().reset_index())
                if line.empty:
                    continue
                degrees = (line.window_index * args.window_us
                           / trial.rotation_period_us * 360)
                label = source if channel == "all" else f"{source} {channel}"
                figure.add_trace(go.Scatter(
                    x=degrees, y=line.distance, mode="lines+markers", name=label,
                    legendgroup=label, showlegend=(row == 1),
                    line=dict(color=SOURCE_COLOURS[source], width=2,
                              dash=dashes.get(channel)),
                    marker=dict(size=5)), row=row, col=1)
        figure.update_yaxes(title_text="distance", row=row, col=1)
    figure.update_xaxes(title_text="position within the rotation (degrees)",
                        row=len(keys), col=1)
    figure.update_layout(title=f"Phase profile &mdash; {args.trial}",
                         template="plotly_white", height=250 * len(keys),
                         margin=dict(l=70, r=30, t=90, b=60),
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    figure.write_html(str(output.figure("phase_profile.html")), include_plotlyjs="cdn")


def placement_figures(placement, ruler, args, output, floor):
    if ruler is None:
        return
    for _, row in placement.iterrows():
        key = row["metric"]
        channel = row.get("polarity_channel", "all")
        block = ruler[ruler.metric == key].sort_values("shift_us")
        if block.empty:
            continue
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=block.shift_us, y=block.mean_distance, mode="lines+markers",
            name="ruler (test 2)",
            line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2)))
        source = row["source"]
        colour = SOURCE_COLOURS.get(source, MUTED)
        figure.add_hline(y=row["mean_distance"], line=dict(color=colour, dash="dash"),
                         annotation_text=f"{source} gap", annotation_position="top left")
        if np.isfinite(row["equivalent_shift_us"]):
            figure.add_vline(x=row["equivalent_shift_us"],
                             line=dict(color=colour, dash="dot"),
                             annotation_text=f"{row['equivalent_shift_us']:.0f} us "
                                             f"= {row['equivalent_shift_deg']:.2f} deg",
                             annotation_position="bottom right")
        if key in floor:
            low, high = floor[key]
            figure.add_hrect(y0=low, y1=high, fillcolor="#9aa0a6", opacity=0.25,
                             line_width=0, annotation_text="test 1 chance band")
        suffix = "" if channel == "all" else f", {channel} events"
        figure.update_layout(
            title=f"Placement of the {source} gap on the ruler &mdash; {key}{suffix}",
            xaxis_title="shift delta (us)", yaxis_title="distance",
            template="plotly_white", height=460,
            margin=dict(l=70, r=30, t=80, b=60))
        name = (f"placement_{source}_{key}.html" if channel == "all" else
                f"placement_{source}_{key}_{channel}.html")
        figure.write_html(str(output.figure(name)), include_plotlyjs="cdn")


def heatmap_figure(frame, key, args, output):
    figure = go.Figure(go.Heatmap(z=frame.values, x=list(frame.columns),
                                  y=list(frame.index), colorscale="Viridis",
                                  colorbar=dict(title="distance")))
    figure.update_layout(title=f"All to all &mdash; {key}, {args.trial}, "
                               f"{args.window_us} us windows",
                         template="plotly_white", height=620, width=740,
                         xaxis=dict(tickangle=-45), margin=dict(l=90, r=40, t=80, b=110))
    figure.write_html(str(output.figure(f"all_to_all_{key}.html")),
                      include_plotlyjs="cdn")


def mds_figure(frame, key, args, output):
    try:
        from sklearn.manifold import MDS
    except ImportError:
        print("   sklearn missing, MDS skipped")
        return
    matrix = np.nan_to_num(frame.values, nan=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    model = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                normalized_stress=False)
    coords = model.fit_transform(matrix)
    figure = go.Figure()
    for source in ("real", "v2e"):
        mask = [i for i, label in enumerate(frame.index) if label.startswith(source)]
        if not mask:
            continue
        figure.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1], mode="markers+text",
            text=[frame.index[i] for i in mask], textposition="top center",
            name=source, marker=dict(size=13, color=SOURCE_COLOURS[source])))
    figure.update_layout(
        title=f"MDS layout &mdash; {key}, stress {model.stress_:.4g}",
        xaxis_title="MDS 1", yaxis_title="MDS 2", template="plotly_white",
        height=600, margin=dict(l=60, r=30, t=80, b=60))
    figure.write_html(str(output.figure(f"mds_{key}.html")), include_plotlyjs="cdn")
    print(f"   MDS {key}: stress {model.stress_:.6g}")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config()
    feature_scales = config["feature_scales"]
    trial = Trial.load(args.trials_dir, args.trial)
    streams = load_streams(trial, sources=("real", "v2e"))
    real, v2e = streams["real"], streams["v2e"]
    simulator_streams = {"v2e": v2e}
    if args.v2ce_path is not None:
        if not args.v2ce_path.exists():
            raise SystemExit(f"V2CE file does not exist: {args.v2ce_path}")
        simulator_streams["v2ce"] = Stream(args.v2ce_path, "v2ce")
    metrics = build_metrics(args.metrics, config)
    count = events_per_comparison(args.window_us, args.events_per_comparison)
    n_periods = whole_periods(trial, [real, *simulator_streams.values()], args.periods)

    ctx = {"feature_scales": feature_scales,
           "t_scale": float(feature_scales.get("t", 1)),
           "centre": trial.ellipse_centre,
           "width": float(config["sensor"]["width"]),
           "height": float(config["sensor"]["height"]),
           "window_us": float(args.window_us)}

    output = RunOutput("test4_synthetic", trial,
                       tag=(("smoke" if args.smoke else f"w{args.window_us}us")
                            + ("_v2ce" if "v2ce" in simulator_streams else "")
                            + ("_polsplit" if args.polarity == "split" else "")),
                       existing=args.resume)
    parameters = dict(vars(args))
    parameters.update(events_per_comparison=count, n_periods_used=n_periods,
                      rotation_period_us=trial.rotation_period_us,
                      simulator_paths={name: str(stream.path)
                                       for name, stream in simulator_streams.items()},
                      modified_sources={k: list(v) for k, v in MODIFIED_SOURCES.items()})
    output.write_config(parameters, args.metrics, config)

    print(f"trial {trial.name}   rotation {trial.rotation_period_us:,} us "
          f"({trial.rotation_period_source})")
    print(f"{n_periods} whole rotations, baseline {args.baseline_period}, "
          f"window {args.window_us} us, {count} events per side, polarity {args.polarity}")
    print(f"metrics: {', '.join(args.metrics)}")
    print(f"output: {output.directory}\n")

    writer = ResultWriter(output.csv("results.csv"),
                          key_fields=("source", "period_index", "window_index",
                                      "polarity_channel", "metric"),
                          resume=bool(args.resume))
    if args.resume:
        print(f"resuming: {writer.n_rows:,} rows already on disk\n")

    run_4a(args, trial, real, simulator_streams, metrics, ctx, count, n_periods, writer)
    results = writer.frame()
    if results.empty:
        print("no rows produced")
        return

    summary = (results.groupby(["source", "period_index", "metric", "polarity_channel"])
               .agg(mean_distance=("distance", "mean"),
                    sum_distance=("distance", "sum"),
                    sd_distance=("distance", "std"),
                    median_distance=("distance", "median"),
                    mean_chamfer_a_to_b=("chamfer_a_to_b", "mean"),
                    mean_chamfer_b_to_a=("chamfer_b_to_a", "mean"),
                    n_windows=("distance", "count"))
               .reset_index())
    summary.to_csv(output.csv("summary.csv"), index=False)
    print(f"\nwrote results.csv ({len(results):,} rows) and summary.csv")

    floor = load_floor(args.floor_from, args.window_us)
    placement, ruler = build_placement(results, args, output, floor)
    baseline_figure(results, args, output, floor)
    phase_figure(results, args, trial, output)
    placement_figures(placement, ruler, args, output, floor)

    if args.all_to_all:
        run_4c(args, trial, real, v2e, metrics, ctx, count, n_periods, output)

    print(f"figures in {output.figures}")
    real.close()
    for stream in simulator_streams.values():
        stream.close()


if __name__ == "__main__":
    main()
