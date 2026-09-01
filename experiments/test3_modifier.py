"""Test 3 - modifier: which kinds of difference does each metric respond to?

3a sweeps a generic corruption applied to one side of a real-against-real comparison.
The unmodified side is the same window, and at magnitude zero the modifier is the
identity and the two sides are the identical point set, so every metric reads exactly
zero. The Test 1 chance level is drawn on the figures as a separate reference.

3b is not a sweep. Each directed modifier is a hypothesis about what the simulator does
wrong, checked by whether it moves real data toward v2e.

    python experiments/test3_modifier.py --smoke
    python experiments/test3_modifier.py --phase 3a
    python experiments/test3_modifier.py --phase 3b
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import (METRIC_KEYS, ResultWriter, RunOutput, Trial, build_metrics,
                    events_per_comparison, load_config, load_streams, measure,
                    select_indices, subsample, to_points)

TRIALS_DIR = r"C:/Users/cxm3593/Academic/Workspace/EventCamCalib/output/trials"

MUTED = "#52514e"
METRIC_COLOURS = {"mmd_rbf03": "#2a78d6", "mmd_rbf15": "#eb6834", "mmd_rbf75": "#1baf7a",
                  "swd": "#eda100", "chamfer": "#4a3aa7"}
PLOT_MAX_POINTS = 8000

# Magnitude zero is the identity in every sweep, and anchors the scale at exactly zero.
GENERIC_SWEEPS = {
    "spatial_offset_x": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "spatial_offset_xy": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "scaling": [1.0, 1.01, 1.02, 1.05, 1.10],
    "subsample": [1.0, 0.90, 0.75, 0.50, 0.25],
    "uniform_noise": [0.0, 0.01, 0.05, 0.10, 0.25, 0.50],
    "temporal_clump_uniform": [0.0, 10.0, 25.0, 50.0, 100.0, 161.0, 250.0],
}

DIRECTED = ["none", "clump_to_v2e_grid", "match_count_v2e_down", "match_count_real_up",
            "clump_and_match"]


def stable_seed(*parts) -> int:
    """A reproducible seed from arbitrary labels.

    Python's hash() is salted per process, so it cannot be used: the same run would
    draw different noise on a second invocation and the results would not reproduce.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="optical_chopper_data_f1")
    parser.add_argument("--trials-dir", default=TRIALS_DIR)
    parser.add_argument("--phase", choices=["3a", "3b"], default="3a")
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--first-period", type=int, default=0)
    parser.add_argument("--window-us", type=int, default=5000)
    parser.add_argument("--max-windows", type=int, default=40,
                        help="windows compared per rotation, spread over the whole turn")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modifiers", nargs="+", default=None,
                        help="trim to a subset; default is all for the chosen phase")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--floor-from", default=None,
                        help="a test1_null run folder, for the chance band on the figures")
    parser.add_argument("--events-per-comparison", default="none",
                        help="events each side is cut to: 'none' (no cap, default), "
                             "'auto' (the old median-based table), or an integer")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None)
    args = parser.parse_args()
    chosen_metrics, chosen_mods = args.metrics, args.modifiers
    if args.smoke:
        args.periods, args.seeds, args.max_windows = 2, [0], 4
        args.metrics = chosen_metrics or ["mmd_rbf15", "chamfer"]
        args.modifiers = chosen_mods or (["spatial_offset_x", "subsample"]
                                         if args.phase == "3a" else DIRECTED)
    else:
        args.metrics = chosen_metrics or list(METRIC_KEYS)
        args.modifiers = chosen_mods or (list(GENERIC_SWEEPS) if args.phase == "3a"
                                         else DIRECTED)
    return args


# ---------------------------------------------------------------------------
# Modifiers - applied to the (N, 3) feature array, in original physical units
# ---------------------------------------------------------------------------

def apply_generic(name: str, magnitude: float, points: np.ndarray, ctx: dict,
                  rng: np.random.Generator) -> np.ndarray:
    """One generic corruption. `subsample` is handled by the caller's target count.

    x and y carry a feature scale of 1, so a magnitude in these coordinates is a
    magnitude in pixels; t is divided by the temporal scale, so it is converted back
    to microseconds before any timing modifier and converted forward again after.
    """
    out = points.copy()
    if name == "subsample":
        return out                                   # size is set by the caller
    if name == "spatial_offset_x":
        out[:, 0] += magnitude
        return out
    if name == "spatial_offset_xy":
        out[:, 0] += magnitude
        out[:, 1] += magnitude
        return out
    if name == "scaling":
        cx, cy = ctx["centre"]
        out[:, 0] = cx + magnitude * (out[:, 0] - cx)
        out[:, 1] = cy + magnitude * (out[:, 1] - cy)
        return out
    if name == "temporal_clump_uniform":
        if magnitude <= 0:
            return out
        scale = ctx["t_scale"]
        micros = out[:, 2] * scale
        out[:, 2] = np.round(micros / magnitude) * magnitude / scale
        return out
    if name == "uniform_noise":
        if magnitude <= 0:
            return out
        n = int(round(magnitude * len(points)))
        if n <= 0:
            return out
        noise = np.empty((n, 3), dtype=out.dtype)
        noise[:, 0] = rng.uniform(0, ctx["width"], n)
        noise[:, 1] = rng.uniform(0, ctx["height"], n)
        noise[:, 2] = rng.uniform(0, ctx["window_us"], n) / ctx["t_scale"]
        return np.concatenate([out, noise])
    raise KeyError(f"unknown modifier {name}")


def is_identity(name: str, magnitude: float) -> bool:
    if name in ("spatial_offset_x", "spatial_offset_xy", "uniform_noise",
                "temporal_clump_uniform"):
        return magnitude == 0
    if name in ("scaling", "subsample"):
        return magnitude == 1.0
    return False


def clump_to_v2e_grid(points: np.ndarray, origin_us: int, ctx: dict) -> np.ndarray:
    """Snap each timestamp to the nearest timestamp v2e actually emitted.

    v2e's grid is irregular - gaps run from 22 to 819 us - so quantising to a uniform
    step is a different corruption from the one v2e applies.
    """
    grid = ctx["v2e_times"]
    scale = ctx["t_scale"]
    out = points.copy()
    absolute = out[:, 2] * scale + origin_us
    idx = np.searchsorted(grid, absolute)
    idx = np.clip(idx, 1, len(grid) - 1)
    left, right = grid[idx - 1], grid[idx]
    nearest = np.where(np.abs(absolute - left) <= np.abs(right - absolute), left, right)
    out[:, 2] = (nearest - origin_us) / scale
    return out


def add_noise_to_count(points: np.ndarray, target: int, ctx: dict,
                       rng: np.random.Generator) -> np.ndarray:
    """Pad with uniform noise events until the cloud holds `target` points."""
    n = int(target) - len(points)
    if n <= 0:
        return points.copy()
    noise = np.empty((n, 3), dtype=points.dtype)
    noise[:, 0] = rng.uniform(0, ctx["width"], n)
    noise[:, 1] = rng.uniform(0, ctx["height"], n)
    noise[:, 2] = rng.uniform(0, ctx["window_us"], n) / ctx["t_scale"]
    return np.concatenate([points, noise])


def load_floor(folder, window_us: int) -> dict:
    if not folder:
        return {}
    path = Path(folder) / "summary.csv"
    if not path.exists():
        print(f"floor: {path} not found, figures drawn without the chance band")
        return {}
    table = pd.read_csv(path)
    table = table[table.window_length_us == window_us]
    if table.empty:
        return {}
    grouped = table.groupby("metric")[["median_distance", "q95_distance"]].mean()
    return {k: (float(v["median_distance"]), float(v["q95_distance"]))
            for k, v in grouped.iterrows()}


# ---------------------------------------------------------------------------
# 3a
# ---------------------------------------------------------------------------

def run_3a(args, trial, real, metrics, ctx, count, writer):
    feature_scales = ctx["feature_scales"]
    plan = [(name, m) for name in args.modifiers for m in GENERIC_SWEEPS[name]]
    print(f"{len(plan)} modifier/magnitude pairs x {len(args.seeds)} seeds "
          f"x {args.periods} rotations\n")

    for period in range(args.first_period, args.first_period + args.periods):
        begin, end = trial.period_bounds(period, real.t_start)
        if end > real.t_end:
            print(f"rotation {period}: past the end of the recording, stopping")
            break
        starts = np.arange(begin, end, args.window_us)
        chosen = select_indices(len(starts), args.max_windows)
        print(f"rotation {period}: {len(chosen)} of {len(starts)} windows")

        cached = {}
        for index in chosen:
            index = int(index)
            start = int(starts[index])
            cached[index] = to_points(real.slice(start, start + args.window_us),
                                      start, feature_scales)

        for name, magnitude in plan:
            for seed in args.seeds:
                for index in chosen:
                    index = int(index)
                    points = cached[index]
                    if len(points) == 0:
                        continue
                    base_seed = seed * 1_000_003 + index
                    side_a = subsample(points, count, np.random.default_rng(base_seed))

                    mod_rng = np.random.default_rng(
                        stable_seed(name, float(magnitude), seed, index))
                    modified = apply_generic(name, magnitude, points, ctx, mod_rng)
                    # The subsample sweep is the one place sizes must differ: it asks
                    # whether a metric responds to event count, and normalising both
                    # sides would erase exactly what it is testing.
                    if name == "subsample":
                        # No cap means the reference size is the window's own count.
                        reference = count if count is not None else len(points)
                        target = int(round(reference * magnitude))
                    else:
                        target = count
                    side_b = subsample(modified, target, np.random.default_rng(base_seed))

                    for key, (metric, settings) in metrics.items():
                        if writer.already_done(name, magnitude, seed, period, index, key):
                            continue
                        value, squared = measure(metric, settings, side_a, side_b)
                        writer.append({
                            "trial": trial.name, "phase": "3a", "modifier": name,
                            "magnitude": magnitude, "seed": seed, "metric": key,
                            "period_index": period, "window_index": index,
                            "window_start_us": start,
                            "n_events_a": len(side_a), "n_events_b": len(side_b),
                            "distance": value, "distance_squared": squared,
                        })
                writer.flush()
            print(f"   {name} = {magnitude}: done")


# ---------------------------------------------------------------------------
# 3b
# ---------------------------------------------------------------------------

def run_3b(args, trial, real, v2e, metrics, ctx, count, writer):
    """Directed modifiers, real against v2e.

    Run twice over. `normalised` holds both sides at EVENTS_PER_COMPARISON, which is
    the protocol's rule everywhere else. `natural` leaves the window counts as they
    fall, because match_count is about the count difference and normalising makes it
    a no-op. Both are recorded; neither is merged into the other.
    """
    feature_scales = ctx["feature_scales"]

    for period in range(args.first_period, args.first_period + args.periods):
        begin, end = trial.period_bounds(period, real.t_start)
        if end > real.t_end or end > v2e.t_end:
            print(f"rotation {period}: past the end of a recording, stopping")
            break
        starts = np.arange(begin, end, args.window_us)
        chosen = select_indices(len(starts), args.max_windows)
        print(f"rotation {period}: {len(chosen)} of {len(starts)} windows")

        for name in args.modifiers:
            for sizing in ("normalised", "natural"):
                for seed in args.seeds:
                    for index in chosen:
                        index = int(index)
                        start = int(starts[index])
                        real_pts = to_points(real.slice(start, start + args.window_us),
                                             start, feature_scales)
                        v2e_pts = to_points(v2e.slice(start, start + args.window_us),
                                            start, feature_scales)
                        if len(real_pts) == 0 or len(v2e_pts) == 0:
                            continue
                        base_seed = seed * 1_000_003 + index
                        mod_rng = np.random.default_rng(
                            stable_seed(name, sizing, seed, index))

                        side_a, side_b = real_pts, v2e_pts
                        if name in ("clump_to_v2e_grid", "clump_and_match"):
                            side_a = clump_to_v2e_grid(side_a, start, ctx)
                        if name in ("match_count_v2e_down", "clump_and_match"):
                            side_b = subsample(side_b, len(real_pts), mod_rng)
                        if name == "match_count_real_up":
                            side_a = add_noise_to_count(side_a, len(v2e_pts), ctx, mod_rng)

                        if sizing == "normalised":
                            side_a = subsample(side_a, count,
                                               np.random.default_rng(base_seed))
                            side_b = subsample(side_b, count,
                                               np.random.default_rng(base_seed + 1))

                        for key, (metric, settings) in metrics.items():
                            if writer.already_done(name, sizing, seed, period, index, key):
                                continue
                            value, squared = measure(metric, settings, side_a, side_b)
                            writer.append({
                                "trial": trial.name, "phase": "3b", "modifier": name,
                                "sizing": sizing, "magnitude": 1.0, "seed": seed,
                                "metric": key, "period_index": period,
                                "window_index": index, "window_start_us": start,
                                "n_events_real": len(real_pts),
                                "n_events_v2e": len(v2e_pts),
                                "n_events_a": len(side_a), "n_events_b": len(side_b),
                                "distance": value, "distance_squared": squared,
                            })
                    writer.flush()
                print(f"   {name} [{sizing}]: done")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config()
    feature_scales = config["feature_scales"]
    trial = Trial.load(args.trials_dir, args.trial)
    sources = ("real",) if args.phase == "3a" else ("real", "v2e")
    streams = load_streams(trial, sources=sources)
    real = streams["real"]
    v2e = streams.get("v2e")
    metrics = build_metrics(args.metrics, config)
    count = events_per_comparison(args.window_us, args.events_per_comparison)

    ctx = {
        "feature_scales": feature_scales,
        "t_scale": float(feature_scales.get("t", 1)),
        "centre": trial.ellipse_centre,
        "width": float(config["sensor"]["width"]),
        "height": float(config["sensor"]["height"]),
        "window_us": float(args.window_us),
        "v2e_times": v2e.t if v2e is not None else None,
    }

    output = RunOutput("test3_modifier", trial,
                       tag=("smoke" if args.smoke else f"{args.phase}_w{args.window_us}us"),
                       existing=args.resume)
    parameters = dict(vars(args))
    parameters.update(events_per_comparison=count,
                      sweeps={k: GENERIC_SWEEPS[k] for k in args.modifiers
                              if k in GENERIC_SWEEPS},
                      rotation_period_us=trial.rotation_period_us,
                      unmodified_side="the same window, so magnitude zero reads zero",
                      subsample_not_normalised=True)
    output.write_config(parameters, args.metrics, config)

    print(f"trial {trial.name}   rotation {trial.rotation_period_us:,} us "
          f"({trial.rotation_period_source})")
    print(f"phase {args.phase}, window {args.window_us} us, {count} events per side")
    print(f"modifiers: {', '.join(args.modifiers)}")
    print(f"metrics: {', '.join(args.metrics)}")
    print(f"output: {output.directory}\n")

    keys_3a = ("modifier", "magnitude", "seed", "period_index", "window_index", "metric")
    keys_3b = ("modifier", "sizing", "seed", "period_index", "window_index", "metric")
    writer = ResultWriter(output.csv("results.csv"),
                          key_fields=keys_3a if args.phase == "3a" else keys_3b,
                          resume=bool(args.resume))
    if args.resume:
        print(f"resuming: {writer.n_rows:,} rows already on disk\n")

    if args.phase == "3a":
        run_3a(args, trial, real, metrics, ctx, count, writer)
    else:
        run_3b(args, trial, real, v2e, metrics, ctx, count, writer)

    results = writer.frame()
    if results.empty:
        print("no rows produced")
        return

    group = (["modifier", "magnitude", "metric"] if args.phase == "3a"
             else ["modifier", "sizing", "metric"])
    summary = (results.groupby(group)
               .agg(mean_distance=("distance", "mean"),
                    sum_distance=("distance", "sum"),
                    sd_distance=("distance", "std"),
                    median_distance=("distance", "median"),
                    mean_distance_squared=("distance_squared", "mean"),
                    mean_n_events_a=("n_events_a", "mean"),
                    mean_n_events_b=("n_events_b", "mean"),
                    n_comparisons=("distance", "count"))
               .reset_index())
    summary.to_csv(output.csv("summary.csv"), index=False)
    print(f"\nwrote results.csv ({len(results):,} rows) and summary.csv")

    floor = load_floor(args.floor_from, args.window_us)
    if args.phase == "3a":
        zero = summary[summary.apply(
            lambda r: is_identity(r.modifier, r.magnitude), axis=1)]
        print("\nidentity magnitudes (must be essentially zero):")
        print(zero[["modifier", "magnitude", "metric", "mean_distance"]]
              .to_string(index=False))
        figures_3a(summary, args, output, floor)
        draw_before_after(real, trial, args, ctx, output)
    else:
        figures_3b(summary, args, output, floor)
    print(f"figures in {output.figures}")

    real.close()
    if v2e is not None:
        v2e.close()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures_3a(summary, args, output, floor):
    keys = [k for k in args.metrics]

    for name in args.modifiers:
        block = summary[summary.modifier == name]
        if block.empty:
            continue
        figure = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                               vertical_spacing=0.035, subplot_titles=keys)
        for row, key in enumerate(keys, start=1):
            line = block[block.metric == key].sort_values("magnitude")
            figure.add_trace(go.Scatter(
                x=line.magnitude, y=line.mean_distance, mode="lines+markers",
                line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2),
                marker=dict(size=7), showlegend=False,
                error_y=dict(type="data", array=line.sd_distance, visible=True,
                             thickness=0.8, width=0)), row=row, col=1)
            if key in floor:
                low, high = floor[key]
                figure.add_hrect(y0=low, y1=high, row=row, col=1, fillcolor="#9aa0a6",
                                 opacity=0.25, line_width=0,
                                 annotation_text="test 1 chance band" if row == 1 else None,
                                 annotation_position="top left")
            figure.update_yaxes(title_text="distance", row=row, col=1)
        figure.update_xaxes(title_text=f"{name} magnitude", row=len(keys), col=1)
        figure.update_layout(title=f"Modifier {name} &mdash; {args.trial}, "
                                   f"{args.window_us} us windows",
                             template="plotly_white", height=240 * len(keys),
                             showlegend=False, margin=dict(l=70, r=30, t=80, b=60))
        figure.write_html(str(output.figure(f"modifier_curve_{name}.html")),
                          include_plotlyjs="cdn")

    # The main figure: modifier by metric, small multiples.
    mods = [m for m in args.modifiers if not summary[summary.modifier == m].empty]
    grid = make_subplots(rows=len(mods), cols=len(keys), shared_xaxes=False,
                         vertical_spacing=0.05, horizontal_spacing=0.04,
                         column_titles=keys, row_titles=mods)
    for r, name in enumerate(mods, start=1):
        for c, key in enumerate(keys, start=1):
            line = summary[(summary.modifier == name) &
                           (summary.metric == key)].sort_values("magnitude")
            grid.add_trace(go.Scatter(
                x=line.magnitude, y=line.mean_distance, mode="lines+markers",
                line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2),
                marker=dict(size=5), showlegend=False), row=r, col=c)
            if key in floor:
                low, high = floor[key]
                grid.add_hrect(y0=low, y1=high, row=r, col=c, fillcolor="#9aa0a6",
                               opacity=0.22, line_width=0)
    grid.update_layout(
        title=f"Selectivity: modifier against metric &mdash; {args.trial}, "
              f"{args.window_us} us windows (grey band = test 1 chance level)",
        template="plotly_white", height=230 * len(mods), width=300 * len(keys),
        margin=dict(l=80, r=120, t=90, b=60))
    grid.write_html(str(output.figure("selectivity_grid.html")), include_plotlyjs="cdn")

    block = summary[summary.modifier == "subsample"]
    if not block.empty:
        figure = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                               vertical_spacing=0.035, subplot_titles=keys)
        for row, key in enumerate(keys, start=1):
            line = block[block.metric == key].sort_values("magnitude")
            figure.add_trace(go.Scatter(
                x=line.magnitude, y=line.mean_distance, mode="lines+markers",
                line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2),
                marker=dict(size=7), showlegend=False,
                error_y=dict(type="data", array=line.sd_distance, visible=True,
                             thickness=0.8, width=0)), row=row, col=1)
            figure.update_yaxes(title_text="distance", row=row, col=1)
        figure.update_xaxes(title_text="fraction of events retained on one side",
                            row=len(keys), col=1)
        figure.update_layout(
            title="Subsample check &mdash; one side reduced, the other left whole",
            template="plotly_white", height=240 * len(keys), showlegend=False,
            margin=dict(l=70, r=30, t=80, b=60))
        figure.write_html(str(output.figure("subsample_check.html")),
                          include_plotlyjs="cdn")


def figures_3b(summary, args, output, floor):
    for sizing in sorted(summary.sizing.unique()):
        block = summary[summary.sizing == sizing]
        keys = [k for k in args.metrics]
        figure = make_subplots(rows=1, cols=len(keys), subplot_titles=keys)
        for c, key in enumerate(keys, start=1):
            line = block[block.metric == key]
            order = [m for m in DIRECTED if m in set(line.modifier)]
            line = line.set_index("modifier").loc[order].reset_index()
            figure.add_trace(go.Bar(
                x=line.modifier, y=line.mean_distance, showlegend=False,
                marker_color=METRIC_COLOURS.get(key, MUTED),
                error_y=dict(type="data", array=line.sd_distance, visible=True)),
                row=1, col=c)
            figure.update_yaxes(title_text="distance" if c == 1 else None, row=1, col=c)
        figure.update_layout(
            title=f"Directed modifiers, real against v2e &mdash; {args.trial}, "
                  f"{sizing} sizing, {args.window_us} us windows",
            template="plotly_white", height=460, width=320 * len(keys),
            margin=dict(l=70, r=30, t=90, b=140))
        figure.update_xaxes(tickangle=-35)
        figure.write_html(str(output.figure(f"directed_bars_{sizing}.html")),
                          include_plotlyjs="cdn")
    # The spec's filename, pointing at the protocol's own sizing rule.
    primary = output.figure("directed_bars_normalised.html")
    if primary.exists():
        output.figure("directed_bars.html").write_text(
            primary.read_text(encoding="utf-8"), encoding="utf-8")


def draw_before_after(real, trial, args, ctx, output):
    """One window before and after, so the corruption is visible."""
    wanted = [("spatial_offset_xy", 8.0), ("uniform_noise", 0.5),
              ("temporal_clump_uniform", 250.0), ("scaling", 1.10)]
    begin, _ = trial.period_bounds(args.first_period, real.t_start)
    start = int(begin + 20 * args.window_us)
    points = to_points(real.slice(start, start + args.window_us), start,
                       ctx["feature_scales"])
    if len(points) == 0:
        return
    step = max(1, len(points) // PLOT_MAX_POINTS)
    for name, magnitude in wanted:
        if name not in args.modifiers:
            continue
        modified = apply_generic(name, magnitude, points, ctx,
                                 np.random.default_rng(0))
        figure = go.Figure()
        for cloud, label, colour in ((points, "before", "#2a78d6"),
                                     (modified, f"after {name} = {magnitude}", "#eb6834")):
            shown = cloud[::step]
            figure.add_trace(go.Scatter3d(
                x=shown[:, 0], y=shown[:, 1], z=shown[:, 2] * ctx["t_scale"] / 1000.0,
                mode="markers", name=f"{label} ({len(cloud):,} events)",
                marker=dict(size=1.6, color=colour, opacity=0.6), hoverinfo="skip"))
        figure.update_layout(
            title=f"{name} = {magnitude} &mdash; {args.window_us} us window at "
                  f"{start:,} us",
            scene=dict(xaxis_title="x (px)", yaxis_title="y (px)",
                       zaxis_title="t within window (ms)"),
            template="plotly_white", height=620, margin=dict(l=0, r=0, t=50, b=0))
        figure.write_html(
            str(output.figure(f"events_3d_{name}_{magnitude}.html")),
            include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
