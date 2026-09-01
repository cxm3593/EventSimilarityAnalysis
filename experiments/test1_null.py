"""Test 1 - null: what does each metric read when there is no difference?

Takes whole rotations, splits each one's events randomly into two halves, and compares
half against half window by window. Both halves come from the same instants, so any
distance is sampling chance alone. That sets the resolution limit: below it, nothing is
measurable at all.

    python experiments/test1_null.py --smoke          # a couple of minutes
    python experiments/test1_null.py                  # the full run
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import (METRIC_KEYS, ResultWriter, RunOutput, Trial, build_metrics,
                    cut_windows, events_per_comparison, load_config, load_streams,
                    measure, select_indices, split_in_half, subsample)

TRIALS_DIR = r"C:/Users/cxm3593/Academic/Workspace/EventCamCalib/output/trials"

INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
METRIC_COLOURS = {"mmd_rbf03": "#2a78d6", "mmd_rbf15": "#eb6834", "mmd_rbf75": "#1baf7a",
                  "swd": "#eda100", "chamfer": "#4a3aa7"}
PLOT_MAX_POINTS = 20000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="optical_chopper_data_f1")
    parser.add_argument("--trials-dir", default=TRIALS_DIR)
    parser.add_argument("--periods", type=int, default=3,
                        help="how many whole rotations to use; at least 2")
    parser.add_argument("--first-period", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="one random half-split per seed")
    parser.add_argument("--windows-us", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap the windows compared per rotation; for quick runs")
    parser.add_argument("--events-per-comparison", default="none",
                        help="events each side is cut to: 'none' (no cap, default), "
                             "'auto' (the old median-based table), or an integer")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration to check the run end to end")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None,
                        help="continue an interrupted run, reusing its folder and config")
    args = parser.parse_args()
    chosen_metrics = args.metrics          # None unless given explicitly
    if args.smoke:
        args.periods, args.seeds = 2, [0]
        args.windows_us, args.max_windows = [1000, 5000], 25
        args.metrics = chosen_metrics or ["mmd_rbf15", "swd"]
    else:
        args.metrics = chosen_metrics or list(METRIC_KEYS)
    if args.periods < 2:
        parser.error("--periods must be at least 2")
    return args


def event_cloud(events, title, path):
    """3D view of one set of events, subsampled so the file stays openable."""
    step = max(1, len(events) // PLOT_MAX_POINTS)
    shown = events[::step]
    figure = go.Figure(go.Scatter3d(
        x=shown["x"], y=shown["y"], z=shown["t"] / 1000.0, mode="markers",
        marker=dict(size=1.2, color=shown["t"] / 1000.0, colorscale="Viridis",
                    showscale=False), hoverinfo="skip"))
    figure.update_layout(
        title=f"{title} &mdash; {len(events):,} events, {len(shown):,} drawn",
        scene=dict(xaxis_title="x (px)", yaxis_title="y (px)", zaxis_title="t (ms)"),
        template="plotly_white", height=620, margin=dict(l=0, r=0, t=50, b=0))
    figure.write_html(str(path), include_plotlyjs="cdn")


def main():
    args = parse_args()
    config = load_config()
    feature_scales = config["feature_scales"]
    trial = Trial.load(args.trials_dir, args.trial)
    streams = load_streams(trial, sources=("real",))
    real = streams["real"]
    metrics = build_metrics(args.metrics, config)

    output = RunOutput("test1_null", trial, tag="smoke" if args.smoke else "",
                       existing=args.resume)
    output.write_config(vars(args), args.metrics, config)

    print(f"trial {trial.name}   rotation {trial.rotation_period_us:,} us "
          f"({trial.rotation_period_source})")
    print(f"{args.periods} rotations from index {args.first_period}, "
          f"seeds {args.seeds}, windows {args.windows_us} us")
    print(f"metrics: {', '.join(args.metrics)}")
    print(f"output: {output.directory}\n")

    # Rows reach disk as they are produced: a run that dies after two hours must
    # leave its results behind rather than lose them.
    writer = ResultWriter(output.csv("results.csv"),
                          key_fields=("period_index", "split_seed", "window_length_us",
                                      "metric", "window_index"),
                          resume=bool(args.resume))
    if args.resume:
        print(f"resuming: {writer.n_rows:,} rows already on disk")

    for period in range(args.first_period, args.first_period + args.periods):
        begin, end = trial.period_bounds(period, real.t_start)
        if end > real.t_end:
            print(f"rotation {period}: past the end of the recording, stopping")
            break
        events = real.slice(begin, end)
        print(f"rotation {period}: {begin:,} to {end:,} us, {len(events):,} events")
        event_cloud(events, f"rotation {period}, all events",
                    output.figure(f"events_3d_period{period}_full.html"))

        for seed in args.seeds:
            if seed == args.seeds[0]:
                first, second = split_in_half(events, seed)
                event_cloud(first, f"rotation {period}, seed {seed}, half A",
                            output.figure(f"events_3d_period{period}_seed{seed}_halfA.html"))
                event_cloud(second, f"rotation {period}, seed {seed}, half B",
                            output.figure(f"events_3d_period{period}_seed{seed}_halfB.html"))

            for window_us in args.windows_us:
                count = events_per_comparison(window_us, args.events_per_comparison)
                windows, starts = cut_windows(events, begin, end, window_us,
                                              feature_scales)
                chosen = select_indices(len(windows), args.max_windows)
                # The null is defined per window: divide that window's events into
                # two exactly equal random samples. Splitting the whole rotation first
                # only balances the rotation globally, leaving unequal A/B counts in
                # individual comparisons.
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, period, window_us])
                )

                for index in chosen:
                    index = int(index)
                    a, b = split_in_half(windows[index], rng)
                    a = subsample(a, count, rng)
                    b = subsample(b, count, rng)
                    for key, (metric, settings) in metrics.items():
                        if writer.already_done(period, seed, window_us, key, index):
                            continue
                        value, squared = measure(metric, settings, a, b)
                        writer.append({
                            "trial": trial.name, "period_index": period,
                            "split_seed": seed, "window_length_us": window_us,
                            "metric": key, "window_index": index,
                            "window_start_us": int(starts[index]),
                            "n_events_a": len(a), "n_events_b": len(b),
                            "distance": value, "distance_squared": squared,
                        })
                writer.flush()
                print(f"   seed {seed}, {window_us:>6} us windows: {len(chosen)} compared")

    results = writer.frame()

    # Quantiles as well as mean and spread: the unbiased MMD estimator clamps negative
    # values to zero, so the null has an atom at zero and mean +/- spread misleads.
    summary = (results.groupby(["period_index", "split_seed", "window_length_us", "metric"])
               ["distance"].agg(mean_distance="mean", sum_distance="sum",
                                sd_distance="std",
                                median_distance="median",
                                q95_distance=lambda s: s.quantile(0.95),
                                zero_fraction=lambda s: float((s == 0).mean()),
                                n_windows="count").reset_index())
    summary.to_csv(output.csv("summary.csv"), index=False)

    print(f"\nwrote results.csv ({len(results):,} rows) and summary.csv")
    print("\nresolution limit, mean over rotations and seeds:")
    table = (summary.groupby(["window_length_us", "metric"])
             [["mean_distance", "sd_distance", "median_distance",
               "q95_distance", "zero_fraction"]].mean().reset_index())
    print(table.to_string(index=False))

    make_figures(results, table, args, output)
    real.close()


def make_figures(results, table, args, output):
    for window_us in args.windows_us:
        subset = results[results.window_length_us == window_us]
        if subset.empty:
            continue
        # x is the window's absolute start time, not its index within the rotation, so
        # successive rotations lay out along one continuous axis instead of overlapping.
        figure = go.Figure()
        for key in args.metrics:
            block = subset[subset.metric == key]
            for (period, seed), run in block.groupby(["period_index", "split_seed"]):
                run = run.sort_values("window_start_us")
                figure.add_trace(go.Scatter(
                    x=run.window_start_us, y=run.distance, mode="lines",
                    line=dict(color=METRIC_COLOURS.get(key, MUTED), width=1),
                    opacity=0.3, showlegend=False,
                    name=f"{key} p{period} s{seed}"))
            first_period = int(block["period_index"].min())
            for period, period_block in block.groupby("period_index"):
                mean_curve = (period_block.groupby("window_start_us")["distance"]
                              .mean().sort_index())
                figure.add_trace(go.Scatter(
                    x=mean_curve.index, y=mean_curve.values, mode="lines", name=key,
                    showlegend=(int(period) == first_period),
                    line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2.5)))
        figure.update_layout(
            title=f"Chance-level distance, {window_us} us windows &mdash; {args.trial}",
            xaxis_title="window start (us)", yaxis_title="distance",
            template="plotly_white", height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        figure.write_html(str(output.figure(f"null_curve_w{window_us}us.html")),
                          include_plotlyjs="cdn")

    figure = go.Figure()
    for key in args.metrics:
        block = table[table.metric == key].sort_values("window_length_us")
        figure.add_trace(go.Scatter(
            x=block.window_length_us, y=block.mean_distance, mode="lines+markers",
            name=key, line=dict(color=METRIC_COLOURS.get(key, MUTED), width=2),
            marker=dict(size=9),
            error_y=dict(type="data", array=block.sd_distance, visible=True)))
    figure.update_layout(
        title=f"Resolution limit against window length &mdash; {args.trial}",
        xaxis_title="window length (us)", yaxis_title="chance-level distance",
        xaxis_type="log", template="plotly_white", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    figure.write_html(str(output.figure("null_summary.html")), include_plotlyjs="cdn")
    print(f"figures in {output.figures}")


if __name__ == "__main__":
    main()
