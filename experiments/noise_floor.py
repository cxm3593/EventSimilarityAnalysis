"""Noise floor from every neighbouring pair of chopper turns.

Each complete window of a turn is compared with the window at the same wheel position one
measured rotation later, for every turn of the recording except the last, which has no
neighbour. This is the 360-degree point of the temporal offset test, extended from the
first turn to the whole recording. Turns further apart are never compared: speed drift
makes them differ more, and that is not the floor.

The whole pathway is in this file, in order: plan, compare, check, summarise, record.
Windowing, the one-turn-later read and the metrics are imported unchanged from the
perturbation study, so every value is computed exactly as it is there.

    python experiments/noise_floor.py --plan-only
    python experiments/noise_floor.py
    python experiments/noise_floor.py --run-dir output/noise_floor/<run>         # continue
    python experiments/noise_floor.py --summarize-only output/noise_floor/<run>
"""

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

if __package__:
    from .common import (METRIC_KEYS, MIN_EVENTS, PROJECT_ROOT, ResultWriter, Stream, Trial,
                         build_metrics, load_config)
    from .perturbation_methods import TemporalOffset
    from .perturbation_shared import evaluate_window, period_windows, prepare_window
else:
    from common import (METRIC_KEYS, MIN_EVENTS, PROJECT_ROOT, ResultWriter, Stream, Trial,
                        build_metrics, load_config)
    from perturbation_methods import TemporalOffset
    from perturbation_shared import evaluate_window, period_windows, prepare_window

TRIALS_DIR = r"C:/Users/cxm3593/Academic/Workspace/EventCamCalib/output/trials"
TRIAL_NAMES = [f"optical_chopper_data_f{i}" for i in range(1, 6)]
OUTPUT_DIR = PROJECT_ROOT / "output" / "noise_floor"
ONE_TURN_DEGREES = 360.0
# The stored 360-degree point of the temporal offset test. Its windows are this run's first
# pair of turns, so the two must agree; see check_against_stored().
STORED_360 = PROJECT_ROOT / "output" / "perturbation" / "temporal_20260923_175125_279651"
WINDOW_KEYS = ["period_index", "window_index"]
IMPORTED_SOURCES = ["experiments/common.py", "experiments/perturbation_shared.py",
                    "experiments/perturbation_methods.py"]


def say(message: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


# --- 1. Plan ---

def neighbouring_pairs(trial: Trial, real) -> int:
    """How many turns have a complete next turn inside the recording."""
    full_turns = int((real.t_end - real.t_start) // trial.rotation_period_us)
    return max(full_turns - 1, 0)


def plan_windows(trial: Trial, real, window_us: int, max_pairs: int | None = None) -> pd.DataFrame:
    """Every complete window of every turn that has a neighbour; period_index is the turn."""
    pairs = neighbouring_pairs(trial, real)
    if max_pairs is not None:
        pairs = min(pairs, max_pairs)
    if pairs < 1:
        raise ValueError(f"{trial.name}: the recording does not hold two complete turns")
    windows = pd.concat([period_windows(trial, real, turn, window_us) for turn in range(pairs)],
                        ignore_index=True)
    if windows.empty:
        raise ValueError(f"{trial.name}: no complete windows; reduce temporal_window")
    shift = TemporalOffset.shift_us(ONE_TURN_DEGREES, trial.rotation_period_us)
    if windows.window_end_us.max() + shift > real.t_end:
        raise ValueError(f"{trial.name}: the last comparison window leaves the recording")
    return windows


# --- 2. Compare ---

class Progress:
    """The update() that evaluate_window expects, reported to the log about once a minute."""

    def __init__(self, total: int, done: int, label: str, every_s: float = 60):
        self.total, self.done, self.label, self.every_s = total, done, label, every_s
        self.started = self.last = time.time()
        self.done_at_start = done

    def update(self, count: int):
        self.done += count
        now = time.time()
        if now - self.last >= self.every_s or self.done >= self.total:
            rate = (self.done - self.done_at_start) / max(now - self.started, 1e-9)
            remaining = (self.total - self.done) / rate / 60 if rate > 0 else float("nan")
            say(f"{self.label}: {self.done:,}/{self.total:,} metric values "
                f"({100 * self.done / self.total:.1f}%), about {remaining:.0f} min left")
            self.last = now


def completed_windows(path: Path, n_metrics: int) -> set:
    """Windows already measured with every metric. Partly written windows are dropped."""
    if not path.exists():
        return set()
    try:
        saved = pd.read_csv(path)
    except Exception:  # noqa: BLE001 - an unreadable partial file starts over
        path.unlink()
        return set()
    counts = saved.groupby(WINDOW_KEYS).metric.nunique()
    done = set(counts[counts == n_metrics].index)
    keep = saved.set_index(WINDOW_KEYS).index.isin(done)
    if not keep.all():
        saved[keep].to_csv(path, index=False)
    return done


def compare(trial: Trial, real, windows: pd.DataFrame, config: dict, metrics: dict, seed: int,
            results_path: Path) -> pd.DataFrame:
    """Measure each window against the window one rotation later; rows reach disk at once.

    evaluate_window with TemporalOffset at 360 degrees is the same call the temporal offset
    test makes, so the read, the separate re-zeroing and the metrics are identical.
    """
    done = completed_windows(results_path, len(metrics))
    if done:
        say(f"{trial.name}: continuing, {len(done):,} of {len(windows):,} windows already saved")
    progress = Progress(len(windows) * len(metrics), len(done) * len(metrics), trial.name)
    with ResultWriter(results_path, resume=True) as writer:
        for row in windows.itertuples():
            if (row.period_index, row.window_index) in done:
                continue
            window = prepare_window(row, trial, real, config)
            evaluate_window(TemporalOffset, window, [ONE_TURN_DEGREES], seed, metrics, writer,
                            progress)
    return writer.frame()


# --- 3. Check ---

def check_recording(results: pd.DataFrame, windows: pd.DataFrame, trial: Trial,
                    n_metrics: int) -> dict:
    """Integrity of one recording's results. Every entry must be True."""
    valid = results[results.status == "ok"]
    mmd = valid[valid.metric.str.startswith("mmd")]
    period = int(trial.rotation_period_us)
    return {
        "expected_comparisons": len(results) == len(windows) * n_metrics,
        "no_duplicates": not bool(results.duplicated([*WINDOW_KEYS, "metric"]).any()),
        "known_statuses": bool(results.status.isin(["ok", "insufficient_events"]).all()),
        "finite_distances": bool(np.isfinite(valid.distance).all()),
        "signed_mmd_matches_distance": bool(np.allclose(
            mmd.distance, np.sqrt(np.maximum(mmd.distance_squared, 0)), atol=1e-12)),
        "one_rotation_later": bool(
            (results.temporal_offset_us == period).all()
            and (results.comparison_start_us == results.window_start_us + period).all()),
    }


def check_against_stored(results: pd.DataFrame, trial_name: str, stored_root: Path = STORED_360) -> dict:
    """The first pair of turns uses the windows of the stored 360-degree result."""
    path = stored_root / trial_name / "temporal_offset" / "results.csv"
    if not path.exists():
        return {"status": "skipped", "reason": f"not found: {path}"}
    stored = pd.read_csv(path)
    stored = stored[(stored.magnitude == ONE_TURN_DEGREES) & (stored.period_index == 0)]
    first = results[results.period_index == 0]
    merged = first.merge(stored, on=["window_index", "metric"], suffixes=("", "_stored"))
    if len(first) == 0 or len(merged) != len(first) or len(stored) != len(first):
        return {"status": "MISMATCH", "reason": "different windows",
                "windows_here": int(first.window_index.nunique()),
                "windows_stored": int(stored.window_index.nunique())}
    same_windows = bool((merged.window_start_us == merged.window_start_us_stored).all())
    close = np.isclose(merged.distance, merged.distance_stored, rtol=1e-6, atol=1e-9, equal_nan=True)
    difference = np.abs(merged.distance - merged.distance_stored)
    return {"status": "match" if same_windows and close.all() else "MISMATCH",
            "comparisons": int(len(merged)),
            "max_abs_difference": float(np.nanmax(difference)) if len(merged) else None}


# --- 4. Summarise ---

def statistics(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median with the 5% and 95% points, and mean with standard deviation, per group.

    Both are saved so they can be compared. On a noise floor about half of the MMD distances
    are clamped to exactly zero, which distorts the mean of the distance, so the same
    statistics of the signed squared estimate are saved beside them.
    """
    values = results.copy()
    values.loc[values.status != "ok", ["distance", "distance_squared"]] = np.nan

    def quantile(level):
        return lambda series: series.quantile(level)

    table = (values.groupby(keys, sort=False)
             .agg(n_comparisons=("distance", "size"), n_valid=("distance", "count"),
                  median_distance=("distance", "median"),
                  q05_distance=("distance", quantile(0.05)),
                  q95_distance=("distance", quantile(0.95)),
                  mean_distance=("distance", "mean"), std_distance=("distance", "std"),
                  share_exactly_zero=("distance", lambda s: float((s == 0).sum() / max(s.count(), 1))),
                  median_distance_squared=("distance_squared", "median"),
                  q05_distance_squared=("distance_squared", quantile(0.05)),
                  q95_distance_squared=("distance_squared", quantile(0.95)),
                  mean_distance_squared=("distance_squared", "mean"),
                  std_distance_squared=("distance_squared", "std"),
                  mean_n_events_a=("n_events_a", "mean"), mean_n_events_b=("n_events_b", "mean"))
             .reset_index())
    table["mean_minus_median"] = table.mean_distance - table.median_distance
    return table


MEDIAN_COLOUR, MEAN_COLOUR, RANGE_COLOUR = "#2a78d6", "#eb6834", "rgba(137,135,129,0.22)"


def plot_floor(summary: pd.DataFrame, per_pair: pd.DataFrame, path: Path, squared: bool = False):
    """Mean and median for each pair of turns along every recording, one row per metric."""
    suffix = "_distance_squared" if squared else "_distance"
    metrics = [m for m in METRIC_KEYS if m.startswith("mmd")] if squared else list(METRIC_KEYS)
    trials = list(dict.fromkeys(summary.trial))
    figure = make_subplots(rows=len(metrics), cols=len(trials), shared_xaxes=False,
                           column_titles=[name.rsplit("_", 1)[-1].upper() for name in trials],
                           row_titles=metrics, horizontal_spacing=0.045, vertical_spacing=0.06)
    for r, metric in enumerate(metrics, start=1):
        for c, trial in enumerate(trials, start=1):
            pairs = per_pair[(per_pair.trial == trial) & (per_pair.metric == metric)].sort_values("period_index")
            overall = summary[(summary.trial == trial) & (summary.metric == metric)].iloc[0]
            x = pairs.period_index.to_numpy()
            first = r == 1 and c == 1
            figure.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                                        y=np.r_[pairs["q95" + suffix], pairs["q05" + suffix][::-1]],
                                        fill="toself", fillcolor=RANGE_COLOUR, line=dict(width=0),
                                        hoverinfo="skip", name="5–95% of windows", legendgroup="range",
                                        showlegend=first), row=r, col=c)
            for statistic, colour, label in (("median", MEDIAN_COLOUR, "Median per pair of turns"),
                                             ("mean", MEAN_COLOUR, "Mean per pair of turns")):
                figure.add_trace(go.Scatter(
                    x=x, y=pairs[statistic + suffix], mode="lines+markers", name=label,
                    legendgroup=statistic, showlegend=first, line=dict(color=colour, width=2),
                    marker=dict(size=7, color=colour, line=dict(width=1.5, color="white")),
                    hovertemplate=(f"{label}<br>turn %{{x}} vs %{{customdata}}<br>%{{y:.5g}}<extra></extra>"),
                    customdata=x + 1), row=r, col=c)
                figure.add_hline(y=overall[statistic + suffix], line=dict(color=colour, width=1),
                                 opacity=0.55, row=r, col=c)
    what = "signed squared MMD estimate" if squared else "distance"
    figure.update_layout(
        title=(f"Noise floor: {what} between neighbouring turns"
               "<br><sup>Each window against the same wheel position one turn later. Points are per "
               "pair of turns; thin horizontal lines are the median and mean over all pairs.</sup>"),
        template="plotly_white", height=230 * len(metrics) + 150, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        margin=dict(l=60, r=60, t=130, b=50))
    figure.update_xaxes(title_text="turn", row=len(metrics))
    figure.write_html(path, include_plotlyjs=True)


def summarise_run(run_dir: Path):
    """Combine the recordings already saved in a run: tables and figures at the run root."""
    frames = [pd.read_csv(path) for path in sorted(run_dir.glob("*/results.csv"))]
    if not frames:
        raise SystemExit(f"no results under {run_dir}")
    results = pd.concat(frames, ignore_index=True)
    summary = statistics(results, ["trial", "metric"])
    per_pair = statistics(results, ["trial", "period_index", "metric"])
    summary.to_csv(run_dir / "summary.csv", index=False)
    per_pair.to_csv(run_dir / "per_pair.csv", index=False)
    plot_floor(summary, per_pair, run_dir / "noise_floor.html")
    plot_floor(summary, per_pair, run_dir / "noise_floor_signed_mmd.html", squared=True)
    shown = ["trial", "metric", "n_valid", "median_distance", "q05_distance", "q95_distance",
             "mean_distance", "std_distance", "mean_minus_median", "share_exactly_zero"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        say("noise floor over all neighbouring pairs of turns:\n"
            + summary[shown].round(5).to_string(index=False))
    return summary, per_pair


# --- 5. Record ---

class Tee:
    """Mirror everything printed into the run's log file."""

    def __init__(self, path: Path, stream):
        self.file, self.stream = path.open("a", encoding="utf-8"), stream

    def write(self, text):
        if self.stream is not None:
            self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        if self.stream is not None:
            self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def write_state(run_dir: Path, status: str, **details):
    """Atomically record where the run is, for monitoring and for the morning after."""
    record = {"status": status, "pid": os.getpid(),
              "updated_at": datetime.now().isoformat(timespec="seconds"), **details}
    temporary = run_dir / "run_state.tmp"
    temporary.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    temporary.replace(run_dir / "run_state.yaml")


def snapshot_sources(run_dir: Path) -> dict:
    """Keep this script and config.yaml beside the results, with hashes of what it imports."""
    source = run_dir / "source"
    source.mkdir(exist_ok=True)
    for path in (Path(__file__).resolve(), PROJECT_ROOT / "config.yaml"):
        shutil.copy2(path, source / path.name)
    relative = [*IMPORTED_SOURCES, "experiments/noise_floor.py", "config.yaml",
                *sorted(path.relative_to(PROJECT_ROOT).as_posix()
                        for path in (PROJECT_ROOT / "src/event_analysis_toolbox").rglob("*.py"))]
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in relative}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def write_config(path: Path, record: dict):
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")


def trial_record(trial: Trial, real, windows: pd.DataFrame, n_metrics: int) -> dict:
    """What was compared for one recording, with the rotation period and where it came from."""
    return {"name": trial.name, "real_path": str(trial.real_path),
            "rotation_period_us": int(trial.rotation_period_us),
            "rotation_period_source": trial.rotation_period_source,
            "recording_span_us": int(real.t_end - real.t_start),
            "full_turns": int((real.t_end - real.t_start) // trial.rotation_period_us),
            "pairs_of_turns": int(windows.period_index.nunique()),
            "windows": int(len(windows)), "eligible_reference_windows": int(windows.eligible.sum()),
            "comparisons": int(len(windows)), "metric_values": int(len(windows) * n_metrics)}


# --- The pathway ---

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", "--trial", nargs="+", default=TRIAL_NAMES)
    parser.add_argument("--trials-dir", default=TRIALS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--temporal-window", type=int, default=5000, help="window length in microseconds")
    parser.add_argument("--seed", type=int, default=0, help="recorded with each row; nothing here is random")
    parser.add_argument("--max-pairs", type=int, default=None, help="limit the pairs of turns, for a short test")
    parser.add_argument("--plan-only", action="store_true", help="print the plan and exit; no metrics")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="continue this run; recordings already summarised are skipped")
    parser.add_argument("--summarize-only", type=Path, default=None, metavar="RUN_DIR",
                        help="rebuild the tables and figures of a saved run")
    args = parser.parse_args(argv)
    if args.temporal_window <= 0 or (args.max_pairs is not None and args.max_pairs < 1):
        parser.error("temporal_window and max_pairs must be positive")
    return args


def print_plan(args):
    """The comparisons a run would make, without computing any metric."""
    total = 0
    for name in args.trials:
        trial = Trial.load(args.trials_dir, name)
        real = Stream(trial.real_path, "real")
        try:
            windows = plan_windows(trial, real, args.temporal_window, args.max_pairs)
            record = trial_record(trial, real, windows, len(METRIC_KEYS))
        finally:
            real.close()
        total += record["comparisons"]
        print(f"{name}: {record['full_turns']} full turns, {record['pairs_of_turns']} pairs, "
              f"{record['windows'] // record['pairs_of_turns']} windows per turn, "
              f"{record['comparisons']:,} comparisons "
              f"(rotation {record['rotation_period_us']:,} us)")
    print(f"total: {total:,} comparisons, {total * len(METRIC_KEYS):,} metric values")


def run_recording(name: str, args, config: dict, metrics: dict, run_dir: Path) -> tuple[dict, dict]:
    """Plan, compare, check and summarise one recording."""
    directory = run_dir / name
    directory.mkdir(exist_ok=True)
    trial = Trial.load(args.trials_dir, name)
    real = Stream(trial.real_path, "real")
    try:
        windows = plan_windows(trial, real, args.temporal_window, args.max_pairs)
        record = trial_record(trial, real, windows, len(metrics))
        write_config(directory / "run_config.yaml", record)
        windows.to_csv(directory / "windows.csv", index=False)
        say(f"{name}: {record['pairs_of_turns']} pairs of turns, {record['comparisons']:,} comparisons")
        results = compare(trial, real, windows, config, metrics, args.seed, directory / "results.csv")
    finally:
        real.close()
    checks = check_recording(results, windows, trial, len(metrics))
    checks["first_pair_equals_stored_360"] = check_against_stored(results, name)
    statistics(results, ["trial", "period_index", "metric"]).to_csv(directory / "per_pair.csv", index=False)
    # summary.csv is written last: its presence marks the recording as complete.
    statistics(results, ["trial", "metric"]).to_csv(directory / "summary.csv", index=False)
    return record, checks


def checks_passed(checks: dict) -> bool:
    for recording in checks.values():
        for key, value in recording.items():
            if key == "first_pair_equals_stored_360":
                if value["status"] == "MISMATCH":
                    return False
            elif value is not True:
                return False
    return True


def main(argv=None):
    args = parse_args(argv)
    if args.summarize_only is not None:
        summarise_run(args.summarize_only)
        return
    if args.plan_only:
        print_plan(args)
        return

    run_dir = args.run_dir or args.output_dir / f"neighbouring_turns_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(run_dir / "run.log", sys.stdout)
    sys.stderr = Tee(run_dir / "run.log", sys.stderr)
    say(f"run directory: {run_dir.resolve()}")

    config = load_config()
    metrics = build_metrics(METRIC_KEYS, config)
    config_path, checks_path = run_dir / "run_config.yaml", run_dir / "checks.yaml"
    record = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {
        "experiment": "noise_floor_neighbouring_turns", "script": "experiments/noise_floor.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "definition": ("each complete window of a turn against the window one measured rotation "
                       "later, for every turn with a complete next turn; each window re-zeroed "
                       "separately; turns further apart are never compared"),
        "feature_scales": config["feature_scales"], "sensor": config["sensor"],
        "metric_settings": {key: settings for key, (_, settings) in metrics.items()},
        "sampling": "all events retained", "minimum_events_per_side": MIN_EVENTS,
        "statistics": ("median with 5% and 95% points and mean with standard deviation over "
                       "valid windows; the signed squared estimate is summarised too"),
        "trials": {}, "launches": []}
    record["launches"].append({"at": datetime.now().isoformat(timespec="seconds"),
                               "git_commit": git_commit(), "source_sha256": snapshot_sources(run_dir),
                               "arguments": {k: str(v) if isinstance(v, Path) else v
                                             for k, v in vars(args).items()}})
    write_config(config_path, record)
    checks = yaml.safe_load(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {}

    completed = []
    try:
        for name in args.trials:
            if (run_dir / name / "summary.csv").exists():
                say(f"{name}: already complete, skipped")
                completed.append(name)
                continue
            write_state(run_dir, "running", trial=name, completed=completed)
            record["trials"][name], checks[name] = run_recording(name, args, config, metrics, run_dir)
            write_config(config_path, record)
            write_config(checks_path, checks)
            completed.append(name)
            say(f"{name}: done; checks {checks[name]}")
        summarise_run(run_dir)
        passed = checks_passed(checks)
        write_state(run_dir, "completed", completed=completed, checks_passed=passed)
        say(f"completed: {run_dir.resolve()} | all checks passed: {passed}")
    except BaseException:
        write_state(run_dir, "failed", completed=completed, error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
