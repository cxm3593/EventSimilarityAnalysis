"""Run calibrated event-stream perturbation evaluations.

Author: Chengyi Ma

Argument loading and the run sequence are here. Physical perturbations are grouped
into static-method classes in perturbation_methods.py; reusable windowing, metrics,
outputs and figures are grouped in perturbation_shared.py.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

if __package__:
    from .common import METRIC_KEYS, Stream, Trial, build_metrics, load_config
    from .perturbation_methods import METHODS
    from .perturbation_shared import load_data, plot_comparison, write_run_config
else:
    from common import METRIC_KEYS, Stream, Trial, build_metrics, load_config
    from perturbation_methods import METHODS
    from perturbation_shared import load_data, plot_comparison, write_run_config

TRIALS_DIR = r"C:/Users/cxm3593/Academic/Workspace/EventCamCalib/output/trials"


# --- Saved command-line arguments ---

def yaml_argument_tokens(path: Path, parser: argparse.ArgumentParser) -> list[str]:
    """Translate a YAML mapping into argv tokens; argparse owns names and types."""
    try:
        with path.open(encoding="utf-8") as handle:
            saved = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        parser.error(f"cannot read {path}: {error}")
    if saved is None:
        return []
    if not isinstance(saved, dict):
        parser.error("the CLA file must contain a mapping of argument names to values")
    tokens = []
    for name, value in saved.items():
        if not isinstance(name, str):
            parser.error("CLA argument names must be strings")
        values = value if isinstance(value, list) else [value]
        if any(not isinstance(item, (str, int, float)) or isinstance(item, bool) for item in values):
            parser.error(f"CLA value for {name!r} must be a string, number, or list of these")
        option = "--" + name.replace("_", "-")
        if isinstance(value, list):
            tokens.append(option)
            tokens.extend(str(item) for item in values)
        else:
            tokens.append(f"{option}={value}")
    return tokens


def validate_args(args, parser):
    """Reject invalid physical sweeps and duplicate work before creating outputs."""
    if args.temporal_window <= 0 or args.periods <= 0 or args.first_period < 0:
        parser.error("temporal_window and periods must be positive; first_period must be non-negative")
    if len(set(args.trials)) != len(args.trials) or len(set(args.evaluations)) != len(args.evaluations):
        parser.error("trials and evaluations must not contain duplicates")
    for method in METHODS.values():
        values = getattr(args, method.sweep_argument)
        if not np.all(np.isfinite(values)) or len(set(values)) != len(values):
            parser.error(f"{method.sweep_argument} must contain unique finite values")
    if min(args.spatial_scaling_sweep) <= 0:
        parser.error("spatial scaling factors must be positive")
    if min(args.spatial_jitter_sweep) < 0 or min(args.uniform_noise_sweep) < 0:
        parser.error("jitter amplitudes and noise ratios must be non-negative")
    if min(args.subsampling_sweep) <= 0 or max(args.subsampling_sweep) > 1:
        parser.error("subsampling fractions must be in (0, 1]")
    if args.temporal_far_step_us <= 0 or min(args.temporal_near_us) < 0:
        parser.error("temporal_far_step_us must be positive; temporal_near_us must be non-negative")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse YAML arguments first and explicit CLI overrides second."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cla", type=Path, metavar="YAML", help="saved command-line arguments")
    parser.add_argument("--trials-dir", default=TRIALS_DIR)
    parser.add_argument("--trials", "--trial", nargs="+", default=["optical_chopper_data_f1"])
    parser.add_argument("--output-dir", type=Path, default=Path("output/perturbation"))
    parser.add_argument("--evaluations", nargs="+", choices=list(METHODS), default=["spatial_offset"])
    parser.add_argument("--temporal-window", type=int, default=5000, help="window length in microseconds")
    parser.add_argument("--first-period", type=int, default=0)
    parser.add_argument("--periods", type=int, default=1, help="rotations; use every complete window")
    parser.add_argument("--seed", type=int, default=0, help="reproducible stochastic perturbations")
    parser.add_argument("--spatial-offset-sweep", type=float, nargs="+", default=[0, 1, 3, 5, 7, 9, 11, 13, 15])
    parser.add_argument("--spatial-scaling-sweep", type=float, nargs="+", default=[1, 1.01, 1.02, 1.05, 1.10])
    parser.add_argument("--spatial-jitter-sweep", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8])
    parser.add_argument("--subsampling-sweep", type=float, nargs="+", default=[1, 0.9, 0.75, 0.5, 0.25])
    parser.add_argument("--uniform-noise-sweep", type=float, nargs="+", default=[0, 0.01, 0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--temporal-offset-sweep", type=float, nargs="+",
                        default=[0, 5, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270, 315, 360],
                        help="phase offsets in full-rotation degrees, converted to microseconds per trial")
    parser.add_argument("--temporal-offset-grid", choices=["degrees", "time"], default="degrees",
                        help="time builds a microsecond sweep through one measured rotation per trial")
    parser.add_argument("--temporal-near-us", type=int, nargs="+",
                        default=[0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    parser.add_argument("--temporal-far-step-us", type=int, default=20000,
                        help="spacing beyond the near-time sweep; quarter-rotation anchors are added")
    known, _ = parser.parse_known_args(argv)
    saved = yaml_argument_tokens(known.cla, parser) if known.cla else []
    args = parser.parse_args(saved + argv)
    validate_args(args, parser)
    return args


# --- Evaluation sequence and durable status ---

def write_status(run_dir: Path, status: str, **details):
    """Atomically record the worker identity and current stage for monitoring."""
    record = {"status": status, "pid": os.getpid(), "updated_at": datetime.now().isoformat(), **details}
    temporary = run_dir / "run_state.tmp"
    temporary.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    temporary.replace(run_dir / "run_state.yaml")


def evaluate_trial(method, name: str, args, config: dict, metrics: dict, run_dir: Path) -> pd.DataFrame:
    """Load one pair, run one evaluation, then release both file-backed readers."""
    write_status(run_dir, "running", evaluation=method.name, trial=name)
    trial, real, v2e = load_data(args.trials_dir, name)
    try:
        return method.evaluate(trial, real, args, config, metrics, run_dir / name / method.name)
    finally:
        real.close()
        v2e.close()


def evaluate_trials(method, args, config: dict, metrics: dict, run_dir: Path) -> pd.DataFrame:
    """Run all recordings sequentially and save their combined comparison plot."""
    summaries = []
    for name in args.trials:
        summaries.append(evaluate_trial(method, name, args, config, metrics, run_dir))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(run_dir / f"{method.name}_summary.csv", index=False)
    plot_comparison(summary, method, run_dir / f"{method.name}.html")
    return summary


def main():
    """Run selected perturbations in order; retain partial CSVs if a job fails."""
    args = parse_args()
    config = load_config()
    metrics = build_metrics(METRIC_KEYS, config)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_config(run_dir, args, config, metrics)
    print(f"Run directory: {run_dir.resolve()}", flush=True)
    summaries = []
    try:
        for name in args.evaluations:
            summaries.append(evaluate_trials(METHODS[name], args, config, metrics, run_dir))
        pd.concat(summaries, ignore_index=True).to_csv(run_dir / "summary.csv", index=False)
        write_status(run_dir, "completed", evaluations=args.evaluations, trials=args.trials)
    except Exception:
        write_status(run_dir, "failed", error=traceback.format_exc())
        raise
    print(f"Completed: {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
