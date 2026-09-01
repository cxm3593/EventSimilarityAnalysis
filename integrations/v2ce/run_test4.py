"""Extend compatible Test 4 runs with V2CE without recomputing real and v2e.

The existing F1--F5 runs already contain the deterministic real and v2e comparisons
for three periods, 40 phase-spread 5 ms windows, all five metrics, and no event cap.
This helper seeds a new run with only those rows, resumes Test 4 to compute V2CE, and
regenerates a combined summary and figures. All-to-all and modified-real sources are
disabled for the resumed run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / "output/tests"
V2CE_ROOT = PROJECT_ROOT / "output/v2ce"
TRIAL_NAMES = tuple(f"optical_chopper_data_f{i}" for i in range(1, 6))
METRICS = ("mmd_rbf03", "mmd_rbf15", "mmd_rbf75", "swd", "chamfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group()
    selected.add_argument("--trial", choices=TRIAL_NAMES)
    selected.add_argument("--all", action="store_true")
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--window-us", type=int, default=5000)
    parser.add_argument("--max-windows", type=int, default=40)
    return parser.parse_args()


def compatible(config: dict, args: argparse.Namespace) -> bool:
    parameters = config.get("parameters", {})
    event_cap = parameters.get("events_per_comparison")
    return (
        parameters.get("window_us") == args.window_us
        and parameters.get("n_periods_used") == args.periods
        and parameters.get("max_windows") == args.max_windows
        and event_cap is None
        and tuple(config.get("metrics", ())) == METRICS
    )


def find_base_run(trial: str, args: argparse.Namespace) -> Path:
    root = TEST_ROOT / trial / "test4_synthetic"
    candidates = []
    for config_path in root.glob("*/run_config.yaml"):
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            continue
        if compatible(config, args) and (config_path.parent / "results.csv").exists():
            candidates.append(config_path.parent)
    if not candidates:
        raise FileNotFoundError(f"no compatible base Test 4 run for {trial}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def seed_run(trial: str, base: Path, args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = base.parent / f"{stamp}_w{args.window_us}us_v2ce"
    destination.mkdir(parents=True, exist_ok=False)

    results = pd.read_csv(base / "results.csv")
    results = results[
        results["source"].isin(["real", "v2e"])
        & (results["period_index"] < args.periods)
    ].copy()
    expected = 2 * args.periods * args.max_windows * len(METRICS)
    if len(results) != expected:
        raise RuntimeError(
            f"{base} has {len(results)} reusable rows; expected {expected}"
        )
    results.to_csv(destination / "results.csv", index=False)
    return destination


def run_trial(trial: str, args: argparse.Namespace) -> Path:
    v2ce_path = V2CE_ROOT / trial / "final_masked_v2ce.h5"
    if not v2ce_path.exists():
        raise FileNotFoundError(v2ce_path)
    base = find_base_run(trial, args)
    destination = seed_run(trial, base, args)
    print(f"\n=== {trial} ===")
    print(f"reusing real/v2e rows from {base.name}")
    print(f"combined run: {destination}")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/test4_synthetic.py"),
        "--trial", trial,
        "--periods", str(args.periods),
        "--window-us", str(args.window_us),
        "--max-windows", str(args.max_windows),
        "--events-per-comparison", "none",
        "--metrics", *METRICS,
        "--v2ce-path", str(v2ce_path),
        "--no-all-to-all",
        "--no-modified",
        "--resume", str(destination),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return destination


def main() -> None:
    args = parse_args()
    trials = TRIAL_NAMES if args.all or args.trial is None else (args.trial,)
    outputs = [run_trial(trial, args) for trial in trials]
    print("\nCompleted combined Test 4 runs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
