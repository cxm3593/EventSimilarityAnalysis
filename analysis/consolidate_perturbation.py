"""Publish a validated six-method result folder from two explicit completed runs.

This is post-processing only: no metric evaluation or automatic 'latest' selection.
Original runs remain untouched. A previous current folder is archived, never deleted.
Run from the repository root with --updated-run PATH --retained-run PATH.
"""

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import METRIC_KEYS
from experiments.perturbation_methods import METHODS
from experiments.perturbation_shared import summarize_distances

UPDATED = ("spatial_offset", "spatial_scaling", "spatial_jitter", "temporal_offset")
RETAINED = ("subsampling", "uniform_noise")
GROUP_KEYS = ["trial", "perturbation", "magnitude", "metric"]


def read_yaml(path: Path) -> dict:
    """Read recorded provenance without inferring settings from directory names."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def completed_config(root: Path) -> dict:
    """Reject unfinished source runs, even when some trial outputs are present."""
    state = read_yaml(root / "run_state.yaml")
    if state.get("status") != "completed":
        raise ValueError(f"Source run is not complete: {root}")
    return read_yaml(root / "run_config.yaml")


def settings_signature(config: dict) -> dict:
    """Select common measurement settings that must agree across source runs."""
    fields = ["feature_scales", "sensor", "metric_settings", "sampling", "minimum_events_per_side"]
    signature = {key: config[key] for key in fields}
    signature["window_arguments"] = {key: config["arguments"][key]
                                     for key in ("temporal_window", "first_period", "periods", "seed")}
    return signature


def verify_counts(results: pd.DataFrame, windows: pd.DataFrame, method: str):
    """Check reference counts and the intended cardinality effect of each method."""
    counts = results.merge(windows[["period_index", "window_index", "n_events"]],
                           on=["period_index", "window_index"], validate="many_to_one")
    if len(counts) != len(results) or not (counts.n_events_a == counts.n_events).all():
        raise ValueError(f"Reference counts do not match window provenance: {method}")
    expected = counts.n_events_a.to_numpy()
    if method == "subsampling":
        expected = np.rint(counts.magnitude * expected)
    elif method == "uniform_noise":
        expected = expected + np.rint(counts.magnitude * expected)
    if method != "temporal_offset" and not np.array_equal(counts.n_events_b, expected):
        raise ValueError(f"Unexpected modified event counts: {method}")


def verify_summary(results: pd.DataFrame, summary: pd.DataFrame):
    """Recompute saved means, quantiles, signed estimates and exclusion counts."""
    recomputed = summarize_distances(results).sort_values(GROUP_KEYS).reset_index(drop=True)
    saved = summary.sort_values(GROUP_KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[recomputed.columns], recomputed,
                                  check_dtype=False, check_exact=False, rtol=1e-8, atol=1e-12)


def validate_trial(source: Path, trial: str, method: str, signature: dict) -> tuple[pd.DataFrame, dict]:
    """Validate one complete trial/method before copying or publishing anything."""
    directory = source / trial / method
    config = read_yaml(directory / "run_config.yaml")
    if settings_signature(config) != signature:
        raise ValueError(f"Incompatible measurement settings: {directory}")
    if config["trial"]["name"] != trial or config["experiment"] != method:
        raise ValueError(f"Incorrect trial or method provenance: {directory}")
    windows = pd.read_csv(directory / "windows.csv")
    results = pd.read_csv(directory / "results.csv")
    summary = pd.read_csv(directory / "summary.csv")
    sweep = config["arguments"][METHODS[method].sweep_argument]
    expected = len(windows) * len(sweep) * len(METRIC_KEYS)
    keys = [*GROUP_KEYS, "period_index", "window_index"]
    if len(results) != expected or results.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate comparisons: {directory}")
    if set(results.trial) != {trial} or set(results.perturbation) != {method}:
        raise ValueError(f"Mixed trial/method rows: {directory}")
    np.testing.assert_allclose(sorted(results.magnitude.unique()), sorted(sweep), rtol=1e-12, atol=1e-12)
    if set(results.metric) != set(METRIC_KEYS):
        raise ValueError(f"Missing metrics: {directory}")
    if not results.status.isin(["ok", "insufficient_events"]).all():
        raise ValueError(f"Unexpected failure status: {directory}")
    low_count = results[["n_events_a", "n_events_b"]].min(axis=1) < config["minimum_events_per_side"]
    if not np.array_equal(low_count, results.status.eq("insufficient_events")):
        raise ValueError(f"Incorrect exclusion statuses: {directory}")
    valid = results[results.status == "ok"]
    if not np.isfinite(valid.distance).all():
        raise ValueError(f"Non-finite valid distances: {directory}")
    mmd = valid[valid.metric.str.startswith("mmd")]
    if not np.isfinite(mmd.distance_squared).all():
        raise ValueError(f"Missing signed MMD estimates: {directory}")
    np.testing.assert_allclose(mmd.distance, np.sqrt(np.maximum(mmd.distance_squared, 0)), atol=1e-12)
    identity = 1 if method in ("spatial_scaling", "subsampling") else 0
    np.testing.assert_allclose(valid.loc[valid.magnitude == identity, "distance"], 0, atol=1e-8)
    verify_counts(results, windows, method)
    verify_summary(results, summary)
    if results.groupby(GROUP_KEYS).size().ne(len(windows)).any():
        raise ValueError(f"Unequal window coverage across magnitudes: {directory}")
    if method == "temporal_offset":
        period = config["trial"]["rotation_period_us"]
        np.testing.assert_array_equal(results.temporal_offset_us, np.rint(results.magnitude * period / 360))
        np.testing.assert_allclose(results.realized_phase_degrees, results.temporal_offset_us * 360 / period)
        summary["temporal_offset_us"] = np.rint(summary.magnitude * period / 360).astype(int)
        summary["realized_phase_degrees"] = summary.temporal_offset_us * 360 / period
    if not all((directory / name).is_file() for name in ("point_cloud.html", f"{method}.html")):
        raise ValueError(f"Missing plots: {directory}")
    summary["source_run"] = str(source.resolve())
    summary["rotation_period_us"] = config["trial"]["rotation_period_us"]
    audit = {"trial": trial, "method": method, "comparisons": len(results), "summary_rows": len(summary),
             "excluded": int((results.status != "ok").sum()), "windows": len(windows),
             "sweep_points": len(sweep), "period_us": config["trial"]["rotation_period_us"]}
    return summary, audit


def copy_method_artifacts(method: str, sources: dict, summary: pd.DataFrame, stage: Path):
    """Copy a completed method overview and write its consolidated summary."""
    shutil.copy2(sources[method] / f"{method}.html", stage / f"{method}.html")
    summary[summary.perturbation == method].to_csv(stage / f"{method}_summary.csv", index=False)


def consolidate(updated_run: Path, retained_run: Path, output: Path) -> Path:
    """Validate all six methods, stage an independent copy, then publish current."""
    updated_run, retained_run, output = updated_run.resolve(), retained_run.resolve(), output.resolve()
    sources = {**dict.fromkeys(UPDATED, updated_run), **dict.fromkeys(RETAINED, retained_run)}
    if any(output == root or output in root.parents or root in output.parents for root in sources.values()):
        raise ValueError("Output must be separate from, not inside or above, either source run")
    new_config, old_config = completed_config(updated_run), completed_config(retained_run)
    signature = settings_signature(new_config)
    if settings_signature(old_config) != signature:
        raise ValueError("Source runs use incompatible measurement settings")
    trials = new_config["arguments"]["trials"]
    if set(trials) != set(old_config["arguments"]["trials"]):
        raise ValueError("Source runs contain different trial selections")
    summaries, audits = [], []
    for trial, method in product(trials, METHODS):
        summary, audit = validate_trial(sources[method], trial, method, signature)
        summaries.append(summary)
        audits.append(audit)
        print(f"Validated {trial}/{method}: {audit['comparisons']} comparisons", flush=True)
    combined = pd.concat(summaries, ignore_index=True)
    if combined.groupby("trial").rotation_period_us.nunique().ne(1).any():
        raise ValueError("Rotation periods disagree between source runs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stage = output.with_name(f"{output.name}_staging_{stamp}")
    stage.mkdir(parents=True, exist_ok=False)
    for trial, method in product(trials, METHODS):
        shutil.copytree(sources[method] / trial / method, stage / trial / method)
    combined.to_csv(stage / "summary.csv", index=False)
    for method in METHODS:
        copy_method_artifacts(method, sources, combined, stage)
    manifest = {"created_at": datetime.now().isoformat(), "status": "completed",
                "method_sources": {method: str(root) for method, root in sources.items()},
                "trials": trials, "audits": audits,
                "total_comparisons": sum(item["comparisons"] for item in audits),
                "summary_rows": len(combined)}
    (stage / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    provenance = {"experiment": "consolidated_perturbations", **signature,
                  "source_configurations": {"updated": new_config, "retained": old_config},
                  "note": "Per-trial run_config.yaml files are unchanged copies of original configurations."}
    (stage / "run_config.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8")
    if output.exists():
        output.rename(output.with_name(f"{output.name}_archive_{stamp}"))
    stage.rename(output)
    return output


def main():
    """Accept explicit run paths; never select a potentially unfinished latest run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updated-run", type=Path, required=True)
    parser.add_argument("--retained-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/perturbation/current")
    args = parser.parse_args()
    print(consolidate(args.updated_run, args.retained_run, args.output))


if __name__ == "__main__":
    main()
