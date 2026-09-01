"""Aggregate the completed 5 ms Test 3 and Test 4 runs across F1--F5.

The script is intentionally read-only with respect to experiment outputs.  It writes
one derived JSON file for inspection and presentation; no metric values are recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = ROOT / "output" / "tests"
OUTPUT = ROOT / "analysis" / "test3_test4_summary.json"

METRICS = ["mmd_rbf03", "mmd_rbf15", "mmd_rbf75", "swd", "chamfer"]
MODIFIERS = [
    "spatial_offset_x",
    "spatial_offset_xy",
    "scaling",
    "subsample",
    "uniform_noise",
    "temporal_clump_uniform",
]


def latest(trial: Path, pattern: str) -> Path:
    matches = sorted(trial.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No run matches {trial / pattern}")
    return matches[-1]


def severity(modifier: str, magnitude: pd.Series) -> pd.Series:
    if modifier == "subsample":
        return 1.0 - magnitude
    if modifier == "scaling":
        return magnitude - 1.0
    return magnitude


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def main() -> None:
    selected_runs = []
    curves_3a = []
    monotonicity_3a = []
    directed_3b = []
    primary_4 = []
    paired_4 = []
    sources_4 = []
    placement_4 = []

    test4_results_by_trial: dict[str, pd.DataFrame] = {}

    for frequency in range(1, 6):
        trial_name = f"F{frequency}"
        trial = TEST_ROOT / f"optical_chopper_data_f{frequency}"
        paths = {
            "3a": latest(trial, "test3_modifier/*_3a_w5000us"),
            "3b": latest(trial, "test3_modifier/*_3b_w5000us"),
            "4": latest(trial, "test4_synthetic/*_w5000us"),
        }

        for test, path in paths.items():
            config = yaml.safe_load((path / "run_config.yaml").read_text())
            parameters = config["parameters"]
            selected_runs.append({
                "frequency": trial_name,
                "test": test,
                "run": path.name,
                "window_us": parameters.get("window_us"),
                "periods": parameters.get("periods", parameters.get("n_periods_used")),
                "max_windows": parameters.get("max_windows"),
                "events_per_comparison": parameters.get("events_per_comparison"),
                "all_to_all": parameters.get("all_to_all"),
            })

        summary_3a = pd.read_csv(paths["3a"] / "summary.csv")
        summary_3a["frequency"] = trial_name
        for modifier in MODIFIERS:
            for metric in METRICS:
                block = summary_3a[
                    (summary_3a.modifier == modifier) & (summary_3a.metric == metric)
                ].copy()
                if block.empty:
                    continue
                block["severity"] = severity(modifier, block.magnitude)
                block = block.sort_values("severity")
                for row in block.itertuples():
                    curves_3a.append({
                        "frequency": trial_name,
                        "modifier": modifier,
                        "metric": metric,
                        "magnitude": float(row.magnitude),
                        "severity": float(row.severity),
                        "mean_distance": float(row.mean_distance),
                        "sd_distance": float(row.sd_distance),
                        "mean_n_events_a": float(row.mean_n_events_a),
                        "mean_n_events_b": float(row.mean_n_events_b),
                    })
                nonzero = block[block.severity > 0]
                rho, _ = spearmanr(block.severity, block.mean_distance)
                endpoint = block.iloc[-1]
                monotonicity_3a.append({
                    "frequency": trial_name,
                    "modifier": modifier,
                    "metric": metric,
                    "spearman_rho": finite_or_none(rho),
                    "endpoint_magnitude": float(endpoint.magnitude),
                    "endpoint_distance": float(endpoint.mean_distance),
                    "first_nonzero_distance": (
                        float(nonzero.iloc[0].mean_distance) if not nonzero.empty else None
                    ),
                })

        summary_3b = pd.read_csv(paths["3b"] / "summary.csv")
        natural = summary_3b[summary_3b.sizing == "natural"].copy()
        for metric in METRICS:
            block = natural[natural.metric == metric]
            baseline_rows = block[block.modifier == "none"]
            if baseline_rows.empty:
                continue
            baseline = float(baseline_rows.mean_distance.iloc[0])
            for row in block.itertuples():
                directed_3b.append({
                    "frequency": trial_name,
                    "metric": metric,
                    "modifier": row.modifier,
                    "mean_distance": float(row.mean_distance),
                    "sd_distance": float(row.sd_distance),
                    "baseline_distance": baseline,
                    "ratio_to_unmodified": (
                        float(row.mean_distance / baseline) if baseline != 0 else None
                    ),
                    "percent_change": (
                        float((row.mean_distance - baseline) / baseline * 100)
                        if baseline != 0 else None
                    ),
                    "mean_n_events_a": float(row.mean_n_events_a),
                    "mean_n_events_b": float(row.mean_n_events_b),
                })

        results_4 = pd.read_csv(paths["4"] / "results.csv")
        results_4["frequency"] = trial_name
        test4_results_by_trial[trial_name] = results_4

        for metric in METRICS:
            metric_rows = results_4[
                (results_4.metric == metric) &
                (results_4.polarity_channel == "all")
            ]
            real = metric_rows[
                (metric_rows.source == "real") & (metric_rows.period_index != 0)
            ].distance.dropna()
            v2e = metric_rows[metric_rows.source == "v2e"].distance.dropna()
            if real.empty or v2e.empty:
                continue
            real_mean = float(real.mean())
            v2e_mean = float(v2e.mean())
            pooled = np.sqrt((real.var(ddof=1) + v2e.var(ddof=1)) / 2)
            primary_4.append({
                "frequency": trial_name,
                "metric": metric,
                "real_mean_excluding_reference": real_mean,
                "real_sd": float(real.std(ddof=1)),
                "real_q95": float(real.quantile(0.95)),
                "v2e_mean": v2e_mean,
                "v2e_sd": float(v2e.std(ddof=1)),
                "gap_ratio": float(v2e_mean / real_mean) if real_mean else None,
                "mean_difference": v2e_mean - real_mean,
                "standardized_difference": (
                    float((v2e_mean - real_mean) / pooled) if pooled > 0 else None
                ),
                "fraction_v2e_above_real_q95": float((v2e > real.quantile(0.95)).mean()),
                "n_real_windows": int(len(real)),
                "n_v2e_windows": int(len(v2e)),
            })

            paired_source = metric_rows[
                metric_rows.source.isin(["real", "v2e"]) &
                (metric_rows.period_index != 0)
            ]
            paired = paired_source.pivot_table(
                index=["period_index", "window_index"],
                columns="source",
                values="distance",
                aggfunc="first",
            ).dropna()
            if not paired.empty:
                delta = paired["v2e"] - paired["real"]
                paired_4.append({
                    "frequency": trial_name,
                    "metric": metric,
                    "mean_paired_difference": float(delta.mean()),
                    "median_paired_difference": float(delta.median()),
                    "q05_paired_difference": float(delta.quantile(0.05)),
                    "q95_paired_difference": float(delta.quantile(0.95)),
                    "fraction_v2e_greater_than_real": float((delta > 0).mean()),
                    "n_paired_windows": int(len(delta)),
                })

            for source, source_rows in metric_rows.groupby("source"):
                if source == "real":
                    source_rows = source_rows[source_rows.period_index != 0]
                values = source_rows.distance.dropna()
                if values.empty:
                    continue
                sources_4.append({
                    "frequency": trial_name,
                    "metric": metric,
                    "source": source,
                    "mean_distance": float(values.mean()),
                    "sd_distance": float(values.std(ddof=1)),
                    "n_windows": int(len(values)),
                })

        placement_path = paths["4"] / "placement.csv"
        if placement_path.exists():
            placement = pd.read_csv(placement_path)
            for row in placement.itertuples():
                placement_4.append({
                    "frequency": trial_name,
                    "metric": row.metric,
                    "v2e_mean_distance": float(row.v2e_mean_distance),
                    "floor": finite_or_none(row.floor),
                    "ceiling": finite_or_none(row.ceiling),
                    "equivalent_shift_ms": finite_or_none(row.equivalent_shift_us / 1000),
                    "equivalent_spatial_offset_px": finite_or_none(
                        row.equivalent_spatial_offset_px
                    ),
                    "equivalent_noise_fraction": finite_or_none(
                        row.equivalent_noise_fraction
                    ),
                })

    primary_frame = pd.DataFrame(primary_4)
    cross_frequency = []
    for metric, block in primary_frame.groupby("metric"):
        cross_frequency.append({
            "metric": metric,
            "mean_gap_ratio": float(block.gap_ratio.mean()),
            "min_gap_ratio": float(block.gap_ratio.min()),
            "max_gap_ratio": float(block.gap_ratio.max()),
            "mean_standardized_difference": float(block.standardized_difference.mean()),
            "frequencies_v2e_above_real": int((block.mean_difference > 0).sum()),
            "mean_fraction_v2e_above_real_q95": float(
                block.fraction_v2e_above_real_q95.mean()
            ),
        })

    output = {
        "selected_runs": selected_runs,
        "test3a_curves": curves_3a,
        "test3a_monotonicity": monotonicity_3a,
        "test3b_directed": directed_3b,
        "test4_primary": primary_4,
        "test4_paired": paired_4,
        "test4_cross_frequency": cross_frequency,
        "test4_sources": sources_4,
        "test4_placement": placement_4,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Test 3a curve rows: {len(curves_3a)}")
    print(f"Test 3b directed rows: {len(directed_3b)}")
    print(f"Test 4 primary rows: {len(primary_4)}")


if __name__ == "__main__":
    main()
