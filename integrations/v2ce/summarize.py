"""Build cross-frequency summaries for the combined real/v2e/V2CE Test 4 runs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / "output/tests"
V2CE_ROOT = PROJECT_ROOT / "output/v2ce"
TRIALS_ROOT = PROJECT_ROOT.parent / "EventCamCalib/output/trials"
TRIAL_NAMES = tuple(f"optical_chopper_data_f{i}" for i in range(1, 6))


def find_combined_run(trial: str) -> Path:
    candidates = []
    root = TEST_ROOT / trial / "test4_synthetic"
    for config_path in root.glob("*/run_config.yaml"):
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        parameters = config.get("parameters", {})
        simulators = parameters.get("simulator_paths", {})
        if (
            "v2ce" in simulators
            and parameters.get("window_us") == 5000
            and parameters.get("n_periods_used") == 3
            and parameters.get("max_windows") == 40
            and parameters.get("all_to_all") is False
            and parameters.get("modified") is False
            and (config_path.parent / "results.csv").exists()
        ):
            candidates.append(config_path.parent)
    if not candidates:
        raise FileNotFoundError(f"no completed combined V2CE run for {trial}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def stream_stats(path: Path, native_pixels: int) -> dict:
    with h5py.File(path, "r") as handle:
        events = handle["events"]
        count = len(events)
        # Index the compound dataset first. ``events["t"][0]`` materialises the
        # complete timestamp column, which is several gigabytes for the larger runs.
        first = int(events[0]["t"])
        last = int(events[-1]["t"])
    duration_s = (last - first) / 1e6
    return {
        "events": count,
        "duration_s": duration_s,
        "total_event_rate_hz": count / duration_s,
        "native_pixels": native_pixels,
        "events_per_pixel_second": count / duration_s / native_pixels,
    }


def main() -> None:
    frames = []
    run_index = {}
    for index, trial in enumerate(TRIAL_NAMES, start=1):
        run = find_combined_run(trial)
        run_index[f"f{index}"] = str(run)
        frame = pd.read_csv(run / "results.csv")
        if len(frame) != 1800:
            raise RuntimeError(f"{run} has {len(frame)} rows, expected 1800")
        frame["frequency"] = f"f{index}"
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    summary = (results.groupby(["frequency", "source", "metric"])
               .agg(mean_distance=("distance", "mean"),
                    sd_distance=("distance", "std"),
                    median_distance=("distance", "median"),
                    valid_comparisons=("distance", "count"),
                    total_comparisons=("distance", "size"),
                    median_events_a=("n_events_a", "median"),
                    median_events_b=("n_events_b", "median"))
               .reset_index())

    wide = summary.pivot(index=["frequency", "metric"], columns="source",
                         values="mean_distance").reset_index()
    wide["v2ce_over_v2e"] = wide["v2ce"] / wide["v2e"]
    wide["v2e_over_real"] = wide["v2e"] / wide["real"]
    wide["v2ce_over_real"] = wide["v2ce"] / wide["real"]

    rates = []
    for index, trial in enumerate(TRIAL_NAMES, start=1):
        paths = {
            "real": TRIALS_ROOT / trial / "final_masked_real.h5",
            "v2e": TRIALS_ROOT / trial / "final_masked_v2e.h5",
            "v2ce": V2CE_ROOT / trial / "final_masked_v2ce.h5",
        }
        for source, path in paths.items():
            native_pixels = 462 * 260 if source == "v2ce" else 1280 * 720
            row = {"frequency": f"f{index}", "source": source, "path": str(path)}
            row.update(stream_stats(path, native_pixels))
            rates.append(row)
    rate_summary = pd.DataFrame(rates)
    real_rates = (rate_summary[rate_summary.source == "real"]
                  .set_index("frequency")[["total_event_rate_hz",
                                           "events_per_pixel_second"]])
    rate_summary["total_rate_relative_to_real"] = rate_summary.apply(
        lambda row: row.total_event_rate_hz
        / real_rates.loc[row.frequency, "total_event_rate_hz"], axis=1)
    rate_summary["per_pixel_rate_relative_to_real"] = rate_summary.apply(
        lambda row: row.events_per_pixel_second
        / real_rates.loc[row.frequency, "events_per_pixel_second"], axis=1)

    output = V2CE_ROOT
    summary.to_csv(output / "test4_comparison_summary.csv", index=False)
    wide.to_csv(output / "test4_comparison_wide.csv", index=False)
    rate_summary.to_csv(output / "event_rate_summary.csv", index=False)
    with (output / "test4_run_index.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_index, handle, sort_keys=False)

    invalid = int(results["distance"].isna().sum())
    print(f"combined rows: {len(results):,}; invalid distances: {invalid}")
    print("\nMean-distance ratios (V2CE / v2e):")
    print(wide.pivot(index="frequency", columns="metric",
                     values="v2ce_over_v2e").round(3).to_string())
    print("\nWrote:")
    for name in ("test4_comparison_summary.csv", "test4_comparison_wide.csv",
                 "event_rate_summary.csv", "test4_run_index.yaml"):
        print(output / name)


if __name__ == "__main__":
    main()
