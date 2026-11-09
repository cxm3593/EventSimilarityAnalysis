"""Monte Carlo error of SWD from its random projections.

SWD squared is the mean of the per-projection costs W_2^2 over L random directions, so its
Monte Carlo standard error is sd(per-projection costs) / sqrt(L). For SWD itself, the
square root, the delta method gives SE(SWD) = SE(SWD^2) / (2 SWD).

For every complete 5 ms window of the first rotation of each recording, the real window
is compared with three partners, using the settings of config.yaml (L = 100, p = 2,
seed 0, the same directions as every other run):

    real_next_turn   the real window one measured rotation later
    v2e              the v2e window over the same time span
    v2ce             the V2CE window over the same time span

A cross-check on a few windows per recording and comparison tests the error model:
the spread of SWD over other seeds at the same L should match the predicted error, and a
reference with many more projections should lie within a few predicted errors.

Writes to output/swd_projection_error/<stamp>/: per_window.csv, summary.csv,
crosscheck.csv, checks.yaml, output.txt and run_config.yaml, which records the rotation
periods and where they came from.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from common import MIN_EVENTS, Stream, Trial, build_metrics, load_config, measure  # noqa: E402
from perturbation_shared import period_windows, read_points  # noqa: E402
# Both set the CUDA path from the pip packages before cupy and POT are imported.
from event_analysis_toolbox.sliced_wasserstein import _require_cupy, _require_pot  # noqa: E402

ot = _require_pot()

TRIALS = ROOT.parent / "EventCamCalib/output/trials"
RECORDINGS = ["f1", "f2", "f3", "f4", "f5"]
COMPARISONS = ["real_next_turn", "v2e", "v2ce"]
WINDOW_US = 5000


class Device:
    """Array module for POT, numpy or cupy, following config.yaml."""

    def __init__(self, backend: str):
        self.name = backend
        if backend == "cupy":
            self.xp = _require_cupy()
        else:
            self.xp = np

    def put(self, array):
        return self.xp.ascontiguousarray(self.xp.asarray(array, dtype=self.xp.float64))

    def get(self, array) -> np.ndarray:
        return self.xp.asnumpy(array) if self.name == "cupy" else np.asarray(array)

    def release(self):
        # Every window has a different size, so cached blocks are never reused; free
        # them after each call, as sliced_wasserstein.py does.
        if self.name == "cupy":
            self.xp.get_default_memory_pool().free_all_blocks()


def projected_costs(device, a, b, p, n_projections=None, seed=None, projections=None):
    """SWD and the per-projection costs W_p^p from POT."""
    value, log = ot.sliced_wasserstein_distance(
        device.put(a), device.put(b), n_projections=n_projections or 1, p=p, seed=seed,
        projections=None if projections is None else device.put(projections), log=True)
    result = float(device.get(value)), device.get(log["projected_emds"]).astype(np.float64)
    device.release()
    return result


def monte_carlo_error(value: float, costs: np.ndarray) -> dict:
    """Standard errors of SWD^2 and SWD from the spread of the per-projection costs."""
    sd = float(np.std(costs, ddof=1))
    se_squared = sd / math.sqrt(len(costs))
    se = se_squared / (2 * value) if value > 0 else float("nan")
    return {"sd_projection_cost": sd, "se_swd_squared": se_squared, "se_swd": se,
            "relative_se_swd": se / value if value > 0 else float("nan"),
            # A translation |delta| gives SWD_2 = |delta| / sqrt(3) in three dimensions.
            "se_as_translation_px": se * math.sqrt(3)}


def reference_swd(device, a, b, p, count, chunk, seed):
    """SWD with many projections, in chunks to bound device memory."""
    rng = np.random.default_rng(seed)
    total, done = 0.0, 0
    while done < count:
        size = min(chunk, count - done)
        directions = rng.normal(size=(a.shape[1], size))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        _, costs = projected_costs(device, a, b, p, projections=directions)
        total += float(costs.sum())
        done += size
    return (total / count) ** (1.0 / p)


def partner_points(name, window, trial, streams, scales):
    """Points of the window compared with the real window, or None if not covered."""
    start, end = int(window.window_start_us), int(window.window_end_us)
    if name == "real_next_turn":
        start, end = start + trial.rotation_period_us, end + trial.rotation_period_us
        stream = streams["real"]
    else:
        stream = streams[name]
    if end > stream.t_end:
        return None
    return read_points(stream, start, end, scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", nargs="+", default=RECORDINGS)
    parser.add_argument("--stride", type=int, default=1, help="use every k-th window")
    parser.add_argument("--crosscheck-windows", type=int, default=2,
                        help="windows per recording and comparison for the cross-check")
    parser.add_argument("--crosscheck-seeds", type=int, default=50)
    parser.add_argument("--reference-projections", type=int, default=10000)
    parser.add_argument("--reference-chunk", type=int, default=500)
    args = parser.parse_args()

    config = load_config()
    scales = config["feature_scales"]
    swd_settings = config["sliced_wasserstein"]
    n_projections, p, seed = int(swd_settings["n_projections"]), float(swd_settings["p"]), int(swd_settings["seed"])
    device = Device(swd_settings.get("backend", "numpy"))
    (swd_metric, metric_settings), = build_metrics(["swd"], config).values()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "output" / "swd_projection_error" / stamp
    out.mkdir(parents=True)

    rows, cross_rows, agreement, periods, skipped = [], [], [], {}, []
    for recording in args.recordings:
        trial = Trial.load(TRIALS, f"optical_chopper_data_{recording}")
        periods[recording] = {"rotation_period_us": trial.rotation_period_us,
                              "source": trial.rotation_period_source}
        streams = {"real": Stream(trial.real_path, "real"),
                   "v2e": Stream(trial.v2e_path, "v2e"),
                   "v2ce": Stream(trial.directory / "final_masked_v2ce.h5", "v2ce")}
        try:
            windows = period_windows(trial, streams["real"], 0, WINDOW_US)
            windows = windows[windows.eligible].iloc[::args.stride]
            picks = set(np.linspace(0, len(windows) - 1, args.crosscheck_windows + 2)
                        .round().astype(int)[1:-1].tolist())
            for position, window in enumerate(tqdm(list(windows.itertuples()), desc=recording)):
                real_points = read_points(streams["real"], int(window.window_start_us),
                                          int(window.window_end_us), scales)
                for name in COMPARISONS:
                    other = partner_points(name, window, trial, streams, scales)
                    if other is None or len(other) < MIN_EVENTS:
                        skipped.append({"recording": recording, "comparison": name,
                                        "window_index": int(window.window_index)})
                        continue
                    value, costs = projected_costs(device, real_points, other, p,
                                                   n_projections=n_projections, seed=seed)
                    error = monte_carlo_error(value, costs)
                    rows.append({"recording": recording, "comparison": name,
                                 "window_index": int(window.window_index),
                                 "n_real": len(real_points), "n_other": len(other),
                                 "swd": value, **error})
                    if position == 0:
                        # The first window must reproduce the value every other run uses.
                        stored, _ = measure(swd_metric, metric_settings, real_points, other)
                        agreement.append({"recording": recording, "comparison": name,
                                          "this_script": value, "common_measure": stored,
                                          "mean_of_costs_root": float(np.mean(costs) ** (1 / p))})
                    if position in picks:
                        spread = [projected_costs(device, real_points, other, p,
                                                  n_projections=n_projections, seed=s)[0]
                                  for s in range(seed + 1, seed + 1 + args.crosscheck_seeds)]
                        reference = reference_swd(device, real_points, other, p,
                                                  args.reference_projections, args.reference_chunk,
                                                  seed=10_000 + position)
                        cross_rows.append({
                            "recording": recording, "comparison": name,
                            "window_index": int(window.window_index), "swd_seed0": value,
                            "predicted_se": error["se_swd"],
                            "observed_sd_over_seeds": float(np.std(spread, ddof=1)),
                            "ratio_observed_to_predicted": float(np.std(spread, ddof=1)) / error["se_swd"],
                            "swd_reference": reference,
                            "deviation_in_se": (value - reference) / error["se_swd"]})
        finally:
            for stream in streams.values():
                stream.close()

    per_window = pd.DataFrame(rows)
    per_window.to_csv(out / "per_window.csv", index=False)
    crosscheck = pd.DataFrame(cross_rows)
    crosscheck.to_csv(out / "crosscheck.csv", index=False)

    def quantiles(group):
        return pd.Series({
            "windows": len(group),
            "swd_median": group.swd.median(),
            "se_swd_median": group.se_swd.median(),
            "se_swd_p95": group.se_swd.quantile(0.95),
            "se_swd_max": group.se_swd.max(),
            "relative_se_median": group.relative_se_swd.median(),
            "relative_se_p95": group.relative_se_swd.quantile(0.95),
            "se_as_translation_px_p95": group.se_as_translation_px.quantile(0.95)})

    summary = per_window.groupby(["comparison", "recording"]).apply(quantiles).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    agreement = pd.DataFrame(agreement)
    checks = {
        "comparisons": int(len(per_window)),
        "skipped": skipped,
        "all_finite": bool(np.isfinite(per_window[["swd", "se_swd"]].to_numpy()).all()),
        "max_abs_difference_to_common_measure": float((agreement.this_script - agreement.common_measure).abs().max()),
        "max_abs_difference_root_of_mean_cost": float((agreement.this_script - agreement.mean_of_costs_root).abs().max()),
        "crosscheck_ratio_observed_to_predicted": {
            "median": float(crosscheck.ratio_observed_to_predicted.median()),
            "min": float(crosscheck.ratio_observed_to_predicted.min()),
            "max": float(crosscheck.ratio_observed_to_predicted.max())},
        "crosscheck_share_within_2_se_of_reference": float((crosscheck.deviation_in_se.abs() <= 2).mean()),
    }
    (out / "checks.yaml").write_text(yaml.safe_dump(checks, sort_keys=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    text = "\n".join([
        f"Monte Carlo error of SWD, L = {n_projections}, p = {p:g}, seed {seed}; "
        "5 ms windows of the first rotation",
        "", summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"),
        "", "Cross-check (observed spread over seeds / predicted error; deviation from the reference in errors):",
        crosscheck.to_string(index=False, float_format=lambda v: f"{v:.4g}"),
        "", yaml.safe_dump(checks, sort_keys=False)])
    print(text)
    (out / "output.txt").write_text(text + "\n", encoding="utf-8")

    run_config = {
        "script": "analysis/swd_projection_error.py", "created": stamp,
        "window_us": WINDOW_US, "rotation": 0, "stride": args.stride,
        "comparisons": COMPARISONS, "n_projections": n_projections, "p": p, "seed": seed,
        "backend": device.name, "feature_scales": scales,
        "error_model": "SE(SWD^2) = sd(per-projection W_p^p, ddof=1) / sqrt(L); SE(SWD) = SE(SWD^2) / (2 SWD)",
        "crosscheck": {"windows_per_recording_and_comparison": args.crosscheck_windows,
                       "seeds": f"{seed + 1}..{seed + args.crosscheck_seeds}",
                       "reference_projections": args.reference_projections,
                       "reference_directions": "normalised Gaussian, numpy seed 10000 + window position"},
        "rotation_periods": periods}
    (out / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
