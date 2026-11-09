"""Express the floor and the simulator distances in the units of each perturbation.

For each recording and metric, the median perturbation curve is inverted by linear
interpolation at its first crossing (every curve is 0 at magnitude 0). A value above
everything the curve reaches in the tested range is capped there and flagged.

Two outputs:
  conversions.csv         floor, v2e and V2CE distances as an equivalent rotation,
                          translation, scaling, jitter and added noise
  rotation_crosswalk.csv  the distance a rotation of 1, 5 or 10 degrees produces, and the
                          other perturbations that produce the same distance

Sources: spatial offset/scaling/jitter from the 22 Sep run, uniform noise from the 21 Sep
run, temporal offset from the 23-24 Sep run, simulator distances from the comparison runs.
"""

import pandas as pd

from _shared import (RECORDINGS, SAMPLING_NOISE_RUN, SPATIAL_RUN, TEMPORAL_RUN,
                     comparison_runs, start, trial_name)

METRICS = ["swd", "mmd_rbf75", "mmd_rbf15"]
ANCHORS_DEG = [1.0, 5.0, 10.0]
# perturbation -> (column label, magnitude transform, which magnitudes form the curve)
UNITS = {
    "temporal_offset": ("rotation_deg", lambda m: m, lambda m: m <= 45),
    "spatial_offset": ("shift_px_per_axis", lambda m: m, lambda m: m >= 0),
    "spatial_scaling": ("scale_pct", lambda m: (m - 1) * 100, lambda m: m >= 1),
    "spatial_jitter": ("jitter_px", lambda m: m, lambda m: m >= 0),
    "uniform_noise": ("noise_pct", lambda m: m * 100, lambda m: m >= 0),
}


def load_curves() -> pd.DataFrame:
    parts = [pd.read_csv(SPATIAL_RUN / f"{name}_summary.csv")
             for name in ("spatial_offset", "spatial_scaling", "spatial_jitter")]
    parts.append(pd.read_csv(SAMPLING_NOISE_RUN / "uniform_noise_summary.csv"))
    parts.append(pd.read_csv(TEMPORAL_RUN / "summary.csv"))
    return pd.concat(parts, ignore_index=True)


def curve_for(curves, recording, metric, perturbation) -> pd.Series:
    _, transform, keep = UNITS[perturbation]
    block = curves[(curves.trial == trial_name(recording)) & (curves.metric == metric)
                   & (curves.perturbation == perturbation)]
    block = block[block.magnitude.map(keep)]
    return pd.Series(block.median_distance.values,
                     index=block.magnitude.map(transform).values).sort_index()


def invert(curve: pd.Series, value: float) -> tuple[float, bool]:
    """Magnitude at which the curve first reaches value, and whether it was capped."""
    xs, ys = curve.index.to_numpy(float), curve.to_numpy(float)
    if value <= 0:
        return 0.0, False
    if value >= ys.max():
        return float(xs[-1]), True
    for i in range(1, len(xs)):
        if ys[i] >= value:
            if ys[i] == ys[i - 1]:
                return float(xs[i - 1]), False
            slope = (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + (value - ys[i - 1]) * slope), False
    return float("nan"), False


def simulator_distances() -> pd.DataFrame:
    """Per-window distances: v2e and V2CE over rotations 0-2, real over rotations 1-2."""
    frames = []
    for recording, run in comparison_runs().items():
        frame = pd.read_csv(run / "results.csv")
        frame = frame[frame.polarity_channel == "all"]
        frame = frame[~((frame.source == "real") & (frame.period_index == 0))].copy()
        frame["recording"] = recording
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    output = start("unit_conversion", __file__,
                   {"metrics": METRICS, "anchors_deg": ANCHORS_DEG,
                    "rotation_curve_range_deg": [0, 45], "scaling_side": "enlargement",
                    "spatial_run": SPATIAL_RUN.name, "noise_run": SAMPLING_NOISE_RUN.name,
                    "temporal_run": TEMPORAL_RUN.name,
                    "comparison_runs": {k: v.name for k, v in comparison_runs().items()}})
    curves = load_curves()
    windows = simulator_distances()
    temporal = curves[curves.perturbation == "temporal_offset"]

    rows, cross = [], []
    for recording in RECORDINGS:
        for metric in METRICS:
            values = {}
            at_360 = temporal[(temporal.trial == trial_name(recording)) & (temporal.metric == metric)
                              & (temporal.magnitude == 360)].iloc[0]
            values["floor_360deg"] = (at_360.median_distance, at_360.q05_distance,
                                      at_360.q95_distance, int(at_360.n_valid))
            block = windows[(windows.recording == recording) & (windows.metric == metric)]
            for source, label in (("real", "floor_rotations_1_2"), ("v2e", "v2e"), ("v2ce", "v2ce")):
                d = block[block.source == source].distance
                values[label] = (d.median(), d.quantile(0.05), d.quantile(0.95), int(d.count()))
            for source, (median, q05, q95, n) in values.items():
                row = {"recording": recording, "metric": metric, "source": source,
                       "median_distance": median, "q05_distance": q05, "q95_distance": q95,
                       "n_windows": n}
                for perturbation, (label, _, _) in UNITS.items():
                    curve = curve_for(curves, recording, metric, perturbation)
                    row[label], row[f"{label}_beyond_range"] = invert(curve, median)
                rotation = curve_for(curves, recording, metric, "temporal_offset")
                row["rotation_deg_q05"], _ = invert(rotation, q05)
                row["rotation_deg_q95"], _ = invert(rotation, q95)
                rows.append(row)

            rotation = temporal[(temporal.trial == trial_name(recording))
                                & (temporal.metric == metric)].set_index("magnitude").median_distance
            for anchor in ANCHORS_DEG:
                row = {"recording": recording, "metric": metric, "rotation_deg": anchor,
                       "distance": rotation.loc[anchor]}
                for perturbation, (label, _, _) in UNITS.items():
                    if perturbation == "temporal_offset":
                        continue
                    curve = curve_for(curves, recording, metric, perturbation)
                    row[label], row[f"{label}_beyond_range"] = invert(curve, rotation.loc[anchor])
                cross.append(row)

    conversions, crosswalk = pd.DataFrame(rows), pd.DataFrame(cross)
    conversions.to_csv(output / "conversions.csv", index=False)
    crosswalk.to_csv(output / "rotation_crosswalk.csv", index=False)

    pd.set_option("display.width", 220)
    shown = ["recording", "source", "median_distance", "rotation_deg", "shift_px_per_axis",
             "scale_pct", "jitter_px", "noise_pct"]
    for metric in METRICS:
        print(f"\n=== {metric}: equivalent magnitude of each perturbation (median distance) ===")
        print(conversions[conversions.metric == metric][shown].round(3).to_string(index=False))
    for metric in ("swd", "mmd_rbf75"):
        print(f"\n=== {metric}: what a rotation is worth in the other units ===")
        print(crosswalk[crosswalk.metric == metric].drop(columns="metric").round(3).to_string(index=False))
    print("\nA value at the end of the tested range (25 px, 20 %, 16 px, 50 %, 45 deg) with "
          "its *_beyond_range flag set means the curve never reaches that distance.")
    print(f"\nWrote {output / 'conversions.csv'}\nWrote {output / 'rotation_crosswalk.csv'}")


if __name__ == "__main__":
    main()
