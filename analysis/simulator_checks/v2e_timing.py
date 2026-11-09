"""Where is the real-vs-v2e distance smallest as v2e is shifted in time?

For each recording, 40 windows of 5 ms are spread over one rotation. Each real window
[s, s+5ms) is compared with the v2e window [s+d, s+d+5ms) for d from -10 to +10 ms, each
re-zeroed to its own start as in the perturbation study. The best shift is the minimum of
the median SWD over windows; a bootstrap over windows gives its 90% range.

This is a check, not a calibration: reported distances keep the calibrated alignment.
"""

import numpy as np
import pandas as pd

from _shared import RECORDINGS, WINDOW_US, event_paths, load_trial, read_span, start, window
from common import build_metrics, load_config, measure, to_points

SHIFTS_MS = np.arange(-10, 11)
N_WINDOWS = 40
BEGIN_US = 1_000_000
BOOTSTRAPS = 500


def main():
    output = start("v2e_timing", __file__,
                   {"metric": "swd", "window_us": WINDOW_US, "n_windows": N_WINDOWS,
                    "begin_us": BEGIN_US, "shifts_ms": SHIFTS_MS.tolist(),
                    "bootstraps": BOOTSTRAPS, "seed": 0})
    config = load_config()
    (swd, settings), = build_metrics(["swd"], config).values()
    scales = config["feature_scales"]
    rng = np.random.default_rng(0)

    curves, best_rows = [], []
    for recording in RECORDINGS:
        period = load_trial(recording).rotation_period_us
        paths = event_paths(recording)
        end = BEGIN_US + period
        real = read_span(paths["real"], BEGIN_US, end)
        v2e = read_span(paths["v2e"], BEGIN_US - 15_000, end + 15_000)
        starts = np.linspace(BEGIN_US, end - WINDOW_US, N_WINDOWS).astype(np.int64)

        distances = np.full((len(starts), len(SHIFTS_MS)), np.nan)
        for i, s in enumerate(starts):
            a = to_points(window(real, s), int(s), scales)
            for j, d in enumerate(SHIFTS_MS):
                shifted = int(s + d * 1000)
                b = to_points(window(v2e, shifted), shifted, scales)
                distances[i, j] = measure(swd, settings, a, b)[0]

        curve = np.nanmedian(distances, axis=0)
        best = int(SHIFTS_MS[np.nanargmin(curve)])
        boots = [SHIFTS_MS[np.nanargmin(np.nanmedian(
            distances[rng.integers(0, len(starts), len(starts))], axis=0))] for _ in range(BOOTSTRAPS)]
        low, high = np.percentile(boots, [5, 95])
        at_zero = float(curve[SHIFTS_MS == 0][0])
        best_rows.append({"recording": recording, "best_shift_ms": best, "range90_low_ms": low,
                          "range90_high_ms": high, "swd_at_0": at_zero, "swd_at_best": float(curve.min()),
                          "reduction_pct": 100 * (1 - curve.min() / at_zero),
                          "best_shift_deg": best * 360 / (period / 1000)})
        curves += [{"recording": recording, "shift_ms": int(d), "median_swd": float(v)}
                   for d, v in zip(SHIFTS_MS, curve)]

    best = pd.DataFrame(best_rows)
    best.to_csv(output / "best_shift.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "shift_curves.csv", index=False)
    pd.set_option("display.width", 200)
    print(best.round(2).to_string(index=False))
    print(f"\nWrote {output / 'best_shift.csv'}\nWrote {output / 'shift_curves.csv'}")


if __name__ == "__main__":
    main()
