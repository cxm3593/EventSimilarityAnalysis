"""Does it matter whether V2CE is scaled up or the real data scaled down?

Five 5 ms windows per recording (F1, F5), three versions of the same comparisons:

  A  full resolution 1280x720 (current): V2CE scaled up, t/42, kernels 3 and 75 px
  B  462x260: real scaled down onto V2CE's pixels, V2CE native, the SAME parameters
  C  462x260 with parameters converted to the larger pixels (1 V2CE px = 2.77 sensor px):
     t/(42*2.77), kernels 3/2.77 and 75/2.77 px, i.e. the same physical scale as A

Floor = the real window one rotation later, treated the same way. If C x 2.77 equals A,
the direction of scaling does not change the measurement.
"""

import numpy as np
import pandas as pd

from _shared import WINDOW_US, event_paths, load_trial, read_span, start, window
from common import get_metric, load_config, measure, to_points

RECORDINGS = ["f1", "f5"]
N_WINDOWS = 5
BEGIN_US = 1_000_000
SX, SY = 1280 / 462, 720 / 260


def mmd(config, max_distance):
    section = dict(config["mmd"])
    section["biased"] = False
    section["kernels"] = [{"rbf_kernel_max_distance": max_distance,
                           "rbf_kernel_target_similarity": 0.5}]
    metric = get_metric("mmd")
    settings = metric.build_kwargs({"mmd": section})
    if metric.supports_inner_progress:
        settings["progress"] = False
    return metric, settings


def down(events):
    """Full-resolution pixel -> V2CE pixel (exact inverse for V2CE's scaled-up events)."""
    out = events.copy()
    out["x"] = np.rint((events["x"] + 0.5) / SX - 0.5).astype(events["x"].dtype)
    out["y"] = np.rint((events["y"] + 0.5) / SY - 0.5).astype(events["y"].dtype)
    return out


def main():
    output = start("v2ce_scale_direction", __file__,
                   {"window_us": WINDOW_US, "n_windows": N_WINDOWS, "begin_us": BEGIN_US,
                    "scale_x": SX, "scale_y": SY, "kernels_px": [3, 75], "time_scale": 42},
                   recordings=RECORDINGS)
    config = load_config()
    swd = get_metric("sliced_wasserstein")
    swd_settings = swd.build_kwargs({"sliced_wasserstein": dict(config["sliced_wasserstein"])})
    versions = {
        "A full-res": (False, {"x": 1, "y": 1, "t": 42},
                       {"mmd_small": mmd(config, 3), "mmd_large": mmd(config, 75)}),
        "B low-res, same parameters": (True, {"x": 1, "y": 1, "t": 42},
                                       {"mmd_small": mmd(config, 3), "mmd_large": mmd(config, 75)}),
        "C low-res, converted": (True, {"x": 1, "y": 1, "t": 42 * SX},
                                 {"mmd_small": mmd(config, 3 / SX), "mmd_large": mmd(config, 75 / SX)}),
    }

    rows = []
    for recording in RECORDINGS:
        shift = load_trial(recording).rotation_period_us
        paths = event_paths(recording)
        real = read_span(paths["real"], BEGIN_US, BEGIN_US + 2 * shift + 10_000)
        v2ce = read_span(paths["v2ce"], BEGIN_US, BEGIN_US + shift + 10_000)
        starts = np.linspace(BEGIN_US, BEGIN_US + shift - WINDOW_US, N_WINDOWS).astype(np.int64)
        for s in starts:
            ref, nxt, sim = window(real, s), window(real, s + shift), window(v2ce, s)
            for version, (low, scales, mmds) in versions.items():
                r, n, v = (down(ref), down(nxt), down(sim)) if low else (ref, nxt, sim)
                a = to_points(r, int(s), scales)
                for label, other, origin in (("floor", n, s + shift), ("v2ce", v, s)):
                    b = to_points(other, int(origin), scales)
                    row = {"recording": recording, "window_start_us": int(s), "version": version,
                           "comparison": label, "n_real": len(ref), "n_v2ce": len(sim),
                           "swd": measure(swd, swd_settings, a, b)[0]}
                    for key, (metric, settings) in mmds.items():
                        row[key] = measure(metric, settings, a, b)[1]
                    rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output / "per_window.csv", index=False)
    pd.set_option("display.width", 200)
    for recording in RECORDINGS:
        block = table[(table.recording == recording) & (table.comparison == "v2ce")]
        wide = block.pivot(index="window_start_us", columns="version", values="swd")
        wide["C x 2.77"] = wide["C low-res, converted"] * SX
        print(f"\n=== {recording.upper()}: SWD real vs V2CE per window ===")
        print(wide.round(2).to_string())
        medians = (table[table.recording == recording]
                   .groupby(["version", "comparison"])[["swd", "mmd_small", "mmd_large"]].median())
        print("Median of the windows (MMD = signed squared estimate):")
        print(medians.round(5).to_string())
    print(f"\nWrote {output / 'per_window.csv'}")


if __name__ == "__main__":
    main()
