"""How much of the real-vs-V2CE distance comes from V2CE's resolution and event count?

Reference: the real window at rotation k (F1 and F5, 40 windows). Every comparison set is
built from independent events, so MMD carries no overlap bias:

  floor              real window at the same phase one rotation later (k+1)
  grid               (k+1) snapped to V2CE's 462x260 grid and mapped back, as generate.py does
  count              (k+1) randomly reduced to V2CE's event count in that window
  grid+count         both
  floor_both_grid    both real windows snapped to the grid (the floor at V2CE's resolution)
  v2ce               V2CE at the same phase as k
  v2ce_fill          V2CE with each event spread over its 2.77 px footprint, then rounded
  v2ce_vs_grid_real  V2CE against the reference snapped to the grid (real scaled down)

MMD columns are the signed squared estimate; SWD is the distance.
"""

import numpy as np
import pandas as pd

from _shared import WINDOW_US, event_paths, load_trial, read_span, start, window
from common import build_metrics, load_config, measure, to_points

RECORDINGS = ["f1", "f5"]
METRICS = ["mmd_rbf03", "mmd_rbf15", "mmd_rbf75", "swd"]
N_WINDOWS = 40
BEGIN_US = 1_000_000
WIDTH, HEIGHT, V2CE_WIDTH, V2CE_HEIGHT = 1280, 720, 462, 260
SX, SY = WIDTH / V2CE_WIDTH, HEIGHT / V2CE_HEIGHT
rng = np.random.default_rng(0)


def to_grid(events):
    out = events.copy()
    gx = np.rint((events["x"] + 0.5) / SX - 0.5)
    gy = np.rint((events["y"] + 0.5) / SY - 0.5)
    out["x"] = np.rint((gx + 0.5) * SX - 0.5).astype(events["x"].dtype)
    out["y"] = np.rint((gy + 0.5) * SY - 0.5).astype(events["y"].dtype)
    return out


def fill(events):
    out = events.copy()
    x = events["x"].astype(float) + rng.uniform(-SX / 2, SX / 2, len(events))
    y = events["y"].astype(float) + rng.uniform(-SY / 2, SY / 2, len(events))
    out["x"] = np.clip(np.rint(x), 0, WIDTH - 1).astype(events["x"].dtype)
    out["y"] = np.clip(np.rint(y), 0, HEIGHT - 1).astype(events["y"].dtype)
    return out


def reduce(events, n):
    if n >= len(events):
        return events
    return events[np.sort(rng.choice(len(events), n, replace=False))]


def main():
    output = start("v2ce_decomposition", __file__,
                   {"metrics": METRICS, "window_us": WINDOW_US, "n_windows": N_WINDOWS,
                    "begin_us": BEGIN_US, "v2ce_grid": [V2CE_WIDTH, V2CE_HEIGHT], "seed": 0},
                   recordings=RECORDINGS)
    config = load_config()
    metrics = build_metrics(METRICS, config)
    scales = config["feature_scales"]

    rows = []
    for recording in RECORDINGS:
        shift = load_trial(recording).rotation_period_us
        paths = event_paths(recording)
        real = read_span(paths["real"], BEGIN_US, BEGIN_US + 2 * shift + 10_000)
        v2ce = read_span(paths["v2ce"], BEGIN_US, BEGIN_US + shift + 10_000)
        starts = np.linspace(BEGIN_US, BEGIN_US + shift - WINDOW_US, N_WINDOWS).astype(np.int64)

        for s in starts:
            ref, nxt, sim = window(real, s), window(real, s + shift), window(v2ce, s)
            # name -> (reference, comparison, comparison origin)
            sets = {
                "floor": (ref, nxt, s + shift),
                "grid": (ref, to_grid(nxt), s + shift),
                "count": (ref, reduce(nxt, len(sim)), s + shift),
                "grid+count": (ref, reduce(to_grid(nxt), len(sim)), s + shift),
                "floor_both_grid": (to_grid(ref), to_grid(nxt), s + shift),
                "v2ce": (ref, sim, s),
                "v2ce_fill": (ref, fill(sim), s),
                "v2ce_vs_grid_real": (to_grid(ref), sim, s),
            }
            for name, (reference, other, origin) in sets.items():
                a = to_points(reference, int(s), scales)
                b = to_points(other, int(origin), scales)
                row = {"recording": recording, "window_start_us": int(s), "comparison": name,
                       "n_real": len(ref), "n_v2ce": len(sim)}
                for key, (metric, settings) in metrics.items():
                    distance, squared = measure(metric, settings, a, b)
                    row[key] = squared if key.startswith("mmd") else distance
                rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output / "per_window.csv", index=False)
    order = list(dict.fromkeys(table.comparison))
    for recording in RECORDINGS:
        block = table[table.recording == recording]
        print(f"\n=== {recording.upper()}: median events per window real "
              f"{int(block.n_real.median())}, V2CE {int(block.n_v2ce.median())} ===")
        print("median over windows; MMD columns are the signed squared estimate")
        print(block.groupby("comparison")[METRICS].median().loc[order].round(5).to_string())
    print(f"\nWrote {output / 'per_window.csv'}")


if __name__ == "__main__":
    main()
