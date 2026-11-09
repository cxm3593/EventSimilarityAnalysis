"""Is V2CE's gap in where its events fall, or in their timing within the frame interval?

For F1 and F5, three versions of the floor, v2e and V2CE comparisons:

  5 ms, space only     (x, y) with time dropped
  5 ms, space-time     (x, y, t/42), the standard comparison
  33.3 ms, space-time  one whole frame interval, aligned to frame boundaries

If V2CE placed events correctly but timed them wrongly within each interval, the first and
third versions would bring it much closer to real. Reference is real rotation k; the floor
is real rotation k+1 at the same phase. MMD columns are the signed squared estimate; large
windows are thinned to 20,000 events for MMD only, which does not bias the estimate.
"""

import numpy as np
import pandas as pd

from _shared import event_paths, load_trial, read_span, start, window
from common import build_metrics, load_config, measure, to_points

RECORDINGS = ["f1", "f5"]
METRICS = ["mmd_rbf15", "mmd_rbf75", "swd"]
BEGIN_US = 1_000_000
FRAME_US = 1e6 / 30
MMD_CAP = 20_000
rng = np.random.default_rng(0)


def thin(points):
    if len(points) <= MMD_CAP:
        return points
    return points[np.sort(rng.choice(len(points), MMD_CAP, replace=False))]


def main():
    output = start("v2ce_space_and_frame_interval", __file__,
                   {"metrics": METRICS, "begin_us": BEGIN_US, "frame_rate": 30,
                    "mmd_event_cap": MMD_CAP, "seed": 0}, recordings=RECORDINGS)
    config = load_config()
    metrics = build_metrics(METRICS, config)
    scales = config["feature_scales"]

    rows = []
    for recording in RECORDINGS:
        shift = load_trial(recording).rotation_period_us
        paths = event_paths(recording)
        real = read_span(paths["real"], BEGIN_US, BEGIN_US + 2 * shift + 50_000)
        sims = {k: read_span(paths[k], BEGIN_US, BEGIN_US + shift + 50_000) for k in ("v2e", "v2ce")}
        five_ms = np.linspace(BEGIN_US, BEGIN_US + shift - 5000, 40).astype(np.int64)
        frames = ((np.ceil(BEGIN_US / FRAME_US) + np.linspace(0, shift / FRAME_US - 2, 12).round())
                  * FRAME_US).round().astype(np.int64)
        versions = [("5 ms, space only", 5000, five_ms, False),
                    ("5 ms, space-time", 5000, five_ms, True),
                    ("33.3 ms frame interval, space-time", int(FRAME_US), frames, True)]
        for label, length, starts, use_time in versions:
            for s in starts:
                s = int(s)
                ref = to_points(window(real, s, length), s, scales)
                sets = {"floor": to_points(window(real, s + shift, length), s + shift, scales),
                        "v2e": to_points(window(sims["v2e"], s, length), s, scales),
                        "v2ce": to_points(window(sims["v2ce"], s, length), s, scales)}
                for name, points in sets.items():
                    a, b = (ref, points) if use_time else (ref[:, :2], points[:, :2])
                    row = {"recording": recording, "version": label, "window_start_us": s,
                           "comparison": name}
                    for key, (metric, settings) in metrics.items():
                        if key.startswith("mmd"):
                            row[key] = measure(metric, settings, thin(a), thin(b))[1]
                        else:
                            row[key] = measure(metric, settings, a, b)[0]
                    rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output / "per_window.csv", index=False)
    pd.set_option("display.width", 200)
    for recording in RECORDINGS:
        print(f"\n=== {recording.upper()} (medians over windows; MMD = signed squared estimate) ===")
        block = table[table.recording == recording]
        for label in dict.fromkeys(block.version):
            part = block[block.version == label]
            print(f"-- {label} ({part.window_start_us.nunique()} windows) --")
            print(part.groupby("comparison")[METRICS].median().loc[["floor", "v2e", "v2ce"]]
                  .round(5).to_string())
    print(f"\nWrote {output / 'per_window.csv'}")


if __name__ == "__main__":
    main()
