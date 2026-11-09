"""Is V2CE aligned in time with the real recording?

For F1 and F5, each real 5 ms window [s, s+5ms) is compared with the simulator window
[s+d, s+d+5ms) for d from -20 to +20 ms; v2e, which was calibrated against real, is the
control. Also reports how each simulator spreads its timestamps within the 33.3 ms frame
interval, and how many distinct timestamps each source uses in 1 ms.
"""

import numpy as np
import pandas as pd

from _shared import WINDOW_US, event_paths, load_trial, read_span, start, window
from common import build_metrics, load_config, measure, to_points

RECORDINGS = ["f1", "f5"]
SHIFTS_MS = [-20, -15, -10, -7.5, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7.5, 10, 15, 20]
N_WINDOWS = 40
BEGIN_US = 1_000_000
FRAME_US = 1e6 / 30


def main():
    output = start("v2ce_alignment", __file__,
                   {"metric": "swd", "window_us": WINDOW_US, "n_windows": N_WINDOWS,
                    "begin_us": BEGIN_US, "shifts_ms": SHIFTS_MS, "frame_rate": 30},
                   recordings=RECORDINGS)
    config = load_config()
    (swd, settings), = build_metrics(["swd"], config).values()
    scales = config["feature_scales"]

    curves, spread = [], []
    for recording in RECORDINGS:
        period = load_trial(recording).rotation_period_us
        end = BEGIN_US + period
        margin = 25_000
        data = {name: read_span(path, BEGIN_US - margin, end + margin)
                for name, path in event_paths(recording).items()}
        starts = np.linspace(BEGIN_US, end - WINDOW_US, N_WINDOWS).astype(np.int64)

        print(f"\n=== {recording.upper()}: rotation {period / 1e3:.0f} ms, {len(starts)} windows ===")
        print("shift_ms " + " ".join(f"{s:>7}" for s in SHIFTS_MS))
        for simulator in ("v2e", "v2ce"):
            medians = []
            for d in SHIFTS_MS:
                shift = int(round(d * 1000))
                values = []
                for s in starts:
                    a = to_points(window(data["real"], s), int(s), scales)
                    b = to_points(window(data[simulator], s + shift), int(s + shift), scales)
                    values.append(measure(swd, settings, a, b)[0])
                medians.append(float(np.nanmedian(values)))
                curves.append({"recording": recording, "simulator": simulator, "shift_ms": d,
                               "median_swd": medians[-1]})
            best = SHIFTS_MS[int(np.nanargmin(medians))]
            print(f"{simulator:<8} " + " ".join(f"{m:7.2f}" for m in medians) + f"   best={best} ms")

        for name in ("real", "v2e", "v2ce"):
            t = data[name]["t"].astype(np.int64)
            one_ms = t[(t >= BEGIN_US) & (t < BEGIN_US + 1000)]
            row = {"recording": recording, "source": name,
                   "distinct_timestamps_in_1ms": int(len(np.unique(one_ms)))}
            if name != "real":
                share, _ = np.histogram((t % FRAME_US) / FRAME_US, bins=10, range=(0, 1))
                share = share / share.sum()
                row.update({f"tenth_{k + 1}": float(v) for k, v in enumerate(share)})
                print(f"{name:<5} share of events per tenth of the frame interval: "
                      + " ".join(f"{v:.3f}" for v in share))
            print(f"{name:<5} distinct timestamps in 1 ms: {row['distinct_timestamps_in_1ms']}")
            spread.append(row)

    pd.DataFrame(curves).to_csv(output / "shift_curves.csv", index=False)
    pd.DataFrame(spread).to_csv(output / "timestamp_spread.csv", index=False)
    print(f"\nWrote {output / 'shift_curves.csv'}\nWrote {output / 'timestamp_spread.csv'}")


if __name__ == "__main__":
    main()
