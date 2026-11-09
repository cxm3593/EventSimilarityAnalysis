"""How much of a window each MMD kernel relates: the share of event pairs it holds similar.

Read-only check for the Choice of Parameters subsection. For every complete 5 ms window
of the first rotation of each recording, it takes the real events as scaled points
(x, y, t/42), and measures the share of event pairs whose kernel similarity is above
one half (distance below d) and above 1/16 (distance below 2d), for the three kernels
d = 3, 15 and 75. It also records the mean similarity over all pairs, the within-set
term that enters MMD.

A window keeps at most --max-events events, drawn uniformly at random. Uniform
sampling keeps the distribution of pair distances, so the shares are unbiased.

Writes to output/kernel_extent/<stamp>/: per_window.csv, summary.csv, output.txt and
run_config.yaml, which records the rotation periods and where they came from.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from common import MIN_EVENTS, Stream, Trial, load_config  # noqa: E402
from perturbation_shared import period_windows, read_points  # noqa: E402

TRIALS = ROOT.parent / "EventCamCalib/output/trials"
RECORDINGS = ["f1", "f2", "f3", "f4", "f5"]
KERNELS_PX = [3, 15, 75]
WINDOW_US = 5000


def pair_statistics(points: np.ndarray, kernels, chunk: int = 1000) -> dict:
    """Shares of distinct pairs within d and 2d, and the mean similarity, per kernel."""
    n = len(points)
    within_d = {d: 0 for d in kernels}
    within_2d = {d: 0 for d in kernels}
    similarity = {d: 0.0 for d in kernels}
    for begin in range(0, n, chunk):
        block = points[begin:begin + chunk]
        squared = ((block[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        rows = np.arange(len(block))
        squared[rows, begin + rows] = np.inf  # leave out each event's pair with itself
        for d in kernels:
            within_d[d] += int((squared < d * d).sum())
            within_2d[d] += int((squared < 4 * d * d).sum())
            similarity[d] += float(np.power(0.5, squared / (d * d)).sum())
    pairs = n * (n - 1)
    return {d: {"share_above_half": within_d[d] / pairs,
                "share_above_sixteenth": within_2d[d] / pairs,
                "mean_similarity": similarity[d] / pairs} for d in kernels}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-events", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    scales = config["feature_scales"]
    rng = np.random.default_rng(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "output" / "kernel_extent" / stamp
    out.mkdir(parents=True)

    rows, periods = [], {}
    for recording in RECORDINGS:
        trial = Trial.load(TRIALS, f"optical_chopper_data_{recording}")
        periods[recording] = {"rotation_period_us": trial.rotation_period_us,
                              "source": trial.rotation_period_source}
        real = Stream(trial.real_path, "real")
        try:
            windows = period_windows(trial, real, 0, WINDOW_US)
            for window in windows.itertuples():
                if window.n_events < MIN_EVENTS:
                    continue
                points = read_points(real, window.window_start_us, window.window_end_us, scales)
                if len(points) > args.max_events:
                    points = points[rng.choice(len(points), args.max_events, replace=False)]
                for d, values in pair_statistics(points.astype(np.float64), KERNELS_PX).items():
                    rows.append({"recording": recording, "window_index": window.window_index,
                                 "n_events": window.n_events, "n_used": len(points),
                                 "kernel_px": d, **values})
        finally:
            real.close()
        print(f"{recording}: {len(windows)} windows")

    per_window = pd.DataFrame(rows)
    per_window.to_csv(out / "per_window.csv", index=False)
    columns = ["share_above_half", "share_above_sixteenth", "mean_similarity"]
    summary = (per_window.groupby(["recording", "kernel_px"])[columns]
               .quantile([0.05, 0.5, 0.95]).unstack())
    summary.columns = [f"{name}_q{int(q * 100):02d}" for name, q in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    lines = ["Share of event pairs per 5 ms window, real events, first rotation",
             "median over windows [5%, 95%]", ""]
    for (recording, d), group in per_window.groupby(["recording", "kernel_px"]):
        parts = [f"{recording} d={d:>2} px"]
        for name in columns:
            q = group[name].quantile([0.05, 0.5, 0.95]).to_numpy()
            parts.append(f"{name} {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]")
        lines.append("  ".join(parts))
    text = "\n".join(lines)
    print(text)
    (out / "output.txt").write_text(text + "\n", encoding="utf-8")

    run_config = {"script": "analysis/kernel_extent.py", "created": stamp,
                  "window_us": WINDOW_US, "rotation": 0, "kernels_px": KERNELS_PX,
                  "similarity": "0.5 ** (r / d) ** 2; above one half for r < d, above 1/16 for r < 2d",
                  "max_events_per_window": args.max_events, "seed": args.seed,
                  "feature_scales": scales, "source": "real", "rotation_periods": periods}
    (out / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False),
                                         encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
