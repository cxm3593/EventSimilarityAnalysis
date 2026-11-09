"""Validate all five temporal reuse sources and live counts before overnight launch."""
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.perturbation import parse_args
from experiments.common import METRIC_KEYS, build_metrics, load_config
from experiments.perturbation_methods import TemporalOffset
from experiments.perturbation_shared import load_data, plan_windows, write_run_config
from experiments.temporal_reuse import TemporalReuse


def main():
    args = parse_args(["--cla", "experiments/temporal_degrees_cla.yaml"])
    config = load_config()
    metrics = build_metrics(METRIC_KEYS, config)
    root = ROOT / "output/perturbation" / datetime.now().strftime("degree_preflight_%Y%m%d_%H%M%S")
    root.mkdir(exist_ok=False)
    audits = []
    for name in args.trials:
        trial, real, v2e = load_data(args.trials_dir, name)
        try:
            resolved = TemporalOffset.resolved_args(trial, args)
            windows = plan_windows(trial, real, resolved)
            TemporalOffset.validate_coverage(trial, real, windows, resolved.temporal_offset_sweep)
            directory = root / name
            directory.mkdir()
            write_run_config(directory, resolved, config, metrics, TemporalOffset, trial, windows)
            cache = TemporalReuse.load(Path(args.temporal_reuse_run), directory, windows,
                                       resolved.temporal_offset_sweep)
            rows = pd.DataFrame(cache.rows.values())
            # Recheck every reused endpoint and event count against the current recording.
            planned = windows.set_index(["period_index", "window_index"])
            keys = pd.MultiIndex.from_frame(rows[["period_index", "window_index"]])
            starts = planned.loc[keys, "window_start_us"].to_numpy()
            ends = planned.loc[keys, "window_end_us"].to_numpy()
            np.testing.assert_array_equal(rows.window_start_us, starts)
            np.testing.assert_array_equal(rows.window_end_us, ends)
            np.testing.assert_array_equal(rows.comparison_start_us, starts + rows.temporal_offset_us)
            np.testing.assert_array_equal(rows.comparison_end_us, ends + rows.temporal_offset_us)
            np.testing.assert_array_equal(rows.n_events_a, np.searchsorted(real.t, ends) - np.searchsorted(real.t, starts))
            np.testing.assert_array_equal(rows.n_events_b, np.searchsorted(real.t, rows.comparison_end_us) - np.searchsorted(real.t, rows.comparison_start_us))
            audit = yaml.safe_load((directory / "reuse_provenance.yaml").read_text())
            # Determine whether the stopped expanded run offers any additional exact keys.
            partial = ROOT / "output/perturbation/20260922_181016_100519" / name / "temporal_offset/results.csv"
            additional = 0
            if partial.exists():
                available = pd.read_csv(partial)
                missing_shifts = set(resolved.resolved_temporal_offsets_us) - set(rows.temporal_offset_us)
                additional = int(available.temporal_offset_us.isin(missing_shifts).sum())
            audit.update(trial=name, additional_matching_rows_in_stopped_run=additional)
            audits.append(audit)
            print(f"{name}: all reuse endpoints/counts verified; stopped run adds {additional} rows", flush=True)
        finally:
            real.close()
            v2e.close()
    (root / "audit.yaml").write_text(yaml.safe_dump(audits, sort_keys=False), encoding="utf-8")
    print(f"PREFLIGHT PASSED: {root}", flush=True)
    print({key: sum(a[key] for a in audits) for key in ("expected_rows", "reused_rows", "new_rows")})


if __name__ == "__main__":
    main()
