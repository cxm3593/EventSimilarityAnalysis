"""Reuse validated temporal comparisons by exact physical windows, never interpolation.

The source must contain a complete trial/method. Reused rows retain their numerical
results; requested angle labels are rewritten only after matching integer time shifts.
Every output row records whether it was reused, and source files are hashed.
"""
from pathlib import Path
import hashlib

import pandas as pd
import yaml


def physical_key(row):
    """Angles can differ after rounding; integer shift and window identify the pair."""
    return tuple(row[k] for k in ("period_index", "window_index", "temporal_offset_us", "metric"))


class TemporalReuse:
    def __init__(self, results, source):
        if results.duplicated(["period_index", "window_index", "temporal_offset_us", "metric"]).any():
            raise ValueError("Duplicate physical temporal keys in reuse source")
        self.rows = {physical_key(row): row for row in results.to_dict("records")}
        self.source = str(Path(source).resolve())

    @classmethod
    def load(cls, source_root, output_dir, windows, sweep):
        """Validate original results/settings/windows before constructing a cache."""
        # Deferred imports avoid the shared evaluator / validator import cycle.
        from analysis.consolidate_perturbation import read_yaml, settings_signature, validate_trial
        target = read_yaml(output_dir / "run_config.yaml")
        trial = target["trial"]["name"]
        source_dir = source_root / trial / "temporal_offset"
        source = read_yaml(source_dir / "run_config.yaml")
        validate_trial(source_root, trial, "temporal_offset", settings_signature(target))
        if source["trial"] != target["trial"]:
            raise ValueError("Incompatible trial provenance for temporal reuse")
        for field in ("window_selection", "boundary_policy", "perturbation_definition", "mmd_squared"):
            if source[field] != target[field]:
                raise ValueError(f"Incompatible temporal convention: {field}")
        pd.testing.assert_frame_equal(pd.read_csv(source_dir / "windows.csv"), windows,
                                      check_dtype=False)
        period = target["trial"]["rotation_period_us"]
        shifts = [int(round(degrees * period / 360)) for degrees in sweep]
        if len(set(shifts)) != len(shifts):
            raise ValueError("Requested degree offsets round to duplicate physical shifts")
        results = pd.read_csv(source_dir / "results.csv", float_precision="round_trip")
        results = results[results.temporal_offset_us.isin(shifts)]
        record = {"source_directory": str(source_dir.resolve()),
                  "matching": "exact period/window/integer microsecond shift/metric; no interpolation",
                  "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                    for name in ("run_config.yaml", "windows.csv", "results.csv", "summary.csv")},
                  "expected_rows": len(windows) * len(sweep) * len(target["metric_settings"]),
                  "reused_rows": len(results)}
        record["new_rows"] = record["expected_rows"] - record["reused_rows"]
        (output_dir / "reuse_provenance.yaml").write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
        print(f"{trial}: {record['reused_rows']:,} reused, {record['new_rows']:,} new comparisons", flush=True)
        return cls(results, source_dir / "results.csv")

    def lookup(self, info, metric, n_a, n_b):
        """Recheck physical endpoints and live event counts when the pair is read."""
        old = self.rows.get(physical_key({**info, "metric": metric}))
        if old is None:
            return None
        for field in ("trial", "perturbation", "seed", "window_start_us", "window_end_us",
                      "comparison_start_us", "comparison_end_us"):
            if old[field] != info[field]:
                raise ValueError(f"Reused comparison has mismatched {field}")
        if old["n_events_a"] != n_a or old["n_events_b"] != n_b:
            raise ValueError("Reused comparison does not match live event counts")
        return {**old, **info, "metric": metric, "reused_from": self.source}

    def measure(self, reference, modified, metrics, info):
        """Run only metrics not present in the validated source, in normal order."""
        if __package__:
            from .perturbation_shared import measure_pair
        else:
            from perturbation_shared import measure_pair
        cached = {key: self.lookup(info, key, len(reference), len(modified)) for key in metrics}
        pending = {key: value for key, value in metrics.items() if cached[key] is None}
        computed = {row["metric"]: {**row, "reused_from": ""}
                    for row in measure_pair(reference, modified, pending, info)}
        return [cached[key] if cached[key] is not None else computed[key] for key in metrics]
