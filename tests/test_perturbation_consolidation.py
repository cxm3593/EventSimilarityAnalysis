"""Post-processing tests: explicit completed sources, validation and safe publication."""
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
import yaml

from analysis import consolidate_perturbation as merge


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def fixture_method(root, method, base):
    """Create two magnitudes and one window, with exact known summary values."""
    sweep = [1, .5] if method == "subsampling" else [1, 1.2] if method == "spatial_scaling" else [0, .5]
    directory = root / "f1" / method
    directory.mkdir(parents=True)
    arguments = {**base["arguments"], merge.METHODS[method].sweep_argument: sweep}
    config = {**base, "arguments": arguments, "experiment": method,
              "trial": {"name": "f1", "rotation_period_us": 10000}}
    write_yaml(directory / "run_config.yaml", config)
    pd.DataFrame([{"period_index": 0, "window_index": 0, "n_events": 100}]).to_csv(directory / "windows.csv", index=False)
    rows = []
    for magnitude, metric in product(sweep, merge.METRIC_KEYS):
        count = round(100 * magnitude) if method == "subsampling" else round(100 * (1 + magnitude)) if method == "uniform_noise" else 100
        identity = magnitude == sweep[0]
        row = {"trial": "f1", "perturbation": method, "magnitude": magnitude, "metric": metric,
               "period_index": 0, "window_index": 0, "n_events_a": 100, "n_events_b": count,
               "status": "ok", "distance": 0 if identity else .1,
               "distance_squared": -.01 if identity else .01}
        if method == "temporal_offset":
            row.update(temporal_offset_us=round(magnitude * 10000 / 360),
                       realized_phase_degrees=round(magnitude * 10000 / 360) * 360 / 10000)
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(directory / "results.csv", index=False)
    merge.summarize_distances(results).to_csv(directory / "summary.csv", index=False)
    (directory / "point_cloud.html").write_text("example")
    (directory / f"{method}.html").write_text("trial figure")
    (root / f"{method}.html").write_text("combined figure")


@pytest.fixture
def runs(tmp_path):
    base = {"feature_scales": {"x": 1, "y": 1, "t": 42}, "sensor": {"width": 1280, "height": 720},
            "metric_settings": {key: {} for key in merge.METRIC_KEYS},
            "sampling": "all events", "minimum_events_per_side": 50,
            "arguments": {"trials": ["f1"], "temporal_window": 5000, "first_period": 0, "periods": 1, "seed": 0}}
    updated, retained = tmp_path / "updated", tmp_path / "retained"
    for root in (updated, retained):
        root.mkdir()
        write_yaml(root / "run_state.yaml", {"status": "completed"})
        write_yaml(root / "run_config.yaml", base)
    for method in merge.METHODS:
        fixture_method(updated if method in merge.UPDATED else retained, method, base)
    return updated, retained, tmp_path / "current"


def test_publish_and_archive_without_changing_sources(runs):
    updated, retained, output = runs
    original = (updated / "f1/spatial_offset/results.csv").read_bytes()
    output.mkdir()
    (output / "keep.txt").write_text("previous current")
    merge.consolidate(updated, retained, output)
    assert (updated / "f1/spatial_offset/results.csv").read_bytes() == original
    assert next(output.parent.glob("current_archive_*/keep.txt")).read_text() == "previous current"
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 60 and set(summary.perturbation) == set(merge.METHODS)
    assert set(summary.loc[summary.perturbation == "subsampling", "source_run"]) == {str(retained.resolve())}
    assert summary.loc[summary.perturbation == "temporal_offset", "temporal_offset_us"].notna().all()
    manifest = merge.read_yaml(output / "manifest.yaml")
    assert manifest["total_comparisons"] == 60 and manifest["status"] == "completed"


def test_reject_incomplete_without_publishing(runs):
    updated, retained, output = runs
    write_yaml(updated / "run_state.yaml", {"status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        merge.consolidate(updated, retained, output)
    assert not output.exists()


def test_reject_duplicate_rows(runs):
    updated, retained, output = runs
    path = updated / "f1/spatial_offset/results.csv"
    values = pd.read_csv(path)
    pd.concat([values, values.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        merge.consolidate(updated, retained, output)
    assert not output.exists()


def test_reject_incompatible_settings(runs):
    updated, retained, output = runs
    path = retained / "run_config.yaml"
    config = merge.read_yaml(path)
    config["feature_scales"]["t"] = 99
    write_yaml(path, config)
    with pytest.raises(ValueError, match="incompatible"):
        merge.consolidate(updated, retained, output)


def test_reject_source_as_output(runs):
    updated, retained, _ = runs
    with pytest.raises(ValueError, match="separate"):
        merge.consolidate(updated, retained, updated)
