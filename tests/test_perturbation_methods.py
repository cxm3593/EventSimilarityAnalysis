"""Physical semantics, complete-window selection and full output regression tests."""
import ast
import inspect

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import perturbation as runner
from experiments import perturbation_shared as shared
from experiments import perturbation_methods as methods
from experiments.common import METRIC_KEYS, Trial, build_metrics, load_config


class MemoryStream:
    def __init__(self, times):
        self.t = np.asarray(times, dtype=np.int64)
        self.t_start, self.t_end = int(self.t[0]), int(self.t[-1])
        self.events = np.zeros(len(times), dtype=[("x", "i4"), ("y", "i4"), ("p", "i1"), ("t", "i8")])
        self.events["t"] = self.t
        self.events["x"] = np.arange(len(times)) % 13
        self.events["y"] = np.arange(len(times)) % 7

    def slice(self, start, end):
        low, high = np.searchsorted(self.t, [start, end])
        return self.events[low:high]


@pytest.fixture
def trial(tmp_path):
    return Trial("tiny", tmp_path, tmp_path / "real.h5", tmp_path / "v2e.h5",
                 250, "test fixture", (4, 3))


@pytest.fixture
def args(tmp_path):
    return runner.parse_args(["--temporal-window", "100", "--periods", "2", "--output-dir", str(tmp_path)])


@pytest.fixture
def config():
    config = load_config()
    config["mmd"]["backend"] = "numpy"
    config["mmd"]["chunk_size"] = 64
    config["sliced_wasserstein"]["backend"] = "numpy"
    config["sliced_wasserstein"]["n_projections"] = 8
    return config


@pytest.fixture
def window(trial, args, config):
    real = MemoryStream(np.arange(1001))
    row = next(shared.plan_windows(trial, real, args).itertuples())
    return shared.prepare_window(row, trial, real, config)


def test_complete_windows_and_coverage(trial, args):
    plan = shared.plan_windows(trial, MemoryStream(np.arange(501)), args)
    assert plan.window_start_us.tolist() == [0, 100, 250, 350]
    assert plan.window_end_us.tolist() == [100, 200, 350, 450]
    assert plan.n_events.tolist() == [100] * 4
    with pytest.raises(ValueError, match="not fully covered"):
        shared.plan_windows(trial, MemoryStream(np.arange(450)), args)
    args.temporal_window = 300
    with pytest.raises(ValueError, match="no complete windows"):
        shared.plan_windows(trial, MemoryStream(np.arange(501)), args)


@pytest.mark.parametrize("method,identity", [
    (methods.SpatialOffset, 0), (methods.SpatialScaling, 1), (methods.SpatialJitter, 0),
    (methods.Subsampling, 1), (methods.UniformNoise, 0), (methods.TemporalOffset, 0),
])
def test_identities_and_reference_immutability(window, method, identity):
    before = window.points.copy()
    result = method.apply(window, identity, np.random.default_rng(0))
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(window.points, before)
    assert not np.shares_memory(result, window.points)


def test_offset_units_no_clipping(window):
    window.scales = {"x": 2, "y": 3, "t": 42}
    result = methods.SpatialOffset.apply(window, 3000, None)
    np.testing.assert_allclose(result - window.points, np.tile([1500, 1000, 0], (100, 1)))
    assert result[:, 0].min() > 1280


def test_scaling_uses_calibrated_centre_and_preserves_time(window):
    window.scales = {"x": 2, "y": 3, "t": 42}
    result = methods.SpatialScaling.apply(window, 1.1, None)
    centre = np.array([2, 1])
    np.testing.assert_allclose(result[:, :2], centre + 1.1 * (window.points[:, :2] - centre))
    np.testing.assert_array_equal(result[:, 2], window.points[:, 2])


def test_jitter_bounds_reproducibility_and_common_directions(window):
    small = methods.SpatialJitter.apply(window, 0.5, shared.window_rng(0, "jitter", window))
    large = methods.SpatialJitter.apply(window, 8, shared.window_rng(0, "jitter", window))
    delta = large - window.points
    assert np.abs(delta[:, :2]).max() <= 8
    assert np.any((delta[:, :2] % 1) != 0)
    np.testing.assert_allclose(delta[:, :2], 16 * (small - window.points)[:, :2], atol=1e-12)
    np.testing.assert_array_equal(large[:, 2], window.points[:, 2])


def test_subsampling_distinct_nested_subsets(window):
    small = methods.Subsampling.apply(window, 0.25, shared.window_rng(0, "subsampling", window))
    large = methods.Subsampling.apply(window, 0.5, shared.window_rng(0, "subsampling", window))
    assert len(small) == 25 and len(large) == 50
    assert len(set(small[:, 2])) == 25
    assert set(small[:, 2]).issubset(set(large[:, 2]))
    assert len(window.points) == 100


def test_noise_bounds_fraction_and_prefix(window):
    result = methods.UniformNoise.apply(window, 0.5, shared.window_rng(0, "noise", window))
    smaller = methods.UniformNoise.apply(window, 0.25, shared.window_rng(0, "noise", window))
    assert len(result) == 150
    np.testing.assert_array_equal(result[:100], window.points)
    np.testing.assert_array_equal(result[:125], smaller)
    noise = result[100:]
    assert (noise >= 0).all()
    assert (noise[:, 0] < window.sensor["width"]).all()
    assert (noise[:, 1] < window.sensor["height"]).all()
    assert (noise[:, 2] < 100 / 42).all()
    assert methods.UniformNoise.details(window, 0.5, result)["actual_noise_fraction"] == pytest.approx(1 / 3)


def test_temporal_offset_reads_later_data_not_timestamp_translation(window, args):
    shifted = methods.TemporalOffset.apply(window, 360, None)
    expected = shared.read_points(window.real, 250, 350, window.scales)
    np.testing.assert_array_equal(shifted, expected)
    assert not np.array_equal(shifted[:, :2], window.points[:, :2])
    np.testing.assert_array_equal(shifted[:, 2], window.points[:, 2])
    details = methods.TemporalOffset.details(window, 360, shifted)
    assert details["temporal_offset_us"] == 250
    assert details["realized_phase_degrees"] == 360
    plan = shared.plan_windows(window.trial, window.real, args)
    methods.TemporalOffset.validate_coverage(window.trial, window.real, plan, [0, 360])
    with pytest.raises(ValueError, match="coverage"):
        methods.TemporalOffset.validate_coverage(window.trial, window.real, plan, [0, 1440])
    with pytest.raises(ValueError, match="before"):
        methods.TemporalOffset.validate_coverage(window.trial, window.real, plan, [-360, 0])


@pytest.mark.parametrize("method,sweep", [
    (methods.SpatialOffset, [0, 3]), (methods.SpatialScaling, [1, 1.1]),
    (methods.SpatialJitter, [0, 2]), (methods.Subsampling, [1, 0.5]),
    (methods.UniformNoise, [0, 0.5]), (methods.TemporalOffset, [0, 45]),
])
def test_all_metrics_outputs_and_means(trial, args, config, tmp_path, method, sweep):
    setattr(args, method.sweep_argument, sweep)
    metrics = build_metrics(METRIC_KEYS, config)
    output = tmp_path / method.name
    summary = method.evaluate(trial, MemoryStream(np.arange(1001)), args, config, metrics, output)
    results = pd.read_csv(output / "results.csv")
    assert len(results) == 4 * 2 * 5
    assert len(summary) == 2 * 5
    assert results.status.eq("ok").all()
    assert results.n_events_a.eq(100).all()
    np.testing.assert_allclose(results[results.magnitude == sweep[0]].distance, 0, atol=1e-7)
    expected = results.groupby(["trial", "perturbation", "magnitude", "metric"]).distance.mean()
    actual = summary.set_index(["trial", "perturbation", "magnitude", "metric"]).mean_distance.sort_index()
    np.testing.assert_allclose(actual, expected)
    assert summary.n_valid.eq(4).all()
    assert results.distance_squared[results.metric.str.startswith("mmd")].notna().all()
    record = yaml.safe_load((output / "run_config.yaml").read_text(encoding="utf-8"))
    assert record["trial"]["rotation_period_us"] == 250
    assert record["metric_settings"]["mmd_rbf03"]["biased"] is False
    assert record["windows"]["total"] == 4
    assert (output / "point_cloud.html").exists()
    assert (output / f"{method.name}.html").exists()


def test_sparse_windows_and_intentional_thinning_exclusions(trial, args, config, tmp_path):
    args.subsampling_sweep = [1, 0.25]
    summary = methods.Subsampling.evaluate(trial, MemoryStream(np.arange(1001)), args, config,
                                           build_metrics(METRIC_KEYS, config), tmp_path / "thin")
    assert summary[summary.magnitude == 0.25].n_valid.eq(0).all()
    assert summary[summary.magnitude == 0.25].n_excluded.eq(4).all()
    args.spatial_offset_sweep = [0, 3]
    summary = methods.SpatialOffset.evaluate(trial, MemoryStream(np.arange(0, 1001, 10)), args,
                                             config, {key: (None, {}) for key in METRIC_KEYS}, tmp_path / "sparse")
    assert summary.mean_distance.isna().all()
    assert summary.n_excluded.eq(4).all()
    assert not (tmp_path / "sparse" / "point_cloud.html").exists()


def test_plot_structure_and_display_only_subsampling(window, monkeypatch, tmp_path):
    figures = []
    monkeypatch.setattr(shared.go.Figure, "write_html", lambda self, *a, **kw: figures.append(self))
    summary = pd.DataFrame({"trial": ["f1"] * 5 + ["f2"] * 5, "metric": list(METRIC_KEYS) * 2,
                            "magnitude": [3] * 10, "mean_distance": [1] * 10,
                            "n_valid": [4] * 10, "n_excluded": [0] * 10})
    shared.plot_comparison(summary, methods.SpatialOffset, tmp_path / "summary.html")
    assert len(figures[0].data) == 10
    assert len({trace.yaxis for trace in figures[0].data}) == 5
    assert len({trace.line.color for trace in figures[0].data if trace.name == "f1"}) == 1
    window.points = np.ones((2100, 3))
    shifted = methods.SpatialOffset.apply(window, 3, None)
    shared.plot_point_cloud(window, shifted, methods.SpatialOffset, 3, tmp_path / "cloud.html")
    reference, changed = figures[1].data
    assert len(reference.x) == 2000 and len(window.points) == 2100
    np.testing.assert_allclose(np.asarray(changed.x) - reference.x, 3)
    np.testing.assert_array_equal(reference.z, changed.z)


def test_full_main_sequence_and_status(trial, config, monkeypatch, tmp_path):
    class TestStream(MemoryStream):
        def close(self):
            pass

    cli = ["perturbation.py", "--trials", "tiny", "--output-dir", str(tmp_path / "run"),
           "--evaluations", *methods.METHODS, "--temporal-window", "100",
           "--spatial-offset-sweep", "0", "3", "--spatial-scaling-sweep", "1", "1.1",
           "--spatial-jitter-sweep", "0", "2", "--subsampling-sweep", "1", "0.5",
           "--uniform-noise-sweep", "0", "0.1", "--temporal-offset-sweep", "0", "45"]
    monkeypatch.setattr(runner.sys, "argv", cli)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "load_data", lambda *a: (trial, TestStream(np.arange(1001)), TestStream(np.arange(1001))))
    runner.main()
    output = next((tmp_path / "run").iterdir())
    assert yaml.safe_load((output / "run_state.yaml").read_text())["status"] == "completed"
    summary = pd.read_csv(output / "summary.csv")
    assert set(summary.perturbation) == set(methods.METHODS)
    assert len(summary) == 6 * 2 * 5


def test_failed_main_records_error_and_closes_streams(trial, config, monkeypatch, tmp_path):
    closed = []
    class TestStream(MemoryStream):
        def close(self):
            closed.append(True)

    def fail(*args):
        raise RuntimeError("intentional test failure")

    monkeypatch.setattr(runner.sys, "argv", ["perturbation.py", "--output-dir", str(tmp_path / "run")])
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "load_data", lambda *a: (trial, TestStream(np.arange(501)), TestStream(np.arange(501))))
    monkeypatch.setattr(methods.SpatialOffset, "evaluate", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        runner.main()
    assert len(closed) == 2
    output = next((tmp_path / "run").iterdir())
    state = yaml.safe_load((output / "run_state.yaml").read_text())
    assert state["status"] == "failed" and "intentional test failure" in state["error"]


@pytest.mark.parametrize("arguments", [
    ["--temporal-window", "0"], ["--periods", "0"], ["--first-period", "-1"],
    ["--spatial-offset-sweep", "nan"], ["--spatial-scaling-sweep", "0"],
    ["--spatial-jitter-sweep", "-1"], ["--subsampling-sweep", "1.1"],
    ["--uniform-noise-sweep", "-1"], ["--trials", "f1", "f1"],
])
def test_invalid_physical_settings(arguments):
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)


@pytest.mark.parametrize("module", [runner, shared, methods])
def test_no_nested_for_loops(module):
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            assert not any(isinstance(child, (ast.For, ast.AsyncFor))
                           for statement in node.body for child in ast.walk(statement))


@pytest.mark.parametrize("period", [1961623, 1000183, 664309, 497822, 401164, 250])
def test_time_grid_roundtrip_and_anchors(period, trial):
    args = runner.parse_args(["--temporal-offset-grid", "time"])
    trial.rotation_period_us = period
    original = args.temporal_offset_sweep.copy()
    resolved = methods.TemporalOffset.resolved_args(trial, args)
    times = resolved.resolved_temporal_offsets_us
    assert times == sorted(set(times))
    assert times[0] == 0 and times[-1] == period
    assert set(round(period * f) for f in (0, .25, .5, .75, 1)) <= set(times)
    assert set(t for t in args.temporal_near_us if t <= period) <= set(times)
    assert set(range(60000, period + 1, 20000)) <= set(times)
    assert args.temporal_offset_sweep == original
    assert not hasattr(args, "resolved_temporal_offsets_us")


def test_degree_grid_backward_compatibility(trial):
    args = runner.parse_args(["--temporal-offset-sweep", "0", "45", "360"])
    resolved = methods.TemporalOffset.resolved_args(trial, args)
    assert resolved.temporal_offset_sweep == [0, 45, 360]
    assert resolved.resolved_temporal_offsets_us == [0, 31, 250]


def test_time_grid_saved_provenance(trial, args, config, tmp_path):
    args.temporal_offset_grid = "time"
    args.temporal_near_us = [0, 100]
    args.temporal_far_step_us = 100
    output = tmp_path / "time_grid"
    methods.TemporalOffset.evaluate(trial, MemoryStream(np.arange(1001)), args,
                                   config, build_metrics(METRIC_KEYS, config), output)
    saved = yaml.safe_load((output / "run_config.yaml").read_text())
    results = pd.read_csv(output / "results.csv")
    assert saved["arguments"]["resolved_temporal_offsets_us"] == [0, 62, 100, 125, 188, 200, 250]
    assert set(results.temporal_offset_us) == {0, 62, 100, 125, 188, 200, 250}
    np.testing.assert_allclose(results.realized_phase_degrees, 360 * results.temporal_offset_us / 250)


@pytest.mark.parametrize("arguments", [
    ["--temporal-far-step-us", "0"], ["--temporal-near-us", "-1"],
])
def test_invalid_time_grid(arguments):
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)
