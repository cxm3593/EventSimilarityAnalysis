"""Planning, comparison, checks and statistics of the neighbouring-turn noise floor."""
import numpy as np
import pandas as pd
import pytest

from experiments import noise_floor
from experiments.common import METRIC_KEYS, Trial, build_metrics, load_config


class MemoryStream:
    """A recording held in memory, with the reader interface the experiment uses."""

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
    return Trial("tiny", tmp_path, tmp_path / "real.h5", tmp_path / "v2e.h5", 250, "test fixture", (4, 3))


@pytest.fixture
def config():
    config = load_config()
    config["mmd"]["backend"] = "numpy"
    config["mmd"]["chunk_size"] = 64
    config["sliced_wasserstein"]["backend"] = "numpy"
    config["sliced_wasserstein"]["n_projections"] = 8
    return config


def test_pairs_are_turns_with_a_complete_neighbour(trial):
    # 0..1000 holds four complete turns of 250, so three of them have a next turn.
    assert noise_floor.neighbouring_pairs(trial, MemoryStream(np.arange(1001))) == 3
    assert noise_floor.neighbouring_pairs(trial, MemoryStream(np.arange(1000))) == 2
    assert noise_floor.neighbouring_pairs(trial, MemoryStream(np.arange(300))) == 0


def test_plan_covers_every_turn_but_the_last(trial):
    real = MemoryStream(np.arange(1001))
    plan = noise_floor.plan_windows(trial, real, 100)
    assert sorted(plan.period_index.unique()) == [0, 1, 2]
    assert plan.window_start_us.tolist() == [0, 100, 250, 350, 500, 600]
    assert plan.window_end_us.max() + trial.rotation_period_us <= real.t_end
    assert noise_floor.plan_windows(trial, real, 100, max_pairs=1).period_index.unique().tolist() == [0]
    with pytest.raises(ValueError, match="two complete turns"):
        noise_floor.plan_windows(trial, MemoryStream(np.arange(300)), 100)


def test_compare_measures_one_rotation_later_and_can_continue(trial, config, tmp_path):
    real = MemoryStream(np.arange(1001))
    metrics = build_metrics(METRIC_KEYS, config)
    windows = noise_floor.plan_windows(trial, real, 100)
    path = tmp_path / "results.csv"
    results = noise_floor.compare(trial, real, windows, config, metrics, 0, path)
    assert len(results) == len(windows) * len(METRIC_KEYS)
    assert (results.comparison_start_us == results.window_start_us + 250).all()
    assert all(noise_floor.check_recording(results, windows, trial, len(METRIC_KEYS)).values())

    # Drop one whole window and one metric of another: both are measured again, nothing else.
    damaged = results[~((results.period_index == 2) & (results.window_index == 1))]
    damaged = damaged[~((damaged.period_index == 1) & (damaged.window_index == 0)
                        & (damaged.metric == "swd"))]
    damaged.to_csv(path, index=False)
    assert len(noise_floor.completed_windows(path, len(METRIC_KEYS))) == len(windows) - 2
    again = noise_floor.compare(trial, real, windows, config, metrics, 0, path)
    keys = [*noise_floor.WINDOW_KEYS, "metric"]
    pd.testing.assert_frame_equal(again.sort_values(keys).reset_index(drop=True),
                                  results.sort_values(keys).reset_index(drop=True))


def test_checks_catch_duplicates_and_wrong_offsets(trial, config, tmp_path):
    real = MemoryStream(np.arange(1001))
    windows = noise_floor.plan_windows(trial, real, 100)
    results = noise_floor.compare(trial, real, windows, config, build_metrics(METRIC_KEYS, config), 0,
                                  tmp_path / "results.csv")
    doubled = pd.concat([results, results.iloc[:1]], ignore_index=True)
    checks = noise_floor.check_recording(doubled, windows, trial, len(METRIC_KEYS))
    assert not checks["no_duplicates"] and not checks["expected_comparisons"]
    shifted = results.assign(temporal_offset_us=251)
    assert not noise_floor.check_recording(shifted, windows, trial, len(METRIC_KEYS))["one_rotation_later"]


def test_statistics_report_median_and_mean_over_valid_rows():
    rows = pd.DataFrame({
        "trial": "tiny", "period_index": [0, 0, 0, 1, 1], "metric": "mmd_rbf03",
        "distance": [0.0, 0.0, 3.0, 4.0, 100.0], "distance_squared": [-1.0, -3.0, 9.0, 16.0, 1e4],
        "status": ["ok", "ok", "ok", "ok", "insufficient_events"],
        "n_events_a": 10, "n_events_b": 10})
    overall = noise_floor.statistics(rows, ["trial", "metric"]).iloc[0]
    assert overall.n_comparisons == 5 and overall.n_valid == 4
    assert overall.median_distance == 1.5 and overall.mean_distance == 1.75
    assert overall.mean_minus_median == 0.25 and overall.share_exactly_zero == 0.5
    # The signed estimate keeps the negative values that the distance clamps to zero.
    assert overall.mean_distance_squared == 5.25 and overall.median_distance_squared == 4.0
    pairs = noise_floor.statistics(rows, ["trial", "period_index", "metric"]).set_index("period_index")
    assert pairs.loc[0, "median_distance"] == 0.0 and pairs.loc[0, "mean_distance"] == 1.0
    assert pairs.loc[1, "n_valid"] == 1 and pairs.loc[1, "mean_distance"] == 4.0


def test_first_pair_is_compared_with_the_stored_result(tmp_path):
    stored = pd.DataFrame({"magnitude": [360.0, 360.0, 90.0], "period_index": 0, "window_index": [0, 1, 0],
                           "metric": "swd", "window_start_us": [0, 100, 0], "distance": [1.0, 2.0, 9.0]})
    folder = tmp_path / "tiny" / "temporal_offset"
    folder.mkdir(parents=True)
    stored.to_csv(folder / "results.csv", index=False)
    here = pd.DataFrame({"period_index": [0, 0, 1], "window_index": [0, 1, 0], "metric": "swd",
                         "window_start_us": [0, 100, 250], "distance": [1.0, 2.0, 5.0]})
    assert noise_floor.check_against_stored(here, "tiny", tmp_path)["status"] == "match"
    here.loc[1, "distance"] = 2.5
    assert noise_floor.check_against_stored(here, "tiny", tmp_path)["status"] == "MISMATCH"
    assert noise_floor.check_against_stored(here, "absent", tmp_path)["status"] == "skipped"
