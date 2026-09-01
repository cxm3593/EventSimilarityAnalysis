import numpy as np

from experiments.common import cut_windows, split_in_half


FEATURE_SCALES = {"x": 1, "y": 1, "t": 42}


def make_events(stop_us):
    events = np.zeros(
        stop_us,
        dtype=[("x", "i4"), ("y", "i4"), ("p", "i1"), ("t", "i8")],
    )
    events["x"] = np.arange(stop_us)
    events["t"] = np.arange(stop_us)
    return events


def test_cut_windows_omits_incomplete_tail():
    windows, starts = cut_windows(make_events(23), 0, 23, 10, FEATURE_SCALES)

    np.testing.assert_array_equal(starts, [0, 10])
    assert [len(window) for window in windows] == [10, 10]


def test_cut_windows_keeps_last_complete_window():
    windows, starts = cut_windows(make_events(20), 0, 20, 10, FEATURE_SCALES)

    np.testing.assert_array_equal(starts, [0, 10])
    assert [len(window) for window in windows] == [10, 10]


def test_split_in_half_accepts_rng_and_balances_each_window():
    points = np.column_stack([np.arange(12), np.zeros(12), np.zeros(12)])
    first, second = split_in_half(points, np.random.default_rng(7))

    assert len(first) == len(second) == 6
    np.testing.assert_array_equal(
        np.sort(np.concatenate([first[:, 0], second[:, 0]])), np.arange(12)
    )
