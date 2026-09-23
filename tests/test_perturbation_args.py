from pathlib import Path
import argparse

import pytest

from experiments.perturbation import TRIALS_DIR, parse_args


def test_defaults_without_yaml():
    args = parse_args([])
    assert args.trials_dir == TRIALS_DIR
    assert args.trials == ["optical_chopper_data_f1"]


def test_saved_arguments_and_cli_precedence(tmp_path):
    cla = tmp_path / "arguments.yaml"
    cla.write_text("trials_dir: 'D:/trial data'\ntrial: optical_chopper_data_f2\n")
    args = parse_args(["--cla", str(cla)])
    assert args.trials_dir == "D:/trial data"
    assert args.trials == ["optical_chopper_data_f2"]
    for command in (["--trial", "optical_chopper_data_f3", "--cla", str(cla)],
                    ["--cla", str(cla), "--trial", "optical_chopper_data_f3"]):
        assert parse_args(command).trials == ["optical_chopper_data_f3"]


def test_partial_and_empty_yaml(tmp_path):
    cla = tmp_path / "arguments.yaml"
    cla.write_text("trial: optical_chopper_data_f4\n")
    assert parse_args(["--cla", str(cla)]).trials_dir == TRIALS_DIR
    cla.write_text("")
    assert parse_args(["--cla", str(cla)]).trials == ["optical_chopper_data_f1"]


@pytest.mark.parametrize("content", [
    "- trial\n", "phase: 3a\n", "trial: null\n", "trial: true\n",
    "trials_dir: []\n", "trial: [\n", "42: trial\n",
    "trials: []\n", "trials: [[f1]]\n", "trials: [null]\n",
    "trials_dir: [one, two]\n",
])
def test_invalid_yaml_is_an_argument_error(tmp_path, content):
    cla = tmp_path / "arguments.yaml"
    cla.write_text(content)
    with pytest.raises(SystemExit) as error:
        parse_args(["--cla", str(cla)])
    assert error.value.code == 2


def test_missing_yaml_is_an_argument_error(tmp_path):
    with pytest.raises(SystemExit) as error:
        parse_args(["--cla", str(tmp_path / "missing.yaml")])
    assert error.value.code == 2


def test_checked_in_argument_file():
    cla = Path(__file__).resolve().parents[1] / "experiments" / "perturbation_cla.yaml"
    args = parse_args(["--cla", str(cla)])
    assert args.trials == [f"optical_chopper_data_f{i}" for i in range(1, 6)]


def test_unknown_cli_argument_is_rejected():
    with pytest.raises(SystemExit) as error:
        parse_args(["--unknown"])
    assert error.value.code == 2


@pytest.mark.parametrize("value", ["42", "", "--literal-name", "a name with spaces"])
def test_yaml_values_follow_cli_string_semantics(tmp_path, value):
    cla = tmp_path / "arguments.yaml"
    cla.write_text(f"trial: '{value}'\n")
    assert parse_args(["--cla", str(cla)]).trials == parse_args([f"--trial={value}"]).trials


def test_trial_list_and_cli_replacement(tmp_path):
    cla = tmp_path / "arguments.yaml"
    cla.write_text("trials: [f1, 'trial with spaces', f3]\n")
    assert parse_args(["--cla", str(cla)]).trials == ["f1", "trial with spaces", "f3"]
    for command in (["--trials", "f4", "f5", "--cla", str(cla)],
                    ["--cla", str(cla), "--trials", "f4", "f5"]):
        assert parse_args(command).trials == ["f4", "f5"]
    assert parse_args(["--cla", str(cla), "--trial", "f2"]).trials == ["f2"]


def test_main_closes_each_pair_before_loading_next(monkeypatch, tmp_path):
    from experiments import perturbation
    from types import SimpleNamespace
    import pandas as pd

    calls = []

    class FakeStream:
        t_start, t_end = 0, 100

        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(("close", self.name))

    def load_pair(directory, name):
        calls.append(("load", name))
        return SimpleNamespace(name=name), FakeStream(f"{name}/real"), FakeStream(f"{name}/v2e")

    monkeypatch.setattr(perturbation.sys, "argv", ["perturbation.py", "--trials", "f1", "f2",
                                                  "--output-dir", str(tmp_path / "output")])
    monkeypatch.setattr(perturbation, "load_data", load_pair)
    monkeypatch.setattr(perturbation, "build_metrics", lambda *args: {})
    monkeypatch.setattr(perturbation, "write_run_config", lambda *args: None)
    monkeypatch.setattr(perturbation, "plot_comparison", lambda *args: None)
    monkeypatch.setattr(perturbation.METHODS["spatial_offset"], "evaluate", lambda *args: pd.DataFrame())
    perturbation.main()
    assert calls == [
        ("load", "f1"), ("close", "f1/real"), ("close", "f1/v2e"),
        ("load", "f2"), ("close", "f2/real"), ("close", "f2/v2e"),
    ]


def test_load_data_returns_open_readers(tmp_path, monkeypatch):
    import h5py
    import numpy as np
    from types import SimpleNamespace
    from experiments import perturbation

    trial = SimpleNamespace(real_path=tmp_path / "real.h5", v2e_path=tmp_path / "v2e.h5")
    events = np.zeros(3, dtype=[("x", "i4"), ("y", "i4"), ("p", "i1"), ("t", "i8")])
    events["t"] = [10, 20, 30]
    for path in (trial.real_path, trial.v2e_path):
        with h5py.File(path, "w") as handle:
            handle.create_dataset("events", data=events)
    monkeypatch.setattr(perturbation.Trial, "load", lambda directory, name: trial)

    metadata, real, v2e = perturbation.load_data(tmp_path, "f1")
    try:
        assert metadata is trial
        np.testing.assert_array_equal(real.slice(10, 30), events[:2])
        np.testing.assert_array_equal(v2e.slice(20, 31), events[1:])
    finally:
        real.close()
        v2e.close()
    assert not real._file.id.valid
    assert not v2e._file.id.valid


def test_load_data_closes_real_if_v2e_fails(monkeypatch):
    from types import SimpleNamespace
    from experiments import perturbation
    from experiments import perturbation_shared

    closed = []
    trial = SimpleNamespace(real_path="real.h5", v2e_path="missing.h5")

    def open_stream(path, label):
        if label == "v2e":
            raise FileNotFoundError(path)
        return SimpleNamespace(close=lambda: closed.append(label))

    monkeypatch.setattr(perturbation.Trial, "load", lambda directory, name: trial)
    monkeypatch.setattr(perturbation_shared, "Stream", open_stream)
    with pytest.raises(FileNotFoundError):
        perturbation.load_data("trials", "f1")
    assert closed == ["real"]


def test_new_argument_is_validated_by_argparse(tmp_path, monkeypatch, capsys):
    # Adding an argument must not require changes to the YAML loader.
    original_init = argparse.ArgumentParser.__init__

    def init_with_count(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_argument("--count", type=int, choices=[1, 2])
        self.add_argument("--seeds", type=int, nargs="+")

    monkeypatch.setattr(argparse.ArgumentParser, "__init__", init_with_count)
    cla = tmp_path / "arguments.yaml"
    cla.write_text("count: 2\n")
    assert parse_args(["--cla", str(cla)]).count == 2
    assert parse_args(["--cla", str(cla), "--count", "1"]).count == 1
    cla.write_text("seeds: [0, 1, 2]\n")
    assert parse_args(["--cla", str(cla)]).seeds == [0, 1, 2]

    for value, message in [("wrong", "invalid int value"), (3, "invalid choice")]:
        cla.write_text(f"count: {value}\n")
        with pytest.raises(SystemExit) as error:
            parse_args(["--cla", str(cla)])
        assert error.value.code == 2
        assert message in capsys.readouterr().err
