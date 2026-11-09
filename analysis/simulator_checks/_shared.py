"""Shared paths and helpers for the read-only simulator checks.

Every check reads existing event files and results. Each writes only to
output/simulator/analysis/<check>/, leaving its printed output in output.txt and its
settings, including the rotation periods and where they came from, in run_config.yaml.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from common import Trial  # noqa: E402

TRIALS = ROOT.parent / "EventCamCalib/output/trials"
PERTURBATION = ROOT / "output/perturbation"
SPATIAL_RUN = PERTURBATION / "spatial_20260922_181016_100519"
SAMPLING_NOISE_RUN = PERTURBATION / "sampling_noise_20260921_210141_079777"
TEMPORAL_RUN = PERTURBATION / "temporal_20260923_175125_279651"
SIMULATOR = ROOT / "output/simulator"
RECORDINGS = ["f1", "f2", "f3", "f4", "f5"]
WINDOW_US = 5000


def trial_name(recording: str) -> str:
    return f"optical_chopper_data_{recording}"


def load_trial(recording: str) -> Trial:
    return Trial.load(TRIALS, trial_name(recording))


def event_paths(recording: str) -> dict:
    folder = TRIALS / trial_name(recording)
    return {"real": folder / "final_masked_real.h5",
            "v2e": folder / "final_masked_v2e.h5",
            "v2ce": folder / "final_masked_v2ce.h5"}


def comparison_runs() -> dict:
    """Recording -> folder of the real/v2e/V2CE comparison run the paper uses."""
    index = yaml.safe_load((SIMULATOR / "run_index.yaml").read_text(encoding="utf-8"))
    return {recording: Path(folder) for recording, folder in index.items()}


def index_of(dataset, timestamp_us: int) -> int:
    """First row with t >= timestamp_us, by bisection, without loading the column."""
    low, high = 0, dataset.shape[0]
    while low < high:
        middle = (low + high) // 2
        if int(dataset[middle]["t"]) < timestamp_us:
            low = middle + 1
        else:
            high = middle
    return low


def read_span(path: Path, start_us: int, end_us: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        events = handle["events"]
        return events[index_of(events, start_us):index_of(events, end_us)]


def window(events: np.ndarray, start_us: int, length_us: int = WINDOW_US) -> np.ndarray:
    low, high = np.searchsorted(events["t"], [start_us, start_us + length_us])
    return events[low:high].copy()


class _Tee:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def start(check: str, script: str, parameters: dict, recordings=RECORDINGS) -> Path:
    """Open the check's output folder, record its settings, and mirror prints to a file."""
    output = SIMULATOR / "analysis" / check
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    periods = {}
    for recording in recordings:
        trial = load_trial(recording)
        periods[recording] = {"rotation_period_us": trial.rotation_period_us,
                              "source": trial.rotation_period_source}
    record = {
        "check": check,
        "script": str(Path(script).resolve().relative_to(ROOT)).replace("\\", "/"),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "parameters": parameters,
        "rotation_periods": periods,
        "trials_dir": str(TRIALS),
    }
    with (output / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)
    sys.stdout = _Tee(output / "output.txt")
    return output
