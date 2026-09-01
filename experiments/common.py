"""Shared pieces for the metric evaluation tests.

Every test script imports from here so that segment boundaries, window construction,
sample sizes and metric settings are identical across tests. Anything that differs
between tests belongs in the test script, not in this file.
"""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from event_analysis_toolbox.feature_preprocessing import window_features  # noqa: E402
from event_analysis_toolbox.metrics import get_metric  # noqa: E402

# Events per side of a comparison, by window length. Set from the median half-window
# event count of the real stream, which is the smaller of the two sources.
DEFAULT_EVENTS_PER_COMPARISON = {1000: 750, 2500: 1900, 5000: 3800, 10000: 7500}
MIN_EVENTS = 50


def events_per_comparison(window_us, setting="none"):
    """How many events each side of a comparison is cut down to.

    "none" means no cap at all: every window keeps the events it actually holds. The
    chopper's density varies several-fold across a rotation, and capping imposes a
    uniformity the instrument does not have. In Test 1 the two halves are equal-size by
    construction anyway, so a cap changes nothing within a comparison there.

    "auto" restores the old table, which was set from the median half-window count.
    """
    if setting is None or str(setting).lower() in ("none", "off", "0"):
        return None
    if str(setting).lower() == "auto":
        return DEFAULT_EVENTS_PER_COMPARISON.get(int(window_us))
    return int(setting)

METRIC_KEYS = ("mmd_rbf03", "mmd_rbf15", "mmd_rbf75", "swd", "chamfer")


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One recording: file paths, the rotation period, and how it was obtained."""

    name: str
    directory: Path
    real_path: Path
    v2e_path: Path
    rotation_period_us: int
    rotation_period_source: str
    ellipse_centre: tuple
    calibration: dict = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, trials_dir: Path, name: str) -> "Trial":
        directory = Path(trials_dir) / name
        with open(directory / "result.yaml", "r") as handle:
            result = yaml.safe_load(handle)

        measured = directory / "rotation_period.yaml"
        if measured.exists():
            with open(measured, "r") as handle:
                record = yaml.safe_load(handle)
            period = int(round(record["rotation_period_us"]))
            source = f"rotation_period.yaml ({record.get('method', 'measured')})"
        else:
            # Fall back to four quarter-periods. The calibration's own
            # rotation_period_us is half a rotation and must not be used.
            quarter = result["temporal_fine_calibration"]["symmetry_period_us"]
            period = int(round(quarter * 4))
            source = "4 x symmetry_period_us from result.yaml (fallback)"

        centre = result["spatial_fine_calibration"]["real_ellipse"]["center"]
        return cls(
            name=name,
            directory=directory,
            real_path=directory / "final_masked_real.h5",
            v2e_path=directory / "final_masked_v2e.h5",
            rotation_period_us=period,
            rotation_period_source=source,
            ellipse_centre=(float(centre[0]), float(centre[1])),
            calibration=result.get("temporal_fine_calibration", {}),
        )

    def period_bounds(self, index: int, start_us: int) -> tuple[int, int]:
        begin = start_us + index * self.rotation_period_us
        return begin, begin + self.rotation_period_us


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

class Stream:
    """One event file. Timestamps are held in memory; everything else stays on disk."""

    def __init__(self, path: Path, label: str):
        self.path = Path(path)
        self.label = label
        self._file = h5py.File(self.path, "r")
        self.dataset = self._file["events"]
        self.t = np.asarray(self.dataset["t"], dtype=np.int64)

    def __len__(self):
        return len(self.t)

    @property
    def t_start(self) -> int:
        return int(self.t[0])

    @property
    def t_end(self) -> int:
        return int(self.t[-1])

    def slice(self, start_us: int, end_us: int) -> np.ndarray:
        """Events with start_us <= t < end_us, as a structured array."""
        low, high = np.searchsorted(self.t, [start_us, end_us])
        return self.dataset[int(low):int(high)]

    def close(self):
        self._file.close()


def load_streams(trial: Trial, sources=("real", "v2e")) -> dict:
    paths = {"real": trial.real_path, "v2e": trial.v2e_path}
    return {name: Stream(paths[name], name) for name in sources}


# ---------------------------------------------------------------------------
# Windows and sampling
# ---------------------------------------------------------------------------

def to_points(events: np.ndarray, origin_us: int, feature_scales: dict) -> np.ndarray:
    """Structured events to the (N, 3) array a metric consumes."""
    points, _ = window_features(events, feature_scales=feature_scales, time_origin=origin_us)
    return points


def cut_windows(events: np.ndarray, start_us: int, end_us: int, window_us: int,
                feature_scales: dict) -> list:
    """Split one stretch into complete, consecutive fixed-length windows.

    A trailing interval shorter than ``window_us`` is deliberately omitted. Treating
    that tail as a full window changes the null sample size and creates an artificial
    distance spike at every segment boundary.
    """
    start_us, end_us, window_us = int(start_us), int(end_us), int(window_us)
    if window_us <= 0:
        raise ValueError("window_us must be positive")
    starts = np.arange(start_us, end_us - window_us + 1, window_us, dtype=np.int64)
    times = events["t"]
    low = np.searchsorted(times, starts)
    high = np.searchsorted(times, starts + window_us)
    return [to_points(events[a:b], origin, feature_scales)
            for a, b, origin in zip(low, high, starts)], starts


def subsample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Reduce to exactly `count` points, or return unchanged if already smaller.

    Every comparison in every test uses the same count so that values are on one
    scale - MMD in particular changes with sample size.
    """
    if count is None or len(points) <= count:
        return points
    return points[rng.choice(len(points), size=count, replace=False)]


def split_in_half(events: np.ndarray, seed) -> tuple[np.ndarray, np.ndarray]:
    """Randomly divide events into two halves of equal size, sorted by time.

    Both halves come from the same instants, so any difference between them is
    sampling chance and nothing else.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    order = rng.permutation(len(events))
    half = len(events) // 2
    first, second = events[np.sort(order[:half])], events[np.sort(order[half:2 * half])]
    return first, second


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_metrics(keys, config: dict) -> dict:
    """Build the requested metrics and their settings from config.yaml.

    MMD always runs unbiased: the biased estimator responds to sample size, which
    is exactly what the subsample check in test 3 is meant to detect.
    """
    def mmd_settings(max_distance):
        section = dict(config["mmd"])
        section["biased"] = False
        section["kernels"] = [{"rbf_kernel_max_distance": max_distance,
                               "rbf_kernel_target_similarity": 0.5}]
        return "mmd", {"mmd": section}

    recipes = {
        "mmd_rbf03": lambda: mmd_settings(3),
        "mmd_rbf15": lambda: mmd_settings(15),
        "mmd_rbf75": lambda: mmd_settings(75),
        "swd": lambda: ("sliced_wasserstein",
                        {"sliced_wasserstein": dict(config["sliced_wasserstein"])}),
        "chamfer": lambda: ("chamfer", {"chamfer": dict(config["chamfer"])}),
    }

    built = {}
    for key in keys:
        name, section = recipes[key]()
        metric = get_metric(name)
        settings = metric.build_kwargs(section)
        if metric.supports_inner_progress:
            settings["progress"] = False
        built[key] = (metric, settings)
    return built


def measure(metric, settings, points_a: np.ndarray, points_b: np.ndarray):
    """One comparison. Returns (distance, secondary).

    `secondary` is the squared value where the metric provides one. For MMD it is the
    unclamped squared estimate, which matters here: the unbiased estimator can go
    negative when there is no real difference, and `distance` clamps that to zero. About
    half the null values come out as exactly zero as a result, so a spread computed on
    `distance` is not meaningful. Build the null on `secondary` for MMD, and use
    quantiles rather than mean and spread for every metric.
    """
    if len(points_a) < MIN_EVENTS or len(points_b) < MIN_EVENTS:
        return float("nan"), float("nan")
    result = metric.compute(points_a, points_b, **settings)
    secondary = metric.secondary_value(result)
    return float(result.value), (float(secondary) if secondary is not None else float("nan"))


def distance(metric, settings, points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Just the distance. Prefer `measure` when the squared value is wanted too."""
    return measure(metric, settings, points_a, points_b)[0]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class RunOutput:
    """One run's folder, with its configuration written before any result."""

    def __init__(self, test_name: str, trial: Trial, tag: str = "",
                 existing: str = None):
        if existing:
            # Resuming: reuse the interrupted run's folder rather than opening a new
            # one, so its partial results.csv is the file that gets continued.
            self.directory = Path(existing)
            if not self.directory.is_absolute():
                self.directory = (PROJECT_ROOT / "output" / "tests" / trial.name
                                  / test_name / existing)
            if not self.directory.exists():
                raise SystemExit(f"cannot resume: {self.directory} does not exist")
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{stamp}_{tag}" if tag else stamp
            self.directory = (PROJECT_ROOT / "output" / "tests" / trial.name
                              / test_name / run_id)
        self.figures = self.directory / "figures"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.test_name = test_name
        self.trial = trial

    def write_config(self, parameters: dict, metric_keys, config: dict) -> Path:
        record = {
            "test": self.test_name,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "trial": {
                "name": self.trial.name,
                "directory": str(self.trial.directory),
                "real_path": str(self.trial.real_path),
                "v2e_path": str(self.trial.v2e_path),
                "rotation_period_us": self.trial.rotation_period_us,
                "rotation_period_source": self.trial.rotation_period_source,
                "ellipse_centre": list(self.trial.ellipse_centre),
            },
            "parameters": _plain(parameters),
            "metrics": list(metric_keys),
            "feature_scales": config.get("feature_scales"),
            "mmd_section": config.get("mmd"),
            "sliced_wasserstein_section": config.get("sliced_wasserstein"),
            "chamfer_section": config.get("chamfer"),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "git_commit": _git_commit(),
            },
        }
        path = self.directory / "run_config.yaml"
        with open(path, "w") as handle:
            yaml.safe_dump(record, handle, sort_keys=False, default_flow_style=False)
        return path

    def csv(self, name: str) -> Path:
        return self.directory / name

    def figure(self, name: str) -> Path:
        return self.figures / name


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Incremental results
# ---------------------------------------------------------------------------

class ResultWriter:
    """Append rows to a CSV as they are produced.

    A long run that dies partway must not lose everything, so rows reach disk as
    soon as they exist rather than at the end. `key_fields` names the columns that
    identify a comparison; on resume, every combination already on disk is skipped.
    """

    def __init__(self, path: Path, key_fields=(), flush_every: int = 200,
                 resume: bool = False):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self.flush_every = int(flush_every)
        self.done = set()
        self._handle = None
        self._writer = None
        self._since_flush = 0
        self.n_rows = 0

        if resume and self.path.exists():
            self._recover()
        elif self.path.exists():
            self.path.unlink()

    def _recover(self):
        """Drop any half-written final line, then remember which keys are present."""
        with open(self.path, "rb") as handle:
            data = handle.read()
        cut = data.rfind(b"\n")
        if cut >= 0 and cut + 1 != len(data):
            with open(self.path, "wb") as handle:
                handle.write(data[:cut + 1])
        try:
            existing = pd.read_csv(self.path)
        except Exception:  # noqa: BLE001 - an unreadable partial file starts over
            self.path.unlink()
            return
        self.n_rows = len(existing)
        if self.key_fields and all(f in existing.columns for f in self.key_fields):
            self.done = set(map(tuple, existing[list(self.key_fields)].values.tolist()))

    def already_done(self, *key) -> bool:
        return tuple(key) in self.done

    def append(self, row: dict):
        if self._writer is None:
            new = not self.path.exists() or self.path.stat().st_size == 0
            self._handle = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row.keys()))
            if new:
                self._writer.writeheader()
        self._writer.writerow(row)
        self.n_rows += 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def extend(self, rows):
        for row in rows:
            self.append(row)

    def flush(self):
        if self._handle is not None:
            self._handle.flush()
            self._since_flush = 0

    def close(self):
        self.flush()
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def frame(self):
        """Read the finished file back, for summarising and plotting."""
        self.close()
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def select_indices(n_total: int, cap) -> np.ndarray:
    """At most `cap` indices, spread evenly over the whole range.

    Capping a run must not restrict it to the start of the rotation - the chopper's
    phase varies across a turn, so a prefix is not a fair sample of it.
    """
    if cap is None or cap >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, int(cap)).round().astype(int))
