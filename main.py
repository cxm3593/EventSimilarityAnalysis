"""Event data distance analysis pipeline entry point.

Each comparison mode (baseline_comparison, all_to_all_comparison, ...) is a
``ComparisonStrategy`` registered by name, mirroring the metric registry in
``event_analysis_toolbox.metrics`` and the modifier registry in
``event_analysis_toolbox.event_modifiers``. Adding a new comparison mode means
adding a new ``ComparisonStrategy`` subclass, not touching ``main()``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml  # pyright: ignore[reportMissingModuleSource]

from event_data_toolbox.event_data_manager import EventDataManager
from event_analysis_toolbox.all_to_all_comparison import (
    all_to_all_comparison,
    plot_all_to_all_comparison,
    save_all_to_all_comparison_results,
)
from event_analysis_toolbox.baseline_comparison import (
    baseline_comparison,
    plot_baseline_comparison,
    save_baseline_comparison_results,
)
from event_analysis_toolbox.comparison_common import prepare_run_dir
from event_analysis_toolbox.event_modifiers import ModifierPipeline, build_pipelines
from event_analysis_toolbox.metrics import BaseMetric, get_metric


LOGGER = logging.getLogger(__name__)

# A window's worth of event records: an h5py Dataset or structured numpy array
# with (at least) "t", "x", "y" fields. Left as Any since both duck-type the
# same indexing/slicing interface.
EventArray = Any

# The dict payload returned by a comparison strategy's `run()` (and consumed
# by its `save()`/`plot()`). Kept as a plain dict since its shape differs by
# strategy; see baseline_comparison()/all_to_all_comparison() for the exact keys.
ComparisonResult = dict[str, Any]

DEFAULT_WINDOW_SCHEMES: list[dict[str, Any]] = [{"name": "consecutive", "stride": None}]
DEFAULT_COMPARISON_MODES: list[str] = ["baseline_comparison"]

# Legacy config names (windowed / mds) map to the current comparison strategy names.
LEGACY_COMPARISON_MODE_ALIASES: dict[str, str] = {
    "windowed": "baseline_comparison",
    "mds": "all_to_all_comparison",
}


@dataclass(frozen=True)
class WindowScheme:
    """One named window-sampling scheme: how far apart comparison windows start."""

    name: str
    stride: int | None  # microseconds between window starts; None = baseline width

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> WindowScheme:
        return cls(name=spec.get("name") or "scheme", stride=spec.get("stride"))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Fully resolved, typed settings for one pipeline run.

    Built once from ``config.yaml`` by :meth:`from_yaml` so every downstream
    function takes this single object instead of a long list of keyword
    arguments.
    """

    run_output_dir: Path
    real_temporal_offset: int
    v2e_temporal_offset: int
    metric: BaseMetric
    metric_kwargs: dict[str, Any]
    window_schemes: list[WindowScheme]
    comparison_modes: list[str]
    real_data_path: Path
    v2e_data_path: Path
    feature_names: list[str] | None
    feature_scales: dict[str, float] | list[float] | None
    modifier_pipelines: list[ModifierPipeline]
    sensor: dict[str, int] | None
    random_generator: np.random.Generator
    random_seed: int | None
    baseline_start_us: int
    baseline_end_us: int
    n_real_windows: int
    n_v2e_windows: int
    all_to_all_settings: dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> PipelineConfig:
        with open(config_path, "r") as config_file:
            config_dict: dict[str, Any] = yaml.safe_load(config_file)

        metric = get_metric(config_dict.get("metric", "mmd"))
        run_output_dir = prepare_run_dir(
            config_dict.get("output_dir", "output"), metric.name, config_dict,
        )

        requested_mode_names = (
            config_dict.get("comparison_modes") or config_dict.get("analysis_modes")
        )
        comparison_modes = _normalize_comparison_mode_names(
            requested_mode_names or DEFAULT_COMPARISON_MODES
        )
        _validate_comparison_mode_names(comparison_modes)

        window_scheme_specs = config_dict.get("window_schemes") or DEFAULT_WINDOW_SCHEMES
        window_schemes = [WindowScheme.from_dict(spec) for spec in window_scheme_specs]

        windows_section = config_dict.get("windows") or {}
        random_seed = config_dict.get("seed")

        return cls(
            run_output_dir=run_output_dir,
            real_temporal_offset=config_dict.get("real_temporal_offset"), 
            v2e_temporal_offset=config_dict.get("v2e_temporal_offset"),
            metric=metric,
            metric_kwargs=metric.build_kwargs(config_dict),
            window_schemes=window_schemes,
            comparison_modes=comparison_modes,
            real_data_path=Path(config_dict["real_data_path"]),
            v2e_data_path=Path(config_dict["v2e_data_path"]),
            feature_names=config_dict.get("feature_names"),
            feature_scales=config_dict.get("feature_scales"),
            modifier_pipelines=build_pipelines(config_dict.get("modifiers")),
            sensor=config_dict.get("sensor"),
            random_generator=np.random.default_rng(random_seed),
            random_seed=random_seed,
            baseline_start_us=int(config_dict["baseline_start"]),
            baseline_end_us=int(config_dict["baseline_end"]),
            n_real_windows=int(windows_section.get("n_real_windows", 9)),
            n_v2e_windows=int(windows_section.get("n_v2e_windows", 10)),
            all_to_all_settings=config_dict.get("all_to_all") or config_dict.get("mds") or {},
        )


def _normalize_comparison_mode_names(mode_names: list[str]) -> list[str]:
    """Map legacy comparison-mode aliases (windowed/mds) to current strategy names."""
    return [LEGACY_COMPARISON_MODE_ALIASES.get(name, name) for name in mode_names]


def _validate_comparison_mode_names(mode_names: list[str]) -> None:
    unknown_names = sorted(set(mode_names) - set(_STRATEGIES_BY_MODE))
    if unknown_names:
        raise ValueError(
            f"Unsupported comparison mode(s): {unknown_names}. "
            f"Available modes: {sorted(_STRATEGIES_BY_MODE)}"
        )


# --------------------------------------------------------------------------
# Comparison strategies
# --------------------------------------------------------------------------


class ComparisonStrategy(ABC):
    """Runs one comparison mode (e.g. baseline-vs-many or all-to-all) for a scheme."""

    mode_name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        real_events: EventArray,
        v2e_events: EventArray,
        scheme: WindowScheme,
        config: PipelineConfig,
    ) -> ComparisonResult:
        """Execute the comparison for one window scheme and return its result."""

    @abstractmethod
    def save(self, result: ComparisonResult, output_dir: Path) -> dict[str, str]:
        """Persist CSV/YAML/plot artifacts and return the written file paths."""

    @abstractmethod
    def plot(self, result: ComparisonResult, config: PipelineConfig, *, show: bool = True) -> None:
        """Render a plot for one result."""


_STRATEGIES_BY_MODE: dict[str, ComparisonStrategy] = {}


def register_strategy(strategy_cls: type[ComparisonStrategy]) -> type[ComparisonStrategy]:
    """Class decorator that registers a ``ComparisonStrategy`` under its ``mode_name``."""
    _STRATEGIES_BY_MODE[strategy_cls.mode_name] = strategy_cls()
    return strategy_cls


def _log_window_distance(label: str, window: dict[str, Any]) -> None:
    LOGGER.debug(
        "  %-14s [%9d, %9d] us  n=%7d  distance=%.6f",
        label, window["start"], window["end"], window["n_events"], window["distance"],
    )


@register_strategy
class BaselineComparisonStrategy(ComparisonStrategy):
    """One fixed baseline window compared against many sliding real/v2e windows."""

    mode_name = "baseline_comparison"

    def run(self, real_events, v2e_events, scheme, config) -> ComparisonResult:
        LOGGER.info(
            "Baseline comparison [%s] scheme=%s stride=%s",
            config.metric.name, scheme.name, scheme.stride,
        )

        result = baseline_comparison(
            real_data=real_events,
            v2e_data=v2e_events,
            baseline_start=config.baseline_start_us,
            baseline_end=config.baseline_end_us,
            n_real_windows=config.n_real_windows,
            n_v2e_windows=config.n_v2e_windows,
            stride=scheme.stride,
            metric=config.metric,
            metric_kwargs=config.metric_kwargs,
            feature_names=config.feature_names,
            feature_scales=config.feature_scales,
            modifier_pipelines=config.modifier_pipelines,
            sensor=config.sensor,
            rng=config.random_generator,
            seed=config.random_seed,
            name=scheme.name,
        )

        baseline_window = result["baseline"]
        LOGGER.debug(
            "Baseline real window [%s, %s] us: %s events",
            baseline_window["start"], baseline_window["end"], baseline_window["n_events"],
        )
        for window in result["real_windows"]:
            _log_window_distance("real", window)
        for window in result["v2e_windows"]:
            _log_window_distance("v2e", window)
        for variant_name, windows in result.get("modified_real_windows", {}).items():
            for window in windows:
                _log_window_distance(f"real[{variant_name}]", window)

        return result

    def save(self, result, output_dir: Path) -> dict[str, str]:
        paths = save_baseline_comparison_results(
            result, output_dir, run_name="", file_prefix=result.get("name") or "scheme",
        )
        LOGGER.info("Saved baseline comparison results to: %s", paths["dir"])
        return paths

    def plot(self, result, config, *, show: bool = True) -> None:
        plot_baseline_comparison(result, show=show)


@register_strategy
class AllToAllComparisonStrategy(ComparisonStrategy):
    """Pairwise distances between every window, with an optional MDS layout."""

    mode_name = "all_to_all_comparison"

    def run(self, real_events, v2e_events, scheme, config) -> ComparisonResult:
        visualizer_name = config.all_to_all_settings.get("visualizer", "mds")
        visualizer_kwargs = {
            key: value
            for key, value in config.all_to_all_settings.items()
            if key not in ("visualizer", "annotate")
        }

        LOGGER.info(
            "All-to-all comparison [%s] visualizer=%s scheme=%s stride=%s",
            config.metric.name, visualizer_name, scheme.name, scheme.stride,
        )

        result = all_to_all_comparison(
            real_data=real_events,
            v2e_data=v2e_events,
            baseline_start=config.baseline_start_us,
            baseline_end=config.baseline_end_us,
            n_real_windows=config.n_real_windows,
            n_v2e_windows=config.n_v2e_windows,
            stride=scheme.stride,
            metric=config.metric,
            metric_kwargs=config.metric_kwargs,
            feature_names=config.feature_names,
            feature_scales=config.feature_scales,
            visualizer=visualizer_name,
            visualizer_kwargs=visualizer_kwargs,
            name=scheme.name,
        )

        LOGGER.debug(
            "All-to-all comparison used %d windows (%d skipped).",
            len(result["windows"]), len(result.get("skipped_windows", [])),
        )
        return result

    def save(self, result, output_dir: Path) -> dict[str, str]:
        paths = save_all_to_all_comparison_results(
            result, output_dir, run_name="", file_prefix=result.get("name") or "scheme",
        )
        LOGGER.info("Saved all-to-all comparison results to: %s", paths["dir"])
        return paths

    def plot(self, result, config, *, show: bool = True) -> None:
        if not result.get("visualizer"):
            return
        annotate = bool(config.all_to_all_settings.get("annotate", False))
        plot_all_to_all_comparison(result, show=show, annotate=annotate)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_event_streams(config: PipelineConfig) -> tuple[EventArray, EventArray]:
    """Load the real and v2e event streams referenced by ``config.yaml``.

    HDF5 datasets are opened lazily, so this only reads metadata up front.
    """
    event_data_manager = EventDataManager()

    real_events = event_data_manager.load_event_data_h5(
        config.real_data_path, dataset_name="events", data_key="real_data",
    )
    LOGGER.info(
        "Real data loaded: %s events, duration %s us",
        real_events.shape[0], real_events["t"].max(),
    )

    v2e_events = event_data_manager.load_event_data_h5(
        config.v2e_data_path, dataset_name="events", data_key="v2e_data",
    )
    LOGGER.info(
        "V2E data loaded: %s events, duration %s us",
        v2e_events.shape[0], v2e_events["t"].max(),
    )

    return real_events, v2e_events


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run_comparisons(
    real_events: EventArray,
    v2e_events: EventArray,
    config: PipelineConfig,
) -> dict[str, list[ComparisonResult]]:
    """Run every configured comparison mode for every window scheme, saving as it goes."""
    results_by_mode: dict[str, list[ComparisonResult]] = defaultdict(list)
    for scheme in config.window_schemes:
        for mode_name in config.comparison_modes:
            strategy = _STRATEGIES_BY_MODE[mode_name]
            result = strategy.run(real_events, v2e_events, scheme, config)
            strategy.save(result, config.run_output_dir)
            results_by_mode[mode_name].append(result)
    return results_by_mode


def render_result_plots(
    results_by_mode: dict[str, list[ComparisonResult]],
    config: PipelineConfig,
) -> None:
    """Plot every collected result, after all comparisons have finished."""
    for mode_name, results in results_by_mode.items():
        strategy = _STRATEGIES_BY_MODE[mode_name]
        for result in results:
            strategy.plot(result, config, show=True)


def configure_logging() -> None:
    """Attach a single console handler: INFO+ by default, DEBUG for per-window detail."""
    LOGGER.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(console_handler)


##########################################
# Main function for the analysis pipeline.
##########################################
def main() -> None:
    configure_logging()
    LOGGER.info("Event Data Distance Analysis Pipeline starting...")

    config = PipelineConfig.from_yaml()
    LOGGER.info("Run output directory: %s", config.run_output_dir.resolve())
    for line in config.metric.describe_settings(config.metric_kwargs):
        LOGGER.info(line)

    # Load data
    real_events, v2e_events = load_event_streams(config)

    results_by_mode = run_comparisons(real_events, v2e_events, config)
    render_result_plots(results_by_mode, config)


if __name__ == "__main__":
    main()
