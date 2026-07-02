import datetime
from pathlib import Path

import numpy as np

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
from event_analysis_toolbox.event_modifiers import build_pipelines
from event_analysis_toolbox.metrics import BaseMetric, get_metric
import yaml  # pyright: ignore[reportMissingModuleSource]


_DEFAULT_SCHEMES = [{"name": "consecutive", "stride": None}]
_DEFAULT_COMPARISON_MODES = ["baseline_comparison"]
_AVAILABLE_COMPARISON_MODES = {"baseline_comparison", "all_to_all_comparison"}
# Legacy config names (windowed / mds) map to the new comparison strategies.
_LEGACY_COMPARISON_MODE_ALIASES = {
    "windowed": "baseline_comparison",
    "mds": "all_to_all_comparison",
}


def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def _normalize_comparison_modes(modes):
    normalized = []
    for mode in modes:
        normalized.append(_LEGACY_COMPARISON_MODE_ALIASES.get(mode, mode))
    return normalized


def _run_baseline_comparison(
    real_data,
    v2e_data,
    scheme,
    *,
    baseline_start: int,
    baseline_end: int,
    n_real_windows: int,
    n_v2e_windows: int,
    metric: str | BaseMetric,
    metric_kwargs,
    feature_names,
    feature_scales,
    modifier_pipelines,
    sensor,
    rng,
    seed,
    output_dir,
):
    name = scheme.get("name") or "scheme"
    stride = scheme.get("stride")
    metric_label = get_metric(metric).name

    print(f"\n=== Baseline comparison [{metric_label}] scheme: {name}  (stride={stride}) ===")

    results = baseline_comparison(
        real_data=real_data,
        v2e_data=v2e_data,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        n_real_windows=n_real_windows,
        n_v2e_windows=n_v2e_windows,
        stride=stride,
        metric=metric,
        metric_kwargs=metric_kwargs,
        feature_names=feature_names,
        feature_scales=feature_scales,
        modifier_pipelines=modifier_pipelines,
        sensor=sensor,
        rng=rng,
        seed=seed,
        name=name,
    )

    baseline = results["baseline"]
    print(
        f"Baseline real window [{baseline['start']}, {baseline['end']}] μs: "
        f"{baseline['n_events']} events"
    )
    for w in results["real_windows"]:
        print(
            f"  real [{w['start']:>9}, {w['end']:>9}] μs  "
            f"n={w['n_events']:>7}  distance={w['distance']:.6f}"
        )
    for w in results["v2e_windows"]:
        print(
            f"  v2e  [{w['start']:>9}, {w['end']:>9}] μs  "
            f"n={w['n_events']:>7}  distance={w['distance']:.6f}"
        )
    for variant, entries in results.get("modified_real_windows", {}).items():
        for w in entries:
            print(
                f"  real[{variant}] [{w['start']:>9}, {w['end']:>9}] μs  "
                f"n={w['n_events']:>7}  distance={w['distance']:.6f}"
            )

    paths = save_baseline_comparison_results(
        results,
        output_dir,
        run_name="",
        file_prefix=name,
    )
    print(f"Saved baseline comparison results to: {paths['dir']}")
    return results


def _run_all_to_all_scheme(
    real_data,
    v2e_data,
    scheme,
    *,
    baseline_start: int,
    baseline_end: int,
    n_real_windows: int,
    n_v2e_windows: int,
    metric: str | BaseMetric,
    metric_kwargs,
    feature_names,
    feature_scales,
    output_dir,
    all_to_all_config,
):
    name = scheme.get("name") or "scheme"
    stride = scheme.get("stride")
    metric_label = get_metric(metric).name
    visualizer = all_to_all_config.get("visualizer", "mds")
    visualizer_kwargs = {
        k: v for k, v in all_to_all_config.items()
        if k not in ("visualizer", "annotate")
    }

    print(
        f"\n=== All-to-all comparison [{metric_label}] "
        f"(visualizer={visualizer}) scheme: {name}  (stride={stride}) ==="
    )

    results = all_to_all_comparison(
        real_data=real_data,
        v2e_data=v2e_data,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        n_real_windows=n_real_windows,
        n_v2e_windows=n_v2e_windows,
        stride=stride,
        metric=metric,
        metric_kwargs=metric_kwargs,
        feature_names=feature_names,
        feature_scales=feature_scales,
        visualizer=visualizer,
        visualizer_kwargs=visualizer_kwargs,
        name=name,
    )

    print(
        f"All-to-all comparison used {len(results['windows'])} windows "
        f"({len(results.get('skipped_windows', []))} skipped)."
    )
    paths = save_all_to_all_comparison_results(
        results,
        output_dir,
        run_name="",
        file_prefix=name,
    )
    print(f"Saved all-to-all comparison results to: {paths['dir']}")
    return results


def _validate_comparison_modes(comparison_modes):
    unknown_modes = sorted(set(comparison_modes) - _AVAILABLE_COMPARISON_MODES)
    if unknown_modes:
        raise ValueError(
            "Unsupported comparison mode(s): "
            f"{unknown_modes}. Available modes: {sorted(_AVAILABLE_COMPARISON_MODES)}"
        )


def main():
    # --- Initialization ---
    config = load_config()

    output_root = Path(config.get('output_dir', 'output'))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f"run_{timestamp}"
    schemes = config.get('window_schemes') or _DEFAULT_SCHEMES

    raw_modes = config.get("comparison_modes") or config.get("analysis_modes")
    comparison_modes = _normalize_comparison_modes(raw_modes or _DEFAULT_COMPARISON_MODES)
    _validate_comparison_modes(comparison_modes)

    metric_impl = get_metric(config.get("metric", "mmd"))
    feature_names = config.get("feature_names")
    feature_scales = config.get("feature_scales")
    seed = config.get("seed")
    sensor = config.get("sensor")
    modifier_pipelines = build_pipelines(config.get("modifiers"))
    rng = np.random.default_rng(seed)
    all_to_all_config = config.get("all_to_all") or config.get("mds") or {}
    windows_config = config.get("windows") or {}
    baseline_start = int(config["baseline_start"])
    baseline_end = int(config["baseline_end"])
    n_real_windows = int(windows_config.get("n_real_windows", 9))
    n_v2e_windows = int(windows_config.get("n_v2e_windows", 10))

    # --- Load Event Data ---
    event_data_manager = EventDataManager()

    real_data = event_data_manager.load_event_data_h5(
        config['real_data_path'], dataset_name="events", data_key="real_data"
    )
    print(
        f"Real data loaded, total number of events: {real_data.shape[0]}, "
        f"duration: {real_data['t'].max()} microseconds"
    )
    v2e_data = event_data_manager.load_event_data_h5(
        config['v2e_data_path'], dataset_name="events", data_key="v2e_data"
    )
    print(
        f"V2E data loaded, total number of events: {v2e_data.shape[0]}, "
        f"duration: {v2e_data['t'].max()} microseconds"
    )

    # --- Build metric arguments ---
    metric_kwargs = metric_impl.build_kwargs(config)
    for line in metric_impl.describe_settings(metric_kwargs):
        print(line)

    # --- Run comparisons ---
    baseline_results = []
    all_to_all_results = []
    for scheme in schemes:
        if "baseline_comparison" in comparison_modes:
            results = _run_baseline_comparison(
                real_data,
                v2e_data,
                scheme,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                n_real_windows=n_real_windows,
                n_v2e_windows=n_v2e_windows,
                metric=metric_impl,
                metric_kwargs=metric_kwargs,
                feature_names=feature_names,
                feature_scales=feature_scales,
                modifier_pipelines=modifier_pipelines,
                sensor=sensor,
                rng=rng,
                seed=seed,
                output_dir=run_dir,
            )
            baseline_results.append(results)

        if "all_to_all_comparison" in comparison_modes:
            results = _run_all_to_all_scheme(
                real_data,
                v2e_data,
                scheme,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                n_real_windows=n_real_windows,
                n_v2e_windows=n_v2e_windows,
                metric=metric_impl,
                metric_kwargs=metric_kwargs,
                feature_names=feature_names,
                feature_scales=feature_scales,
                output_dir=run_dir,
                all_to_all_config=all_to_all_config,
            )
            all_to_all_results.append(results)

    for results in baseline_results:
        plot_baseline_comparison(results, show=True)

    annotate = all_to_all_config.get("annotate", False)
    for results in all_to_all_results:
        if results.get("visualizer"):
            plot_all_to_all_comparison(results, show=True, annotate=annotate)


if __name__ == "__main__":
    main()
