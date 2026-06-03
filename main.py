import datetime
from pathlib import Path

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


def _run_baseline_scheme(
    real_data,
    v2e_data,
    scheme,
    *,
    baseline_start: int,
    baseline_end: int,
    n_real_windows: int,
    n_v2e_windows: int,
    metric: str,
    metric_kwargs,
    output_dir,
):
    name = scheme.get("name") or "scheme"
    stride = scheme.get("stride")

    print(f"\n=== Baseline comparison [{metric}] scheme: {name}  (stride={stride}) ===")

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
    metric: str,
    metric_kwargs,
    output_dir,
    all_to_all_config,
):
    name = scheme.get("name") or "scheme"
    stride = scheme.get("stride")
    visualizer = all_to_all_config.get("visualizer", "mds")
    visualizer_kwargs = {
        k: v for k, v in all_to_all_config.items()
        if k not in ("visualizer", "annotate")
    }

    print(
        f"\n=== All-to-all comparison [{metric}] "
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


def _shared_feature_kwargs(config):
    """Feature settings shared across distance / similarity metrics."""
    return {
        "feature_names": config.get("feature_names"),
        "feature_scales": config.get("feature_scales"),
    }


def _build_metric_kwargs(config, metric: str):
    if metric == "mmd":
        mmd_config = config.get("mmd")
        if not mmd_config:
            raise ValueError("config.yaml must define an 'mmd' section when metric is 'mmd'.")
        return {
            **_shared_feature_kwargs(config),
            "chunk_size": mmd_config["chunk_size"],
            "rbf_kernel_max_distance": mmd_config["rbf_kernel_max_distance"],
            "rbf_kernel_target_similarity": mmd_config["rbf_kernel_target_similarity"],
            "backend": mmd_config["backend"],
        }
    raise ValueError(f"Unsupported metric: {metric!r}")


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

    metric = config.get("metric", "mmd")
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
    metric_kwargs = _build_metric_kwargs(config, metric)
    if metric == "mmd":
        print(
            "RBF kernel max distance:",
            metric_kwargs["rbf_kernel_max_distance"],
            (
                f"(similarity <= {metric_kwargs['rbf_kernel_target_similarity']} "
                "beyond this scaled distance)"
            ),
        )

    # --- Run comparisons ---
    baseline_results = []
    all_to_all_results = []
    for scheme in schemes:
        if "baseline_comparison" in comparison_modes:
            results = _run_baseline_scheme(
                real_data,
                v2e_data,
                scheme,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                n_real_windows=n_real_windows,
                n_v2e_windows=n_v2e_windows,
                metric=metric,
                metric_kwargs=metric_kwargs,
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
                metric=metric,
                metric_kwargs=metric_kwargs,
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
