import datetime
from pathlib import Path

from event_data_toolbox.event_data_manager import EventDataManager
from event_analysis_toolbox.mds_mmd import (
    mds_mmd_analysis,
    plot_mds_mmd,
    save_mds_mmd_results,
)
from event_analysis_toolbox.windowed_mmd import (
    plot_windowed_mmd,
    save_windowed_mmd_results,
    windowed_mmd_analysis,
)
import yaml  # pyright: ignore[reportMissingModuleSource]


_DEFAULT_SCHEMES = [{"name": "consecutive", "stride": None}]
_DEFAULT_ANALYSIS_MODES = ["windowed"]
_AVAILABLE_ANALYSIS_MODES = {"windowed", "mds"}


def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def _run_scheme(
    real_data,
    v2e_data,
    scheme,
    *,
    baseline_start: int,
    baseline_end: int,
    mmd_kwargs,
    output_dir,
):
    name = scheme.get("name") or "scheme"
    stride = scheme.get("stride")

    print(f"\n=== Windowed MMD scheme: {name}  (stride={stride}) ===")

    results = windowed_mmd_analysis(
        real_data=real_data,
        v2e_data=v2e_data,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        n_real_windows=9,
        n_v2e_windows=10,
        stride=stride,
        mmd_kwargs=mmd_kwargs,
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
            f"n={w['n_events']:>7}  MMD={w['mmd']:.6f}"
        )
    for w in results["v2e_windows"]:
        print(
            f"  v2e  [{w['start']:>9}, {w['end']:>9}] μs  "
            f"n={w['n_events']:>7}  MMD={w['mmd']:.6f}"
        )

    paths = save_windowed_mmd_results(
        results,
        output_dir,
        run_name="",
        file_prefix=name,
    )
    print(f"Saved windowed MMD results to: {paths['dir']}")
    return results


def _run_mds_scheme(
    real_data,
    v2e_data,
    scheme,
    *,
    baseline_start: int,
    baseline_end: int,
    mmd_kwargs,
    output_dir,
    mds_config,
):
    name = scheme.get("name") or "scheme"
    stride = scheme.get("stride")

    print(f"\n=== MDS all-to-all MMD scheme: {name}  (stride={stride}) ===")

    results = mds_mmd_analysis(
        real_data=real_data,
        v2e_data=v2e_data,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        n_real_windows=9,
        n_v2e_windows=10,
        stride=stride,
        mmd_kwargs=mmd_kwargs,
        name=name,
        n_components=mds_config.get("n_components", 2),
    )

    print(
        f"MDS analysis used {len(results['windows'])} windows "
        f"({len(results.get('skipped_windows', []))} skipped)."
    )
    paths = save_mds_mmd_results(
        results,
        output_dir,
        run_name="",
        file_prefix=name,
    )
    print(f"Saved MDS MMD results to: {paths['dir']}")
    return results


def _validate_analysis_modes(analysis_modes):
    unknown_modes = sorted(set(analysis_modes) - _AVAILABLE_ANALYSIS_MODES)
    if unknown_modes:
        raise ValueError(
            "Unsupported analysis mode(s): "
            f"{unknown_modes}. Available modes: {sorted(_AVAILABLE_ANALYSIS_MODES)}"
        )


def main():
    config = load_config()
    output_root = Path(config.get('output_dir', 'output'))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f"run_{timestamp}"
    schemes = config.get('window_schemes') or _DEFAULT_SCHEMES
    analysis_modes = config.get("analysis_modes") or _DEFAULT_ANALYSIS_MODES
    _validate_analysis_modes(analysis_modes)
    mds_config = config.get("mds") or {}
    baseline_start = int(config["baseline_start"])
    baseline_end = int(config["baseline_end"])

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

    mmd_kwargs = {
        "chunk_size": 20_000,
        "rbf_kernel_max_distance": config["rbf_kernel_max_distance"],
        "rbf_kernel_target_similarity": config.get("rbf_kernel_target_similarity", 0.01),
        "feature_names": None,
        "feature_scales": config.get("feature_scales"),
        "backend": "cupy",
    }
    print(
        "RBF kernel max distance:",
        mmd_kwargs["rbf_kernel_max_distance"],
        (
            f"(similarity <= {mmd_kwargs['rbf_kernel_target_similarity']} "
            "beyond this scaled distance)"
        ),
    )

    windowed_results = []
    mds_results = []
    for scheme in schemes:
        if "windowed" in analysis_modes:
            results = _run_scheme(
                real_data,
                v2e_data,
                scheme,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                mmd_kwargs=mmd_kwargs,
                output_dir=run_dir,
            )
            windowed_results.append(results)

        if "mds" in analysis_modes:
            results = _run_mds_scheme(
                real_data,
                v2e_data,
                scheme,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                mmd_kwargs=mmd_kwargs,
                output_dir=run_dir,
                mds_config=mds_config,
            )
            mds_results.append(results)

    for results in windowed_results:
        plot_windowed_mmd(results, show=True)

    for results in mds_results:
        plot_mds_mmd(results, show=True, annotate=mds_config.get("annotate", False))


if __name__ == "__main__":
    main()
