# Results index

Runs from the optical-chopper dataset, chopper frequencies f1-f5.
Renamed 2026-08-19 from timestamped `run_<metric>_<date>_<time>` folders.

## Folder name format

```
<metric>_<frequency>_<metric setting>_w<window width>_d<duration>
```

| Part | Meaning |
| --- | --- |
| `mmd` / `swd` | Metric: Maximum Mean Discrepancy / Sliced Wasserstein Distance |
| `f1` .. `f5` | Optical-chopper frequency of the source recording (`optical_chopper_data_fN`) |
| `rbfNN` | MMD only: RBF kernel `max_distance` (zero-padded so 03 < 15 < 75 sorts correctly) |
| `pNNN` | SWD only: number of random projections |
| `w1000us` | Comparison window width in microseconds (= `baseline_end - baseline_start`) |
| `d999ms` | Total span covered = window width x number of comparison windows |

## Settings shared by all 20 runs

These are constant everywhere below, so they are **not** in the folder names.
If you ever rerun with a different value, add it to the name or the runs stop being comparable.

| Setting | Value | Why it matters |
| --- | --- | --- |
| `feature_scales.t` | 42 | Time is divided by 42 before any distance, so 42 us ~ 1 pixel. Changing it rescales every MMD/SWD number. |
| `rbf_kernel_target_similarity` | 0.5 | Second RBF kernel parameter; `rbfNN` alone is ambiguous without it. |
| `baseline_start` / `baseline_end` | 15000 / 16000 us | Which slice of the recording is the reference window. |
| Window scheme | `consecutive`, `stride: null` | Non-overlapping windows. Contrast with the `periodic` (stride 315000) runs in `output/`. |
| `n_real_windows` / `n_v2e_windows` | 999 / 1000 | Number of comparison windows. |
| MMD `biased` | true | Biased estimator (kernel diagonal kept). |
| SWD `p` | 2 | Order of the Wasserstein distance. |
| `seed` | 0 | |
| Sensor | 1280 x 720 | |

## Runs

| Folder | Metric | Freq | Metric setting | Original folder |
| --- | --- | --- | --- | --- |
| `mmd_f1_rbf03_w1000us_d999ms` | MMD | f1 | rbf_max_distance=3, target_similarity=0.5 | `run_mmd_20260819_140627` |
| `mmd_f1_rbf15_w1000us_d999ms` | MMD | f1 | rbf_max_distance=15, target_similarity=0.5 | `run_mmd_20260819_182543` |
| `mmd_f1_rbf75_w1000us_d999ms` | MMD | f1 | rbf_max_distance=75, target_similarity=0.5 | `run_mmd_20260819_142433` |
| `mmd_f2_rbf03_w1000us_d999ms` | MMD | f2 | rbf_max_distance=3, target_similarity=0.5 | `run_mmd_20260819_140742` |
| `mmd_f2_rbf15_w1000us_d999ms` | MMD | f2 | rbf_max_distance=15, target_similarity=0.5 | `run_mmd_20260819_182717` |
| `mmd_f2_rbf75_w1000us_d999ms` | MMD | f2 | rbf_max_distance=75, target_similarity=0.5 | `run_mmd_20260819_142546` |
| `mmd_f3_rbf03_w1000us_d999ms` | MMD | f3 | rbf_max_distance=3, target_similarity=0.5 | `run_mmd_20260819_140957` |
| `mmd_f3_rbf15_w1000us_d999ms` | MMD | f3 | rbf_max_distance=15, target_similarity=0.5 | `run_mmd_20260819_182917` |
| `mmd_f3_rbf75_w1000us_d999ms` | MMD | f3 | rbf_max_distance=75, target_similarity=0.5 | `run_mmd_20260819_142851` |
| `mmd_f4_rbf03_w1000us_d999ms` | MMD | f4 | rbf_max_distance=3, target_similarity=0.5 | `run_mmd_20260819_141320` |
| `mmd_f4_rbf15_w1000us_d999ms` | MMD | f4 | rbf_max_distance=15, target_similarity=0.5 | `run_mmd_20260819_183509` |
| `mmd_f4_rbf75_w1000us_d999ms` | MMD | f4 | rbf_max_distance=75, target_similarity=0.5 | `run_mmd_20260819_143145` |
| `mmd_f5_rbf03_w1000us_d999ms` | MMD | f5 | rbf_max_distance=3, target_similarity=0.5 | `run_mmd_20260819_141809` |
| `mmd_f5_rbf15_w1000us_d999ms` | MMD | f5 | rbf_max_distance=15, target_similarity=0.5 | `run_mmd_20260819_183957` |
| `mmd_f5_rbf75_w1000us_d999ms` | MMD | f5 | rbf_max_distance=75, target_similarity=0.5 | `run_mmd_20260819_143613` |
| `swd_f1_p100_w1000us_d999ms` | SWD | f1 | n_projections=100, p=2 | `run_sliced_wasserstein_20260819_135412` |
| `swd_f2_p100_w1000us_d999ms` | SWD | f2 | n_projections=100, p=2 | `run_sliced_wasserstein_20260819_135719` |
| `swd_f3_p100_w1000us_d999ms` | SWD | f3 | n_projections=100, p=2 | `run_sliced_wasserstein_20260819_135909` |
| `swd_f4_p100_w1000us_d999ms` | SWD | f4 | n_projections=100, p=2 | `run_sliced_wasserstein_20260819_140049` |
| `swd_f5_p100_w1000us_d999ms` | SWD | f5 | n_projections=100, p=2 | `run_sliced_wasserstein_20260819_140340` |

## Files inside each run folder

Filenames are unchanged, so any existing globs still work:

| File | Contents |
| --- | --- |
| `run_config.yaml` | Full config the run was launched with -- the authoritative record |
| `consecutive_baseline_metadata.yaml` | Resolved window/baseline settings for this scheme |
| `consecutive_baseline_results.csv` | Distance-to-baseline curve |
| `consecutive_baseline_kernel_results.csv` | MMD only: per-kernel breakdown |
| `consecutive_baseline_plot.png` | Rendered curve |
