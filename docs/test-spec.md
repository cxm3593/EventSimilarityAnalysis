# Test specification — four scripts

Draft, 23 August 2026. One script per test, each runnable alone.

---

## Shared

### Segmentation

Cut segments with the **measured** rotation period (1,961,623 µs on f1, from
`rotation_period.yaml`), not 4x the quarter period.

Per-segment alignment is not a prerequisite. Tests 1, 2 and 3 do not compare different
rotations, and Test 4 cuts real and v2e on the same boundaries so any phase error is
shared. It remains a later refinement, and it affects Test 4's real-vs-real figure.

### Two reference levels

They measure different things. Report both; do not merge them.

- **Test 1 — resolution limit.** One window split in half, so the only difference is
  sampling chance. Says whether a metric can measure anything at all.
- **Test 4 — real against real across rotations.** Sampling chance plus whatever genuinely
  differs between rotations. A separate quantity, analysed later.

### One sample size for every comparison

Test 1 compares half a window against the other half, so it uses half the events. Every
other comparison must use the same number, or the values are not on one scale — MMD in
particular changes with sample size.

| name | default | meaning |
|---|---|---|
| `EVENTS_PER_COMPARISON` | set from data | events used on each side of every comparison |
| `MIN_EVENTS` | 50 | below this a window is skipped and recorded as not-a-number |

Pick `EVENTS_PER_COMPARISON` from the smallest typical half-window count across sources.
Real is the binding constraint, since v2e carries about 2.7x more. Subsample both sides
down to it, with a recorded seed.

**One exception.** The subsample modifier in Test 3a is deliberately about unequal sizes,
so it must not be normalised — hold one side at `EVENTS_PER_COMPARISON` and reduce the
other. Normalising it would make the check do nothing.

### Polarity

Everything below drops polarity, matching the pipeline. But the calibration found ON and
OFF events offset from each other by 1.5 to 5 µs·10³ between real and v2e, so polarity
carries a defect the metrics currently cannot see. Worth one run of Test 4 with ON and
OFF compared separately, as a check on whether that is being thrown away.

### Metric settings — all tests run all five

| key | metric | settings |
|---|---|---|
| `mmd_rbf03` | MMD | rbf max distance 3, target similarity 0.5, **unbiased** |
| `mmd_rbf15` | MMD | rbf max distance 15, target similarity 0.5, **unbiased** |
| `mmd_rbf75` | MMD | rbf max distance 75, target similarity 0.5, **unbiased** |
| `swd` | sliced Wasserstein | 100 projections, p = 2, seed recorded |
| `chamfer` | Chamfer | symmetric; also record each direction separately. CPU, so the slowest — check its cost before the all-to-all run |

Feature space is (x, y, t / `TIME_SCALE`) with `TIME_SCALE = 42` from `config.yaml`.

### Every script records its configuration

Alongside its outputs, each script writes `run_config.yaml` holding: trial, source files,
rotation period and how it was obtained, segment boundaries, every parameter below,
`EVENTS_PER_COMPARISON`, all seeds, the metric settings, package versions, and a
timestamp. No result file without its config.

### Output layout

```
output/tests/<trial>/<test_name>/<run_id>/
    run_config.yaml
    results.csv
    figures/
```

`run_id` is the timestamp plus the parameters that distinguish the run.

---

## Test 1 — Null

### Parameters

| name | default | meaning |
|---|---|---|
| `N_PERIODS` | 3 | how many full rotations to use; must be ≥ 2 |
| `PERIOD_START_INDEX` | 0 | which rotation to start from |
| `SPLIT_SEEDS` | [0, 1, 2, 3, 4] | one half-split per seed |
| `WINDOW_LENGTHS_US` | [1000, 5000, 10000] | window lengths to compare at |
| `WINDOW_STRIDE_US` | = window length | non-overlapping |

### Method

For each rotation, split its events into two halves at random using each seed. Then
compare half A against half B window by window at each window length.

### Results — `results.csv`

One row per (period, seed, window length, metric, window index):

```
trial, period_index, split_seed, window_length_us, metric,
window_index, window_start_us, n_events_a, n_events_b, distance
```

Plus `summary.csv`, one row per (period, seed, window length, metric):

```
mean_distance, sum_distance, sd_distance, n_windows
```

### Figures

- `events_3d_period{k}_full.html` — event cloud, x/y/t, whole rotation
- `events_3d_period{k}_seed{s}_halfA.html`, `..._halfB.html` — the two halves
- `null_curve_w{len}us.html` — distance against window index, one line per metric,
  faint lines per seed and a bold mean
- `null_summary.html` — mean distance against window length, one line per metric,
  with the spread as a band. **This is the figure that picks the window length.**

---

## Test 2 — Ruler

### Parameters

| name | default | meaning |
|---|---|---|
| `BASELINE_START_US` | segment start | where the reference window begins |
| `N_BASELINE_WINDOWS` | 20 | how many reference positions to average over |
| `WINDOW_LENGTH_US` | chosen from Test 1 | |
| `NEAR_STEPS_US` | [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50] ms | fine sampling of the rise |
| `FAR_STEP_US` | 20000 | fixed step beyond the near field |
| `MAX_SHIFT_PERIODS` | 2 | how far out to sweep |
| `FIXED_STEP_ONLY` | False | set True for a uniform sweep instead |

### Method

Reference window at a chosen start. Comparison window at start + δ. Sweep δ. Average
across the reference positions.

### Results — `results.csv`

```
trial, baseline_start_us, window_length_us, metric,
shift_us, shift_deg, n_baseline_windows, mean_distance, sd_distance
```

`shift_deg` = shift / rotation period × 360.

### Figures

- `ruler_near.html` — distance against shift, 0 to 50 ms, log x, one line per metric,
  with the Test 1 floor drawn as a horizontal band
- `ruler_full.html` — distance against shift out to `MAX_SHIFT_PERIODS`, linear x,
  showing the rise, the ceiling at half a rotation, and the return at a full one
- `ruler_deg.html` — the same with degrees of rotation on the x axis
- `windows_3d_shift{δ}.html` — reference and shifted window overlaid in x/y/t, at a few
  chosen δ, so the geometry behind a distance value is visible

---

## Test 3 — Modifier

### 3a. Generic sweeps

| modifier | sweep |
|---|---|
| `spatial_offset` | 0.5, 1, 2, 4, 8, 16 px, x only and x+y |
| `scaling` | 1.01, 1.02, 1.05, 1.10 about the ellipse centre |
| `subsample` | keep 90, 75, 50, 25% of one side only — see the sample-size note above |
| `uniform_noise` | add 1, 5, 10, 25, 50% of the window's event count |
| `temporal_clump_uniform` | quantise t to 10, 25, 50, 100, 161, 250 µs |

### 3b. Directed modifiers

| modifier | definition |
|---|---|
| `clump_to_v2e_grid` | snap each real timestamp to the **nearest actual v2e timestamp**. v2e's grid is irregular — gaps 22 to 819 µs — so a uniform step is a different corruption |
| `match_event_count` | subsample v2e down to real's count, and separately add noise to real up to v2e's count |
| `clump_and_match` | both together |

### Parameters

| name | default |
|---|---|
| `N_PERIODS` | 3 |
| `WINDOW_LENGTH_US` | from Test 1 |
| `MODIFIER_SEEDS` | [0, 1, 2] for anything random |
| `MODIFIERS` | all of the above; trim to run a subset |

### Results — `results.csv`

```
trial, modifier, magnitude, seed, metric, period_index,
window_index, distance
```

Plus `summary.csv` per (modifier, magnitude, metric): mean, sum, sd.

### Figures

- `modifier_curve_<name>.html` — distance against magnitude, one line per metric, floor
  drawn in
- `selectivity_grid.html` — small multiples, modifier by metric. **The main figure.**
- `subsample_check.html` — distance against retained fraction. A flat line is correct;
  a rising line means that metric is responding to sample size
- `directed_bars.html` — real vs v2e gap before and after each directed modifier
- `events_3d_<modifier>_<magnitude>.html` — one window before and after, for a few
  modifiers, so the corruption is visible

---

## Test 4 — Synthetic data

v2e only for now.

### Parameters

| name | default |
|---|---|
| `N_PERIODS` | all available (7 on f1) |
| `WINDOW_LENGTH_US` | from Test 1 |
| `BASELINE_PERIOD_INDEX` | 0 |
| `RUN_ALL_TO_ALL` | True |

### 4a / 4b — baseline comparison and placement

Compare against the baseline rotation, over full periods: real, v2e, and each modified
real from Test 3a.

`results.csv`:

```
trial, source, period_index, metric, window_index, window_start_us, distance
```

`placement.csv` — the v2e gap converted through the Test 2 and 3a curves:

```
metric, v2e_mean_distance, floor, ceiling,
equivalent_shift_us, equivalent_shift_deg,
equivalent_spatial_offset_px, equivalent_noise_fraction
```

### 4c — all to all

Pairwise distances among all real and all v2e periods. Slow; runs only if
`RUN_ALL_TO_ALL`.

`all_to_all_<metric>.csv` — the full matrix with row and column labels.

### Figures

- `baseline_comparison.html` — mean distance against period index, one line per source
- `phase_profile.html` — mean distance against position within the rotation, real and
  v2e, showing **where in the cycle** they diverge
- `placement_<metric>.html` — the ruler curve with the v2e gap marked on it
- `all_to_all_<metric>.html` — heatmap
- `mds_<metric>.html` — layout of all periods, coloured by source, stress printed

---

## Run order

Test 1 → choose window length → Test 2 → Test 3a → **stop and look at
`selectivity_grid.html`** → Test 3b → Test 4. f1 only until the shape is known.
