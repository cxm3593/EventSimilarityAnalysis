# EventSimilarityAnalysis

Research code for a paper on measuring the gap between real and simulated event-camera
data. The contribution is a **calibrated benchmark**: a controlled recording plus a
protocol that gives a distance measure a floor, a scale and physical units.

## The experiment

An optical chopper wheel recorded simultaneously by a real event camera and by
simulators (v2e now; DVS-Voltmeter and V2CE later) driven from the same video. Five
chopper speeds, f1 to f5.

Data lives in a sibling repo:

```
../EventCamCalib/output/trials/optical_chopper_data_f1/
    final_masked_real.h5     real events, (x, y, p, t) structured, time-sorted
    final_masked_v2e.h5      v2e events, same layout
    final_masked_v2ce.h5     V2CE events, same layout (integrations/v2ce/generate.py)
    result.yaml              calibration output
    rotation_period.yaml     written by tools/measure_rotation_period.py - preferred
```

## Chopper geometry - read this before touching periods

The wheel has **2 apertures**, therefore **4 blade edges per rotation** (each aperture
has an opening edge and a closing edge). The event rate bursts at every edge.

`result.yaml` is misleading on this and cost several days:

- `symmetry_period_us` is a **quarter** rotation, not a half. On f1: 491,342 us.
- `rotation_period_us` there is `symmetry_period_us * apertures`, giving **half** a
  rotation while naming it a whole one. The multiplier should be `2 * apertures`.

**A full rotation on f1 is about 1,961,623 us (1.96 s).** Prefer the measured value in
`rotation_period.yaml`. `experiments/common.py` reads it and falls back to
`4 * symmetry_period_us`, recording which it used.

## Known traps

**Biased MMD.** The biased estimator adds roughly 1/n per sample, so it responds to event
count rather than distribution. v2e carries about 2.7x real's events, so this
contaminates every real-vs-v2e comparison. **Always run unbiased.** `common.build_metrics`
forces `biased: False`.

**Clamped nulls.** `mmd.py` returns `sqrt(max(mmd_squared, 0))`. The unbiased estimator
goes negative when there is no real difference, so about half of all null values come out
as exactly zero. Mean and standard deviation are therefore meaningless on the null - use
the unclamped `distance_squared` column, and quantiles rather than mean +/- spread.

**Sample size.** MMD changes with sample size, so every comparison in every test must use
the same number of events on each side. See `DEFAULT_EVENTS_PER_COMPARISON` in
`common.py`. The one exception is the subsample sweep in test 3, which is deliberately
about unequal sizes.

**CUDA path bug.** `_configure_cuda_path_from_python_packages` in `mmd.py` calls
`importlib.util.find_spec("nvidia.something")`. That **raises** when the parent `nvidia`
package is absent, rather than returning None. Any machine without the `nvidia-*` pip
packages crashes here, even on the numpy backend. Fix:

```python
try:
    spec = importlib.util.find_spec(package_name)
except (ImportError, ValueError):
    continue
```

**Timestamp resolution.** In a 1 ms window, real uses about 811 distinct timestamps and
v2e about 6. v2e's grid is irregular - gaps 22 to 819 us - so clumping real timestamps to
imitate it must snap to v2e's actual timestamps, not to a uniform step.

**Event rate.** v2e emits about 2.7x real's events. MMD and sliced Wasserstein compare
normalised distributions, so they are structurally blind to this. Report it separately.

## Layout

```
config.yaml                 shared settings: feature scales, metric sections
main.py                     the original baseline-comparison pipeline
src/event_analysis_toolbox/ metrics, mmd, sliced_wasserstein, chamfer, preprocessing
experiments/                the four evaluation tests - common.py plus test1..test4
notebook/periodic_analysis.ipynb   exploratory work on segmentation
notebook/perturbation.ipynb        figures for the perturbation study
notebook/simulator_comparison.ipynb  figures for real vs v2e vs V2CE
analysis/simulator_checks/  read-only checks on the simulator comparison
docs/                       the plan and specification these tests implement
output/perturbation/        the perturbation study: three source runs, figures, provenance
output/simulator/           real vs v2e vs V2CE: comparison/<trial>/<run_id>/, analysis/
output/noise_floor/         every turn against the next one (experiments/noise_floor.py)
output/README.md            which results the paper uses and where each came from
output/tests/<trial>/<test>/<run_id>/   where tests 1-3 write if rerun; absent otherwise
../EventSimilarityAnalysis_archive/     everything older, moved out on 2026-10-02
```

## Conventions

- Every run writes `run_config.yaml` beside its results, recording the rotation period
  and where it came from. No result file without its configuration.
- Metrics: `mmd_rbf03`, `mmd_rbf15`, `mmd_rbf75`, `swd`, `chamfer`. All five, always.
- Feature space is (x, y, t / 42); the temporal scale comes from `config.yaml`.
- Figures are plotly HTML. Subsample before plotting - a rotation holds 1.4 M events.
- Backends come from `config.yaml` (`cupy` on this machine). Falling back to `numpy`
  should work once the CUDA path bug above is fixed.

## Writing with the user

- When drafting or rewriting paper text and outlines, follow the user's own language
  style: plain, clear, concise language and familiar technical terms. Concise means
  avoiding unnecessary wording, not requiring short sentences. Use the sentence length
  needed to express the reasoning clearly. Improve grammar and clarity without replacing
  their voice with elaborate academic prose.
- Preserve the user's observations, explanations, and intended emphasis. Do not silently
  remove an explanation or substitute implementation details for the point they are making.
- Explain causes in concrete steps. Prefer wording such as "This is because..." and
  "An earlier test showed..." when supported, rather than abstract terminology.
- State established observations directly. Do not add unnecessary hedging, but do not
  turn an unverified causal explanation into a fact. Explain any necessary factual
  correction briefly and separately from the rewritten passage.
- During an outline pass, help finish the outline first. Keep deeper interpretation
  and mathematical derivations for when the user asks for them.
- For writing requests, provide the draft together with a concise explanation of the
  wording, structure, and how they express the user's intended meaning. Do not return
  only the drafted text unless the user explicitly requests text only.
- Review the draft for clarity, accuracy, concision, and preservation of the user's
  meaning. If there are worthwhile improvements, explain them and provide a revised
  version in the same response. Do not stop at suggesting revisions or manufacture
  changes just to produce another version.
