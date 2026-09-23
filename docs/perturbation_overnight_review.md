# Perturbation implementation and overnight run

Updated 22 September 2026, after the suite finished at 06:26 America/New_York.
All five new evaluations are complete and the saved results have been checked.

## Status

- The original spatial-offset run finished successfully. All 40,635 comparisons
  are present, all statuses are `ok`, and the combined summary has 225 rows.
  Original outputs are untouched: `output/perturbation/20260921_181528_347872`.
- The refactored code passed 64 tests, including the complete six-evaluation
  sequence, physical transform semantics, CSV summaries, plots and failure cleanup.
- A bounded real-data smoke check used one F1 5 ms window containing 7,614 events.
  All six perturbations produced finite distances with the five configured metrics.
  These are engineering checks, not full experimental results:
  `output/perturbation/suite_smoke_20260921_205850_218549`.
- The five new evaluations ran from 21:01:40 to 06:26:06 (about 9 h 24 min), with
  exit code 0. They ran sequentially in the order
  scaling, jitter, subsampling, uniform noise, temporal phase offset.
  Run outputs: `output/perturbation/20260921_210141_079777`.
  Logs and process manifest: `output/perturbation/overnight_20260921_210140`.
  A source/settings snapshot is saved under that directory's `source` folder.
- All 162,540 new comparisons are present: scaling 22,575; jitter 27,090;
  subsampling 22,575; uniform noise 27,090; temporal offset 63,210. All statuses
  are `ok`, with finite distances, zero exclusions and no duplicate comparison
  keys. The final combined summary contains 900 rows.
- Checks passed for every trial/magnitude/metric row count, recomputed per-trial
  summaries, the final combined summary, identity cases, event-count rules,
  phase/time conversion, common reference windows and signed MMD-squared values.
  Configurations and source-snapshot hashes were checked. All 55 HTML files have
  the expected trace counts; this is structural validation, not visual layout QA.

## Where to read the implementation

- `experiments/perturbation.py`: argument parsing, validation and the run sequence.
- `experiments/perturbation_methods.py`: `SpatialOffset`, `SpatialScaling`,
  `SpatialJitter`, `Subsampling`, `UniformNoise`, and `TemporalOffset`. Each class
  groups its definition, transformation and evaluation entry point. No instances
  are needed. The small `LocalPerturbation` base only supplies shared defaults.
- `experiments/perturbation_shared.py`: data loading/window preparation, measurement,
  aggregation, output metadata and plotting, in separate labelled sections.
  `ReferenceWindow` is a typed data container, not an experiment controller.
- `experiments/perturbation_cla.yaml`: editable argument values. The existing
  `config.yaml` still supplies feature scaling and metric settings.
- `experiments/overnight_runner.py`: starts one suite, saves logs/PIDs/source hashes,
  waits for its exit status, and records completion or failure. It is not a scheduler.

The sequence is: select an evaluation, load a trial, prepare one window, apply its
sweep, measure all metrics, flush rows, then summarize and plot. Iteration is split
into named functions; no function contains nested `for` loops. Every evaluation
has a tqdm bar counting completed metric comparisons.

## Parameters and meaning

All trials use every complete non-overlapping 5 ms window in the first full
rotation: F1=392, F2=200, F3=132, F4=99, F5=80. Partial tails are omitted. No cap is
applied to metric input counts. Deliberate subsampling is the only thinning.

- Spatial offset: the user's `[0,1,3,5,7,9,11,13,15]` px sweep is retained.
  Each value is added to both x and y; total displacement is sqrt(2) times its
  absolute value. It was not rerun overnight.
- Scaling: `[1,1.01,1.02,1.05,1.10]`, about the calibrated chopper centre.
  Coordinates are not scaled about the image origin.
- Jitter: `[0,0.5,1,2,4,8]` px maximum independent uniform displacement per spatial
  axis. Subpixel values are retained. This is a local spatial dispersion test,
  not a physical model of changes in sensor bias.
- Subsampling: retain `[1,0.9,0.75,0.5,0.25]` of the original events, rounding
  the retained count to an integer. Sampling is without replacement and subsets
  are nested across the sweep. The reference side keeps all original events.
- Uniform noise: add `[0,0.01,0.05,0.1,0.25,0.5]` times the original event count.
  Noise is continuous uniform x/y/t over the full sensor and the window duration;
  it is not masked back to the stimulus. Ratio 0.5 means nominal final noise
  fraction 1/3, not 1/2. Realized counts and mixture fractions are recorded.
- Temporal offset: `[0,5,15,30,45,60,75,90,135,180,225,270,315,360]` mechanical
  rotation degrees. The measured period converts each angle to integer
  microseconds. The comparison reads a later recorded window and re-zeros it
  relative to its own start. It does **not** translate timestamps within the
  reference point cloud. No wraparound is used; 360 degrees compares the next
  recorded rotation and need not return zero. This is a coarse exploratory phase
  curve, not a precise inverse calibration or a new real-real noise-floor estimate.

Spatial transforms leave timestamps unchanged and do not clip or remove events
outside the sensor bounds. Temporal sweeps require the same baseline windows to
fit every offset; insufficient overall coverage raises an error before evaluation.

Seed 0 gives one reproducible realization per window. Random draws are shared
across magnitudes, so jitter directions and subsampling/noise prefixes remain
consistent. These are not independent repeated-seed experiments.

## Output and interpretation

Each trial/evaluation directory contains `run_config.yaml`, `windows.csv`,
`results.csv`, `summary.csv`, `point_cloud.html` and the evaluation's comparison
HTML. The run root also receives combined summaries and figures per evaluation,
and a final combined `summary.csv` only after the full suite succeeds.

- Every attempted window/magnitude/metric comparison is saved with event counts,
  period/window indices, magnitude and validity status.
- Comparisons with fewer than 50 events on either side have missing distances and
  an explicit status. Summary counts distinguish attempted, valid and excluded
  comparisons. Temporal offsets and subsampling can change eligibility.
- Means weight each valid window equally. The 5th/95th percentiles describe window
  variation; they are not confidence intervals or significance thresholds.
- Signed unbiased MMD-squared values are preserved. A zero plotted MMD distance
  can result from clamping a negative estimate; it does not establish insensitivity.
- Plotting alone downsamples each cloud to at most 2,000 points. Equal-size spatial
  transforms use corresponding display indices; other comparisons need not have
  pointwise correspondence. The example is the highest-count eligible reference
  window, at the strongest perturbation (45 degrees when available for phase offset).
- Each comparison plot has one subplot per metric with stable recording colours.
  HTML generation and figure structure are tested. Local-file browser policy
  prevented an automated visual preview; inspect the HTML layout when reviewing.

## Findings to review

These observations describe this configuration and the first rotation of each
recording. Numerical ranges below are across F1–F5 window means, not uncertainty
intervals. Raw values from different metrics should not be compared directly.

- **Scaling:** SWD increases almost proportionally over the tested range: about
  0.908–0.913 at 1% expansion and 9.073–9.118 at 10%. MMD responses depend on
  kernel settings and clamping near the identity. Chamfer also increases, but
  its 10% response spans 7.182–9.987 across recordings. This describes consistency
  across these recordings; it does not isolate event density as the sole cause.
- **Spatial jitter:** SWD and Chamfer increase throughout the sweep. At 8 px
  maximum displacement per axis, their means are respectively 0.402–0.546 and
  5.133–7.127. All plotted MMD means are zero through 4 px; at 8 px the smaller
  kernels show some positive responses, while RBF-75 remains clamped to zero.
  Signed estimates change even where the plotted distance stays zero. The
  current plots therefore cannot support a claim of complete insensitivity.
- **Subsampling:** retaining 25% raises SWD to 1.580–3.364 and Chamfer to
  3.180–5.180. RBF-3 and RBF-15 remain clamped to zero; RBF-75 has small positive
  means, 0.000283–0.001305. Random thinning preserves the generating distribution
  in expectation, not the exact finite empirical distribution. These results
  concern finite-sample thinning, not a direct measurement of event rate.
- **Uniform noise:** SWD, Chamfer and the larger-kernel MMD settings increase
  at sufficiently large additions. At added/original ratio 0.5, SWD is
  98.385–99.245, Chamfer 69.338–70.188 and RBF-75 0.1240–0.1264. RBF-3 is
  clamped to zero throughout the sweep. Report the full-sensor noise domain and
  final noise fraction (one third here); neither is a physical sensor-noise model.
- **Temporal phase:** RBF-75 and SWD show broad peaks around 45 degrees and
  corresponding later phases, whereas the smaller kernels largely plateau away
  from the recurring minima. At 45 degrees SWD is 44.670–45.898; at 360 degrees
  it is 1.529–8.640. The latter compares another recorded rotation, not the same
  points, so a nonzero return is expected. Minima vary more across recordings
  than the broad peaks. Periodicity and plateaus prevent a unique inverse from
  distance to angle; finer sampling is needed before treating a branch as a ruler.

### Important MMD interpretation caveat

The code uses the requested estimator with within-sample diagonals removed and
retains its signed squared value. However, the original and modified clouds share
events: the usual independent-sample unbiasedness interpretation does not directly
apply to these paired comparisons. For an identical cloud, the cross term still
includes self-pairs, so a negative squared estimate is expected, not a zero-centred
independent-sampling null. Across the jitter sweep, this negative starting value
and subsequent square-root clamping can hide small responses. Keeping all events
does not remove this dependence.

Before making sensitivity claims from the flat MMD curves, inspect signed response
curves alongside the current plots and decide whether a separate independent-sample
or paired-estimator control is appropriate. Neither a zero-clamped curve nor a
baseline-subtracted signed curve alone establishes statistical significance. No
estimator or scientific parameters were changed during this run.

## Where to start today

Open the combined [jitter figure](../output/perturbation/20260921_210141_079777/spatial_jitter.html)
and its per-trial point-cloud examples first, then the
[scaling](../output/perturbation/20260921_210141_079777/spatial_scaling.html),
[subsampling](../output/perturbation/20260921_210141_079777/subsampling.html),
[noise](../output/perturbation/20260921_210141_079777/uniform_noise.html), and
[phase](../output/perturbation/20260921_210141_079777/temporal_offset.html) figures.
The [combined summary](../output/perturbation/20260921_210141_079777/summary.csv)
contains the descriptive values; individual comparisons remain in each trial's
evaluation folder. Review signed MMD values before writing the perturbation findings.

The current defaults deliberately preserve the earlier sweeps. Potential follow-up
choices are additional random seeds/rotations, finer small-angle phase steps and
different jitter amplitudes. Do not interpret this single-rotation run as evidence
of a universal metric winner or statistical significance.
