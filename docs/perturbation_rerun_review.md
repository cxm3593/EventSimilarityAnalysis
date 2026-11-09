# Perturbation rerun review — 22–24 September 2026

## Fresh temporal degree sweep — completed 24 September

The user-approved **fresh temporal-only run** started on 23 September at 17:51:24
and finished on 24 September at **00:27:46 America/New_York** (6 h 36 min 21 s).
The supervisor exited with code 0; the worker also recorded completion. The four
recorded process IDs were absent at the completion check. This run supersedes the
stopped expanded temporal sweep below; it does not complete that older suite.

- [Fresh results and saved configurations](../output/perturbation/20260923_175125_279651/).
- [Five-metric temporal degree figure](../output/perturbation/20260923_175125_279651/temporal_offset.html).
- [Combined temporal summary](../output/perturbation/20260923_175125_279651/summary.csv).
- [Supervisor manifest, logs and immutable source/settings snapshot](../output/perturbation/overnight_degrees_fresh_20260923_1752/).

All five recordings used the 25 requested angles:
`[0,0.25,0.5,1,2,5,10,30,45,60,90,120,135,150,175,180,185,210,225,240,270,300,315,330,360]`.
Every comparison was computed in this run: `temporal_reuse_run` is null in the root
and all five trial configurations, with no reuse provenance or reused-row columns.
No previous temporal distances were copied or interpolated. All events in every
complete 5 ms reference window of one rotation were retained, with seed 0 and the
unchanged five metrics, scales and backends. Comparison windows are actual later
recorded windows, independently re-zeroed, without wrapping or sample caps.

### Completion validation

The read-only audit
[analysis/validate_fresh_temporal_completion.py](../analysis/validate_fresh_temporal_completion.py)
passed on 24 September. It did not evaluate metrics or rewrite original outputs.

- **112,875 metric rows**: F1 49,000; F2 25,000; F3 16,500; F4 12,375; F5 10,000.
  These cover 392, 200, 132, 99 and 80 reference windows, respectively.
- **625 summary rows**, 125 per recording; all comparisons have status `ok`, with
  **zero exclusions** and finite valid distances.
- Full angle/metric/window coverage; unique logical and physical comparison keys;
  reference and comparison endpoints; rounded degree-to-time and realized-angle
  conversions; measured rotation periods; and settings agreement all passed.
- Both event-count columns were independently checked against the source recordings
  using the saved half-open window endpoints. The saved reference windows match
  every complete first-rotation 5 ms window, without capping or omitted windows.
- Zero-offset distances are exactly zero for all five metrics. All signed MMD
  estimates are finite; **4,752 are negative**, consistent with the unbiased
  estimator. Their reported distances obey `sqrt(max(distance_squared, 0))`.
  Negative estimates are retained, not treated as failures or evidence of equality.
- All per-trial summaries were recomputed from their metric rows and matched;
  both root summaries match the concatenated trial summaries.
- All **22 source/settings SHA-256 hashes** match the immutable launch snapshot
  and the corresponding workspace files at validation time.
- Saved HTML curve data match the summaries, with five metric rows and all 25
  recording/metric curves in the overview. All five point-cloud examples have
  finite coordinates and at most 2,000 display points per cloud. Plot checks are
  structural/data checks; browser-rendered visual inspection was not performed.
- No traceback, error, exception or warning was found in the evaluation stderr log.
  The 52 pre-launch regression tests passed before this fresh run started.

No evaluation failures were found. Window summaries are descriptive, not confidence
intervals or significance tests. These checks do not establish density independence.
At 360 degrees, comparison is with the next recorded rotation, not an identity copy.

The monitoring heartbeat is paused after completion validation. The notebook and
original result folders remain unchanged by this completion audit; the notebook
still shows the explicitly labelled previous temporal sweep described below.
`output/perturbation/current/` was not published. Any later consolidation must use
this fresh temporal run, the completed three spatial methods from the stopped
22 September suite, and subsampling/noise from the completed 21 September suite,
with exact per-method provenance. The existing two-source consolidation command
must not be used as-is for this three-source selection.

## Expanded 22 September suite — historical status

Launched **22 September, 18:10:15 America/New_York**. **Stopped at the user's request
on 23 September, 10:26:39 America/New_York**, because the temporal sweep was taking
too long. The worker, supervisor and both launchers were terminated and their absence
verified. All saved results and source snapshots remain intact; no restart is scheduled.

Spatial offset, scaling and jitter are complete across F1–F5 (198,660 comparisons).
Expanded temporal offset completed F1 (215,600 comparisons); F2 stopped with 36,365
saved rows out of 62,000; F3–F5 were not started. The new suite is **incomplete**.

- New result root: [20260922_181016_100519](../output/perturbation/20260922_181016_100519/).
- Supervisor, stdout/stderr and source/settings snapshot: [overnight_20260922_181015](../output/perturbation/overnight_20260922_181015/).
- Startup launcher PID 72624; supervisor PID 59012; evaluation launcher PID 69104;
  actual evaluation worker PID 74316. Verify creation time and command, not PID alone.
- The task heartbeat was paused following the stop request, then reconfigured for
  the fresh run documented above.
- `output/perturbation/current/` was **not published**; full-suite consolidation is deferred.

## Notebook refresh after stopping

[notebook/perturbation.ipynb](../notebook/perturbation.ipynb) now reads only complete
trial/method outputs: expanded spatial offset, scaling and jitter from the new run;
subsampling, uniform noise and the **previous complete temporal sweep** from the
21 September suite. Expanded temporal results, including the completed F1, are not
mixed into those figures. The older temporal sweep is labelled in notebook text and
figure titles. Scientific settings are unchanged.

All 30 selected trial/method outputs passed comparison coverage, unique-key,
status/exclusion, finite-distance, identity, event-count, signed-MMD/clamping,
recomputed-summary and required-plot checks. The selected set contains **311,535
comparisons and 1,725 summary rows, with zero exclusions**. This is a deliberately
mixed-source notebook view, not completion of the expanded temporal study.

Every notebook code cell executed successfully and its outputs were saved. Existing
F1–F5 colours and figure layouts were preserved; spatial plots were structurally
checked for all 25 recording/metric curves, colours and 1450 × 420 dimensions.
New HTML figures are separate from original runs:
[notebook_20260923_spatial_refresh](../output/perturbation/notebook_20260923_spatial_refresh/).
The combined overview and six separate figures were regenerated. Browser-rendered
visual inspection was not performed. This refresh preceded the fresh temporal run
documented above; its new results have not yet been incorporated into the notebook.

## What changed

Only four evaluations are rerun, sequentially across F1–F5:

- Spatial offset, pixels per axis: `[0,1,3,5,7,9,11,13,15,17,19,21,23,25]`.
- Scaling: `[0.90,0.92,0.94,0.96,0.98,1.00,1.01,1.02,1.04,1.05,1.06,1.08,1.10,1.12,1.14,1.16,1.18,1.20]`.
  This retains the original 1.01 and 1.05 points alongside the 0.02 grid.
- Spatial jitter, maximum uniform displacement per axis in pixels:
  `[0,0.5,1,2,3,4,5,6,7,8,12,16]`. This refines **amplitude**, not timestamp resolution.
- Temporal offset uses a time-based grid: `[0,100,200,500,1000,2000,5000,10000,20000,50000]` µs,
  followed by 60000, 80000, … µs at 20000 µs spacing through one measured rotation.
  Rounded quarter-, half-, three-quarter- and full-rotation offsets are also included.
  Duplicate offsets are removed. The actual per-trial times and equivalent degrees
  are saved in its configuration and comparison rows.

The temporal sweep contains 110, 62, 45, 36 and 32 offsets for F1–F5 respectively.
These remain comparisons with **later recorded windows**, independently re-zeroed;
there is no timestamp translation of the same events or wraparound.
The original degree-grid mode remains available for backwards compatibility.

All events, 5 ms non-overlapping windows, one full rotation, seed 0, physical
definitions, feature scales and five metric settings are unchanged. Full temporal
coverage was checked for all five trials before launch. No event-count caps were added.
The expanded temporal sweep may keep the suite running into the following day.

## Checks before launch and publication

- 74 regression tests passed before launch; **79 tests passed** after adding the
  independent consolidation tests.
- A bounded real-data smoke check retained all 7,614 events in one median-count F1
  reference window. All five metrics returned finite valid results at offset 25 px,
  scales 0.9 and 1.2, jitter 16 px, and temporal offsets 100 µs and one rotation.
- Existing subsampling and uniform-noise results from
  [20260921_210141_079777](../output/perturbation/20260921_210141_079777/) were revalidated:
  **49,665 comparisons**, no exclusions. Counts, identities, signed MMD values,
  recomputed summaries and required plot files passed the consolidation checks.
- Expected new comparisons: **536,580**, comprising 198,660 spatial comparisons
  and 337,920 temporal comparisons. Expected new summary rows: **2,525**.
- Expected consolidated total: **586,245 comparisons**, **2,800 summary rows**.
  These are planned counts, not a claim of completion.

## Consolidation and notebook handoff

[analysis/consolidate_perturbation.py](../analysis/consolidate_perturbation.py) is a
separate post-processing script. No consolidation logic was added to the evaluator.
It accepts explicit source runs, rejects unfinished or incompatible runs, checks
all per-trial outputs, and publishes an independent copy with an exact source manifest.
Any existing `current` folder is archived rather than deleted.

Original full-suite plan (deferred after the stop request; do not run this command
against the incomplete source):

```powershell
.venv/Scripts/python.exe analysis/consolidate_perturbation.py --updated-run output/perturbation/20260922_181016_100519 --retained-run output/perturbation/20260921_210141_079777
```

The four updated methods will come from the new run. Subsampling and uniform noise
will come from the completed previous suite. Each copied trial retains its original
`run_config.yaml`; the consolidated summary records `source_run` and rotation period,
with time and angle columns for temporal offset. Original outputs are never rewritten.

The full-suite `current` publication and expanded temporal time/degree figures are
deferred pending a revised user-approved strategy. The completed spatial notebook
refresh is described above. No partial outputs are labelled as a final current dataset.
