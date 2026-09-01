# Plan: periodic segmentation & period-to-period comparison

Prototype as a notebook (`notebook/period_comparison.ipynb`). Promote to a
`ComparisonStrategy` in the toolbox once the design settles.

## Why

The current baseline comparison measures real and v2e each against a *third*
object (one real baseline window). Two streams can be equidistant from a
baseline and still differ from each other, so that design cannot answer "how
different are real and v2e." This one compares them directly, at matched phase,
against a noise floor built from real-vs-real.

## Notebook sections

**1. Load**
Real and v2e via `EventDataManager.load_event_data_h5` (lazy). Read
`symmetry_period_us`, `rotation_period_us`, `apertures` from the trial's
`result.yaml`.

**2. Segment — real data only**
Fixed signature so the method is swappable:

```python
segment(events, **params) -> list[tuple[int, int]]   # (start_us, end_us)
```

- First implementation: fixed period from `result.yaml`.
- Later candidates: flank-crossing landmarks, burst detection.
- Segment on **real only**. v2e inherits the same boundaries, so any v2e timing
  error surfaces as measured difference rather than being absorbed silently.

Write boundaries to a file with provenance (method, params, trial). Every
metric must run on the identical segmentation or the metric comparison is
confounded.

**3. Compare periods**
For each period pair, cut both into 1 ms sub-windows at matched phase and
compute distance per phase bin -> a profile `D(phi)`. Integrate over the cycle
for a scalar. Keep both; the profile shows *where* in the cycle v2e fails.

Metrics: MMD (rbf03 / rbf15 / rbf75), SWD, Chamfer.

**4. Matrices**
Full joint matrix over all 2N periods:

```
M = [ M_RR   M_RV  ]
    [ M_RV^T M_VV  ]
```

`M_VV` is required — for the embedding and for the self-consistency test below.

**5. Summary statistics**

| Quantity | Meaning |
|---|---|
| `D_RR` | real vs real — the noise floor |
| `D_VV` | v2e vs v2e — the simulator's own variability |
| `D_RV` | cross term |
| `E = 2*D_RV - D_RR - D_VV` | contrast, self-normalised, comparable across metrics |

- **`D_VV < D_RR` means v2e is too self-consistent** — it undermodels
  shot-to-shot sensor variability. Candidate headline finding.
- Stratify everything by `|i-j|`. Drift affects real and v2e equally at matched
  separation, so it cancels; pooling across all `|i-j|` breaks that.
- `E >= 0` is guaranteed only for distances of negative type. Do not assume it
  for SWD(p=2) or Chamfer; a negative value is information about the metric.

**6. Visualise** — heatmap and/or MDS, decided later. Heatmap checks
segmentation quality (`M_RR` flat = clean; brightening with `|i-j|` = residual
phase error). MDS shows whether real and v2e periods separate or interleave;
report stress, and feed it a symmetric matrix.

**7. Extra channels**

- **Event rate per phase bin.** MMD and SWD compare *normalised* distributions,
  so the ~3x v2e event excess is structurally invisible to them. Report it
  alongside.
- **Chamfer, directions kept separate.** real->v2e large = v2e missing
  structure; v2e->real large = v2e has spurious structure. Localises the excess.
  MMD and SWD cannot give this.
- **Metric rank agreement.** Spearman correlation between the MMD / SWD /
  Chamfer matrices over the same pairs. Agreement means one metric suffices;
  divergence means they are complementary, and says how.

## Decisions deferred

- Rotation period vs symmetry period. 2 apertures, so symmetry-period segments
  alternate between two physically distinct openings. Test: is `D_RR` larger at
  odd `|i-j|` than even? If yes, use the rotation period.
- Fixed 1 ms sub-windows from segment start, vs N equal phase bins per period.
  Identical under fixed-period segmentation; diverges once segments have
  unequal durations.
- Heatmap, MDS, or both.
- MMD `biased: true` -> `false`. The biased estimator adds +1/n per sample, and
  v2e windows hold ~3x the events, so it contributes a sample-size artefact to
  every real-vs-v2e number.

## Escapes the notebook

1. The segmentation boundaries file (with provenance).
2. The distance matrices, as CSV.

Everything else can stay in the notebook until the design settles.

## Promotion path

`main.py` has a `ComparisonStrategy` ABC with a `register_strategy` decorator
and a `_STRATEGIES_BY_MODE` registry. A third mode alongside
`baseline_comparison` and `all_to_all_comparison` is a new module plus one
import line — no restructuring.
