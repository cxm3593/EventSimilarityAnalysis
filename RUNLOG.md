# Run log — 23/24 August 2026

Overnight session. Facts only: what ran, how long, what was fixed, where the output is.
No findings, no interpretation of any number.

Trial: `optical_chopper_data_f1`, f1 only. Machine: Windows 11, RTX 5080, CuPy backend.

---

## Answers I acted on

You were asked four questions before work started. Recorded here because three of them
changed how the code was written.

1. **Window length for tests 2–4.** You handed it back with a lean toward 1000 or 5000.
   I used **5000 µs** as the primary, on the basis of your own note in
   `docs/research-remaining.md` that at 1 ms the v2e gap sat 2.2 SD above chance. Test 1
   still sweeps all three lengths (1000, 5000, 10000). A 1000 µs variant of tests 2–4 was
   held as a stretch goal; see "What was not run".
2. **Test 2 at shift zero.** Identical point set, so it reads exactly 0. Implemented by
   drawing the comparison-side subsample from a fresh generator with the *same* seed at
   every shift: at delta = 0 the comparison window is the reference window and the draw
   is the same draw. Verified: 0.0 for every metric.
3. **Test 3a's unmodified side.** You replied "one side unmodified, one side modified —
   what is the question?" The question was *which* unmodified side. I used **the same
   window**, so magnitude 0 is the identity and reads exactly 0, consistent with your
   answer to (2). The Test 1 chance level is drawn on the figures as a separate reference
   band rather than being the curve's own zero point. **If you wanted the half-split
   instead, this is the one thing to change.**
4. **Scope.** Your later message replaced "stop after 3a" with "make the implementation
   and try to get the result", so the chain runs 1 → 2 → 3a → 3b → 4.

---

## Step 1 — test 1 on this machine

### Rotation period

`rotation_period.yaml` was **missing from every trial**. Ran, as instructed:

```
cd ../EventCamCalib && python tools/measure_rotation_period.py output/trials
```

Wrote one file per trial. f1 = **1,961,600 µs**, matching the 1,961,623 µs in `CLAUDE.md`.
`common.Trial.load` picks it up and every `run_config.yaml` in this session records
`rotation_period_source: rotation_period.yaml (event rate in 2000 us bins, Hann window,
spectrum zero-padded 32x, ...)`. No run used the `4 x symmetry_period_us` fallback.

### The three smoke runs

All three passed on the first attempt.

| command | result | wall time |
|---|---|---|
| `test1_null.py --smoke --metrics mmd_rbf15` | pass | 6 s |
| `test1_null.py --smoke --metrics swd` | pass | 7 s |
| `test1_null.py --smoke --metrics chamfer` | pass | 4 s |

**The CUDA path bug did not fire, so I did not apply the fix.** `nvidia.cuda_nvrtc`,
`nvidia.cublas` and `nvidia.curand` are all installed in this venv, so
`importlib.util.find_spec` returns a spec instead of raising. `src/event_analysis_toolbox/`
is **unmodified** — no file under `src/` was touched in this session.

Worth knowing: a bare `import cupy` on this machine fails with "Failed to auto-detect
CUDA root directory". It only works because `_configure_cuda_path_from_python_packages`
sets `CUDA_PATH` from those pip packages first. Remove the `nvidia-*` packages and both
the crash you predicted and this one appear together.

### Metric timings

Per comparison, 20-comparison mean after a warm-up call, real data, both sides
subsampled to `DEFAULT_EVENTS_PER_COMPARISON`.

| window | events/side | mmd_rbf03 | mmd_rbf15 | mmd_rbf75 | swd | **chamfer** | all five |
|---|---|---|---|---|---|---|---|
| 1 ms | 731 | 4.32 ms | 4.34 ms | 4.25 ms | 19.12 ms | **0.80 ms** | 32.8 ms |
| 5 ms | 3610 | 19.70 ms | 18.77 ms | 18.98 ms | 17.96 ms | **2.71 ms** | 78.1 ms |
| 10 ms | 7240 | 63.73 ms | 62.34 ms | 62.01 ms | 19.17 ms | **5.02 ms** | 212.3 ms |

Chamfer at 1 ms is **0.80 ms** and at 10 ms is **5.02 ms** per comparison. It is the
cheapest of the five, not the binding cost — MMD is, scaling quadratically with events
per side. No scope was cut on account of Chamfer.

---

## Crash safety

`test1_null.py` collected every row in memory and wrote `results.csv` once at the end.
Changed before any long run, and tests 2–4 were written the same way from the start.

- **`common.ResultWriter`** (new) appends each row to `results.csv` as it is produced,
  flushing every 200 rows. A run that dies leaves everything computed so far on disk.
- **`--resume <RUN_DIR>`** on all four scripts. Reuses the interrupted run's folder,
  truncates a half-written final line, reads back which comparison keys are already
  present and skips them. Verified on a smoke run: 200 rows on disk, nothing recomputed,
  still 200 rows after.
- **`common.select_indices`** (new) replaces the old `--max-windows` behaviour of taking
  the *first* N windows. It now spreads the chosen windows evenly over the whole
  rotation, because a prefix samples one part of the turn only.
- **`RunOutput(existing=...)`** so a resumed run writes into the original folder.

`common.build_metrics` and the sample-size handling (`DEFAULT_EVENTS_PER_COMPARISON`,
`subsample`, `MIN_EVENTS`) are **untouched**, as instructed.

---

## Files added and changed

| file | what |
|---|---|
| `experiments/test2_ruler.py` | new — test 2 |
| `experiments/test3_modifier.py` | new — test 3, `--phase 3a` or `3b` |
| `experiments/test4_synthetic.py` | new — test 4 |
| `experiments/run_all.py` | new — driver: waits for test 1, then runs 2 → 3a → 3b → 4 |
| `experiments/common.py` | changed — `ResultWriter`, `select_indices`, `RunOutput(existing=)` |
| `experiments/test1_null.py` | changed — incremental writes, `--resume`, even window sampling |
| `RUNLOG.md` | new — this file |

Nothing under `src/` and nothing in `notebook/` was touched.

A shell version of the driver was written first and deleted: Git Bash `nohup ... &` does
not survive its parent shell being reaped on Windows, and it died silently after 90
seconds. The Python driver is launched with PowerShell `Start-Process`, which detaches
properly, and was confirmed alive across several shell exits.

---

## Choices inside the specification that were open

Conventions, not bugs. Listed so you can change them if you wanted something else.

- **Test 2 reference positions.** `N_BASELINE_WINDOWS = 20` spread evenly over one whole
  rotation (spacing 98,081 µs), not packed consecutively. Twenty adjacent 5 ms windows
  would cover 100 ms of a 1.96 s turn and sample one phase only.
- **Test 2 sweep grid.** 0, then the spec's near steps 100 µs to 50 ms, then a fixed
  20,000 µs step out to 2 rotations. 203 shifts total.
- **Test 3 window cap.** `--max-windows 40` per rotation. Uncapped, 3a is 3 rotations x
  393 windows x 37 magnitude-configs x 3 seeds, about 2.4 hours. At 40 evenly-spread
  windows it is about 15 min. **This is a scope cut and it is the only one made by
  choice.**
- **Test 3b sizing — flagged for your decision.** `match_event_count` is a no-op under the
  protocol's rule that both sides sit at `EVENTS_PER_COMPARISON`: if both sides are
  already normalised, matching their counts changes nothing. Rather than silently pick
  one, 3b runs **both** and records which in a `sizing` column: `normalised` (both sides
  at 3800, the protocol's rule) and `natural` (raw window counts, where the real-vs-v2e
  counts are about 7500 against 19357 and the modifier actually does something). Figures
  are written per sizing; `directed_bars.html` is a copy of the `normalised` one.
- **Figure layout.** The five metrics span three orders of magnitude, so a shared y axis
  flattens MMD onto the baseline. The ruler and modifier-curve figures use one subplot
  row per metric, each with its own y axis. Nothing is rescaled or normalised.
  `selectivity_grid.html` is modifier x metric small multiples as specified.
- **Chamfer's two directions.** `chamfer_analysis` in `src/` returns only the symmetric
  sum and `src/` was off-limits, so test 4 computes the two directions locally with a
  scipy KD-tree into `chamfer_a_to_b` / `chamfer_b_to_a`. Checked against Open3D's
  symmetric value on 84 comparisons: max relative difference **2.5e-7**.
- **Test 4 modified sources.** One representative magnitude per Test 3a modifier, since
  4a needs one curve each: offset_xy 2 px, scaling 1.02, subsample 0.50, noise 0.10,
  clump 161 µs.
- **Test 4 placement.** `placement.csv` inverts the Test 2 and Test 3a curves by linear
  interpolation. Where a curve never reaches the v2e value the entry is left as
  not-a-number rather than extrapolated.
- **Reproducible seeds.** Modifier randomness is seeded with a CRC32 of the labels, not
  Python's `hash()`, which is salted per process and would not reproduce.

---

## Stopped and reporting — not worked around

Your health-check list includes "`n_events_a` and `n_events_b` not equal to
`EVENTS_PER_COMPARISON` where the spec says they should be". That check does not pass,
and the cause sits in `common.py`'s sample-size handling, which you told me not to
change. So it is reported, not patched. **No parameter was tuned and no code was
altered in response to this.**

Measured on test 1's `results.csv` at 87,170 rows:

| window | target | rows | side A below target | rows where the two sides differ |
|---|---|---|---|---|
| 1000 µs | 750 | 68,670 | 46.5% | 50.8% |
| 5000 µs | 3800 | 13,755 | 49.3% | 51.0% |
| 10000 µs | 7500 | 5,910 | 47.3% | 49.3% |

`n_events_a` quantiles at 5000 µs: 1% = 2028, 5% = 2164, 25% = 3311, 50% = 3800.

The mechanism: `common.subsample` returns the array unchanged when it already holds
`count` or fewer points, and `DEFAULT_EVENTS_PER_COMPARISON` was set from the **median**
half-window event count. Half the windows therefore start below the target, keep their
natural counts, and the two sides differ by whatever the random half-split happened to
give — for example 354 against 358 at 1 ms. Zero rows were skipped by `MIN_EVENTS` and
there are no not-a-number distances.

This applies to all four tests, since all of them subsample through the same function.
Two things you might decide to do; I have done neither:

- lower `EVENTS_PER_COMPARISON` to a low quantile of the half-window count rather than
  the median, so nearly every window can reach it; or
- drop windows that cannot reach the target, recording them as not-a-number the way
  `MIN_EVENTS` already does.

---

## Stage results

All five stages exited 0. Trial f1, window 5000 µs for tests 2–4.

| stage | started | wall | exit | `results.csv` rows | expected |
|---|---|---|---|---|---|
| test 1 null | 00:01:17 | 28m 29s | 0 | 191,400 | 3 x 5 x 5 x (1962+393+197) = 191,400 |
| test 2 ruler | 00:29:46 | 7m 37s | 0 | 1,020 | 204 shifts x 5 metrics = 1,020 |
| test 3a sweeps | 00:37:23 | 24m 13s | 0 | 66,600 | 37 x 3 x 3 x 40 x 5 = 66,600 |
| test 3b directed | 01:01:36 | 57m 45s | 0 | 18,000 | 5 x 2 x 3 x 3 x 40 x 5 = 18,000 |
| test 4 synthetic | 01:59:21 | 10m 18s | 0 | 9,800 | 7 x 7 x 40 x 5 = 9,800 |

Every row count matches its configuration exactly. **Zero not-a-number distances in any
of the five results files.** Every folder has its `run_config.yaml`.

### Where the output is

Paths are under `output/tests/optical_chopper_data_f1/`.

| stage | folder |
|---|---|
| test 1 | `test1_null/20260824_000117/` |
| test 2 | `test2_ruler/20260824_003001_w5000us/` |
| test 3a | `test3_modifier/20260824_003736_3a_w5000us/` |
| test 3b | `test3_modifier/20260824_010141_3b_w5000us/` |
| test 4 | `test4_synthetic/20260824_015926_w5000us/` |

**The figure you asked to be pointed at:**
`output/tests/optical_chopper_data_f1/test3_modifier/20260824_003736_3a_w5000us/figures/selectivity_grid.html`

### Every stage ran twice — read this

I launched the driver twice by mistake. The first was a shell script; I checked whether
it had survived being backgrounded, concluded from `ps` that it had not, and launched a
Python replacement. The check was wrong — Git Bash reports the interpreter, not the
script name, so the shell driver was still alive and waiting. Both drivers then ran the
same four stages concurrently.

Consequences, in full:

- **Every stage after test 1 has two run folders**, about 10 seconds apart. Test 1 itself
  ran once.
- **The two copies are bit-for-bit identical.** I compared every row of all four pairs on
  their full key: keys identical, maximum absolute difference 0.0, 100% exact matches.
  The pipeline is fully deterministic, and this is an unintended but complete
  reproducibility check.
- **The stage timings above are inflated** by the two processes contending for one GPU.
  Test 3a and 3b in particular would be considerably faster run alone. Do not read the
  wall times as clean benchmarks.
- **Nothing is corrupted.** Each process wrote only into its own folder.
- I left both copies in place rather than deleting data. The duplicates are
  `test2_ruler/20260824_002948_w5000us`, `test3_modifier/20260824_003725_3a_w5000us`,
  `test3_modifier/20260824_010133_3b_w5000us`, `test4_synthetic/20260824_015913_w5000us`.
- One detail: test 4's canonical `run_config.yaml` records `ruler_from` and `sweeps_from`
  pointing at the *duplicate* test 2 and 3a folders, because the driver picked whichever
  was newest by modification time. Since the pairs are bit-identical this changes no
  number, but the recorded path is not the folder listed above.

### Health checks

| check | result |
|---|---|
| all runs: empty or all-not-a-number `results.csv` | no — 0 NaN in all five |
| all runs: missing `run_config.yaml` | no — present in every folder |
| all runs: fewer rows than configured | no — all exact, see table |
| all runs: `n_events_a` / `n_events_b` equal to `EVENTS_PER_COMPARISON` | **fails — see "Stopped and reporting" above** |
| test 2: distance at shift zero essentially zero | **exactly 0.0** for all five metrics, 20 reference positions each |
| test 2: completely flat curve across shifts | no — 201 to 204 distinct values per metric over 204 shifts |
| test 3: every modifier completely flat | no — Chamfer and SWD move on all six modifiers; some individual metric/modifier cells are flat, which your note says to leave alone |
| test 3a: identity magnitude reads zero | **exactly 0.0** for every modifier and metric |
| test 3b: modifiers distinguishable | 5 distinct values per metric under both sizings |
| figures: wrong or empty axes | none seen; test 2 sweep spans 0 to 3,920,000 µs = 719.4 degrees = 2 rotations |

Test 2's `results.csv` is the spec's averaged schema; the per-reference-position rows are
in `results_raw.csv` (20,400 rows) alongside it, which is also what makes the run
resumable.

---

## Second pass at 1000 µs

With budget left, the whole chain was run again at `--window-us 1000`, so you can
compare window lengths on tests 2–4 rather than only on test 1. This one ran alone, and
the timings below are therefore the clean ones.

| stage | wall | exit | rows | folder |
|---|---|---|---|---|
| test 2 ruler | 1m 34s | 0 | 1,020 | `test2_ruler/20260824_022145_w1000us/` |
| test 3a sweeps | 4m 48s | 0 | 66,600 | `test3_modifier/20260824_022319_3a_w1000us/` |
| test 3b directed | 6m 10s | 0 | 18,000 | `test3_modifier/20260824_022810_3b_w1000us/` |
| test 4 synthetic | 2m 10s | 0 | 9,800 | `test4_synthetic/20260824_023419_w1000us/` |

Whole chain 14m 42s, against 1h 40m for the same four stages at 5000 µs with two
processes contending. Same health checks pass: 0 not-a-number, shift zero exactly 0.0
for all five metrics, identity magnitudes exactly 0.0, `run_config.yaml` everywhere,
row counts exact.

`EVENTS_PER_COMPARISON` is 750 at this window length against 3800 at 5000 µs, so the two
passes are **not** on one scale and should not be pooled — that is the sample-size point
in `CLAUDE.md`, and it is why the spec wants one window length chosen rather than
several averaged.

---

## Test 4 with polarity split

The spec notes ON and OFF are offset by 1.5–5 µs·10³ between real and v2e and asks for
"one run of Test 4 with ON and OFF compared separately". `test4_synthetic.py` takes
`--polarity split`, which compares the ON channel against ON and OFF against OFF and
records which in the `polarity_channel` column. That run was launched at 02:48 at
5000 µs with `--no-all-to-all`, since 4c does not split polarity.

Folder: `test4_synthetic/20260824_025140_w5000us_polsplit/`, 19,600 rows
(9,800 x 2 channels), exit 0. `placement.csv` carries 10 rows, one per metric per
channel. 4c was disabled for this run only.

The first attempt at it exposed two faults **in code I wrote tonight**. Both are fixed
and both were re-verified; recorded here because one of them briefly damaged two output
files.

1. **Empty figures in split mode.** `baseline_comparison`, `phase_profile` and
   `placement.csv` all filtered on `polarity_channel == "all"`, which does not exist when
   the channels are `on` and `off`. The first split run therefore wrote a 0-row
   `placement.csv` and two figures with no data — the "figures with wrong or empty axes"
   case on your list. Fixed to iterate whatever channels the run produced; ON is drawn
   solid and OFF dashed. After the fix `baseline_comparison` holds 70 traces
   (7 sources x 5 metrics x 2 channels) and `phase_profile` 20.
2. **A resume key mismatch that duplicated rows.** In polarity-drop mode the row is
   written with `polarity_channel = "all"` but the resume check was passing the raw
   internal `""`, so no key ever matched. Re-running the two drop-mode test 4 folders
   with `--resume` to regenerate their derived files appended a second full copy:
   19,600 rows where there should have been 9,800.

   **Repaired, not left.** I verified the duplicate pairs were bit-identical (maximum
   difference 0.0) before removing them, so the deduplication was lossless, then
   regenerated the derived files with the corrected code. Both folders are back to
   exactly 9,800 rows and the resume now reports "9,800 rows already on disk" and adds
   none. No other run was ever resumed, so no other file was affected.

Because of fix (1), `placement.csv` in every test 4 folder now carries an extra
`polarity_channel` column, `"all"` in the drop-mode runs. The drop-mode numbers are
unchanged — only the column was added.

---

## Final verification

Every folder re-checked after all fixes:

| run | rows | expected | NaN | `run_config.yaml` | CSVs | figures |
|---|---|---|---|---|---|---|
| test 1, window sweep | 191,400 | 191,400 | 0 | yes | 2 | 13 |
| test 2, w5000 | 1,020 | 1,020 | 0 | yes | 3 | 9 |
| test 3a, w5000 | 66,600 | 66,600 | 0 | yes | 2 | 12 |
| test 3b, w5000 | 18,000 | 18,000 | 0 | yes | 2 | 3 |
| test 4, w5000 | 9,800 | 9,800 | 0 | yes | 8 | 17 |
| test 2, w1000 | 1,020 | 1,020 | 0 | yes | 3 | 9 |
| test 3a, w1000 | 66,600 | 66,600 | 0 | yes | 2 | 12 |
| test 3b, w1000 | 18,000 | 18,000 | 0 | yes | 2 | 3 |
| test 4, w1000 | 9,800 | 9,800 | 0 | yes | 8 | 17 |
| test 4, polarity split | 19,600 | 19,600 | 0 | yes | 3 | 12 |

All pass.

---

## What was not run

- **f2 to f5.** f1 only, per the spec's run order: "f1 only until the shape is known."
- **Test 2 `FIXED_STEP_ONLY`.** The uniform-sweep variant. The two-phase sweep is the
  default and the one that was run; `--fixed-step-only` is implemented if you want it.
- **4c in the polarity-split run only.** All-to-all ran in both the 5000 µs and 1000 µs
  chains; it was disabled only for the polarity-split run, which does not split it.
- **Nothing was cut for time apart from the test 3 window cap** (`--max-windows 40`,
  described above). Test 1, test 2 and test 4 ran at their full specified scope.

---

## How to resume or re-run

Every script takes `--resume <RUN_DIR>` and continues into that folder, skipping
comparisons already on disk:

```
python experiments/test3_modifier.py --phase 3a --resume output/tests/optical_chopper_data_f1/test3_modifier/20260824_003736_3a_w5000us
```

To re-run a whole chain at another window length, with one driver only:

```
python experiments/run_all.py output/tests/optical_chopper_data_f1/test1_null/20260824_000117 2500
```

Launch it detached with PowerShell `Start-Process`, not Git Bash `nohup` — and check it
is the only driver running before you walk away.

---

## Total

Started 23:47, finished 03:12. About 3h 25m against the four-hour budget. Ten result
folders, all verified. Nothing under `src/` or `notebook/` was touched.
