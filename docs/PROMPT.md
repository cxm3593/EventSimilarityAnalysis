# Prompt for Claude Code

Paste this into Claude Code, run from the `EventSimilarityAnalysis` folder.

---

Read `CLAUDE.md` and `docs/test-spec.md` first. The spec defines four tests; test 1 is
already implemented in `experiments/`. Your job is to get test 1 running properly on this
machine, then implement tests 2 to 4 against the same spec, and produce their output
files.

**Your job is to produce results, not to interpret them.** Do not draw conclusions, do
not tell me what a number means, do not add analysis sections to anything you write. I
will do the analysis myself. The checks below exist only so you can tell a broken run
from a working one.

## Step 1 — get test 1 running

`experiments/test1_null.py` was written and checked against MMD only, on a machine
without POT or Open3D. Verify it here:

```
python experiments/test1_null.py --smoke --metrics mmd_rbf15
python experiments/test1_null.py --smoke --metrics swd
python experiments/test1_null.py --smoke --metrics chamfer
```

Expect the SWD and Chamfer runs to fail in `_configure_cuda_path_from_python_packages`
in `src/event_analysis_toolbox/mmd.py` if the `nvidia-*` packages are not importable.
The fix is in `CLAUDE.md` under "CUDA path bug". Apply it only if it actually fires.

Then time Chamfer — it runs on CPU per `config.yaml` and may be the binding cost for the
later tests. Report seconds per comparison at 1 ms and at 10 ms windows.

Run `python tools/measure_rotation_period.py output/trials` in `../EventCamCalib` first
if `rotation_period.yaml` is missing; `common.py` prefers it and says in `run_config.yaml`
which value it used.

Finally the full run:

```
python experiments/test1_null.py
```

When it finishes, tell me the output folder path and confirm `results.csv`,
`summary.csv`, `run_config.yaml` and the figures are all there. Nothing more.

## Step 2 — implement tests 2, 3 and 4

Follow `docs/test-spec.md`. Each is one script in `experiments/`, importing from
`common.py`, with the same shape as `test1_null.py`: argparse with a `--smoke` flag,
`RunOutput` for the folder and `run_config.yaml`, `results.csv` plus `summary.csv`, and
plotly figures.

Build them one at a time and run `--smoke` on each before moving on.

Do not change `common.py`'s metric construction or sample-size handling without saying
so — those encode decisions that took a while to get right.

Two things the spec calls out that are easy to get wrong:

- **Test 2's sweep** is two-phase: fine geometric steps from 0 to about 50 ms, then a
  fixed coarse step out to one or two rotations. Nearly all the change happens below
  10 ms, so a uniform step misses it.
- **Test 3's subsample sweep** must not normalise both sides to the same event count.
  Hold one side at `EVENTS_PER_COMPARISON` and reduce the other. It is specifically
  testing whether a metric responds to sample size, and normalising would erase that.

## Step 3 — the pilot

Run tests 1, 2 and 3a on f1 only. Then stop, point me at
`figures/selectivity_grid.html` from test 3, and leave it at that — do not describe what
the grid shows.

## While I am away — is the run healthy?

I am starting this late and going home. Work unattended. The checks below are only to
tell a working run from a broken one, so you do not spend the night building on a broken
one. They are not analysis and I do not want your reading of them.

### Before anything long, make the scripts crash-safe

`test1_null.py` collects every row in memory and writes `results.csv` at the very end. A
run that dies after two hours loses everything. Before the full runs, change it — and
write tests 2 to 4 the same way — so results are appended to disk as they are produced,
and a run can resume or at least leave partial results behind.

### Signs a run is broken

**All runs.** Empty or all-not-a-number `results.csv`. Missing `run_config.yaml`. Figures
with wrong or empty axes. Fewer rows than the configuration asked for. `n_events_a` and
`n_events_b` not equal to `EVENTS_PER_COMPARISON` where the spec says they should be.

**Test 2.** Distance at shift zero must be essentially zero — that comparison is a window
against itself, so anything else means the shift or the indexing is wrong. A completely
flat curve across all shifts means the shift is not being applied at all.

**Test 3.** If *every* modifier gives a completely flat curve, the modifier is not being
applied. Individual modifiers being flat is not a bug — leave those alone.

### Fix these yourself

Crashes, wrong paths, out-of-memory (reduce periods or window count and say what you
reduced), all-not-a-number results, obviously wrong plot axes, missing `run_config.yaml`.

### Stop and report, do not work around

Anything where the code looks fine and it is the numbers that look odd. Do not tune
parameters to make output look better. In particular, do not change `common.py`'s metric
construction or sample-size handling — those are conceptual and I would rather decide
them myself than find them silently patched.

### Budget and log

Keep the whole thing under about four hours. If a single run looks like exceeding an
hour, cut its scope, run the smaller version, and say so.

Leave a `RUNLOG.md` at the repo root: what you ran, how long each took, what you fixed
and why, what you stopped on, and where each output folder is. Facts only — no findings,
no conclusions, no interpretation of the numbers. That is the first thing I will read.

## Ground rules

- Do not touch `notebook/periodic_analysis.ipynb`.
- Do not modify anything under `src/event_analysis_toolbox/` except the CUDA path fix,
  and say so if you do.
- Every result folder gets its `run_config.yaml`. No exceptions.
- Report timings. Some of these runs are long and I need to plan around them.
- No analysis. Produce the files, report what ran, stop there.
