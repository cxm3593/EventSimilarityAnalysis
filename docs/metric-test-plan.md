# Test plan — four tests

Draft, 23 August 2026.

---

## Before starting: two things that gate everything

**Per-segment alignment.** Segments must be cut at each rotation's actual start, not at a
constant period. Measured: fixed-period cutting leaves segments several windows out of
phase in a way that does not grow with index, so it is wheel wander and no period value
fixes it. Until this is done, every "real vs real" number is measuring the wheel.

**One fixed sample size for every comparison.** The null test compares half a window
against the other half, so it uses n/2 events. If later tests compare full windows they
are not on the same scale and nothing is comparable. Pick a fixed event count per
comparison — subsample every window down to it — and use it in all four tests.

---

## Test 1 — Null

**Question.** What does each metric read when there is genuinely no difference?

**Method.** Take one window. Split its events randomly into two halves. Compute the
distance between the halves. Repeat across many windows and several random splits per
window.

**Sweep.** Window length: 0.5, 1, 2, 5, 10 ms. More events per window narrows the
spread, so this is also how the window length gets chosen.

**Output.** Per metric, per window length: mean and spread of the chance distance, and a
band (mean ± 2 spread) that later results are judged against.

**Read it as.** The window length where the band is narrow enough for the differences you
care about to sit outside it. On f1 at 1 ms the v2e gap was only 2.2 spreads above
chance, so 1 ms is probably too short.

**Bad result.** The band stays wide at every window length — the metric cannot resolve
anything on this data.

---

## Test 2 — Ruler

**Question.** What does a known amount of difference read as?

**Method.** Take a window at phase φ. Compare it against the window at phase φ + δ from
the same recording. Sweep δ. Average over many φ and many segments.

**Sweep.** δ = 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100 ms, then out to half a rotation.
Fine steps at the low end: measured, a 1 ms shift already produced most of the
separation, so 10 ms steps would miss where everything happens.

**Output.** Distance against δ, per metric. Floor comes from Test 1; ceiling is the value
at half a rotation, where the scene is maximally out of phase. Express δ in degrees of
wheel rotation as well as milliseconds.

**Read it as.** The conversion from a distance into a physical equivalent. Any later
result can be reported as "equivalent to N degrees of rotation."

**Bad result.** Not monotonic, so a reading maps to more than one δ and cannot be
converted. Or it saturates below the values you need to express.

---

## Test 3 — Modifier

Two halves, doing different jobs. Keep them separate in the write-up.

### 3a. Generic sweeps — characterise the metrics

Apply each modifier to one side of a real-vs-real comparison and sweep its magnitude.

| modifier | sweep | what it probes |
|---|---|---|
| spatial offset | 0.5, 1, 2, 4, 8, 16 px | sensitivity to position |
| scaling | 1.01, 1.02, 1.05, 1.10 | sensitivity to size |
| subsample | keep 90, 75, 50, 25% | **should read no change** |
| uniform noise | add 1, 5, 10, 25, 50% of the window's count | sensitivity to spurious events |
| temporal clumping | quantise t to 10, 25, 50, 100, 161, 250 µs | sensitivity to timing resolution |

**Output.** Distance against magnitude, per metric per modifier — the selectivity table.

**The one to watch.** Subsampling changes sample size, not distribution, so a correct
metric should not move. Anything that does is responding to event count. The biased MMD
estimator is expected to fail here, and that failure is what shows the protocol
discriminates rather than passing everything.

**Bad result.** Every metric responds identically to every modifier — they are redundant
and one would do.

### 3b. Directed modifiers — test mechanisms

Not sweeps. Each is a hypothesis about what the simulator does wrong, checked by whether
it moves real data *toward* the simulator.

- **Clump real timestamps to 161 µs.** Measured: in a 1 ms window v2e uses about six
  distinct timestamps against real's 811. Does distance(clumped real, v2e) fall well
  below distance(real, v2e)?
- **Match event count.** v2e emits 2.7x real's events. Subsample v2e to real's count, or
  add noise to real to match v2e's, and see how much of the gap closes.
- **Both together.**

**Output.** For each hypothesis, the gap before and after. A large drop names a cause.

---

## Test 4 — Synthetic data

v2e only for now; DVS-Voltmeter and V2CE later.

**4a. Baseline comparison.** On one set of axes: real vs real (the floor), real vs v2e,
and real vs each modified real from Test 3a. Reports where v2e sits relative to both the
noise floor and the known corruptions.

**4b. Placement on the ruler.** Convert the v2e gap into its equivalent in degrees of
rotation, pixels, and percent density change, using the Test 2 and Test 3a curves.

**4c. All-to-all.** Pairwise distances among all real and all v2e segments, laid out with
MDS. Be aware this is thin with one simulator — it can only show whether v2e segments
sit apart from real ones. Its value arrives with the second and third simulators, where
it shows whether they fail the same way or differently.

---

## Run order

1. Alignment and the fixed sample size — nothing is trustworthy before these.
2. Test 1 across window lengths → choose the window length, get the floor.
3. Test 2 at that window length → get the scale, check monotonicity.
4. Test 3a → the selectivity table. **Stop and look.** If it is flat, the diagnostic
   claim is gone and the paper narrows; better to know now.
5. Test 3b and Test 4 → the v2e result, placed on the scale, with named causes.

Run 1–4 on f1 first. Only spread to f2–f5 once the shape of the result is known.
