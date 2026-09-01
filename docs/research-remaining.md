# Remaining work

Status as of 23 August 2026. Ordered by what blocks what.

---

## A. Fix before any result is trustworthy

**1. Per-segment alignment.** The real-vs-real floor currently measures the wheel's
unsteadiness, not the sensor's. Tested: realigning segments flattens the 7.5 → 12.4
climb to roughly 6.7–7.3, but the required shifts do not grow with segment index
(+2, +2, −4 for segments 1, 3, 6), so this is wander of about 25 ms, not accumulating
drift. **A better period will not fix it.** Build the flank-crossing splitter — locate
each rotation's actual start rather than assuming a constant period.

**2. Window size sweep.** At 1 ms windows the v2e gap sat 2.2 standard deviations
above chance, and its mean was barely outside the chance band. Longer windows hold
more events and tighten that band. Until this is settled we do not know whether there
is enough signal to support anything downstream of it.

**3. Unbiased MMD estimator.** The biased estimator adds roughly 1/n per sample. v2e
carries 2.7x the events of real, so this contaminates every real-vs-v2e number
asymmetrically. It is also the likely cause of the dip-then-rise seen when sweeping
added noise — check that before concluding the kernel width is at fault.

---

## B. The evaluation — seven properties

Each stated as something a metric should do, with its test.

**4. Reads at chance level when nothing differs.** Split one window's events in half,
compare the halves. Prototyped as a throwaway; needs to live in the pipeline.

**5. Responds when the data changes.** Controlled change, reading moves clearly out of
the chance band.

**6. Moves in one direction as the change grows.** Sweep one change type, check the
reading rises and keeps rising.

**7. Separates a small real change from luck.** Smallest controlled change sitting
clearly outside the chance band, reported as a function of window size.

**8. Responds to the kinds of difference we care about — and we know which.**
*The centre of the paper.* Sweep time shift, spatial shift, dropout and added noise
separately; tabulate metric against change type.

**9. Produces an interpretable value.** A ruler per change type, not only for temporal
shift. Converts a reading into a physical equivalent — degrees of rotation, percent of
events dropped, and so on.

**10. Behaves consistently across conditions.** Repeat at f1 through f5; a ranking that
reverses with chopper speed is not safe to recommend.

The modifier machinery for 5–9 already exists in `config.yaml` (`add_noise`,
`subsample`, `jitter`, `transform`, `scaling`, with sweep support) and is commented out.

---

## C. Scope

**11. Add DVS-Voltmeter and V2CE.** Turns "we characterised v2e" into "simulators
differ in identifiable ways," which cannot be dismissed as one-off. Both need the same
input video and the same spatial and temporal calibration already built.

**12. Period-recovery anchor.** Does the metric ranking of simulators match how well
each reproduces the chopper period? The calibration already records
`v2e_period_disagreement` per trial. No models, no training — connects the metric to a
physical outcome someone would want to recover.

**13. Chamfer's two directions, kept separate.** real→sim large means the simulator is
missing structure; sim→real large means it invents structure. Neither MMD nor sliced
Wasserstein can give this.

**14. Event rate as a parallel channel.** MMD and sliced Wasserstein compare normalised
distributions, so the 2.7x event excess is structurally invisible to them. Report it
alongside rather than expecting a metric to catch it.

---

## D. Calibration code — deferred, cheap

**15.** `rotation_period_us = symmetry_period_us * apertures` should be
`* 2 * apertures`. Each aperture presents two blade edges.

**16.** Zero-pad before the transform in `estimate_period_fft`. Removes about 0.19% of
period error — measured, real, and not currently worth a re-run, but do it before the
next trial is processed.

---

## E. Paper

**17.** Frame the chopper as a deliberately calibrated instrument **in the
introduction**, not as a limitation in the discussion. Same facts, very different
reading.

**18.** Pitch as an evaluation / protocol paper. The seven properties are the
contribution, not the metrics, which are all off the shelf. Judged on whether the
protocol is sound and reveals something new.

**19.** Plan to release the protocol and the chopper dataset. A protocol only matters
if others can run it.

**20.** Open from `references.md`: choose the Chamfer citation, verify Carpenter (1988)
and Bahill, Clark & Stark (1975), confirm the target venue.

---

## Before committing to the full grid

Run **f1 x two simulators x three metrics** first and look at the selectivity table.
If different metrics respond differently to different change types, the paper works.
If the table is flat, it does not, and that is worth knowing in a day rather than a
month.
