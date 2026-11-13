# Paper to-do list

Target: first full draft, then send to Gabe. The 25 October 2026 target has passed; new date not set.
Tick an item by changing `[ ]` to `[x]`.

## Writing the draft

- [ ] Evaluation prose: perturbation study. Opening paragraph, sweep table, and Spatial Translation and Spatial Scaling are done; jitter, subsampling, uniform noise, temporal offset and the summary remain.
- [ ] Evaluation prose: simulator study
- [ ] Discussion (no outline yet). Includes the Example Study, moved here from the Evaluation as a recommendation or ablation study.
- [ ] Conclusion
- [ ] Abstract (150 to 250 words)
- [ ] Introduction prose: update to the new outline structure, and write the road map paragraph
- [ ] Perturbation figures: one per subsection with its own caption. Translation and scaling are done; jitter, subsampling, uniform noise and temporal offset remain.

## Content still open

| Item | Section | Notes |
|---|---|---|
| [ ] Radiometric calibration of the scene | Methodology, Framework Overview | Done. Need to review and report only, so the scene can be reproduced. It does not change the v2e or V2CE data. Data: `Data/Radiometric_Camera_Calib_08062026` |
| [ ] Uncertainty of the perturbation responses | Evaluation, Perturbation Study | The whole-recording noise floor run can serve as the basis |
| [ ] Parameter tuning example | Discussion | Promised in the introduction. Moved from the Evaluation's Example Study to the Discussion, as a recommendation or ablation study |

## Fixes in the outline

- [ ] Subsampling bullet: still describes the mean ("The plotted mean MMD distances...", "The largest kernel shows a small increase at the strongest subsampling level"). The figures now use the median, under which that increase is zero. Recheck the bullet against the median figure.
- [ ] Subsampling and Uniform noise bullets: `%` is not escaped (write `\%`). LaTeX reads `%` as the start of a comment, so the rest of those lines is missing from the PDF.
- [ ] Chamfer is still called a "baseline" (background bullet, methodology overview, the "Chamfer baseline" subsection). It is now an independent verification.
- [ ] Framework bullet: "support both types of comparison" comes before either type is named.
- [ ] Framework bullet: the recommendation (SWD guiding the MMD kernel) and the dependence on settings are not mentioned.
- [ ] First bullet: "commercial deployment", "obstacle avoidance / detecting obstacles" and "low-power always-on" are not backed by the cited papers.
- [ ] Simulation bullet: "v2e remains the most widely used" needs a source.
- [ ] Background bullet 2: run-on sentence ("...bias settings the bias settings are specific to systems...").
- [ ] Small errors:
  - Framework bullet: the bold sentence has no period; "quantifying the real and simulated event streams" should be "quantifying the difference between..."
  - EQS sentences: "an machine learning network", "unitless an unintepretable"
  - Version 3 bullet: "event cameras has", "make them operates", "extreme lighting condition", "But, to preserve ... requires"
  - First bullet: "in context", double space in "efficient  high frequency"
  - Simulation bullet: "in different perspectives"

## Fixes in the prose

- [ ] `sections/methodology.tex` loose ends: the window-length reference to Section~ef{sec:parameters} in Event Representation is circular with Choice of Parameters; "A finite set of directions only approximates SWD..." in the SWD subsection repeats the Monte Carlo error text; the radiometric placeholder (`% Leave a space for`) in the Framework Overview.
- [ ] `sections/evaluation.tex`, Spatial Translation and Spatial Scaling: rework the MMD and Chamfer paragraphs so each states its point first and the numbers follow as evidence for it, rather than listing values.
- [ ] `sections/background.tex`: Chamfer is called a "baseline"; the sentence "This motivates the sampling and real--real references used in the evaluation" refers to a test that no longer exists; no closing paragraph.

## Fixes in the figures

- [ ] Perturbation PNGs (`Images/perturbation_*.png`): the text inside the images does not match the paper. Re-export from `notebook/perturbation.ipynb` once fixed, for all six figures.
  - The subtitle says "whiskers: window 5–95%". The paper avoids "whiskers"; the captions describe the vertical lines instead.
  - Each image has an in-figure title (e.g. "Spatial offset x+y: responses across F1–F5"). IEEE figures carry no title inside the image, and "Spatial offset x+y" does not match "spatial translation" in the text. Remove the title and subtitle.
  - Panel names read "MMD RBF-3", "MMD RBF-15", "MMD RBF-75", while the text says "the 3-pixel kernel". Rename to match, e.g. "MMD, $d$ = 3 px".

## Done

- [x] Methodology prose, from the outline (pushed 6 Nov 2026).
- [x] Kernel bandwidths and how they were chosen: 3, 15 and 75 px, in Choice of Parameters.
- [x] Choice of parameters: space-time scale, window length and SWD-guided kernel choice.
- [x] Which noise floor to use: the neighbouring-turns run, `output/noise_floor/neighbouring_turns_20261002_175852` (every turn against the next, same phase).
- [x] Perturbation curves use the median over windows, not the mean (decided 9 Nov 2026); the mean is kept in `notebook/perturbation.ipynb` for comparison.
- [x] v2e timing check: v2e matches real best when shifted by 3 to 7 ms. Not significant, no text needed.
- [x] Manuscript skeleton: section files for Methodology, Evaluation, Discussion and Conclusion.
