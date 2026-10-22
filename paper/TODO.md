# Paper to-do list

Target: first full draft by Sunday 25 October 2026, then send to Gabe.
Tick an item by changing `[ ]` to `[x]`.

## Writing the draft

- [ ] Methodology prose, from the outline
- [ ] Evaluation prose: perturbation study
- [ ] Evaluation prose: simulator study
- [ ] Discussion (no outline yet)
- [ ] Conclusion
- [ ] Abstract (150 to 250 words)
- [ ] Introduction prose: update to the new outline structure, and write the road map paragraph
- [ ] Caption for the combined perturbation figure

## Content still open

| Item | Section | Notes |
|---|---|---|
| [ ] Radiometric calibration of the scene | Methodology, Framework Overview | Done. Need to review and report only, so the scene can be reproduced. It does not change the v2e or V2CE data. Data: `Data/Radiometric_Camera_Calib_08062026` |
| [ ] Kernel bandwidths and how they were chosen | Methodology, MMD | 3, 15 and 75 px. The outline still says "remain to be determined" |
| [ ] Choice of parameters | Methodology, Choice of Parameters | Space-time scale, window length, kernel bandwidth: what they are, how they affect results, SWD-guided choice of MMD kernels |
| [ ] Uncertainty of the perturbation responses | Evaluation, Perturbation Study | The whole-recording noise floor run can serve as the basis |
| [ ] Which noise floor to use | Evaluation, Comparison Setup | Now the 360-degree point of the temporal offset test; the whole-recording run is the alternative. F2 is unstable |
| [ ] Parameter tuning example | Evaluation, Example Study | Promised in the introduction |

## Fixes in the outline

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

- [ ] `sections/background.tex`: Chamfer is called a "baseline"; the sentence "This motivates the sampling and real--real references used in the evaluation" refers to a test that no longer exists; no closing paragraph.

## Done

- [x] v2e timing check: v2e matches real best when shifted by 3 to 7 ms. Not significant, no text needed.
- [x] Manuscript skeleton: section files for Methodology, Evaluation, Discussion and Conclusion.
