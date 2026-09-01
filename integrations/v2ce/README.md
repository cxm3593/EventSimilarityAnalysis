# V2CE integration

This folder keeps V2CE's runtime and project-specific conversion code separate from
the metric environment. The upstream checkout is expected at
`C:/Users/cxm3593/Academic/Workspace/V2CE-Toolbox`.

The generator uses the calibrated trial's `frame_warped.avi`, runs upstream V2CE in
`pano` mode, maps its resized event coordinates back to the original 1280 x 720 image
plane, applies the same fitted chopper ellipse used by `final_masked_real.h5` and
`final_masked_v2e.h5`, and writes an HDF5 `events` dataset with named fields
`(x, y, p, t)` and the same `uint16, uint16, int16, int64` field types.

Create the isolated environment:

```powershell
uv sync --project integrations/v2ce
```

Generate a short F1 smoke test:

```powershell
uv run --project integrations/v2ce python integrations/v2ce/generate.py `
  --trial optical_chopper_data_f1 --max-frames 33
```

Generate complete F1--F5 recordings:

```powershell
uv run --project integrations/v2ce python integrations/v2ce/generate.py --all
```

Outputs are written under `output/v2ce/<trial>/`. The Test 4 input is
`final_masked_v2ce.h5`; the upstream NPZ and generation log are kept beside it for
traceability.

Extend the existing compatible F1--F5 Test 4 runs with V2CE, without recomputing
their real and v2e rows or running all-to-all:

```powershell
.venv/Scripts/python.exe integrations/v2ce/run_test4.py --all
```

Build cross-frequency metric and event-rate summaries:

```powershell
.venv/Scripts/python.exe integrations/v2ce/summarize.py
```

Generate the three-source comparison figure:

```powershell
.venv/Scripts/python.exe integrations/v2ce/plot_test4.py
```

The figure is written to `output/v2ce/test4_real_v2e_v2ce.html`.
