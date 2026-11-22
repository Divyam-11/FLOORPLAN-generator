# Evaluation

## What the application measures

The Evaluation page compares requested and generated bedroom, bathroom and balcony counts from the saved result objects. It checks that the measured clear area is finite and positive for each functional room. It applies the upstream minimum of 2 m² to kitchen and bathroom. Clear areas subtract generated wall footprints, and the total includes balconies.

The original production backend requires valid room geometry, doors for specified functional connections, an exterior entrance and consistently formed opening records. The dashboard shows its reported rejected candidates, the final accepted seed, and errors for exhausted or interrupted jobs. Count matching measures whether the topology condition survived generation and reconstruction. Building-code compliance, barrier-free circulation, structural adequacy and dataset-level generalization remain outside the measured scope. [Geometry policy](../original_source/Text2Floorplan/data/geometry_policy.json) · [Upstream checks](../original_source/Text2Floorplan/backend_cli.py)

## Tested outputs

Verified on 22 November 2026 on Windows x64, Python 3.12, PyTorch 2.11.0+cpu and Transformers 5.14.1. The FastAPI worker reported both weights loaded, CPU, four PyTorch threads and the production 200-step schedule setting.

| Execution | Requested beds / baths / balconies | Final seed | Clear area | Observed outcome |
| --- | --- | ---: | ---: | --- |
| Upstream CLI, medium, supplied seed 42 | 3 / 2 / 1 | 4032486707 | 89.20 m² | Three candidates rejected. PNG decoded at 1620 × 1620 and JSON held 29 wall and 15 opening records. |
| Upstream CLI smoke test, large | 3 / 2 / 2 | 3968430468 | 128.29 m² | PNG and editable JSON verified in 22.2 s including command startup. |
| FastAPI, medium | 3 / 2 / 1 | 4032486707 | 89.21 m² | 17.00 s. Room counts and area checks passed. |
| React form through FastAPI, large | 3 / 2 / 2 | 3968430468 | 128.29 m² | 17.09 s. Room counts and area checks passed. |
| FastAPI retry exercise, medium, supplied seed 42 | 3 / 2 / 1 | 2145797810 | 95.67 m² | 120.50 s across four candidates. Three rejected layouts were retained in the job diagnostics. |

These timings are local observations. The solver may retry under a different seed, and runtime varies by hardware, thread scheduling and geometry checks. The slight medium-plan area difference was observed between the standalone and resident-worker configurations.

## Verification scope

Setup reverified the PTH checksum and reused the cached CLIP encoder. The Vite production build succeeded and its bundle was served by FastAPI. Vite development mode also served the app and proxied the ready API. Restarting FastAPI restored finished jobs. Browser checks exercised generation, zoom, saved history, model-stage expansion, the dataset preview, training status and side-by-side evaluation. The dashboard also fit a 390 px viewport. Both browser downloads matched the generated PNG and JSON by SHA-256. API checks verified a missing ID returns 404, invalid seed or topology budget returns 422, and upstream dataset artifacts are served successfully. An isolated startup check confirmed that a missing checkpoint reports an actionable readiness error.

The original checkout remained clean in Git. The inference process did not load the upstream trainer module. No training checkpoint directory was created, and no training, fine-tuning or retraining was run by setup, startup, inference or testing.
