# FloorPlan AI

Generate editable 2D apartment plans locally with the released Text2Floorplan conditional Flow Matching model. Choose apartment size, bedroom, bathroom and balcony counts, and optionally a seed. The dashboard streams job state through polling and saves each genuine output as a labelled PNG and editable JSON.

**Training code is preserved. Training execution is disabled. All generated plans use the released pretrained weights.**

## Start on Windows

Open PowerShell in this folder. Setup requires Git, 64-bit Python 3.12, Node.js 22.12 or later, and internet access for the initial dependency and model downloads. Allow several GB for dependencies and the model cache. Setup is already complete in the delivered workspace.

```powershell
py -3.12 scripts/setup.py
.\.venv\Scripts\python.exe scripts/launch.py
```

Open [FloorPlan AI](http://127.0.0.1:8000/). Wait for **Engine ready**, then generate a plan. For a repeatable demo on the tested Windows CPU build, leave the medium size and default room counts, enter seed `4032486707`, and select **Generate floor plan**. Previous Generations and Evaluation load the saved results. A rejected layout may retry with a new seed. The actual accepted seed appears in the result.

## Start on Linux or macOS

Use a supported 64-bit Python 3.12 installation, Git and Node.js.

```bash
python3.12 scripts/setup.py
.venv/bin/python scripts/launch.py
```

Windows x64 is the verified platform. Setup selects the CPU wheel index on Windows and Linux and uses PyPI on macOS. Inference uses CPU by default. [Device and API details](docs/inference.md)

### Inference smoke test

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py --seed 4032486707
```

On Linux or macOS, use `.venv/bin/python` in the same command. This runs the original `infer.py` on the CPU and independently verifies the image, JSON structure and positive area under `outputs/smoke/`. [Measured verification results](docs/evaluation.md)

## Project layout

```text
FloorPlan-GenAI/
├── backend/                 FastAPI, job history and one cached inference worker
├── frontend/                React, Vite, Lucide and local dashboard styles
├── original_source/
│   └── Text2Floorplan/      Pinned Git checkout, MIT license, source and model assets
├── dataset/
│   └── samples/            Provenance and links to the original sample artifacts
├── docs/                   Architecture, mathematics, data, training and verification
├── outputs/
│   ├── generated/          Dashboard jobs, PNG and editable JSON
│   └── smoke/              Direct upstream CLI verification runs
├── scripts/                Setup, launch and CPU smoke test
└── source-lock.json        Pinned commit and checkpoint identity
```

## Viva notes

1. **Problem.** Generate an apartment layout from a compact program of room counts and a size category. Keep the result editable at the geometry level.
2. **Model.** A pretrained conditional Flow Matching Transformer predicts how XY corner coordinates move from Gaussian noise toward a plausible plan. A discrete head refines coordinates near the clean endpoint.
3. **Data.** Upstream uses normalized polygon corners, room identities, adjacency and attention masks from documented ResPlan and RPLAN sources, with pooled CLIP prompt features aligned by plan ID.
4. **Objective.** The trainer combines continuous velocity regression with a masked low-noise loss on quantized XY bits.
5. **Inference.** The prompt conditions frozen CLIP and a room graph sampler. The learned velocity is integrated with the Heun solver, then the runtime repairs polygons, calibrates scale and places doors and windows.
6. **Deployment.** A loopback FastAPI server serves React and queues jobs in a single worker. It loads the checksum-verified Flow Matching state and local CLIP encoder once.
7. **Validation.** The upstream runtime rejects certain invalid room geometries, doorway failures and undersized kitchen or bathroom clear areas. The dashboard compares actual room counts and measured areas and records reported rejections.
8. **Limits.** Size is a category and the application supports only the exposed room counts. Areas are calibrated estimates. The checks cover geometry and count consistency. The original raw training data and full training checkpoint are separate assets.

## Documentation

- [Operator runbook](docs/runbook.md)
- [Model architecture](docs/architecture.md)
- [Flow Matching mathematics](docs/flow_matching.md)
- [Dataset and preprocessing](docs/dataset.md)
- [Training pipeline](docs/training_pipeline.md)
- [Inference and API](docs/inference.md)
- [Evaluation and measured results](docs/evaluation.md)

## Provenance

The pinned Text2Floorplan checkout is commit `12d00c17e405e229e5a196c0c90a4d2dd2b14bdf` from Simon H.'s repository. It retains the original MIT license, model, training and data code, solver, geometry checks, opening placement, renderer and export schema. The application adds the React dashboard, FastAPI routes, a resident inference worker, the setup and verification scripts, and the academic documentation. The project setup can recreate the ignored nested source checkout and download the pinned weights to their original locations. [Upstream repository](https://github.com/simerionon/Text2Floorplan/tree/12d00c17e405e229e5a196c0c90a4d2dd2b14bdf) · [Source license](original_source/Text2Floorplan/LICENSE)

The checkpoint, CLIP and datasets have separate provenance and applicable terms. Check the release notes and third-party notices before redistributing upstream assets. [Flow Matching release](https://github.com/Simon777hh/Text2Floorplan/releases/tag/v1.1.0) · [CLIP model card](https://huggingface.co/openai/clip-vit-large-patch14) · [Third-party notices](original_source/Text2Floorplan/THIRD_PARTY_NOTICES.md)
