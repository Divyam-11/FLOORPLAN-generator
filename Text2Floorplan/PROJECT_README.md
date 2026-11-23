# FloorPlan AI: Generative AI (UCS748) project

| Name | Roll no. | Group |
|---|---|---|
| Lakksh Sharma | 102303836 | 4C24 |
| Ishita Gupta | 102303377 | 4C24 |
| Divyam Jund | 102303445 | 4C26 |
| Akhil Gupta | 102315121 | 4O15 |

Repository: https://github.com/lkksharma/FLOORPLAN-generator/tree/main

FloorPlan AI turns a short English sentence ("a 3BHK with a pooja room, a study and a
balcony garden") into an editable vector floor plan with doors, windows and room areas.
It runs offline on a laptop CPU.

How it works:
- A local LLM (Qwen2.5-1.5B) and a rule reader decide the rooms.
- Rules build the room graph.
- A flow-matching Transformer generates the layout. We used the released Text2Floorplan weights.
- Geometry rules add doors and windows.

The full write-up is `report/FloorPlanAI_UCS748_Report.pdf`.

## What is in this zip

| Path | Contents |
|---|---|
| `prompt_planner.py` | **Our main addition**: LLM + rule prompt planner, room-graph rules, en-suites, staged fallback, connection repair |
| `pipeline_generate.py`, `backend_cli.py`, `infer.py`, `topology_rules.py` | Generation pipeline. We modified these to use the planner, show room labels, retry and record metrics |
| `model.py`, `flow_solver.py`, `train.py`, `losses.py`, `inference_runtime.py` | Flow-matching model, Heun solver, training code (from Text2Floorplan) |
| `plan_postprocess.py`, `door_window_rules.py`, `geometry_cleanup.py`, `plan_quality.py`, `revit_geometry.py` | Geometry clean-up, doors, windows, checks |
| `web_editor/` | Browser editor with 2D and 3D views (we added custom room labels) |
| `data/` | Small statistical priors the model needs at inference |
| `report/` | Evaluation scripts, raw results, charts, UI screenshots, LaTeX report and PDF |
| `outputs/` | Example generated plans (PNG + JSON) |
| `scripts/download_models.py` | Downloads the model weights (not included because of size) |

The model weights (about 3.7 GB in total) are **not** included. The download script fetches them.

## Setup (Python 3.12)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_models.py
```

## Run

One plan from the command line. The PNG and JSON are saved to `outputs/inference/`:

```bash
.venv/bin/python infer.py --prompt "2 bedroom flat, master bedroom with en-suite, a study and a balcony garden" --device cpu
```

Browser editor:

```bash
.venv/bin/python -m web_editor.server --open
```

Add `--planner rules` to `infer.py` to skip the LLM and use only the rule reader.

## Reproduce the evaluation

```bash
.venv/bin/python report/eval_planner.py
T2F_PLANNER=rules .venv/bin/python report/eval_compat.py 12
.venv/bin/python report/eval_generation.py 5
.venv/bin/python report/make_charts.py
```

- `eval_planner.py` measures prompt reading on 30 labelled prompts.
- `eval_compat.py` measures GED, overlap and Room ID for 5–8 rooms.
- `eval_generation.py` measures validity and speed.

## Credits

The generative core is Text2Floorplan (MIT): https://github.com/simerionon/Text2Floorplan.
`LICENSE` and `THIRD_PARTY_NOTICES.md` are kept from upstream.
