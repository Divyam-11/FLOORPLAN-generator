# FloorPlan AI operator runbook

Use this runbook to install, launch, verify, and recover the local FloorPlan AI application. Run commands from the FloorPlan-GenAI project folder. The application uses pretrained inference, with training execution disabled.

## Before the demo

Use Git, 64-bit Python 3.12, and Node.js 22.12 or newer. The first setup needs internet access for packages and model downloads. The verified environment was Windows x64 on CPU.

On Windows, run first-time setup in PowerShell.

```powershell
py -3.12 scripts/setup.py
```

On Linux or macOS, use the corresponding Python executable.

```bash
python3.12 scripts/setup.py
```

Setup retrieves the pinned Text2Floorplan checkout, builds the frontend, verifies the released Flow Matching checkpoint, and prepares the local CLIP encoder. Subsequent launches use the cached models. The source revision and expected checkpoint digest are in `source-lock.json`.

## Start and verify the service

1. Start the app in a terminal and keep that terminal open.

   On Windows

   ```powershell
   .\.venv\Scripts\python.exe scripts/launch.py
   ```

   On Linux or macOS

   ```bash
   .venv/bin/python scripts/launch.py
   ```

2. Open [the dashboard](http://127.0.0.1:8000/) and wait for **Engine ready**.
3. If readiness is unclear, check [the health endpoint](http://127.0.0.1:8000/api/health). Expect `state` to be `ready`, both `checkpoint` and `clip` to be `loaded`, and `device` to be `cpu` on the default installation.

   ```powershell
   Invoke-RestMethod 'http://127.0.0.1:8000/api/health'
   ```

## Run the demo

1. Select **Medium**, **3 bedrooms**, **2 bathrooms**, and **1 balcony**. Use seed `4032486707` for the Windows CPU demo case.
2. Select **Generate floor plan**. Leave the server running while the job is queued or generating. The verified demo case took about 17 seconds. A run that needed three retries took about two minutes.
3. Confirm the preview appears and the result lists the final seed, clear area, and validation status. The upstream generator may use another seed after rejecting a candidate.
4. Download the PNG and editable JSON. Open **Evaluation** to compare requested and generated counts, inspect the area checks, and review any candidate rejections.

Jobs and exports persist in `outputs/generated/<job-id>/`. The JSON contains room polygons, walls, openings, and calibrated clear areas. The evaluation checks geometry and room counts. Building-code and architectural certification are outside its scope.

### Independent CPU smoke test

Use this if the dashboard or API fails and you need to test inference directly.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py --seed 4032486707
```

On Linux or macOS, substitute `.venv/bin/python`. Success prints the final seed, estimated clear area, and paths to the verified PNG and JSON under `outputs/smoke/`.

## Diagnose and recover

| Symptom | Action |
| --- | --- |
| Port 8000 is in use | Check `/api/health` to see whether FloorPlan AI is already running. If another service owns the port, add `--port 8001` to the launch command and open `http://127.0.0.1:8001/`. |
| The home page reports that the frontend is unbuilt | Run setup from the project folder, then restart the service. The launcher serves `frontend/dist/`. |
| The engine is unavailable | Read `/api/health` and the launcher terminal for the worker error. Rerun setup for missing assets. If setup identifies a checkpoint checksum mismatch, move the invalid checkpoint aside and rerun setup. |
| A job shows a geometry rejection | Leave the worker running while it retries. If the run becomes **Failed**, read the diagnostic and submit a new seed or the demo configuration above. |
| A previously running job is marked failed after restart | The worker has restarted. Resubmit the conditions. Completed jobs remain in **Previous Generations**. |

To stop the service, wait for active work to complete, then press **Ctrl+C** in the launcher terminal. Keep `outputs/generated/` if you want to preserve the demo history. The standard launcher binds the service to the local machine.
