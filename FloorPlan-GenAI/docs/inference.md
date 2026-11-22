# Local inference and API

## Assets and readiness

`scripts/setup.py` resolves project paths from its own location, provisions the pinned checkout and isolated environment, and builds the frontend. It installs and checks the inference-only release at `original_source/Text2Floorplan/checkpoints/flow_matching_best.pth`. The file is **193,053,154 bytes**.

```text
SHA-256  11fe79209c433ec189b1d19b8626c20ff7319b71aac79915b54e570e89a3c6cb
```

The upstream `scripts/setup_models.py` saves the CLIP text encoder and tokenizer to `original_source/Text2Floorplan/models/clip/`. The worker checks the PTH checksum before loading, then exercises the local CLIP model and verifies a finite 768-dimensional embedding. The readiness endpoint reports success after both models load. Normal generation uses local assets and sends no prompts to hosted services. [Upstream inference instructions](../original_source/Text2Floorplan/docs/inference.md)

## Application request path

The React form maps the size and counts to a dataset-style English prompt. For medium it omits the size adjective, as upstream does. The model always includes a living room and a kitchen. A single FastAPI executor starts an isolated Python worker subprocess and sends one job at a time using JSON over pipes. The worker invokes the original `backend_cli.main()` for topology, sampling, retries, PNG rendering and editable JSON. It injects its resident Flow model and a cached implementation of the same upstream CLIP tokenization, pooling and output conversion.

The upstream backend can try up to ten candidates, including when a required doorway is missing or a clear-area check fails. The supplied seed is the initial attempt. A rejected attempt can use a new seed, so use the recorded final seed to revisit an accepted configuration. Numerical geometry can vary across devices and runtime versions. There is no application wall-clock inference timeout.

## API reference

FastAPI also serves [interactive API docs](http://127.0.0.1:8000/docs) while running.

| Method and path | Result |
| --- | --- |
| `GET /api/health` | Worker state, loaded assets, device and settings |
| `POST /api/jobs` | HTTP 202 with queued job ID |
| `GET /api/jobs` | Saved job history and comparison summaries |
| `GET /api/jobs/{id}` | Queued, running, completed or failed job with progress and rejection messages |
| `GET /api/jobs/{id}/files/png` | Generated preview |
| `GET /api/jobs/{id}/files/json` | Editable geometry |

Append `?download=true` to either result URL for an attachment. The JSON retains the upstream editor schema. The included `preview_image` field is informational and records the path from the machine that generated the plan.

```json
{
  "size": "medium",
  "bedrooms": 3,
  "bathrooms": 2,
  "balconies": 1,
  "seed": 4032486707
}
```

Choose at least one bedroom and one bathroom, zero or more balconies and at most 17 of these rooms combined to fit the topology budget. The optional seed is an unsigned 32-bit integer. Invalid requests return HTTP 422. A worker whose models have not loaded returns HTTP 503 for new jobs. The server binds to loopback and uses one process. Finished jobs survive restarts. Queued or running jobs interrupted by a stop are marked failed when the service starts again.

## Development and devices

The built frontend and API share port 8000 when launched with `scripts/launch.py`. To use Vite during development, run `npm run dev` from `frontend/` while the API is running. Vite proxies `/api` to `127.0.0.1:8000`. `FLOORPLAN_API_URL` can override that development target. `scripts/setup.py --backend-only` skips the frontend package step.

The default setup uses a CPU build of PyTorch on Windows and Linux. Use `FLOORPLAN_THREADS` to change the default four worker threads. If the environment already has a compatible CUDA-enabled PyTorch build, `FLOORPLAN_DEVICE=auto` allows the upstream runtime to choose CUDA and fall back to CPU. CUDA was not exercised during verification. CPU remains the default on Apple systems.
