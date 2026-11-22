from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference_service import Conditions, InferenceService
from .paths import OUTPUTS, ROOT, SOURCE


@asynccontextmanager
async def lifespan(app):
    app.state.inference = InferenceService()
    try:
        yield
    finally:
        app.state.inference.close()


app = FastAPI(title="FloorPlan AI", version="1.0", lifespan=lifespan)


@app.get("/api/health")
def health(request: Request):
    return request.app.state.inference.snapshot()


@app.get("/api/jobs")
def history(request: Request):
    return request.app.state.inference.list_jobs()


@app.post("/api/jobs", status_code=202)
def generate(conditions: Conditions, request: Request):
    try:
        return request.app.state.inference.submit(conditions)
    except RuntimeError as error:
        raise HTTPException(503, detail=str(error)) from error


@app.get("/api/jobs/{job_id}")
def job(job_id: UUID, request: Request):
    result = request.app.state.inference.get_job(str(job_id))
    if result is None:
        raise HTTPException(404, detail="Generation not found.")
    return result


@app.get("/api/jobs/{job_id}/files/{format}")
def artifact(job_id: UUID, format: Literal["png", "json"], request: Request, download: bool = False):
    if job(job_id, request)["state"] != "completed":
        raise HTTPException(409, detail="Generation has no completed output yet.")
    path = OUTPUTS / str(job_id) / f"plan.{format}"
    if not path.is_file():
        raise HTTPException(404, detail="The generated file is missing.")
    filename = f"floorplan-{str(job_id)[:8]}.{format}" if download else None
    return FileResponse(path, media_type="image/png" if format == "png" else "application/json",
                        filename=filename)


SAMPLES = {
    "browser.png": "floorplan.png", "browser.json": "plan.json",
    "revit.png": "revit-floorplan.png", "revit.json": "revit-plan.json",
}
PRIORS = {
    "combined_area_stats_natural.json", "corner_count_distributions.json", "geometry_policy.json",
    "raw_recovery_scale_stats.json", "room_aspect_ranges.json", "room_aspect_samples.npz",
}


@app.get("/api/dataset/samples/{filename}")
def sample(filename: str):
    name = SAMPLES.get(filename)
    path = SOURCE / "sample" / name if name else None
    if path is None or not path.is_file():
        raise HTTPException(404, detail="Upstream example not found.")
    return FileResponse(path)


@app.get("/api/dataset/priors/{filename}")
def prior(filename: str):
    path = SOURCE / "data" / filename
    if filename not in PRIORS or not path.is_file():
        raise HTTPException(404, detail="Upstream prior not found.")
    return FileResponse(path, filename=filename)


frontend = ROOT / "frontend" / "dist"
if (frontend / "index.html").is_file():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
else:
    @app.get("/")
    def unbuilt_frontend():
        return {"message": "Build the frontend with scripts/setup.py or run the Vite development server."}
