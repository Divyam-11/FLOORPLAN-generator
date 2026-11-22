import json
import logging
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event, RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paths import OUTPUTS, ROOT, SOURCE

LOGGER = logging.getLogger(__name__)
ROOM_TYPES = ("bedroom", "bathroom", "balcony")


class Conditions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    size: Literal["small", "medium", "large"] = "medium"
    bedrooms: int = Field(default=3, ge=1, le=17)
    bathrooms: int = Field(default=2, ge=1, le=17)
    balconies: int = Field(default=1, ge=0, le=17)
    seed: int | None = Field(default=None, ge=0, le=4294967295)

    @model_validator(mode="after")
    def validate_room_budget(self):
        # Reserve node slots for the living room, kitchen and entrance.
        if self.bedrooms + self.bathrooms + self.balconies > 17:
            raise ValueError("Choose up to 17 bedrooms, bathrooms and balconies combined.")
        return self

    def prompt(self):
        size = "" if self.size == "medium" else f"{self.size} "
        bathroom = "bathroom" if self.bathrooms == 1 else "bathrooms"
        balcony = "balcony" if self.balconies == 1 else "balconies"
        return (f"A {size}{self.bedrooms}-bedroom apartment with "
                f"{self.bathrooms} {bathroom} and {self.balconies} {balcony}.")


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def summarize_plan(plan, conditions):
    counts = Counter(room["type"] for room in plan["rooms"])
    requested = dict(zip(ROOM_TYPES, (conditions["bedrooms"], conditions["bathrooms"], conditions["balconies"])))
    policy = json.loads((SOURCE / "data" / "geometry_policy.json").read_text(encoding="utf-8"))
    limits = policy["minimum_room_areas_m2"]
    rooms = []
    for room in plan["rooms"]:
        kind = room["type"]
        if kind in {"front_door", "door"}:
            continue
        area = plan["room_areas_m2"].get(str(room["room_id"]))
        minimum = limits.get(kind)
        valid = area is not None and math.isfinite(area) and area > 0
        if valid and minimum is not None:
            valid = area + 1e-8 >= minimum
        rooms.append({
            "id": room["room_id"], "type": kind, "area_m2": area,
            "minimum_m2": minimum, "area_valid": valid,
        })
    generated = {kind: counts[kind] for kind in ROOM_TYPES}
    return {
        "seed": plan["seed"],
        "clip_prompt": plan["clip_prompt"],
        "total_area_m2": plan["total_area_m2"],
        "area_definition": plan["area_definition"],
        "rooms": rooms,
        "validation": {
            "upstream_passed": True,
            "requested": requested,
            "generated": generated,
            "counts_match": generated == requested,
            "areas_valid": all(room["area_valid"] for room in rooms),
        },
    }


class InferenceService:
    def __init__(self):
        self.lock = RLock()
        self.stopping = Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="floorplan")
        self.process = None
        self.jobs = {}
        self.health = {
            "state": "loading", "checkpoint": "pending", "clip": "pending",
            "device": None, "message": "Starting the local inference worker.",
        }
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        for path in OUTPUTS.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["id"] != path.parent.name:
                    continue
                self.jobs[job["id"]] = job
                if job["state"] in {"queued", "running"}:
                    self.update_job(job["id"], state="failed", finished_at=utc_timestamp(),
                                    message="Generation was interrupted when the server stopped.")
            except (OSError, ValueError, KeyError):
                LOGGER.exception("Unable to restore history from %s", path)
        self.executor.submit(self.start_worker)

    def snapshot(self):
        with self.lock:
            if self.health["state"] == "ready" and self.process.poll() is not None:
                self.health.update(state="error", message="Inference worker stopped. Restart the application.")
            return dict(self.health)

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The inference worker stopped. Check the server log and restart the application.")
        return json.loads(line)

    def start_worker(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "backend.worker"],
                cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            )
            if self.stopping.is_set():
                self.process.terminate()
                return
            while True:
                event = self.receive()
                event.pop("event")
                with self.lock:
                    self.health.update(event)
                if event.get("state") in {"ready", "error"}:
                    return
        except Exception as error:
            LOGGER.exception("Inference worker initialization failed")
            with self.lock:
                self.health.update(state="error", message=str(error))

    def update_job(self, job_id, **fields):
        with self.lock:
            job = {**self.jobs.get(job_id, {}), **fields, "id": job_id}
            path = OUTPUTS / job_id / "job.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            self.jobs[job_id] = job
            return dict(job)

    def submit(self, conditions):
        with self.lock:
            status = self.snapshot()
            if status["state"] != "ready":
                raise RuntimeError(status["message"])
            job_id = str(uuid4())
            (OUTPUTS / job_id).mkdir()
            job = self.update_job(
                job_id, state="queued", created_at=utc_timestamp(), conditions=conditions.model_dump(),
                prompt=conditions.prompt(), message="Waiting for the inference worker.", rejections=[],
            )
            self.executor.submit(self.run_job, job_id)
            return job

    def list_jobs(self):
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job["created_at"], reverse=True)

    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def run_job(self, job_id):
        if self.stopping.is_set():
            return
        started = time.perf_counter()
        job = self.update_job(job_id, state="running", started_at=utc_timestamp(),
                              message="Sampling room geometry.")
        directory = OUTPUTS / job_id
        rejections = []
        try:
            request = {"prompt": job["prompt"], "seed": job["conditions"]["seed"], "language": "en"}
            self.process.stdin.write(json.dumps({"directory": str(directory), "request": request}) + "\n")
            self.process.stdin.flush()
            while True:
                event = self.receive()
                kind = event.pop("event")
                if kind == "health":
                    with self.lock:
                        self.health.update(event)
                elif kind == "progress":
                    message = event["message"]
                    if message.startswith(("Geometry check failed:", "Layout check failed:")):
                        rejections.append(message)
                    self.update_job(job_id, message=message, rejections=list(rejections))
                elif kind == "failed":
                    self.finish_job(job_id, started, "failed", event["message"])
                    return
                elif kind == "completed":
                    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
                    if not (directory / "plan.png").is_file():
                        raise RuntimeError("The upstream runtime did not write the preview PNG.")
                    self.finish_job(job_id, started, "completed", "Generation complete.",
                                    result=summarize_plan(plan, job["conditions"]))
                    return
        except Exception as error:
            LOGGER.exception("Generation %s failed", job_id)
            if self.process is None or self.process.poll() is not None:
                with self.lock:
                    self.health.update(state="error", message="Inference worker stopped. Restart the application.")
            self.finish_job(job_id, started, "failed", str(error))

    def finish_job(self, job_id, started, state, message, **fields):
        self.update_job(job_id, finished_at=utc_timestamp(), state=state, message=message,
                        duration_seconds=round(time.perf_counter() - started, 2),
                        device=self.snapshot()["device"], **fields)

    def close(self):
        self.stopping.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.executor.shutdown(cancel_futures=True)
        if self.process is not None:
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
