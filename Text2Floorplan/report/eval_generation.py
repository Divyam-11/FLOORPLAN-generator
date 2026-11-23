"""Measure single-sample validity, room-count accuracy, topology GED and speed.

Each (prompt, seed) is one model sample with no retries, so the numbers describe
the generator itself. Checks are the same ones the app applies before showing a
plan (geometry clean-up, doors, wall geometry, minimum areas).

raw_connection_recall is the share of requested room connections whose rooms
already share a wall long enough for a door in the model's output, before the
door-repair step. It measures how well the generative model itself follows the
requested topology.

GED is the graph edit distance between the room graph the model was asked for
(planned connections) and the graph of doors actually generated, with unit
costs: one per missing/extra room plus one per missing/extra connection. This
mirrors the "compatibility" GED of House-GAN++ / HouseDiffusion, but on our own
prompts, so it is not directly comparable to their RPLAN test split.

Usage: .venv/bin/python report/eval_generation.py [seeds_per_prompt]
Writes report/results_generation.json.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("TEXT2REVIT_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import backend_cli as backend  # noqa: E402

# Reuse LLM replies cached by eval_planner.py so model-loading time does not
# distort the per-sample timings (LLM time is reported separately).
_CACHE = ROOT / "report" / "llm_cache.json"
_planner = backend.pipeline.prompt_planner
_ask = _planner._ask_llm


def _cached_ask(prompt):
    cache = json.loads(_CACHE.read_text()) if _CACHE.is_file() else {}
    if prompt not in cache:
        started = time.perf_counter()
        cache[prompt] = {"reply": _ask(prompt), "seconds": round(time.perf_counter() - started, 2)}
        _CACHE.write_text(json.dumps(cache, indent=1))
    return cache[prompt]["reply"]


_planner._ask_llm = _cached_ask

PROMPTS = [
    # Original fixed-format prompts (what the released model was trained on).
    "A small 1-bedroom apartment with 1 bathroom.",
    "A small 2-bedroom apartment with 1 bathroom and 1 balcony.",
    "A 2-bedroom apartment with 1 bathroom and 1 balcony.",
    "A 3-bedroom apartment with 2 bathrooms and 1 balcony.",
    "A large 3-bedroom apartment with 2 bathrooms and 2 balconies.",
    "A large 4-bedroom apartment with 2 bathrooms and 2 balconies.",
    # Free-form prompts handled by prompt_planner.py.
    "2 bedroom flat with a dining room, a kitchen, 1 bathroom and a terrace",
    "3bhk with a pooja room, a study, 2 bathrooms and a balcony garden",
]
FUNCTIONAL = {"living", "kitchen", "bedroom", "bathroom", "balcony", "storage"}


def evaluate(prompt: str, seed: int) -> dict:
    started = time.perf_counter()
    record = {"prompt": prompt, "seed": seed}
    try:
        processed, overlays, metadata = backend.generate_plan_data(prompt, seed, 0)
    except ValueError as error:  # rejected during geometry clean-up
        record.update(valid=False, stage="geometry", error=str(error).split(":")[0][:60])
        record["seconds"] = time.perf_counter() - started
        return record
    planned_all = [p for p in metadata["planned_edge_lists"]
                   if all(metadata["node_type_ids"][i] != 7 for i in p)]
    record["raw_connection_recall"] = round(
        1 - metadata["restore"].get("pre_repair_unfit", 0) / max(1, len(planned_all)), 3)
    try:
        openings = backend.flatten_openings(overlays)
        backend.validate_door_connections(metadata["rooms"], openings, metadata["edge_lists"])
        walls, _ = backend.build_revit_geometry(
            metadata["rooms"], openings, metadata["pixel_to_meter"], metadata["seed"])
        areas = backend.validate_room_areas(metadata["rooms"], metadata["pixel_to_meter"], walls=walls)
    except ValueError as error:  # rejected by door / wall / area checks
        record.update(valid=False, stage="layout", error=str(error).split(":")[0][:60])
        record["seconds"] = time.perf_counter() - started
        return record

    rooms = [r for r in metadata["rooms"] if r["type"] in FUNCTIONAL]
    present = {r["room_id"] for r in rooms}
    generated = collections.Counter(r["type"] for r in rooms)
    requested = {k: v for k, v in metadata["requested_slots"].items() if v}
    requested["living"] = 1
    planned_nodes = {i for pair in metadata["planned_edge_lists"] for i in pair
                     if metadata["node_type_ids"][i] != 7}  # 7 = front_door
    planned = {frozenset(p) for p in metadata["planned_edge_lists"]
               if all(metadata["node_type_ids"][i] != 7 for i in p)}
    doors = {frozenset((o["owner_room_id"], o["target_room_id"])) for o in openings
             if o["type"] != "window" and o.get("target_room_id") is not None
             and o["owner_room_id"] in present and o["target_room_id"] in present}
    node_cost = len(planned_nodes ^ present)
    edge_cost = len(planned ^ doors)
    record.update(
        valid=True,
        counts_match=all(generated.get(k, 0) == v for k, v in requested.items())
        and sum(generated.values()) == sum(requested.values()),
        ged=node_cost + edge_cost,
        rooms=len(present),
        area_m2=round(sum(areas.values()), 1),
        final_seed=metadata["seed"],
    )
    record["seconds"] = time.perf_counter() - started
    return record


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    results = []
    for prompt in PROMPTS:
        backend.pipeline.prompt_planner.plan_prompt(prompt)  # warm the LLM cache
        for seed in range(1000, 1000 + seeds):
            result = evaluate(prompt, seed)
            results.append(result)
            print(json.dumps(result), flush=True)
    out = ROOT / "report" / "results_generation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
