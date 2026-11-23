"""Measure our generator on the metrics prior work reports (one sample, no retries).

Groups of 5, 6, 7 and 8 rooms, as in House-GAN++, HouseDiffusion and LLM+RLVR.
Per sample, on the flow model's own output:
  GED       graph edit distance between the requested room graph and the graph of
            the generated plan: one per missing room (plus its connections) and one
            per requested connection whose rooms do not share a wall long enough for
            a door (measured after generic geometry clean-up, before door repair).
            Doors are only placed on requested connections, so extra edges cannot occur.
  Overlap   1 if any two rooms overlap by more than 1% of the smaller room (RLVR).
  %Overlap  total pairwise overlap area / total room area (RLVR).
  Room ID   1 if every requested room is present in the output (RLVR).
  Valid     1 if the plan passes all of the app's checks without a retry.
Usage: T2F_PLANNER=rules .venv/bin/python report/eval_compat.py [seeds_per_group]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("TEXT2REVIT_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import backend_cli as backend  # noqa: E402

GROUPS = {
    5: "A 2-bedroom apartment with a living room, a kitchen and 1 bathroom.",
    6: "A 2-bedroom apartment with a living room, a kitchen, 1 bathroom and 1 balcony.",
    7: "A 3-bedroom apartment with a living room, a kitchen and 2 bathrooms.",
    8: "A 3-bedroom apartment with a living room, a kitchen, 2 bathrooms and 1 balcony.",
}


def sample(prompt, seed):
    metrics = backend.pipeline.LAST_METRICS
    metrics.clear()
    started = time.perf_counter()
    valid = True
    try:
        processed, overlays, md = backend.generate_plan_data(prompt, seed, 0)
        openings = backend.flatten_openings(overlays)
        backend.validate_door_connections(md["rooms"], openings, md["edge_lists"])
        walls, _ = backend.build_revit_geometry(md["rooms"], openings, md["pixel_to_meter"], md["seed"])
        backend.validate_room_areas(md["rooms"], md["pixel_to_meter"], walls=walls)
    except ValueError:
        valid = False
    seconds = time.perf_counter() - started
    if "requested" not in metrics:
        return None
    requested, present = set(metrics["requested"]), set(metrics["present"])
    missing = requested - present
    planned = [tuple(p) for p in metrics["planned"]]
    lost_edges = sum(1 for p in planned if set(p) & missing)
    ged = len(missing) + lost_edges + metrics.get("unfit", 0) if "unfit" in metrics else None
    return {"seed": seed, "valid": valid, "ged": ged, "overlap": metrics["overlap"],
            "overlap_fraction": metrics["overlap_fraction"], "room_id": not missing,
            "rooms": len(requested), "seconds": seconds}


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = {}
    for rooms, prompt in GROUPS.items():
        rows = [r for r in (sample(prompt, s) for s in range(2000, 2000 + seeds)) if r]
        geds = [r["ged"] for r in rows if r["ged"] is not None]
        out[rooms] = {
            "prompt": prompt, "samples": len(rows),
            "ged": round(statistics.mean(geds), 2) if geds else None, "ged_n": len(geds),
            "overlap": round(statistics.mean(r["overlap"] for r in rows), 2),
            "overlap_pct": round(statistics.mean(r["overlap_fraction"] for r in rows), 3),
            "room_id": round(statistics.mean(r["room_id"] for r in rows), 2),
            "valid": round(statistics.mean(r["valid"] for r in rows), 2),
            "seconds": round(statistics.mean(r["seconds"] for r in rows), 1),
            "rows": rows,
        }
        g = out[rooms]
        print(f"{rooms} rooms: GED {g['ged']} (n={g['ged_n']}) | Overlap {g['overlap']} | "
              f"%Overlap {g['overlap_pct']} | Room ID {g['room_id']} | valid {g['valid']} | "
              f"{g['seconds']}s", flush=True)
    (ROOT / "report" / "results_compat.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
