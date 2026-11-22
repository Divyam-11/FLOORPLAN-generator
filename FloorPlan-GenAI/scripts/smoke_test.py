"""Run the original command-line inference on CPU and verify its PNG and JSON."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.paths import CHECKPOINT, CLIP, ROOT, SOURCE, checkpoint_matches, clip_files_present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A 3-bedroom apartment with 2 bathrooms and 1 balcony.")
    parser.add_argument("--seed", type=int, default=4032486707)
    args = parser.parse_args()
    if not checkpoint_matches() or not clip_files_present():
        parser.error("Run scripts/setup.py to prepare the verified checkpoint and CLIP encoder.")
    output = ROOT / "outputs" / "smoke"
    existing = set(output.glob("floorplan_*.json"))
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "4")
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, str(SOURCE / "infer.py"), "--prompt", args.prompt,
         "--device", "cpu", "--seed", str(args.seed), "--output-dir", str(output),
         "--checkpoint", str(CHECKPOINT), "--clip-dir", str(CLIP)],
        cwd=SOURCE, env=environment, check=True,
    )
    plans = set(output.glob("floorplan_*.json")) - existing
    if not plans:
        raise RuntimeError("Inference returned success without an editable JSON file.")
    plan_path = max(plans, key=lambda path: path.stat().st_mtime)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    png_path = plan_path.with_suffix(".png")
    from PIL import Image
    with Image.open(png_path) as image:
        image.verify()
    if not plan["rooms"] or not plan["walls"] or not plan["openings"] or plan["total_area_m2"] <= 0:
        raise RuntimeError("The result is missing its rooms, walls, openings or positive floor area.")
    print(f"Verified {len(plan['rooms'])} room records and {plan['total_area_m2']:.2f} m², "
          f"final seed {plan['seed']}, {time.perf_counter() - started:.1f} seconds.")
    print(plan_path)
    print(png_path)


if __name__ == "__main__":
    main()
