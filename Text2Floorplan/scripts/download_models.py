"""Download the three model files FloorPlan AI needs (about 3.7 GB in total).

1. Flow-matching checkpoint -> checkpoints/flow_matching_best.pth (193 MB, SHA-256 checked)
2. CLIP ViT-L/14 text encoder -> models/clip/ (about 470 MB)
3. Qwen2.5-1.5B-Instruct for the prompt planner -> models/llm/qwen2.5-1.5b-instruct/ (about 3 GB)

All downloads and caches stay inside the project folder.
Usage: python scripts/download_models.py
"""
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

CHECKPOINT = ROOT / "checkpoints" / "flow_matching_best.pth"
CHECKPOINT_URL = ("https://github.com/Simon777hh/Text2Floorplan/releases/download/"
                  "v1.1.0/flow_matching_best.pth")
CHECKPOINT_SHA256 = "11fe79209c433ec189b1d19b8626c20ff7319b71aac79915b54e570e89a3c6cb"
QWEN_DIR = ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct"
QWEN_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/"
QWEN_FILES = ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"]


def fetch(url: str, target: Path) -> None:
    if target.is_file():
        print(f"  already present: {target.relative_to(ROOT)}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("1/3 flow-matching checkpoint")
    fetch(CHECKPOINT_URL, CHECKPOINT)
    if digest(CHECKPOINT) != CHECKPOINT_SHA256:
        CHECKPOINT.unlink()
        print("  checksum mismatch, file removed; run again")
        return 1
    print("  checksum OK")

    print("2/3 CLIP text encoder")
    if (ROOT / "models" / "clip" / "config.json").is_file():
        print("  already present: models/clip")
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "setup_models.py")], check=True)

    print("3/3 Qwen2.5-1.5B-Instruct (prompt planner)")
    for name in QWEN_FILES:
        fetch(QWEN_URL + name, QWEN_DIR / name)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
