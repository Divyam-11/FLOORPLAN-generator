import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "original_source" / "Text2Floorplan"
LOCK = json.loads((ROOT / "source-lock.json").read_text(encoding="utf-8"))
ASSET = LOCK["checkpoint"]
CHECKPOINT = SOURCE / "checkpoints" / ASSET["filename"]
CLIP = SOURCE / "models" / "clip"
OUTPUTS = ROOT / "outputs" / "generated"


def checkpoint_matches(path=CHECKPOINT):
    if not path.is_file() or path.stat().st_size != ASSET["size"]:
        return False
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest() == ASSET["sha256"]


def clip_files_present():
    return all((CLIP / filename).is_file() for filename in (
        "config.json", "model.safetensors", "tokenizer_config.json",
    ))
