"""Provision the pinned upstream source, local model assets and application dependencies."""
import argparse
import os
import platform
import shutil
import struct
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.paths import (
    ASSET, CHECKPOINT, CLIP, LOCK, ROOT, SOURCE, checkpoint_matches, clip_files_present,
)


def run(*command, cwd=ROOT, env=None):
    subprocess.run(command, cwd=cwd, env=env, check=True)


def prepare_source():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", LOCK["url"], str(SOURCE))
        run("git", "checkout", "--detach", LOCK["commit"], cwd=SOURCE)
    actual = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != LOCK["commit"]:
        raise SystemExit(f"Expected upstream commit {LOCK['commit']}, found {actual} at {SOURCE}.")


def prepare_checkpoint():
    if checkpoint_matches():
        print(f"Verified cached Flow Matching checkpoint at {CHECKPOINT}", flush=True)
        return
    if CHECKPOINT.exists():
        raise SystemExit(f"Invalid checkpoint at {CHECKPOINT}. Move the file aside, then run setup again.")
    temporary = CHECKPOINT.with_suffix(".pth.partial")
    print(f"Downloading {ASSET['size']:,} bytes of Flow Matching weights", flush=True)
    request = urllib.request.Request(ASSET["url"], headers={"User-Agent": "FloorPlan-GenAI"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    if not checkpoint_matches(temporary):
        raise SystemExit(f"Checkpoint download failed size or SHA-256 verification at {temporary}.")
    temporary.replace(CHECKPOINT)
    print(f"Verified SHA-256 {ASSET['sha256']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true", help="Skip the frontend install and build.")
    args = parser.parse_args()
    print(f"Setup on {platform.system()} {platform.machine()} with Python {platform.python_version()}", flush=True)
    if sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8:
        parser.error("Use 64-bit Python 3.12 for the pinned dependency set.")
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not args.backend_only and not npm:
        parser.error("Install Node.js 22.12 or newer before running setup.")
    prepare_source()
    executable = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not executable.exists():
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    if sys.platform in {"win32", "linux"}:
        run(str(executable), "-m", "pip", "install", "torch==2.11.0",
            "--index-url", "https://download.pytorch.org/whl/cpu")
    run(str(executable), "-m", "pip", "install", "-r", str(ROOT / "backend" / "requirements.txt"))
    prepare_checkpoint()
    if clip_files_present():
        print(f"Using cached CLIP text encoder at {CLIP}", flush=True)
    else:
        environment = os.environ.copy()
        if sys.platform == "win32":
            environment.setdefault("HF_HUB_DISABLE_XET", "1")
        run(str(executable), str(SOURCE / "scripts" / "setup_models.py"), cwd=SOURCE, env=environment)
    if not args.backend_only:
        # npm.cmd needs Windows's command interpreter. All arguments are literals.
        npm_command = ["cmd.exe", "/d", "/c", npm] if sys.platform == "win32" else [npm]
        run(*npm_command, "ci", cwd=ROOT / "frontend")
        run(*npm_command, "run", "build", cwd=ROOT / "frontend")


if __name__ == "__main__":
    main()
