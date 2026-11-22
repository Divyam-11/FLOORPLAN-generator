"""Run Raster-to-Graph inference on an image or a directory of images."""

import argparse
from collections import Counter
from pathlib import Path
import random
from time import perf_counter

import torch
from PIL import UnidentifiedImageError

from deployment.download import CHECKPOINT
from deployment.export import make_graph, write_outputs
from deployment.predict import load_model, predict, prepare_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="image file or directory of images")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="model checkpoint (.pth)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--confidence", type=float, default=0.5, help="junction confidence threshold")
    parser.add_argument("--max-steps", type=int, default=64, help="maximum prediction iterations per image")
    args = parser.parse_args()

    if args.threads < 1 or args.max_steps < 1:
        parser.error("--threads and --max-steps must be positive")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run python -m deployment.download first.")
    if args.input.is_dir():
        images = sorted(
            path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    elif args.input.is_file():
        images = [args.input]
    else:
        parser.error(f"input not found: {args.input}")
    if not images:
        parser.error(f"no supported images found in {args.input}")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable. Use --device cpu.")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    size = len(images)
    stems = Counter(image.stem.casefold() for image in images)
    output_names = [
        image.name if stems[image.stem.casefold()] > 1 else image.stem for image in images
    ]
    if len({name.casefold() for name in output_names}) != size:
        parser.error("one or more filenames would produce the same output name")
    input_paths = {path.resolve() for path in images}
    for name in output_names:
        destinations = {(args.output / f"{name}.{extension}").resolve() for extension in ("png", "json")}
        if destinations & input_paths:
            parser.error("an output would overwrite an input image. Choose a different --output directory.")

    print(f"Loading model on {device.type}...", flush=True)
    model = load_model(args.checkpoint, device)
    failures = 0
    for index, (image, name) in enumerate(zip(images, output_names), start=1):
        started = perf_counter()
        try:
            canvas, transform = prepare_image(image)
        except (UnidentifiedImageError, OSError) as error:
            print(f"[{index}/{size}] Could not read {image.name}: {error}", flush=True)
            failures += 1
            continue
        random.seed(42)
        prediction = predict(model, canvas, device, args.confidence, args.max_steps)
        graph = make_graph(prediction, image.name, transform)
        write_outputs(args.output, name, canvas, graph)
        print(
            f"[{index}/{size}] {image.name}: {len(graph['nodes'])} junctions, "
            f"{len(graph['edges'])} walls, {len(graph['rooms'])} rooms "
            f"({graph['stop_reason']}, {perf_counter() - started:.1f}s)",
            flush=True,
        )
    print(f"Results: {args.output.resolve()}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
