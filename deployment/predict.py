"""Decode wall junctions, wall segments and room labels from a floor plan."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from zipfile import is_zipfile

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as transforms

from args import get_args_parser
from models.build import build_model, build_postprocessor
from util.data_utils import draw_preds_on_tensors, get_edges_amount, initialize_tensors
from util.data_utils import is_stop, point_inside, random_keep, remove_points
from util.edges_utils import edges as edge_directions
from util.mean_std import mean, std
from util.misc import NestedTensor, nested_tensor_from_tensor_list


CANVAS_SIZE = 512
ALIGNMENT_TOLERANCE = 5
# Direction index follows the checkpoint's edge classes: up, left, down, right.
DIRECTIONS = ((1, -1), (0, -1), (1, 1), (0, 1))
ARCHITECTURE_ARGS = (
    "backbone", "num_feature_maps", "num_classes", "num_classes_edges", "enc_layers",
    "dec_layers", "dim_feedforward", "hidden_dim", "dropout", "nheads", "num_queries",
    "dec_n_points", "enc_n_points",
)


def load_model(checkpoint: Path, device: torch.device):
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=is_zipfile(checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError("Expected a model state dictionary or a checkpoint containing 'model'.")

    args = get_args_parser().parse_args([])
    saved_args = state.get("args", {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    if isinstance(saved_args, Mapping):
        for name in ARCHITECTURE_ARGS:
            if name in saved_args:
                setattr(args, name, saved_args[name])
    args.device = "cpu"
    model = build_model(args, pretrained_backbone=False)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to(device).eval()


def prepare_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")

    width, height = image.size
    scale = CANVAS_SIZE / max(width, height)
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    offset = tuple((CANVAS_SIZE - side) // 2 for side in resized)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    canvas.paste(image.resize(resized, Image.Resampling.LANCZOS), offset)
    transform = {"source_size": [width, height], "resized_size": resized, "offset": offset}
    return canvas, transform


def position(point):
    return tuple(point["points"].tolist())


def nearest_in_direction(origin, candidates, direction):
    axis, sign = DIRECTIONS[direction]
    cross_axis = 1 - axis
    nearest = None
    best_distance = float("inf")
    for candidate, location in candidates:
        distance = sign * (location[axis] - origin[axis])
        if 0 < distance < best_distance and abs(location[cross_axis] - origin[cross_axis]) <= ALIGNMENT_TOLERANCE:
            nearest, best_distance = candidate, distance
    return nearest


def connect_layer(points, neighbors, edge_field, reciprocal=False):
    edges = {}
    rejected = []
    candidates = [(point, position(point)) for point in neighbors]
    for point in points:
        flags = edge_directions.get(point[edge_field].item())
        if flags is None:
            rejected.append(point)
            continue
        origin = position(point)
        for direction, present in enumerate(flags):
            if present == "0":
                continue
            neighbor = nearest_in_direction(origin, candidates, direction)
            if neighbor is None:
                rejected.append(point)
                break
            if reciprocal:
                reverse_flags = edge_directions.get(neighbor[edge_field].item(), "0000")
                if reverse_flags[(direction + 2) % 4] == "0":
                    rejected.append(point)
                    break
            key = tuple(sorted((origin, position(neighbor))))
            edges.setdefault(key, (point, neighbor))

    valid_edges = [
        (start, end) for start, end in edges.values()
        if not point_inside(start, rejected) and not point_inside(end, rejected)
    ]
    return remove_points(rejected, points), valid_edges, rejected


@torch.inference_mode()
def predict(model, canvas, device, confidence, max_steps):
    postprocess = build_postprocessor()
    image = transforms.normalize(transforms.to_tensor(canvas), mean, std)
    samples = nested_tensor_from_tensor_list([image]).to(device)
    raster, mask = samples.decompose()
    raster, _ = initialize_tensors(raster)
    target_size = torch.tensor([[CANVAS_SIZE, CANVAS_SIZE]], device=device)
    layers = []
    given = []
    stop = 0

    for _ in range(max_steps):
        output = model(NestedTensor(raster, mask))
        current = {key: value.cpu() for key, value in postprocess(output, target_size)[0].items()}
        edge_indices = set(torch.where(current["edges"] != 0)[0].tolist())
        score_indices = set(torch.where(current["scores"] > confidence)[0].tolist())
        points = random_keep([
            {key: values[index] for key, values in current.items()}
            for index in edge_indices.intersection(score_indices)
        ])
        stop = is_stop(points)
        if stop:
            break

        previous_edges = []
        if given:
            points, previous_edges, _ = connect_layer(points, given, "last_edges")
        stop = is_stop(points)
        if stop:
            break

        current_edges = []
        if len(points) > 1:
            points, current_edges, rejected = connect_layer(points, points, "this_edges", reciprocal=True)
            previous_edges = [
                (start, end) for start, end in previous_edges
                if not point_inside(start, rejected) and not point_inside(end, rejected)
            ]
        stop = is_stop(points)
        if stop:
            break

        layers.append((points, previous_edges, current_edges))
        given.extend(points)
        raster, _ = draw_preds_on_tensors(layers, raster)

    return stop, get_edges_amount(layers), layers
