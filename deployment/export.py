"""Write graph data and a side-by-side raster and graph preview."""

import json

from PIL import Image, ImageDraw

from util.graph_utils import get_cycle_basis_and_semantic
from util.metric_utils import get_results_float_with_semantic
from util.semantics_dict import semantics_dict_rev


QUADRANTS = ("upper_left", "upper_right", "lower_right", "lower_left")
ROOM_COLORS = {
    "living_room": "#F7DDAD", "kitchen": "#F7C9A9", "bedroom": "#CCE5DA",
    "bathroom": "#C4DCF2", "restroom": "#D6D7F3", "balcony": "#D8E8BF",
    "closet": "#E7D8EE", "corridor": "#EBE5D7", "washing_room": "#C8EAEB",
}
STOP_REASONS = {0: "step_limit", 1: "no_valid_junctions", 2: "end_token"}


def label(class_id):
    return semantics_dict_rev.get(class_id, "no_type")


def make_graph(prediction, input_name, transform):
    stop, _, layers = prediction
    junctions, segments = get_results_float_with_semantic(prediction)
    junctions = list(dict.fromkeys(junctions))
    ids = {junction: index for index, junction in enumerate(junctions)}
    edges = sorted({tuple(sorted((ids[start], ids[end]))) for start, end in segments if start != end})
    _, cycles, room_classes = get_cycle_basis_and_semantic(prediction)
    return {
        "input": input_name,
        "coordinate_space": "512x512 image canvas (origin is top left, x right, y down)",
        "input_transform": transform,
        "stop_reason": STOP_REASONS[stop],
        "accepted_layers": len(layers),
        "nodes": [
            {"id": index, "x": x, "y": y, "quadrants": dict(zip(QUADRANTS, map(label, classes)))}
            for index, (x, y, *classes) in enumerate(junctions)
        ],
        "edges": [{"source": source, "target": target} for source, target in edges],
        "rooms": [
            {"label": label(class_id), "boundary": [ids[vertex] for vertex in cycle]}
            for cycle, class_id in zip(cycles, room_classes)
        ],
    }


def render(canvas, graph):
    width, height = canvas.size
    header = 34
    gap = 16
    preview = Image.new("RGB", (width * 2 + gap, height + header), "#F0F3F6")
    preview.paste(canvas, (0, header))
    vector = Image.new("RGB", canvas.size, "white")
    draw = ImageDraw.Draw(vector)
    locations = {node["id"]: (round(node["x"]), round(node["y"])) for node in graph["nodes"]}

    for room in graph["rooms"]:
        polygon = [locations[node_id] for node_id in room["boundary"]]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=ROOM_COLORS.get(room["label"], "#E1E8F0"))
    for edge in graph["edges"]:
        draw.line((locations[edge["source"]], locations[edge["target"]]), fill="#204A66", width=3)
    for x, y in locations.values():
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#B6423A")
    if not locations:
        draw.text((18, 18), "No junctions detected", fill="#204A66")

    preview.paste(vector, (width + gap, header))
    draw = ImageDraw.Draw(preview)
    draw.text((12, 10), "Input floor plan", fill="#152B3A", font_size=14)
    draw.text(
        (width + gap + 12, 10),
        f"Predicted graph  |  {len(locations)} junctions  |  {len(graph['edges'])} walls  |  {len(graph['rooms'])} rooms",
        fill="#152B3A",
        font_size=14,
    )
    return preview


def write_outputs(output_dir, name, canvas, graph):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps(graph, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    render(canvas, graph).save(output_dir / f"{name}.png")
