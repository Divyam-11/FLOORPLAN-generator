"""Turn a free-form apartment description into a room program for the model.

The released Flow Matching model knows six room shapes ("slots"): living,
kitchen, bedroom, bathroom, balcony and storage. This module lets the user ask
for any rooms in plain English ("a 4 bedroom apartment with one dining room, a
kitchen, 5 bathrooms and 3 balconies", "a balcony garden", "a pooja room"):

1. A small local LLM (Qwen2.5-1.5B-Instruct in ``models/llm``) reads the prompt
   and decides which rooms to draw, what to call them, which slot each one is
   drawn with, which rooms it should open onto, and the apartment size.
2. A rule-based reader extracts the numbers the user typed explicitly. Those
   numbers always win, so a miscount by the small LLM cannot change them. The
   reader is also the complete fallback when the LLM is missing or fails.
3. ``topology_from_program`` builds the room graph the model is conditioned on,
   gives every graph node its display label, and wires special rooms (a dining
   room opens onto the living room and kitchen, a pantry onto the kitchen).

Labels are carried through to the PNG, the JSON (``room["label"]``) and the
browser editor. The model itself still draws each room with its slot's shape.
Set ``T2F_PLANNER=rules`` to skip the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import gc
import json
import os
from pathlib import Path
import re

import numpy as np

from topology_rules import (
    NUM_ROOMS,
    TYPE_NAMES,
    TYPE_TO_ID,
    PromptRoomCounts,
    RuleTopology,
    _add_edge,
    _repair_connectivity,
    make_clip_prompt,
    sample_rule_topology,
)

ROOT = Path(__file__).resolve().parent
LLM_DIR = Path(os.environ.get(
    "T2F_LLM_DIR", ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct"))
SLOTS = ("living", "kitchen", "bedroom", "bathroom", "balcony", "storage")
# Upper bounds that keep requests inside what the model saw during training.
SLOT_LIMITS = {"living": 1, "kitchen": 2, "bedroom": 8, "bathroom": 6,
               "balcony": 4, "storage": 3}
SLEEPING = {"Bedroom", "Master bedroom", "Guest room", "Kids room"}
HUB_LABEL = "Hall"
# A bathroom whose only door opens into a bedroom.
ENSUITE = "En-suite bathroom"

# (pattern, label, slot, rooms it should open onto). Earlier entries win when
# two patterns could match at the same place, so specific phrases come first.
VOCABULARY = (
    (r"master\s+bed\s*rooms?", "Master bedroom", "bedroom", ()),
    (r"guest\s+(?:bed\s*)?rooms?", "Guest room", "bedroom", ()),
    (r"(?:kids'?|children'?s?|child'?s?)\s+(?:bed\s*)?rooms?|nurser(?:y|ies)",
     "Kids room", "bedroom", ()),
    (r"bed\s*rooms?|beds?\b", "Bedroom", "bedroom", ()),
    (r"dining(?:\s+(?:rooms?|areas?|halls?|spaces?))?", "Dining room", "bedroom",
     ("living", "kitchen")),
    (r"(?:home\s+)?offices?|study\s+rooms?|study|studies|work\s*rooms?",
     "Study", "bedroom", ("living",)),
    (r"(?:home\s+)?gyms?|workout\s+rooms?", "Gym", "bedroom", ("living",)),
    (r"librar(?:y|ies)", "Library", "bedroom", ("living",)),
    (r"(?:media|tv|cinema|movie)\s+rooms?|(?:home\s+)?theat(?:er|re)s?",
     "Media room", "bedroom", ("living",)),
    (r"play\s*rooms?|play\s+areas?", "Play room", "bedroom", ("living",)),
    (r"(?:family|drawing|sitting)\s+rooms?", "Family room", "bedroom", ("living",)),
    (r"living(?:\s+(?:rooms?|areas?|spaces?))?|lounges?|halls?\b",
     "Living room", "living", ()),
    (r"kitchens?", "Kitchen", "kitchen", ()),
    (r"(?:attached|en-?suite|private)\s+(?:(?:bath|wash|rest)\s*rooms?|baths?\b|toilets?)"
     r"|en-?suites?|ensuites?", ENSUITE, "bathroom", ("bedroom",)),
    (r"(?:bath|wash|rest)\s*rooms?|baths?\b|toilets?|powder\s+rooms?|w\.?c\.?s?\b",
     "Bathroom", "bathroom", ()),
    (r"balcony\s+gardens?|garden\s+balcon(?:y|ies)", "Balcony garden", "balcony",
     ("living",)),
    (r"(?:roof\s+)?terraces?", "Terrace", "balcony", ("living",)),
    (r"gardens?|patios?|decks?|verandah?s?|courtyards?|lawns?", "Garden",
     "balcony", ("living",)),
    (r"balcon(?:y|ies)", "Balcony", "balcony", ()),
    (r"(?:pooja|puja|mandir)(?:\s+rooms?)?", "Pooja room", "storage", ("living",)),
    (r"(?:prayer|meditation)\s+rooms?", "Prayer room", "storage", ("living",)),
    (r"(?:utility|laundry)(?:\s+rooms?)?", "Utility room", "storage", ("kitchen",)),
    (r"pantr(?:y|ies)", "Pantry", "storage", ("kitchen",)),
    (r"(?:walk-?in\s+)?(?:closets?|wardrobes?)|dressing\s+rooms?",
     "Walk-in closet", "storage", ("bedroom",)),
    (r"store\s*rooms?|storage(?:\s+rooms?)?", "Storage", "storage", ()),
)
_ROOM_RE = re.compile(
    "|".join(f"(?P<r{i}>\\b(?:{pattern}))" for i, (pattern, *_) in enumerate(VOCABULARY)))
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUMBER_RE = re.compile(r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\b")
# A number only belongs to the next room word if none of these come between them.
_BREAK_RE = re.compile(r"[,;.:()]|\b(?:and|with|for|of|plus|including|in|on|to|near|by)\b")
_NEGATION_RE = re.compile(r"\b(?:no|without|zero|not)\b")
_SIZE_WORDS = {
    "small": "small", "compact": "small", "tiny": "small", "cozy": "small",
    "cosy": "small", "medium": "medium", "mid-sized": "medium",
    "large": "large", "big": "large", "spacious": "large", "huge": "large",
    "luxury": "large", "luxurious": "large",
}
_SIZE_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _SIZE_WORDS)) + r")\b(?:\s+[\w-]+){0,3}?\s+"
    r"(?:apartment|flat|house|home|unit|condo|villa|plan|floor\s*plan)\b")


@dataclass
class RoomGroup:
    label: str
    slot: str
    count: int
    explicit: bool = True
    next_to: tuple[str, ...] = ()
    family: str = ""  # vocabulary label this room belongs to ("Bedroom" for "Bedroom 2")


@dataclass
class FloorProgram:
    prompt: str
    size: str
    rooms: list[RoomGroup]
    source: str
    notes: list[str] = field(default_factory=list)
    llm_reply: str | None = None

    def slot_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(SLOTS, 0)
        for group in self.rooms:
            counts[group.slot] += group.count
        return counts

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "size": self.size,
            "rooms": [{"label": g.label, "slot": g.slot, "count": g.count,
                       "next_to": list(g.next_to)} for g in self.rooms],
            "notes": self.notes,
            "llm_reply": self.llm_reply,
        }


def lookup_room(name: str) -> tuple[str, str, tuple[str, ...]] | None:
    """Return (label, slot, next_to) for a known room phrase, else None."""
    match = _ROOM_RE.search(name.lower())
    if not match:
        return None
    _, label, slot, next_to = VOCABULARY[int(match.lastgroup[1:])]
    return label, slot, next_to


# ---------------------------------------------------------------- rule reader
def read_rules(prompt: str) -> tuple[list[RoomGroup], str | None]:
    """Extract room counts and size the user wrote explicitly."""
    text = prompt.lower().replace("–", "-")
    groups: dict[str, RoomGroup] = {}

    def add(label, slot, count, explicit, next_to):
        if label in groups:
            current = groups[label]
            if explicit and not current.explicit:
                groups[label] = RoomGroup(label, slot, count, True, next_to)
            elif explicit == current.explicit:
                current.count += count
        else:
            groups[label] = RoomGroup(label, slot, count, explicit, next_to)

    for match in re.finditer(r"(\d+)\s*-?\s*bhk\b", text):
        add("Bedroom", "bedroom", int(match.group(1)), True, ())
        add("Living room", "living", 1, True, ())
        add("Kitchen", "kitchen", 1, True, ())
    text = re.sub(r"\d+\s*-?\s*bhk\b", " ", text)

    previous_end = 0
    for match in _ROOM_RE.finditer(text):
        label, slot, next_to = VOCABULARY[int(match.lastgroup[1:])][1:]
        segment = text[previous_end:match.start()]
        previous_end = match.end()
        breaks = [m.end() for m in _BREAK_RE.finditer(segment)]
        tail = segment[breaks[-1]:] if breaks else segment
        if _NEGATION_RE.search(tail):
            add(label, slot, 0, True, next_to)
            continue
        numbers = list(_NUMBER_RE.finditer(tail))
        if numbers:
            token = numbers[-1].group(1)
            count = int(token) if token.isdigit() else _NUMBER_WORDS[token]
            add(label, slot, count, True, next_to)
        elif label not in groups:  # "…where every bedroom has…" is not a new room
            plural = match.group(0).rstrip().endswith("s")
            add(label, slot, 2 if plural else 1, not plural, next_to)

    ensuite = groups.get(ENSUITE)
    every_bedroom = re.search(r"\b(?:every|each|all)\s+(?:of\s+the\s+)?(?:bed\s*rooms?|rooms?)", text)
    if ensuite is not None and (not ensuite.explicit or every_bedroom):
        # "4 bedrooms with attached bathrooms": one per sleeping room.
        sleeping = sum(g.count for g in groups.values() if g.label in SLEEPING)
        ensuite.count = max(1, sleeping)

    size_match = _SIZE_RE.search(text)
    size = _SIZE_WORDS[size_match.group(1)] if size_match else None
    return list(groups.values()), size


# ----------------------------------------------------------------- LLM reader
_SYSTEM_PROMPT = """You plan apartment floor plans. Read the user's description and list the rooms to draw.
Reply with JSON only, in this format:
{"size": "small|medium|large", "rooms": [{"name": "...", "shape": "...", "count": 1, "next_to": ["..."]}]}

"shape" must be one of:
- living: the single main living room, lounge or hall. Exactly one room has this shape.
- bedroom: any separate closed room of bedroom size: bedroom, study, office, dining room, guest room, gym, library, media room, play room, or any other room.
- bathroom: bathroom, toilet, washroom, powder room. A bathroom inside a bedroom (attached, en-suite) is named "En-suite bathroom" with next_to that bedroom.
- kitchen: kitchen.
- balcony: anything outdoors: balcony, terrace, garden, balcony garden, patio, deck.
- storage: small rooms: storage, pantry, utility, laundry, walk-in closet, pooja room.

Rules:
- Use exactly the numbers the user gives.
- If the user lists rooms, draw only those rooms. If no living room is listed, add {"name": "Hall", "shape": "living", "count": 1}.
- If the user describes needs instead of listing rooms, choose sensible rooms yourself, including a kitchen and bathrooms.
- "name" is a short English room name. "next_to" optionally lists names of rooms it should open onto.
- "size": use the size the user states; otherwise small for 0-1 bedrooms, medium for 2-3, large for 4 or more."""

_EXAMPLES = (
    ("Cosy flat with two bedrooms, a study and a balcony garden for my plants",
     {"size": "small", "rooms": [
         {"name": "Hall", "shape": "living", "count": 1},
         {"name": "Bedroom", "shape": "bedroom", "count": 2},
         {"name": "Study", "shape": "bedroom", "count": 1, "next_to": ["Hall"]},
         {"name": "Balcony garden", "shape": "balcony", "count": 1, "next_to": ["Hall"]}]}),
    ("A home for a family of four where both parents work from home",
     {"size": "large", "rooms": [
         {"name": "Living room", "shape": "living", "count": 1},
         {"name": "Kitchen", "shape": "kitchen", "count": 1},
         {"name": "Bedroom", "shape": "bedroom", "count": 3},
         {"name": "Home office", "shape": "bedroom", "count": 1, "next_to": ["Living room"]},
         {"name": "Bathroom", "shape": "bathroom", "count": 2},
         {"name": "Balcony", "shape": "balcony", "count": 1}]}),
)


@lru_cache(maxsize=16)
def _ask_llm(prompt: str) -> str | None:
    if os.environ.get("T2F_PLANNER", "auto").lower() == "rules":
        return None
    if not (LLM_DIR / "config.json").is_file() or not any(LLM_DIR.glob("*.safetensors")):
        return None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    device = os.environ.get("T2F_LLM_DEVICE") or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_DIR, dtype=dtype, local_files_only=True).to(device)
    model.eval()
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for user, answer in _EXAMPLES:
        messages += [{"role": "user", "content": user},
                     {"role": "assistant", "content": json.dumps(answer)}]
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, do_sample=False,
                                temperature=None, top_p=None, top_k=None)
    reply = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del model, inputs, output
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return reply.strip()


def _loads_lenient(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Small models sometimes emit single quotes, trailing commas or a
        # missing closing bracket.
        fixed = re.sub(r",\s*([}\]])", r"\1", text.replace("'", '"'))
        fixed += "]" * (fixed.count("[") - fixed.count("]"))
        fixed += "}" * (fixed.count("{") - fixed.count("}"))
        return json.loads(fixed)


def _parse_llm(reply: str) -> tuple[list[RoomGroup], str | None]:
    start, end = reply.find("{"), reply.rfind("}")
    payload = _loads_lenient(reply[start:end + 1] if end > start else reply[start:])
    groups = []
    for room in payload.get("rooms", []):
        # "Bedroom 2" and "Bedroom" are the same kind of room.
        name = re.sub(r"\s*#?\d+$", "", " ".join(str(room.get("name", "")).split()))[:28]
        if not name:
            continue
        known = lookup_room(name)
        shape = str(room.get("shape", "")).lower().strip()
        if known and known[0].lower() == name.lower() or known and shape not in SLOTS:
            label, slot, next_to = known
        else:
            label = name[0].upper() + name[1:]
            slot = shape if shape in SLOTS else (known[1] if known else "bedroom")
            next_to = known[2] if known else (("living",) if slot == "bedroom" else ())
        if known and known[0] == "Living room" and slot == "living":
            label = "Living room" if name.lower() in {"living", "living room"} else label
        extra = tuple(str(n) for n in room.get("next_to") or () if str(n).strip())
        try:
            count = int(room.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        groups.append(RoomGroup(label, slot, max(0, count), True,
                                tuple(dict.fromkeys(extra + tuple(next_to))),
                                known[0] if known else label))
    size = str(payload.get("size", "")).lower()
    return groups, size if size in {"small", "medium", "large"} else None


# ------------------------------------------------------------------- planning
def plan_prompt(prompt: str) -> FloorProgram:
    """Decide the rooms, labels and size for a free-form prompt."""
    rule_groups, rule_size = read_rules(prompt)
    notes: list[str] = []
    llm_groups, llm_size, reply, source = [], None, None, "rules"
    try:
        reply = _ask_llm(prompt)
        if reply:
            llm_groups, llm_size = _parse_llm(reply)
            source = "llm+rules"
    except Exception as error:  # the rule reader is a complete fallback
        notes.append(f"LLM planner unavailable ({type(error).__name__}); used the rule reader.")
        llm_groups = []

    # Room kinds the user gave a number for are decided by the rule reader;
    # the LLM decides everything else (extra rooms, names, size).
    counted = {g.label for g in rule_groups if g.explicit}
    mentioned = {g.label for g in rule_groups}
    listed = bool(counted)  # the user listed rooms rather than describing needs
    merged: dict[str, RoomGroup] = {}
    for group in llm_groups:
        if group.family in counted or group.label in counted:
            continue
        if (listed and group.slot != "living" and lookup_room(group.label) is not None
                and group.family not in mentioned):
            continue  # a standard room the user did not ask for (LLM over-adding)
        if group.label in merged:
            merged[group.label].count += group.count
        else:
            merged[group.label] = group
    for group in rule_groups:
        current = merged.get(group.label)
        if current is None:
            merged[group.label] = group
        elif group.explicit:
            # Numbers typed by the user override the LLM's count.
            current.count, current.slot = group.count, group.slot

    groups = [g for g in merged.values() if g.count > 0]
    master = sum(g.count for g in groups if g.label == "Master bedroom")
    for group in groups:
        if group.label == "Bedroom" and master and any(
                r.label == "Bedroom" and r.explicit for r in rule_groups):
            group.count = max(0, group.count - master)  # "4 bedrooms incl. a master"
    groups = [g for g in groups if g.count > 0]

    living = [g for g in groups if g.slot == "living"]
    if len(living) > 1:
        living = [g for g in living if g.label != HUB_LABEL] or living[:1]
        keep = next((g for g in living if g.label == "Living room"), living[0])
        for group in groups:
            if group.slot == "living" and group is not keep:
                if group.label == HUB_LABEL:
                    group.count = 0
                else:
                    group.slot, group.next_to = "bedroom", group.next_to or ("living",)
        keep.count = 1
    elif living:
        living[0].count = 1
    else:
        groups.insert(0, RoomGroup(HUB_LABEL, "living", 1, False))
    groups = [g for g in groups if g.count > 0]

    for slot, limit in SLOT_LIMITS.items():
        total = sum(g.count for g in groups if g.slot == slot)
        if total > limit:
            notes.append(f"Reduced {slot}-shaped rooms from {total} to {limit} (model limit).")
            for group in reversed([g for g in groups if g.slot == slot]):
                cut = min(group.count, total - limit)
                group.count -= cut
                total -= cut
    groups = [g for g in groups if g.count > 0]
    while sum(g.count for g in groups) > NUM_ROOMS - 1:
        largest = max((g for g in groups if g.slot != "living"), key=lambda g: g.count)
        largest.count -= 1
        notes.append(f"Removed one {largest.label} to stay within {NUM_ROOMS - 1} rooms.")
        groups = [g for g in groups if g.count > 0]

    bedroom_slots = sum(g.count for g in groups if g.slot == "bedroom")
    size = rule_size or llm_size or (
        "small" if bedroom_slots <= 1 else "medium" if bedroom_slots <= 3 else "large")
    return FloorProgram(prompt, size, groups, source, notes, reply)


OPEN_PLAN = {"Dining room", "Family room"}
# Counts the released model draws reliably (measured: 4 bedrooms + 5 bathrooms
# + 3 balconies passed 0/32 attempts, 4 + 3 + 2 passed 2/12).
RELAXED_CAPS = (("bathroom", 3, "bathrooms"), ("balcony", 2, "outdoor spaces"),
                ("bedroom", 5, "bedroom-sized rooms"))


def relax_program(program: FloorProgram, level: int) -> FloorProgram:
    """Simplify a program the model keeps failing on. Level 0 changes nothing.

    1: open-plan rooms (dining, family room) merge into the living room.
    2: additionally cap counts at what the model draws reliably.
    Every change is recorded in ``notes`` so the output can say what happened.
    """
    if level <= 0:
        return program
    rooms = [replace(group) for group in program.rooms]
    notes = list(program.notes)
    living = next(g for g in rooms if g.slot == "living")
    merged = [g for g in rooms if g.label in OPEN_PLAN]
    if merged:
        base = "Living" if living.label in {HUB_LABEL, "Living room"} else living.label
        living.label = " / ".join([base] + [g.label.removesuffix(" room") for g in merged])
        rooms = [g for g in rooms if g not in merged]
        notes.append(f"Drew {', '.join(g.label.lower() for g in merged)} as open plan "
                     "with the living room to get a valid layout.")
    if level >= 2:
        for slot, cap, words in RELAXED_CAPS:
            group_list = [g for g in rooms if g.slot == slot]
            total = sum(g.count for g in group_list)
            if total <= cap:
                continue
            excess = total - cap
            for group in reversed(group_list):
                cut = min(group.count, excess)
                group.count -= cut
                excess -= cut
            notes.append(f"Drew {cap} {words} instead of {total}: the model could not "
                         "fit them all in a valid layout.")
        rooms = [g for g in rooms if g.count > 0]
    return replace(program, rooms=rooms, notes=notes)


def topology_from_program(
    program: FloorProgram,
    rng: np.random.Generator,
) -> tuple[RuleTopology, dict[int, str]]:
    """Build the conditioning room graph and a label for every node."""
    slot_counts = program.slot_counts()
    counts = PromptRoomCounts(
        living=1, kitchen=slot_counts["kitchen"], bedroom=slot_counts["bedroom"],
        bathroom=slot_counts["bathroom"], balcony=slot_counts["balcony"],
        storage=slot_counts["storage"],
        size=program.size if program.size in {"small", "large"} else None)
    topology = sample_rule_topology("", rng, counts=counts)
    adjacency = topology.adjacency.copy()
    node_types = topology.node_type_ids
    ids_by_type = topology.room_ids_by_type
    living = ids_by_type["living"][0]

    labels: dict[int, str] = {ids_by_type["front_door"][0]: "Front door"}
    wants: dict[int, tuple[str, ...]] = {}
    for slot in SLOTS:
        ids = list(ids_by_type.get(slot, []))
        names = [(g.label, g.next_to) for g in program.rooms if g.slot == slot
                 for _ in range(g.count)]
        if slot == "bedroom":
            # Give non-sleeping rooms (dining, study…) the nodes that open onto
            # the living room and have no en-suite bathroom.
            def score(node):
                bath = any(adjacency[node, other] and TYPE_NAMES[node_types[other]] == "bathroom"
                           for other in range(NUM_ROOMS))
                return (not adjacency[node, living], bath)
            names.sort(key=lambda item: item[0] in SLEEPING)
            ids.sort(key=score)
        for node, (label, next_to) in zip(ids, names):
            labels[int(node)] = label
            wants[int(node)] = next_to

    by_label = {}
    for node, label in labels.items():
        by_label.setdefault(label.lower(), node)
    sleeping = [n for n in ids_by_type.get("bedroom", []) if labels.get(n) in SLEEPING]

    for node, label in list(labels.items()):
        if TYPE_NAMES[node_types[node]] != "bedroom" or label in SLEEPING:
            continue
        # A dining room or study should not own an en-suite bathroom.
        for other in np.flatnonzero(adjacency[node]).tolist():
            if TYPE_NAMES[node_types[other]] != "bathroom":
                continue
            adjacency[node, other] = adjacency[other, node] = False
            free = [b for b in sleeping if not any(
                adjacency[b, x] and TYPE_NAMES[node_types[x]] == "bathroom"
                for x in range(NUM_ROOMS))]
            _add_edge(adjacency, other, free[0] if free else living, node_types)

    # Requested en-suite bathrooms open only into a sleeping room, the master
    # bedroom first, then bedrooms that do not have a bathroom yet.
    def has_bathroom(room):
        return any(adjacency[room, x] and TYPE_NAMES[node_types[x]] == "bathroom"
                   for x in range(NUM_ROOMS))

    ensuites = [n for n, label in labels.items() if label == ENSUITE]
    if ensuites:
        # Clear the rule-based bathroom edges first so hosts are judged fairly;
        # the other bathrooms become the shared ones off the living room.
        for node in ensuites:
            for other in np.flatnonzero(adjacency[node]).tolist():
                adjacency[node, other] = adjacency[other, node] = False
        for node in [n for n, label in labels.items() if label == "Bathroom"]:
            for other in np.flatnonzero(adjacency[node]).tolist():
                if TYPE_NAMES[node_types[other]] == "bedroom":
                    adjacency[node, other] = adjacency[other, node] = False
            _add_edge(adjacency, node, living, node_types)
    for node in ensuites:
        hosts = sorted(sleeping, key=lambda b: (labels[b] != "Master bedroom", has_bathroom(b)))
        _add_edge(adjacency, node, hosts[0] if hosts else living, node_types)
        wants.pop(node, None)

    for node, targets in wants.items():
        for target in targets:
            key = target.lower().strip()
            other = by_label.get(key)
            if other is None and key in {"hall", "living", "lounge"}:
                other = living
            if other is None and key in ids_by_type and ids_by_type[key]:
                other = ids_by_type[key][0]
            if other is None:
                known = lookup_room(key)
                if known:
                    other = by_label.get(known[0].lower())
                    if other is None and ids_by_type.get(known[1]):
                        other = ids_by_type[known[1]][0]
            if other is not None and other != node:
                _add_edge(adjacency, node, int(other), node_types)

    for node in ids_by_type.get("bedroom", []):
        # Every room drawn with the bedroom shape needs a door that is not
        # through a bathroom or balcony.
        if not any(adjacency[node, other] and TYPE_NAMES[node_types[other]]
                   not in {"bathroom", "balcony"} for other in range(NUM_ROOMS)):
            _add_edge(adjacency, node, living, node_types)

    _repair_connectivity(adjacency, topology.valid_mask, node_types, ids_by_type)
    edge_lists = tuple(
        (a, b) for a in range(NUM_ROOMS) for b in range(a + 1, NUM_ROOMS)
        if topology.valid_mask[a] and topology.valid_mask[b] and adjacency[a, b])
    return replace(topology, adjacency=adjacency, edge_lists=edge_lists,
                   text=make_clip_prompt(counts)), labels


def fit_edges_to_geometry(records, pairs, config, living_id):
    """Adapt the planned room connections to the geometry the model drew.

    A planned connection needs a shared wall long enough for a door. With many
    rooms the model sometimes places two connected rooms apart. Such a
    connection is dropped when every room stays reachable from the living room;
    a room that would be cut off is connected to the neighbour it actually
    touches instead. Returns the new pair set and a list of changes.
    """
    from plan_postprocess import _max_shared_boundary_segment

    by_id = {r.room_id: r for r in records}
    minimum = config.min_door_shared_length - 1e-7
    doors = {"front_door", "door"}
    pairs = {tuple(sorted(map(int, p))) for p in pairs}
    changes = []

    def fits(a, b):
        return _max_shared_boundary_segment(by_id[a], by_id[b]) >= minimum

    def reachable(edges):
        seen, queue = {living_id}, [living_id]
        while queue:
            node = queue.pop()
            for a, b in edges:
                other = b if a == node else a if b == node else None
                if other is not None and other not in seen:
                    seen.add(other)
                    queue.append(other)
        return seen

    rooms = [r.room_id for r in records if r.type_name not in doors]
    for a, b in sorted(pairs):
        if a not in by_id or b not in by_id:
            continue
        if by_id[a].type_name in doors or by_id[b].type_name in doors or fits(a, b):
            continue
        remaining = pairs - {(a, b)}
        if set(rooms) <= reachable(remaining):
            pairs = remaining
            changes.append({"removed": [a, b]})
            continue
        # Reconnect the side that would be cut off to a room it touches.
        lost = [n for n in (a, b) if n not in reachable(remaining)]
        for node in lost:
            touching = sorted(
                (other for other in rooms if other != node and fits(node, other)
                 and not (by_id[node].type_name == by_id[other].type_name == "bedroom")),
                key=lambda o: (by_id[o].type_name in {"bathroom", "balcony", "storage"},
                               o != living_id))
            if touching:
                remaining = remaining | {tuple(sorted((node, touching[0])))}
        if set(rooms) <= reachable(remaining):
            changes.append({"removed": [a, b],
                            "added": [list(p) for p in sorted(remaining - pairs)]})
            pairs = remaining
    return pairs, changes


if __name__ == "__main__":
    import sys

    result = plan_prompt(" ".join(sys.argv[1:]) or
                         "A 4 bedroom apartment, with one dining room, 1 living room, "
                         "a kitchen and 5 bathrooms and 3 balconies")
    print(json.dumps(result.as_dict(), indent=2))
