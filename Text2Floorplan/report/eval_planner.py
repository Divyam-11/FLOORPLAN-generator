"""Measure how accurately each prompt reader turns a sentence into rooms.

Readers compared on 30 hand-labelled prompts:
  original   - upstream Text2Floorplan regex (topology_rules.parse_prompt_room_counts)
  rules      - our rule reader only (prompt_planner.read_rules)
  llm        - Qwen2.5-1.5B-Instruct only (prompt_planner._parse_llm)
  llm+rules  - final planner (prompt_planner.plan_prompt)

Only room kinds mentioned in a prompt are scored (living room and kitchen that a
reader adds by default are not penalised). Metrics:
  count accuracy - mentioned room kinds whose count is exactly right
  coverage       - mentioned room kinds the reader represents at all
  exact match    - prompts where every mentioned kind is right and nothing
                   unrequested (other than living/kitchen) is added
Usage: .venv/bin/python report/eval_planner.py   -> report/results_planner.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import prompt_planner as pp  # noqa: E402

# Cache LLM replies on disk so re-runs (and eval_generation.py) do not reload the
# model for every prompt. Replies are deterministic (greedy decoding).
_CACHE = ROOT / "report" / "llm_cache.json"
_cache = json.loads(_CACHE.read_text()) if _CACHE.is_file() else {}
_ask = pp._ask_llm


def _cached_ask(prompt):
    if prompt not in _cache:
        started = time.perf_counter()
        _cache[prompt] = {"reply": _ask(prompt), "seconds": round(time.perf_counter() - started, 2)}
        _CACHE.write_text(json.dumps(_cache, indent=1))
    return _cache[prompt]["reply"]


pp._ask_llm = _cached_ask
from topology_rules import parse_prompt_room_counts  # noqa: E402

TESTS = [
    ("A 3-bedroom apartment with 2 bathrooms and 1 balcony.", {"Bedroom": 3, "Bathroom": 2, "Balcony": 1}),
    ("A small 2-bedroom apartment with 1 bathroom and 1 balcony.", {"Bedroom": 2, "Bathroom": 1, "Balcony": 1}),
    ("A large 4-bedroom apartment with 2 bathrooms and 2 balconies.", {"Bedroom": 4, "Bathroom": 2, "Balcony": 2}),
    ("3bhk with 2 bathrooms", {"Bedroom": 3, "Living room": 1, "Kitchen": 1, "Bathroom": 2}),
    ("two bedroom flat with one bathroom and a balcony", {"Bedroom": 2, "Bathroom": 1, "Balcony": 1}),
    ("A 4 bedroom apartment, with one dining room, 1 living room, a kitchen and 5 bathrooms and 3 balconies",
     {"Bedroom": 4, "Dining room": 1, "Living room": 1, "Kitchen": 1, "Bathroom": 5, "Balcony": 3}),
    ("3 bedroom apartment where every bedroom has an attached bathroom, plus one common bathroom, a kitchen, "
     "a dining room and a balcony garden",
     {"Bedroom": 3, "En-suite bathroom": 3, "Bathroom": 1, "Kitchen": 1, "Dining room": 1, "Balcony garden": 1}),
    ("2 bed 2 bath flat with a study and a pantry", {"Bedroom": 2, "Bathroom": 2, "Study": 1, "Pantry": 1}),
    ("3bhk with a pooja room, a study, 2 bathrooms and a balcony garden",
     {"Bedroom": 3, "Living room": 1, "Kitchen": 1, "Pooja room": 1, "Study": 1, "Bathroom": 2, "Balcony garden": 1}),
    ("studio apartment with a kitchen, one bathroom and a walk-in closet",
     {"Kitchen": 1, "Bathroom": 1, "Walk-in closet": 1}),
    ("A 1 bedroom apartment with a home office, 1 bathroom and a terrace",
     {"Bedroom": 1, "Study": 1, "Bathroom": 1, "Terrace": 1}),
    ("four bedrooms, three bathrooms, a utility room and two balconies",
     {"Bedroom": 4, "Bathroom": 3, "Utility room": 1, "Balcony": 2}),
    ("2 bedroom apartment with a master bedroom with en-suite, a common bathroom and a garden",
     {"Bedroom": 1, "Master bedroom": 1, "En-suite bathroom": 1, "Bathroom": 1, "Garden": 1}),
    ("A small flat with 1 bedroom, 1 bathroom, a kitchen and a storage room",
     {"Bedroom": 1, "Bathroom": 1, "Kitchen": 1, "Storage": 1}),
    ("3 bedrooms, 2 bathrooms, a gym and a balcony", {"Bedroom": 3, "Bathroom": 2, "Gym": 1, "Balcony": 1}),
    ("A 2-bedroom apartment with a guest room, 2 bathrooms and a laundry room",
     {"Bedroom": 2, "Guest room": 1, "Bathroom": 2, "Utility room": 1}),
    ("large 5 bedroom house with 4 bathrooms, a library, a dining room and 2 terraces",
     {"Bedroom": 5, "Bathroom": 4, "Library": 1, "Dining room": 1, "Terrace": 2}),
    ("1 bhk with a balcony", {"Bedroom": 1, "Living room": 1, "Kitchen": 1, "Balcony": 1}),
    ("apartment with 3 bedrooms, no balcony, 2 bathrooms and a kitchen",
     {"Bedroom": 3, "Balcony": 0, "Bathroom": 2, "Kitchen": 1}),
    ("2 bedroom apartment with a kids room, one bathroom and a pantry",
     {"Bedroom": 2, "Kids room": 1, "Bathroom": 1, "Pantry": 1}),
    ("A 3-bedroom apartment with 2 bathrooms, a media room and a patio",
     {"Bedroom": 3, "Bathroom": 2, "Media room": 1, "Garden": 1}),
    ("a cosy 1 bedroom flat with a study, a bathroom and a small balcony",
     {"Bedroom": 1, "Study": 1, "Bathroom": 1, "Balcony": 1}),
    ("two bedrooms with attached bathrooms, a kitchen, a living room and a balcony",
     {"Bedroom": 2, "En-suite bathroom": 2, "Kitchen": 1, "Living room": 1, "Balcony": 1}),
    ("3 bedroom flat with 3 toilets, a family room and a balcony",
     {"Bedroom": 3, "Bathroom": 3, "Family room": 1, "Balcony": 1}),
    ("4bhk with 3 bathrooms, a servant room and a pooja room",
     {"Bedroom": 4, "Living room": 1, "Kitchen": 1, "Bathroom": 3, "Servant room": 1, "Pooja room": 1}),
    ("an apartment with 2 bedrooms, a music room, one bathroom and a terrace",
     {"Bedroom": 2, "Music room": 1, "Bathroom": 1, "Terrace": 1}),
    ("2 bedroom apartment with a home theatre, 2 bathrooms and a deck",
     {"Bedroom": 2, "Media room": 1, "Bathroom": 2, "Garden": 1}),
    ("a 3 bedroom home with a playroom, 2 bathrooms, kitchen, dining area and a veranda",
     {"Bedroom": 3, "Play room": 1, "Bathroom": 2, "Kitchen": 1, "Dining room": 1, "Garden": 1}),
    ("flat with three bedrooms, a sauna, two bathrooms and a balcony",
     {"Bedroom": 3, "Sauna": 1, "Bathroom": 2, "Balcony": 1}),
    ("1 bedroom apartment with an art studio, a bathroom and a rooftop terrace",
     {"Bedroom": 1, "Art studio": 1, "Bathroom": 1, "Terrace": 1}),
]
DEFAULTS = {"living room", "kitchen"}


def family(label: str) -> str:
    known = pp.lookup_room(label)
    if known:
        name = known[0]
        return "living room" if name in {"Living room", pp.HUB_LABEL} else name.lower()
    if label in {pp.HUB_LABEL}:
        return "living room"
    return re.sub(r"'s\b|\broom\b", "", label.lower()).strip()


def collect(groups) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, count in groups:
        key = family(label)
        out[key] = out.get(key, 0) + count
    return out


def original(prompt):
    counts = parse_prompt_room_counts(prompt).as_dict()
    names = {"living": "Living room", "kitchen": "Kitchen", "bedroom": "Bedroom",
             "bathroom": "Bathroom", "balcony": "Balcony", "storage": "Storage"}
    return collect((names[k], v) for k, v in counts.items())


def rules(prompt):
    return collect((g.label, g.count) for g in pp.read_rules(prompt)[0])


def llm(prompt):
    reply = pp._ask_llm(prompt)
    return collect((g.label, g.count) for g in pp._parse_llm(reply)[0])


def combined(prompt):
    return collect((g.label, g.count) for g in pp.plan_prompt(prompt).rooms)


def matches(gt_key: str, produced: dict[str, int]) -> int:
    key = family(gt_key)
    if key in produced:
        return produced[key]
    first = key.split()[0]
    return sum(v for k, v in produced.items() if k.split() and k.split()[0] == first)


def score(reader) -> dict:
    total = correct = covered = mentioned = exact = 0
    rows, started = [], time.perf_counter()
    for prompt, truth in TESTS:
        try:
            produced = reader(prompt)
        except Exception as error:  # a reader that crashes scores zero on that prompt
            produced = {}
            print(f"  {reader.__name__} failed on {prompt!r}: {error}")
        ok_all = True
        for key, count in truth.items():
            got = matches(key, produced)
            total += 1
            correct += got == count
            if count > 0:
                mentioned += 1
                covered += got > 0
            ok_all &= got == count
        wanted = {family(k) for k in truth} | DEFAULTS
        extras = [k for k, v in produced.items() if v and k not in wanted
                  and not any(k.split()[0] == w.split()[0] for w in wanted if w)]
        exact += ok_all and not extras
        rows.append({"prompt": prompt, "produced": produced, "correct": ok_all, "extras": extras})
    return {
        "count_accuracy": round(100 * correct / total, 1),
        "coverage": round(100 * covered / mentioned, 1),
        "exact_match": round(100 * exact / len(TESTS), 1),
        "seconds_per_prompt": round((time.perf_counter() - started) / len(TESTS), 2),
        "llm_seconds_per_prompt": round(sum(v["seconds"] for v in _cache.values()) / max(1, len(_cache)), 2),
        "rows": rows,
    }


def main():
    results = {}
    for name, reader in (("original", original), ("rules", rules),
                         ("llm", llm), ("llm+rules", combined)):
        results[name] = score(reader)
        r = results[name]
        print(f"{name:10s} count acc {r['count_accuracy']:5.1f}%  coverage {r['coverage']:5.1f}%  "
              f"exact {r['exact_match']:5.1f}%  {r['seconds_per_prompt']}s/prompt", flush=True)
    (ROOT / "report" / "results_planner.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
