"""Draw the report charts as PNGs in report/charts/ (emphasis style: ours in blue, others gray)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)
# Matches the Canva "Blue Minimalist Project Presentation" template: Poppins, one teal accent.
for font in (ROOT / "fonts").glob("Poppins-*.ttf"):
    font_manager.fontManager.addfont(str(font))
SURFACE, INK, MUTED, OURS, OTHER = "#ffffff", "#1f1f1f", "#5c5c5c", "#4f9a93", "#cfdcda"
plt.rcParams.update({"font.family": "Poppins", "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})


def hbar(path, title, subtitle, labels, values, ours, fmt, xmax, xlabel):
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(labels) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    colors = [OURS if i in ours else OTHER for i in range(len(labels))]
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + xmax * 0.012, bar.get_y() + bar.get_height() / 2, fmt(value),
                va="center", color=INK, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, labelpad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


planner = json.loads((ROOT / "results_planner.json").read_text())
names = {"original": "Original Text2Floorplan regex", "rules": "Our rule reader",
         "llm": "Qwen2.5-1.5B alone", "llm+rules": "Ours: LLM + rules"}
hbar(OUT / "planner_accuracy.png", "Reading a free-form prompt",
     "Room counts read correctly, 30 hand-labelled prompts (higher is better)",
     [names[k] for k in names], [planner[k]["count_accuracy"] for k in names], {3},
     lambda v: f"{v:.1f}%", 112, "room counts read correctly, 30 prompts")

# Published compatibility (GED to the input room graph, lower is better), RPLAN 8-room group.
methods = ["House-GAN (2020)", "House-GAN++ (2021)", "HouseDiffusion (2023)",
           "HouseTune (2024)", "LLM + RLVR (2025)"]
ged = [5.3, 3.9, 2.5, 0.32, 0.15]
hbar(OUT / "published_ged.png", "How well generated plans follow the requested room graph",
     "Published GED on RPLAN, 8-room plans (lower is better). Numbers from each paper.",
     methods, ged, set(range(5)), lambda v: f"{v:g}", 6.2, "graph edit distance, lower is better")
print("charts written to", OUT)


# Head-to-head GED by plan size (one sample per plan). Ours from results_compat.json.
compat = json.loads((ROOT / "results_compat.json").read_text())
sizes = [5, 6, 7, 8]
series = [("House-GAN++", [1.9, 2.2, 2.4, 3.9], "#cfd6d5"),
          ("HouseDiffusion", [1.5, 1.2, 1.7, 2.5], "#93a3a1"),
          ("Ours", [compat[str(s)]["ged"] for s in sizes], OURS)]
fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
width = 0.26
for k, (name, values, color) in enumerate(series):
    xs = [i + (k - 1) * (width + 0.02) for i in range(len(sizes))]
    bars = ax.bar(xs, values, width=width, color=color, label=name)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06, f"{value:g}",
                ha="center", va="bottom", fontsize=11, color=INK)
ax.set_xticks(range(len(sizes)), [f"{s} rooms" for s in sizes])
ax.tick_params(axis="x", length=0, colors=INK)
ax.set_yticks([])
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d9dddd")
ax.set_ylim(0, 4.5)
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=12)
ax.set_xlabel("graph edit distance to the requested room graph, one sample, lower is better", labelpad=10)
fig.tight_layout()
fig.savefig(OUT / "ged_by_size.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
