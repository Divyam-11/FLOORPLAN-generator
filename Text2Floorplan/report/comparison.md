# FloorPlan AI: results and comparison with related work

| Name | Roll no. | Group |
|---|---|---|
| Lakksh Sharma | 102303836 | 4C24 |
| Ishita Gupta | 102303377 | 4C24 |
| Divyam Jund | 102303445 | 4C26 |
| Akhil Gupta | 102315121 | 4O15 |

## 1. Why we are doing this

People describe homes in sentences ("a 3BHK with a pooja room and a balcony garden").
The strongest published generators don't take sentences. They need **expert inputs**:
- a **bubble diagram**, meaning a room graph (House-GAN, House-GAN++, HouseDiffusion, GSDiff)
- or a **building outline** (Graph2Plan, WallPlan)

The text-based systems that do exist have three problems:
- they need **long structured descriptions** (Tell2Design)
- they rely on **cloud or very large LLMs**: GPT-4o (HouseTune), GPT-4 / Moonshot (ChatHouseDiffusion), Llama-3.3-70B (RLVR)
- they produce **raster images or bare walls**, with no doors, windows or areas

**Gap:** a short sentence in, an editable vector plan out. Doors, windows and m² areas
included. Runs offline on a laptop CPU, and accepts any room type the user names.

## 2. Who we compare with, and on what

| Group | Methods | Input | Key metrics (2–3 each) |
|---|---|---|---|
| Graph-conditioned | House-GAN (2020), House-GAN++ (2021), HouseDiffusion (2023), GSDiff (2025) | room graph | FID ↓, GED ↓, realism ↑ |
| LLM-assisted | HouseTune (2024), LLM+RLVR (2025) | text / JSON spec | FID ↓, GED ↓ |
| Text-conditioned | Tell2Design T2D (2023), ChatHouseDiffusion (2024) | text + outline | micro / macro IoU ↑ |
| Outline-conditioned | Graph2Plan (2020), WallPlan (2022), GSDiff | outline | FID ↓ |

Definitions:
- **FID** measures how realistic and diverse plan images are compared with real ones.
- **GED** is the graph edit distance between the requested room graph and the generated one.
- **IoU** is pixel overlap with a ground-truth plan.

## 3. Published numbers (copied from each paper)

**RPLAN, 8-room plans, protocol of HouseDiffusion Table 1**

| Method | FID ↓ | GED ↓ | Realism ↑ |
|---|---|---|---|
| House-GAN | 66.4 | 5.3 | −0.95 |
| House-GAN++ | 32.9 | 3.9 | −0.52 |
| HouseDiffusion | 9.5 | 2.5 | −0.19 |
| HouseTune (GPT-4o + diffusion) | 9.0 | 0.32 | −0.03 |
| LLM + RLVR (Llama-3.3-70B, best of 10) | 7.0 | 0.15 | 0.03 |

**GSDiff protocol (757 test graphs)**

| Method | FID ↓ | GED ↓ |
|---|---|---|
| HouseDiffusion | 11.87 | 2.59 |
| GSDiff | 6.82 | 0.49 |

**Outline-conditioned**

| Method | FID ↓ |
|---|---|
| Graph2Plan | 8.40 |
| WallPlan | 9.07 |
| GSDiff | 7.50 |

**Tell2Design test set**

| Method | Micro IoU ↑ | Macro IoU ↑ |
|---|---|---|
| Imagen | 12.17 | 14.96 |
| T2D | 54.34 | 53.30 |
| ChatHouseDiffusion (Moonshot LLM) | 60.57 | 57.34 |

Human designers reach 64.67 micro / 62.32 macro IoU on a 100-plan subset.

## 4. Our measurements

**Prompt reading.** 30 hand-labelled free-form prompts, `report/eval_planner.py`.

| Reader | Count accuracy | Rooms covered | Exact program |
|---|---|---|---|
| Original Text2Floorplan regex | 45.7% | 46.1% | 13.3% |
| Qwen2.5-1.5B alone | 79.8% | 85.2% | 56.7% |
| Our rule reader | 96.1% | 96.9% | 83.3% |
| **Ours: LLM + rules** | **100%** | **100%** | **100%** |

The LLM takes about 14 s per prompt on an Apple M4.

**Generation.** One sample per seed, no retries, CPU. See `report/eval_generation.py`.

| Prompt type (5 seeds each) | Valid on first sample | Raw connection recall | Accepted plans: counts right, GED |
|---|---|---|---|
| 1 bedroom | 100% (5/5) | 100% | 5/5, GED 0 |
| 2 bedrooms (2 prompts) | 60% (6/10) | 90% | 6/6, GED 0 |
| 3 bedrooms (2 prompts) | 60% (6/10) | 95% | 6/6, GED 0 |
| 4 bedrooms | 40% (2/5) | 89% | 2/2, GED 0 |
| Free-form with extra rooms (2 prompts) | 20% (2/10) | 87% | 2/2, GED 1 and 5 |
| **All 40 samples** | **52%** | **92%** | **21/21, mean GED 0.29** |

**Raw connection recall** is the share of requested room connections the flow model's
geometry already honours before any repair. When a free-form plan can't fit a door
where it was requested, the connection is re-routed, which is why GED is above 0 there.
Each sample takes about 10 s on CPU, so retries explain the 1.5–4 min per finished plan.
The most common rejections are a bathroom under 2 m², a missing doorway, and a thin
sliver of a room.

**Footprint.**
- 48.2 M-parameter flow model plus a 1.5 B LLM
- CPU only, offline
- a whole plan takes about 1.5–4 min including retries

## 4b. Head-to-head on shared metrics (one sample per plan, no retries)

These are measured with `report/eval_compat.py`: 12 samples per size, fixed-type prompts.

| Metric | House-GAN++ | HouseDiffusion | LLM + RLVR (n=1) | **Ours** |
|---|---|---|---|---|
| GED ↓, 5 rooms | 1.9 | 1.5 | – | **0** |
| GED ↓, 6 rooms | 2.2 | 1.2 | – | **0.33** |
| GED ↓, 7 rooms | 2.4 | 1.7 | – | **0.33** |
| GED ↓, 8 rooms | 3.9 | 2.5 | 1.89 | **1.0** |
| Overlap rate ↓, 8 rooms | – | – | **0.26** | 1.00 |
| % area overlapping ↓, 8 rooms | – | – | 0.01 | **0.005** |
| Room ID ↑, 8 rooms | – | – | 1.00 | 1.00 |
| Parameters | 2 M | 27 M | 70 B | 48 M + 1.5 B LLM |

Caveats:
- **GED test graphs differ.** Ours are room graphs our planner builds from prompts; the papers use RPLAN test graphs. Ours: 12 samples per size; theirs: thousands.
- **Our GED cannot count extra connections.** Doors are only placed on requested connections, so only missing rooms and missing connections are counted.
- **Overlap is measured on the raw model output.** Our clean-up step then removes overlaps, so accepted plans have none.
- **The RLVR numbers are for one sample** (their Table 3). Their best-of-10 numbers are GED 0.15 and overlap 0.13.

## 5. How to read the comparison (be honest in the viva)

- **Our numbers and the published numbers come from different test sets.** We had
  no RPLAN access, so we can't compute FID or IoU. Don't put our numbers in the
  same column as theirs.
- **On accepted plans, our compatibility is guaranteed.** Doors are placed exactly
  on the requested connections, and plans that fail are rejected. The honest
  measure of the model itself is *raw connection recall* plus the *validity rate*.
- **The prompt test set was written by our group.** The rule reader was refined
  while building it, so 100% is optimistic. Unknown room types (sauna, music room,
  art studio, servant room) are where the LLM adds value: coverage goes from
  96.9% to 100%.
- **We did not train the flow model.** It is the released Text2Floorplan checkpoint.
  Our work is the prompt planner, custom room types and en-suites, the fallback
  strategy, and the evaluation.

## Sources

- HouseDiffusion, Shabani et al., CVPR 2023: https://arxiv.org/abs/2211.13287
- GSDiff, Hu et al., AAAI 2025: https://arxiv.org/abs/2408.16258
- Tell2Design, Leng et al., ACL 2023: https://arxiv.org/abs/2311.15941
- ChatHouseDiffusion, Qin et al., 2024: https://arxiv.org/abs/2410.11908
- HouseTune, Zong et al., 2024: https://arxiv.org/abs/2411.12279
- Floor plans with LLMs + RLVR, Lara et al., NeurIPS 2025 workshop: https://arxiv.org/abs/2605.14117
- Text2Floorplan (our base model): https://github.com/simerionon/Text2Floorplan
