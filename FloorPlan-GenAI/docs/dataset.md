# Dataset and representation

## Sources and available assets

| Source | Representation | Inputs expected by Text2Floorplan |
| --- | --- | --- |
| ResPlan | Residential polygon geometry and connectivity | `data/sources/ResPlan/ResPlan.pkl` |
| RPLAN | Annotated residential floor-plan images from Wu et al., SIGGRAPH Asia 2019 | `data/sources/RPLAN/dataset/floorplan_dataset/` |

ResPlan's repository publishes a CC BY 4.0 grant covering the authors' contributions to its data, with a separate MIT grant for code. RPLAN and the optional RPLAN toolbox have separate access and usage terms. The training datasets and feature caches are obtained or rebuilt separately. [ResPlan license](https://github.com/m-agour/ResPlan/blob/main/LICENSE) · [RPLAN project](https://ustc-gcl-f.github.io/projects/DeepLayout/index.html) · [Text2Floorplan third-party notices](../original_source/Text2Floorplan/THIRD_PARTY_NOTICES.md)

The checkout retains six small data files. Training prompt generation uses canvas area quantiles. Inference uses the committed corner-count distributions, room aspect statistics, physical scale statistics and geometry policy. The [sample index](../dataset/samples/README.md) links to the released browser and native Revit example outputs.

## Preprocessing and alignment

1. **Source preparation.** ResPlan cleans geometry, reconstructs and snaps wall segments, refreshes topology, filters records, and fits plans into a 0–255 canvas while preserving aspect ratio. RPLAN converts labelled PNGs to geometry, filters by geometry/connectivity and records the ResPlan/RPLAN area calibration for prompt labels. Its original coordinates remain in pixel units for GT extraction.
2. **Raw extraction.** `extract_raw_data.py` derives corner and room graph tensors, edge lists and a plan-ID manifest.
3. **Normalization.** `prepare_plan_data.py` canonicalizes and normalizes corner rings, with recovery metadata for canvas transformations. Each plan has up to 128 corners. Each corner has 52 channels: XY, room-centroid XY, 20 room identity channels and 28 ring-corner identity channels.
4. **Topology GT.** `prepare_topology_gt.py` pads to 20 room nodes, removes bedroom-to-bedroom edges per upstream policy, and recomputes graph positional encodings. The 61-channel room features include room identity, semantic type and graph positional encodings.
5. **Prompt and attention features.** `prepare_prompts.py` writes structured English prompts and area metadata. `prepare_attention.py` writes masks plus next-corner indices and edge validity.
6. **CLIP and validation.** `encode_clip.py` pools the frozen CLIP model to 768 values per plan and aligns deduplicated prompts back to plan IDs. `validate_final_training_data.py` checks ID, topology, padding, masks and edge mapping across files.

The trainer expects `plan_norm_gt.npz`, `topology_gt.npz`, `masks.npz`, `edge_mapping.npz` and `clip/pooled_embeddings.npy` with aligned CLIP `plan_ids.npy`. Training needs separately obtained data and preprocessing dependencies. Upstream notes that exact regeneration of every historical training sample from raw data has not been established. [Upstream preparation and validation instructions](../original_source/Text2Floorplan/docs/data_pipeline.md)
