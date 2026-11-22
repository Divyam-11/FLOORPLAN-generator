# Preserved training pipeline

**Training code is available. Training execution is disabled in the application. Inference uses the released pretrained checkpoint.**

The upstream checkout preserves `train.py`, `losses.py`, the model definition, configuration and checkpoint logic, normalization/mask utilities and preprocessing modules. Setup, the FastAPI worker and the smoke test invoke the separate inference entry points. [Training source](../original_source/Text2Floorplan/train.py) · [Loss source](../original_source/Text2Floorplan/losses.py)

| Stage | Source behaviour |
| --- | --- |
| Load and split | Loads aligned normalized coordinates, masks, next-corner maps, room graph features and pooled CLIP embeddings. Checks plan IDs and performs a seeded shuffle. The held-out split is capped by the configured validation size and roughly one tenth of the records, with at least one for valid input. |
| Form the batch | Moves tensors to the training device. Corner identity and edge indices permit grouping tokens into room rings. Topology and precomputed text supply conditions. |
| Sample geometry and noise | Draws Gaussian XY and σ from a mix of uniform and time-shifted samples, with endpoint coverage and a low-σ grid. The source enables D4 rotations/reflections and small σ-dependent perturbations by default. |
| Predict continuous/discrete outputs | The room transformer encodes the graph. The corner transformer predicts velocity with room, text and time conditioning. The refinement head predicts coordinate bits from the estimated clean coordinates and polygon-edge structure. |
| Calculate losses | Uses masked velocity MSE plus low-σ discrete-bit MSE, with default weights of 1.0 and 1.0 and bit threshold σ < 0.12. Bit accuracy is logged separately. |
| Optimize and save | Uses AdamW with warmup and cosine learning-rate decay via `LambdaLR`. Tracks exponential moving average where enabled. Periodic and best-validation checkpoints capture model, EMA, optimizer, scheduler, arguments and epoch state. |

These are code behaviours and defaults. The exact original training size, hardware, runtime, executed epoch count and loss history are not established by this project. The 193 MB release contains the deployed model and architecture arguments. An exact resume would need the original full checkpoint, aligned data and training environment. [Original checkpoint details](../original_source/Text2Floorplan/checkpoints/README.md)

Read [Dataset and representation](dataset.md) for the input contract and [Flow Matching mathematics](flow_matching.md) for the noise convention, objective and solver.
