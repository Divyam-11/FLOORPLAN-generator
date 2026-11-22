# Flow Matching mathematics

## The probability path

Flow Matching learns a time-dependent vector field that transports a simple distribution into a data distribution. In a common convention, sample a noise point `z` and a clean data point `y`, form `x_t = (1 − t)z + ty`, and train against the conditional path velocity `u_t = y − z`. The standard background objective is

```text
L_FM = E[ || v_theta(x_t, t, c) - u_t ||² ]
```

The learned vector field can turn a new noise sample into a new output by numerical integration. A condition `c` changes the distribution that the field models. Here it includes the CLIP prompt embedding and the structural conditions of the requested apartment. [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

## The implementation's noise convention and objectives

The code calls the clean normalized corner coordinates `x0` and independent Gaussian coordinate noise `x1`. Its time variable is the noise fraction σ. That reverses the direction of the background convention.

```text
x_sigma = (1 - sigma) * x0 + sigma * x1
u_sigma = d(x_sigma)/d(sigma) = x1 - x0
x0_estimate = x_sigma - sigma * v_theta(x_sigma, sigma, c)
```

Only XY is sampled from noise. Room, ring and graph conditions identify the corner tokens and their relationships. The trainer samples σ from uniform and time-shifted values, includes endpoints and discretizes low σ. It supports D4 augmentation and a σ-scaled perturbation in the network input.

`losses.py` computes a sum of squared XY velocity errors per corner and averages it over valid corners. The discrete term quantizes each target axis from normalized coordinates to an 8-bit value, then uses 16 targets in {−1, +1} per corner. The discrete head's logits use mean squared error against those targets on valid polygon edges at low σ. The implementation normalizes across eligible corners per sample and averages over active samples. The full objective and diagnostic are

```text
L_total = flow_weight * L_velocity + discrete_weight * L_bits
bit_accuracy = fraction of eligible bits with the correct sign
```

The training parser defaults to both weights at 1.0 and a low-σ threshold of 0.12. Bit accuracy is a diagnostic. The released checkpoint retains architecture arguments and learned weights. [Training sampling](../original_source/Text2Floorplan/train.py) · [Exact loss implementation](../original_source/Text2Floorplan/losses.py) · [Discrete head](../original_source/Text2Floorplan/model.py)

## Generation with the released model

Inference initializes the valid corner coordinates from Gaussian noise at σ = 1 and follows the vector field toward σ = 0. The 200-step production setting builds a time-shifted schedule with a fixed grid near zero. At a typical step, Heun's method predicts a tentative coordinate with the current velocity and then applies the average of the velocities at the two endpoints. The solver uses an Euler step into zero, with the configured low-σ discrete coordinate refinement and final decoding. It returns vector coordinates for room reconstruction and validation. [Heun implementation](../original_source/Text2Floorplan/flow_solver.py)

The application loads the released weights and frozen CLIP once per worker. Size and room counts inform the text and a sampled room graph. The model predicts geometry under those conditions, followed by deterministic geometry operations and candidate validation. An upstream rejection can trigger a different seed and sampling trajectory. [Production sampling and geometry](../original_source/Text2Floorplan/pipeline_generate.py)

## Relationship to other generative models

| Approach | Typical training and generation mechanism |
| --- | --- |
| GAN | A discriminator and generator learn through an adversarial objective. The generator maps noise to an output. |
| VAE | An encoder and decoder optimize a reconstruction and latent regularization objective. The decoder maps samples from a latent prior to outputs. |
| Conventional diffusion | A forward noising process defines the training marginals. A learned denoiser or score model enables a reverse SDE or corresponding probability-flow ODE sampler. |
| Flow Matching here | Direct velocity regression on interpolated vector corner coordinates yields a conditional field for ODE integration, with low-noise coordinate-bit supervision. |

Flow Matching can also be defined with diffusion probability paths. This released production checkpoint uses a conditional vector transformer and a corner-geometry integration path.
