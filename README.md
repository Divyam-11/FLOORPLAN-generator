# Raster-to-Graph Lab Demo

Raster-to-Graph converts a floor-plan image into wall junctions, wall segments, and labeled room boundaries. The original training and evaluation code is included. The separate `deployment/` module runs inference on a single image or a directory.

## Deployment

Open a terminal in this project folder. For a Windows CPU setup with Python 3.12, run these commands in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r deployment/requirements.txt
.\.venv\Scripts\python.exe -m deployment.download
.\.venv\Scripts\python.exe -m deployment deployment/example/floorplan.png --device cpu
```

On Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. CPU inference uses the existing PyTorch implementation of deformable attention and requires no CUDA extension. For a CUDA environment, install a matching PyTorch and torchvision build and use `--device cuda`. [PyTorch installation options](https://pytorch.org/get-started/locally/)

The setup command downloads and verifies the model file (~415 MB) in `deployment/weights/`. It reuses a verified local copy on subsequent runs.

Use your own PNG or JPG by replacing the input path. To process a directory and choose an output location, run the following with the environment's Python.

```text
python -m deployment path/to/floorplans --device cpu --output deployment/results
```

Each image produces a side-by-side PNG and a JSON file in `deployment/results/` by default. The JSON contains junction coordinates and quadrant predictions, wall edges, and room boundaries as lists of junction IDs. Coordinates are in the 512 × 512 inference canvas. To recover source coordinates, subtract the recorded `offset` and multiply each axis by `source_size / resized_size`. The JSON also records whether decoding reached the end token, ran out of valid junctions, or reached the step limit.

### Lab example

The included illustration yielded **13 junctions, 18 wall segments, and 6 room regions** on CPU. The room types are model predictions. For example, the model labeled the illustrated living/kitchen area as a bedroom. [Preview](deployment/example/result/floorplan.png) · [Graph JSON](deployment/example/result/floorplan.json)

## Training and benchmark evaluation

`train.py` is the original training entry point, `engine.py` contains the training and evaluation loops, and `test.py` runs the benchmark on the annotated dataset. These scripts use the upstream dataset, argument conventions, and CUDA-oriented environment. Follow the preserved [upstream setup and data instructions](UPSTREAM_README.md) for dataset access, preparation, and training configuration.

## Source and license

The source is based on upstream commit `04f14b36a26ccc0986d300243f64db8f388dd6ef` and retains the [GPL-3.0 license](LICENSE). The lab adaptation added the deployment module and example and updated model construction and deformable attention dispatch on November 22, 2026. [Original project documentation](UPSTREAM_README.md)
