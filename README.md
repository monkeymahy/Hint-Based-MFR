# Hint-Based MFR

Python OCC implementation of a practical subset of the paper
“Hint-based generic shape feature recognition from three-dimensional B-rep models”.

The implementation builds a face-edge graph from STEP B-Rep topology, detects
the paper's reusable hints, and maps the relevant generic features to three MFR
labels:

- `1 hole`: internal-loop or inward round-side features.
- `2 boss`: internal-loop or face-partition protrusions with outward side walls.
- `3 chamfer`: conical or oblique planar transitional faces.

## Run

Use `conda run` so OpenCASCADE DLL paths are activated:

```powershell
F:\miniforge\Scripts\conda.exe run -n mfr python -m mfr_recognizer.cli E:\dataset\MFR\MFR\step\01010028.step --verbose
```

Evaluate the sample dataset:

```powershell
F:\miniforge\Scripts\conda.exe run -n mfr python -m mfr_recognizer.evaluate --dataset E:\dataset\MFR\MFR --details
```

Batch-generate prediction labels:

```powershell
F:\miniforge\Scripts\conda.exe run -n mfr python -m mfr_recognizer.batch_predict --step-dir E:\dataset\MFR\MFR\step --out-dir E:\dataset\MFR\MFR\pred_label --overwrite
```
