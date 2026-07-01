# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python/OCC (OpenCASCADE) recognizer for manufacturing features on STEP B-rep models, following the hint-based approach from Li et al., "Hint-based generic shape feature recognition from three-dimensional B-rep models" (see `docs/`). It labels every face of a STEP part as one of four classes:

- `0 other`, `1 hole`, `2 boss`, `3 chamfer` (see `data/labelmap.txt`, `LABELS` in `geometry.py`)

The recognizer is a hand-tuned **rule pipeline** over a face-adjacency graph — there is no ML. The recognition rules and the reasoning behind each threshold are documented in detail in `README.md`; treat that file as the spec when modifying rules.

## Environment & running

- Conda env `mfr` with `pythonocc` (OpenCASCADE) is required. Invoke through `conda run -n mfr` so OCC DLLs are on the path. The env manager is conda (see `.vscode/`).
- **Flat imports**: `recognizer.py`, `evaluate.py`, `batch_predict.py`, `_eval_flat.py` all use `from geometry import ...` / `from recognizer import ...` (no package prefix). They must be run **from inside the `mfr_recognizer/` directory**. `python -m mfr_recognizer.cli` from the repo root does **not** work because of these flat imports. (`cli.py` is the exception — it uses relative imports and is run via `python -m cli` from inside the package dir.)

Run a single file (from inside `mfr_recognizer/`):

```powershell
F:\miniforge\Scripts\conda.exe run -n mfr python -m cli ..\data\step\<file>.step --verbose
```

Evaluate against labeled dataset:

```powershell
F:\miniforge\Scripts\conda.exe run -n mfr python evaluate.py --dataset "../data"
```

(pass `../data` with forward slashes and quoted — a backslash form like `..\data` gets mangled by `conda run` on Windows and evaluate silently reports `files: 0`.)

## Verification after every rule change (required)

After **any** change to a recognition rule, threshold, or pass in `recognizer.py`, do all of the following before claiming the change is done — do not skip even for a "small" tweak:

1. **Run the evaluation** on `../data` and report the resulting `files`, `faces`, `accuracy`, and the confusion matrix. Rule changes routinely fix one sample and regress another; the dataset is the source of truth. If accuracy drops, iterate on the gates before reverting (see the standing feedback on pushing through regressions).
2. **Sync `README.md`**. The rules and thresholds documented there are the human-facing spec. Every rule change must be reflected in README.md — the relevant prose section (e.g. 凸台的识别规则), the parameter table if a default constructor arg changed, and the "其他固定辅助规则" bullets if an inline literal or new sub-rule was introduced. Never land a rule change with README.md unchanged.

The baseline as of the current tree is **441/441 = 1.000** across 15 labeled samples (see `data/label/`); a change is only acceptable if it holds or improves that.

Batch-generate predictions (process pool, CPU-bound):

```powershell
F:\miniforge\Scripts\conda.exe run -n mfr python batch_predict.py --step-dir ..\data\step --out-dir ..\data\pred_label --overwrite
```

There is **no test suite, no linter, no build step**. Correctness is measured by `evaluate.py` face-label accuracy plus manual inspection via the `scripts/sample_inspection.py` → `build_inspection_csv.py` → `copy_inspection_samples.py` workflow (random per-instance sampling of minority classes for human review).

## Architecture

Two files hold all the logic:

- **`geometry.py`** — OCC interop and the B-rep graph. `read_step` loads the shape; `BrepGraph.from_shape` enumerates faces, builds `FaceInfo` records (surface type, area, center, normal, axis, U/V spans, radial direction, edge counts, inner-loop flag), and computes the face↔face adjacency via `MapShapesAndAncestors`. It also marks `inner_loop_neighbors` (adjacencies that come from a face's inner wire, not its outer wire — the key hint for holes/bosses). All `Vec3` math helpers and OCC-wrapping functions live here.
- **`recognizer.py`** — the recognition pipeline. `HintBasedRecognizer.recognize_graph` runs a fixed sequence of passes, each producing `FeatureInstance`s that are merged into a single `labels[]` array via `_apply` (which never overwrites an already-labeled face — first pass wins). The passes, in order:
  1. `_recognize_holes` — a single definition-driven pass: enumerate inward cylinders → group by shared-edge BFS with matching axis/radius (`_enumerate_wall_groups`, `_coaxial_same_radius`) → require `u_span` sum ≥ 2π to close a complete circumference → require a carrier hint (direct inner-loop, or one cone/torus transition hop; `_wall_group_has_carrier`) before admitting a blind bottom → find bottoms whose shared circle arcs sum to 2π at a single axial z on the plane's outer wire (`_find_blind_bottoms`) → merge stepped holes that share a shelf plane (`_merge_stepped_holes`)
  2. `_recognize_chamfers` — edge-elimination transitional planes (only planar chamfers, per AP224 "flat cross section"; cone-shaped mouth transitions are left as `other`)
  3. `_recognize_structural_bosses` — the shape-agnostic structural boss pass (closed side-wall ring + top + base; distinguishes boss vs. slot by protrusion direction)
  4. `_group_chamfer_instances` — merges chamfer faces that share a solid-feature anchor into one instance

`RecognitionResult` exposes `labels` (per-face int), `instance_ids` (per-face int, grouping faces into feature instances), and `full_payload` (the `[[sampleid, {seg, inst}]]` format the inspection scripts consume). `cli.py --mode full` emits that instance format; default `labels` mode emits a flat label list.

### Rule-tuning guidance

The recognition rules are the core of the project and are tuned against regressions on the dataset. When changing a rule:

- All thresholds live as `HintBasedRecognizer.__init__` defaults (e.g. `radial_threshold=0.2`, `axis_alignment_threshold=0.7`, `hole_angular_coverage_tolerance=0.05`). `README.md` has a parameter table — keep it in sync when changing a default.
- Several numeric constants are not constructor params but inline literals in `recognizer.py` (e.g. the `0.35` side-wall orthogonality bound, area-ratio caps, and the `1.0e-5` radius/coaxial tolerances scaled by model diagonal). These are also documented in `README.md`'s "其他固定辅助规则" section.
- A change that fixes one sample often regresses another. Iterate on the gates rather than reverting at the first regression — see the project's standing feedback on pushing through local optima when restructuring rules.

## Data layout

- `data/step/` — input `.step` files (gitignored under `data/`)
- `data/label/` — ground-truth per-face label JSON (one list per file, indexed by face)
- `data/full_label/`, `data/pred_label/` — prediction outputs
- `docs/` — the source paper, an extracted text version, a definitions file, and a Chinese rules explainer PPTX

## Commit style

Commits are short, imperative, lowercase-tolerant, and describe the rule change (e.g. "Tighten blind-hole bottom: must cover full circle and be a termination", "Boss side-wall ring: admit free-form walls and ring walls with inner loops"). Match that style.
