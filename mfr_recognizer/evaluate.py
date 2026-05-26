from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .geometry import LABELS
from .recognizer import HintBasedRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hint-based recognizer on an MFR dataset folder")
    parser.add_argument("--dataset", default=r"E:\dataset\MFR\MFR", help="Dataset root with step/ and label/")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of files to evaluate")
    parser.add_argument("--details", action="store_true", help="Print per-file labels")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    step_dir = dataset / "step"
    label_dir = dataset / "label"
    recognizer = HintBasedRecognizer()

    files = sorted(step_dir.glob("*.step"))
    if args.limit:
        files = files[: args.limit]

    total = 0
    correct = 0
    confusion: dict[int, Counter[int]] = defaultdict(Counter)

    for step_path in files:
        label_path = label_dir / f"{step_path.stem}.json"
        if not label_path.exists():
            continue
        expected = json.loads(label_path.read_text(encoding="utf-8"))
        result = recognizer.recognize_step(str(step_path))
        predicted = result.labels
        if len(expected) != len(predicted):
            print(f"{step_path.name}: face count mismatch expected={len(expected)} predicted={len(predicted)}")
            continue

        file_correct = sum(1 for a, b in zip(expected, predicted) if a == b)
        total += len(expected)
        correct += file_correct
        for a, b in zip(expected, predicted):
            confusion[a][b] += 1

        if args.details:
            print(f"{step_path.name}: {file_correct}/{len(expected)}")
            print("  expected:", expected)
            print("  predicted:", predicted)

    print(f"files: {len(files)}")
    print(f"faces: {total}")
    print(f"accuracy: {correct}/{total} = {(correct / total if total else 0):.3f}")
    print("confusion rows=true cols=pred")
    labels = sorted(LABELS)
    header = "true\\pred " + " ".join(f"{LABELS[label]:>8}" for label in labels)
    print(header)
    for true_label in labels:
        row = [confusion[true_label][pred_label] for pred_label in labels]
        print(f"{LABELS[true_label]:>9} " + " ".join(f"{value:8d}" for value in row))


if __name__ == "__main__":
    main()
