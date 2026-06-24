import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import LABELS
from recognizer import HintBasedRecognizer


def main(dataset: str = "data", details: bool = False):
    root = Path(dataset)
    step_dir = root / "step"
    label_dir = root / "label"
    r = HintBasedRecognizer()
    files = sorted(step_dir.glob("*.step"))
    total = correct = 0
    for sp in files:
        lp = label_dir / f"{sp.stem}.json"
        if not lp.exists():
            continue
        expected = json.loads(lp.read_text(encoding="utf-8"))
        res = r.recognize_step(str(sp))
        predicted = res.labels
        if len(expected) != len(predicted):
            print(f"{sp.name}: mismatch {len(expected)} vs {len(predicted)}")
            continue
        fc = sum(1 for a, b in zip(expected, predicted) if a == b)
        total += len(expected)
        correct += fc
        if details and fc != len(expected):
            print(f"{sp.name}: {fc}/{len(expected)}")
            print("  exp:", expected)
            print("  pred:", predicted)
    print(f"files:{len(files)} faces:{total} acc:{correct}/{total}={correct/total if total else 0:.4f}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(ds, details=True)
