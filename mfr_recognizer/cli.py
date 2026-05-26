from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry import LABELS
from .recognizer import HintBasedRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Hint-based STEP feature recognizer")
    parser.add_argument("step", help="Path to a STEP file")
    parser.add_argument("--json", dest="json_out", help="Optional output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print recognized feature instances")
    args = parser.parse_args()

    result = HintBasedRecognizer().recognize_step(args.step)
    payload = {
        "step": str(Path(args.step)),
        "labels": result.labels,
        "label_names": [LABELS[label] for label in result.labels],
        "features": [
            {
                "kind": feature.kind,
                "label": feature.label,
                "faces": result.one_based_faces(feature),
                "hint_faces": [idx + 1 for idx in sorted(feature.hint_faces)],
                "reason": feature.reason,
            }
            for feature in result.features
        ],
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload if args.verbose else {"labels": payload["labels"]}, indent=2))


if __name__ == "__main__":
    main()
