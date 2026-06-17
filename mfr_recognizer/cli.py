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
    parser.add_argument(
        "--mode",
        choices=("labels", "full"),
        default="labels",
        help="Output labels only, or full [[sampleid, {seg, inst}]] instance JSON",
    )
    parser.add_argument(
        "--face-index-base",
        choices=(0, 1),
        type=int,
        default=0,
        help="Face id base used by full-mode seg keys",
    )
    parser.add_argument("--verbose", action="store_true", help="Print recognized feature instances")
    args = parser.parse_args()

    result = HintBasedRecognizer().recognize_step(args.step)
    step_path = Path(args.step)
    if args.mode == "full":
        payload = result.full_payload(step_path.stem, face_index_base=args.face_index_base)
    else:
        payload = {
            "step": str(step_path),
            "labels": result.labels,
            "instance_ids": result.instance_ids,
            "label_names": [LABELS[label] for label in result.labels],
            "features": [
                {
                    "id": feature.instance_id,
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
    if args.mode == "full" or args.verbose:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps({"labels": payload["labels"]}, indent=2))


if __name__ == "__main__":
    main()
