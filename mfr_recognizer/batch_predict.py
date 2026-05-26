from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recognizer import HintBasedRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict STEP face labels")
    parser.add_argument(
        "--step-dir",
        default=r"E:\dataset\MFR\MFR\step",
        help="Directory containing .step/.stp files",
    )
    parser.add_argument(
        "--out-dir",
        default=r"E:\dataset\MFR\MFR\pred_label",
        help="Directory to write predicted JSON label files",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction files")
    parser.add_argument("--verbose", action="store_true", help="Print each processed file")
    args = parser.parse_args()

    step_dir = Path(args.step_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_files = sorted(list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp")))
    recognizer = HintBasedRecognizer()

    written = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for step_path in step_files:
        out_path = out_dir / f"{step_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f"skip existing: {out_path}")
            continue

        try:
            result = recognizer.recognize_step(str(step_path))
            out_path.write_text(json.dumps(result.labels, indent=2), encoding="utf-8")
            written += 1
            if args.verbose:
                print(f"wrote {out_path} ({len(result.labels)} faces)")
        except Exception as exc:  # Keep batch jobs moving and report all failures.
            failed.append((step_path.name, str(exc)))
            print(f"failed {step_path.name}: {exc}")

    print(f"step files: {len(step_files)}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"failed: {len(failed)}")
    if failed:
        print("failures:")
        for name, message in failed:
            print(f"  {name}: {message}")


if __name__ == "__main__":
    main()
