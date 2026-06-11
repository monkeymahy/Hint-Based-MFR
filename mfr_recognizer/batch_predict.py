from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .recognizer import HintBasedRecognizer


def predict_one(step_path: Path, out_dir: Path, overwrite: bool) -> tuple[str, Path, int, str | None]:
    out_path = out_dir / f"{step_path.stem}.json"
    if out_path.exists() and not overwrite:
        return ("skipped", out_path, 0, None)

    try:
        recognizer = HintBasedRecognizer()
        result = recognizer.recognize_step(str(step_path))
        out_path.write_text(json.dumps(result.labels, indent=2), encoding="utf-8")
        return ("written", out_path, len(result.labels), None)
    except Exception as exc:  # Keep batch jobs moving and report all failures.
        return ("failed", out_path, 0, str(exc))


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
    parser.add_argument(
        "--threads",
        "--workers",
        type=int,
        default=1,
        help="Number of local worker threads to use",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")

    step_dir = Path(args.step_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_files = sorted(list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp")))

    written = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    def handle_result(step_path: Path, status: str, out_path: Path, face_count: int, error: str | None) -> None:
        nonlocal written, skipped
        if status == "skipped":
            skipped += 1
            if args.verbose:
                print(f"skip existing: {out_path}")
        elif status == "written":
            written += 1
            if args.verbose:
                print(f"wrote {out_path} ({face_count} faces)")
        else:
            message = error or "unknown error"
            failed.append((step_path.name, message))
            print(f"failed {step_path.name}: {message}")

    if args.threads == 1:
        for step_path in step_files:
            handle_result(step_path, *predict_one(step_path, out_dir, args.overwrite))
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(predict_one, step_path, out_dir, args.overwrite): step_path
                for step_path in step_files
            }
            for future in as_completed(futures):
                step_path = futures[future]
                handle_result(step_path, *future.result())

    print(f"step files: {len(step_files)}")
    print(f"threads: {args.threads}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"failed: {len(failed)}")
    if failed:
        print("failures:")
        for name, message in failed:
            print(f"  {name}: {message}")


if __name__ == "__main__":
    main()
