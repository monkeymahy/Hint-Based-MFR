"""按抽检清单 JSON，把对应的 STEP 样本和预测标签复制到目的目录。

读取 sample_inspection.py 产出的清单，取其中所有 sample_id，
把每个样本的 STEP 文件和 full 模式预测 JSON 一起复制到 --out-dir。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

STEP_SUFFIXES = (".step", ".stp")


def find_step(step_dir: Path, sample_id: str) -> Path | None:
    for suffix in STEP_SUFFIXES:
        candidate = step_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="按抽检清单复制 STEP 与预测标签")
    parser.add_argument("--id-json", default="inspection_sample.json", help="抽检清单 JSON")
    parser.add_argument(
        "--step-dir",
        default=r"E:\dataset\MFR\MFR\step",
        help="STEP 样本目录",
    )
    parser.add_argument(
        "--pred-dir",
        default=r"E:\dataset\MFR\MFR\pred_label",
        help="full 模式预测结果目录",
    )
    parser.add_argument("--out-dir", default="inspection_set", help="目的目录")
    args = parser.parse_args()

    id_path = Path(args.id_json)
    spec = json.loads(id_path.read_text(encoding="utf-8"))
    sample_ids = spec.get("sample_ids", [])
    if not sample_ids:
        raise SystemExit("清单为空，无可复制的样本")

    step_dir = Path(args.step_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_step = out_dir / "step"
    out_label = out_dir / "pred_label"
    out_step.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    copied_step = 0
    copied_pred = 0
    missing_step: list[str] = []
    missing_pred: list[str] = []

    for sample_id in sample_ids:
        step_src = find_step(step_dir, sample_id)
        if step_src is None:
            missing_step.append(sample_id)
        else:
            shutil.copy2(step_src, out_step / step_src.name)
            copied_step += 1

        pred_src = pred_dir / f"{sample_id}.json"
        if not pred_src.exists():
            missing_pred.append(sample_id)
        else:
            shutil.copy2(pred_src, out_label / pred_src.name)
            copied_pred += 1

    # 顺手把清单本身也放进去，便于核对。
    shutil.copy2(id_path, out_dir / id_path.name)

    print(f"目的目录: {out_dir}")
    print(f"复制 STEP : {copied_step}/{len(sample_ids)}")
    print(f"复制预测  : {copied_pred}/{len(sample_ids)}")
    if missing_step:
        print(f"缺失 STEP ({len(missing_step)}): {missing_step[:10]}{' ...' if len(missing_step) > 10 else ''}")
    if missing_pred:
        print(f"缺失预测 ({len(missing_pred)}): {missing_pred[:10]}{' ...' if len(missing_pred) > 10 else ''}")


if __name__ == "__main__":
    main()
