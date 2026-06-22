"""从 full 模式预测结果里，按少数类每类随机抽样，生成抽检样本 id JSON。

读取一批 full-mode 预测文件（格式：[[sample_id, {"seg": {face_id: label}, "inst": ...}]]），
收集 label 属于少数类（hole=1 / boss=2 / chamfer=3）的 face，每类随机抽 N 个，
写出一个抽检清单 JSON，供 copy_inspection_samples.py 使用。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MINORITY_LABELS = {1: "hole", 2: "boss", 3: "chamfer"}


def load_predictions(pred_dir: Path) -> list[tuple[str, Path, dict]]:
    """返回 [(sample_id, pred_file_path, seg_dict), ...]。"""
    items: list[tuple[str, Path, dict]] = []
    for path in sorted(pred_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"[skip] 无法解析 {path.name}")
            continue
        if not isinstance(data, list) or not data:
            continue
        sample_id, payload = data[0]
        seg = payload.get("seg", {}) if isinstance(payload, dict) else {}
        items.append((sample_id, path, seg))
    return items


def collect_face_pool(items):
    """按 label 分组所有少数类 face：{label: [(sample_id, face_id, pred_file), ...]}。"""
    pool: dict[int, list[tuple[str, str, Path]]] = defaultdict(list)
    for sample_id, pred_file, seg in items:
        for face_id, label in seg.items():
            if label in MINORITY_LABELS:
                pool[label].append((sample_id, face_id, pred_file))
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="从 full 模式预测结果抽样生成抽检清单")
    parser.add_argument(
        "--pred-dir",
        default=r"E:\dataset\MFR\MFR\pred_label",
        help="full 模式预测结果目录",
    )
    parser.add_argument("--per-class", type=int, default=100, help="每个少数类抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--out", default="inspection_sample.json", help="输出清单 JSON 路径")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"预测目录不存在: {pred_dir}")

    items = load_predictions(pred_dir)
    print(f"载入预测文件 {len(items)} 个")

    pool = collect_face_pool(items)
    rng = random.Random(args.seed)

    sampled: list[dict] = []
    summary: dict[str, dict] = {}
    for label, name in MINORITY_LABELS.items():
        candidates = pool.get(label, [])
        n_take = min(args.per_class, len(candidates))
        picks = rng.sample(candidates, n_take) if n_take else []
        for sample_id, face_id, pred_file in picks:
            sampled.append(
                {
                    "sample_id": sample_id,
                    "face_id": face_id,
                    "label": label,
                    "label_name": name,
                    "pred_file": pred_file.name,
                }
            )
        summary[name] = {"label": label, "available": len(candidates), "sampled": n_take}

    sample_ids = sorted({s["sample_id"] for s in sampled})

    out = {
        "seed": args.seed,
        "per_class": args.per_class,
        "pred_dir": str(pred_dir),
        "summary": summary,
        "total_faces_sampled": len(sampled),
        "total_sample_ids": len(sample_ids),
        "sample_ids": sample_ids,
        "samples": sorted(sampled, key=lambda s: (s["label"], s["sample_id"], int(s["face_id"]))),
    }

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("抽样汇总：")
    for name, info in summary.items():
        print(f"  {name:8s}(label={info['label']}): 可选 {info['available']:6d}，抽样 {info['sampled']}")
    print(f"共抽样 face {len(sampled)} 个，涉及样本 {len(sample_ids)} 个")
    print(f"清单已写出 -> {args.out}")


if __name__ == "__main__":
    main()
