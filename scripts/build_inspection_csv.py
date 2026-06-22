"""根据 inspection_sample.json 生成人工抽检用的 CSV 表格。

产出两个 CSV（UTF-8 BOM，Excel 友好）：
- inspection_instances.csv：每行一个抽样实例，用于记录 precision 与逐实例对错。
- inspection_samples.csv：每行一个 sample_id，预填预测计数，留空真值/命中计数，
  用于记录 recall 与每样本情况。

两个文件里需要人工填的列留空，预填列由 inspection_sample.json 和预测目录算出。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sample_inspection import MINORITY_LABELS, parse_pred_file


LABEL_NAMES = {0: "other", 1: "hole", 2: "boss", 3: "chamfer"}


def per_sample_instance_counts(pred_dir: Path, sample_ids: set[str]) -> dict[str, Counter[int]]:
    """从预测文件重算每个 sample 的实例计数，返回 {sample_id: Counter(label)}。

    仅统计少数类实例；other 不计（other 的每个面自成一块，数量无意义）。
    """
    counts: dict[str, Counter[int]] = {}
    for sid in sample_ids:
        pred_path = pred_dir / f"{sid}.json"
        counter: Counter[int] = Counter()
        if pred_path.exists():
            try:
                _, seg, inst = parse_pred_file(pred_path)
                from sample_inspection import (
                    instances_from_inst,
                    instances_from_labels_flat,
                )
                instances = (
                    instances_from_inst(seg, inst)
                    if inst is not None
                    else instances_from_labels_flat(seg)
                )
                for label, _faces in instances:
                    if label in MINORITY_LABELS:
                        counter[label] += 1
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        counts[sid] = counter
    return counts


def write_instances_csv(spec: dict, out_path: Path) -> int:
    header = [
        "instance_no",
        "sample_id",
        "pred_label",
        "pred_label_name",
        "faces",
        "face_count",
        "is_real_feature",
        "true_label",
        "is_class_correct",
        "boundary_correct",
        "notes",
    ]
    samples = spec.get("samples", [])
    # 按 sample_id 聚合后给连续序号，便于一次性开一个样本检查其所有实例。
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_sample[s["sample_id"]].append(s)

    rows = []
    instance_no = 0
    for sample_id in sorted(by_sample):
        for s in sorted(by_sample[sample_id], key=lambda x: (x["label"], x["faces"][0] if x["faces"] else 0)):
            instance_no += 1
            rows.append(
                {
                    "instance_no": instance_no,
                    "sample_id": sample_id,
                    "pred_label": s["label"],
                    "pred_label_name": s["label_name"],
                    "faces": ",".join(str(f) for f in s["faces"]),
                    "face_count": len(s["faces"]),
                    "is_real_feature": "",
                    "true_label": "",
                    "is_class_correct": "",
                    "boundary_correct": "",
                    "notes": "",
                }
            )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_samples_csv(spec: dict, pred_dir: Path | None, out_path: Path) -> int:
    sample_ids = sorted(spec.get("sample_ids", []))

    # 该样本被抽中的实例数（按类）。
    sampled_by_sample: dict[str, Counter[int]] = defaultdict(Counter)
    for s in spec.get("samples", []):
        sampled_by_sample[s["sample_id"]][s["label"]] += 1

    pred_counts: dict[str, Counter[int]] = {}
    if pred_dir is not None and pred_dir.is_dir():
        pred_counts = per_sample_instance_counts(pred_dir, set(sample_ids))

    header = [
        "sample_id",
        "sampled_instances",
        "sampled_hole",
        "sampled_boss",
        "sampled_chamfer",
        "pred_count_hole",
        "pred_count_boss",
        "pred_count_chamfer",
        "true_count_hole",
        "true_count_boss",
        "true_count_chamfer",
        "correct_count_hole",
        "correct_count_boss",
        "correct_count_chamfer",
        "notes",
    ]

    rows = []
    for sid in sample_ids:
        sampled = sampled_by_sample.get(sid, Counter())
        pred = pred_counts.get(sid, Counter())
        rows.append(
            {
                "sample_id": sid,
                "sampled_instances": sum(sampled.values()),
                "sampled_hole": sampled.get(1, 0),
                "sampled_boss": sampled.get(2, 0),
                "sampled_chamfer": sampled.get(3, 0),
                "pred_count_hole": pred.get(1, ""),
                "pred_count_boss": pred.get(2, ""),
                "pred_count_chamfer": pred.get(3, ""),
                "true_count_hole": "",
                "true_count_boss": "",
                "true_count_chamfer": "",
                "correct_count_hole": "",
                "correct_count_boss": "",
                "correct_count_chamfer": "",
                "notes": "",
            }
        )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="从抽检清单生成人工检查 CSV 表格")
    parser.add_argument("--id-json", default="inspection_sample.json", help="抽检清单 JSON")
    parser.add_argument(
        "--pred-dir",
        default=r"E:\dataset\MFR\MFR\pred_label",
        help="预测结果目录（用于预填每样本预测实例计数；不存在则留空）",
    )
    parser.add_argument("--out-instances", default="inspection_instances.csv", help="逐实例 CSV 输出路径")
    parser.add_argument("--out-samples", default="inspection_samples.csv", help="逐样本 CSV 输出路径")
    args = parser.parse_args()

    id_path = Path(args.id_json)
    if not id_path.exists():
        raise SystemExit(f"找不到抽检清单: {id_path}")
    spec = json.loads(id_path.read_text(encoding="utf-8"))

    pred_dir = Path(args.pred_dir)

    n_inst = write_instances_csv(spec, Path(args.out_instances))
    n_samp = write_samples_csv(spec, pred_dir if pred_dir.is_dir() else None, Path(args.out_samples))

    print(f"逐实例表 -> {args.out_instances}（{n_inst} 行）")
    print(f"逐样本表 -> {args.out_samples}（{n_samp} 行）")
    print()
    print("填写说明：")
    print("  inspection_instances.csv 填: is_real_feature(1/0) true_label(0/1/2/3) "
          "is_class_correct(1/0) boundary_correct(exact/partial/wrong) notes")
    print("  inspection_samples.csv  填: true_count_* correct_count_* notes")
    print("  precision_k = Σ is_class_correct(pred=k) / Σ 抽样数(k)   [来自逐实例表]")
    print("  recall_k    = Σ correct_count_k / Σ true_count_k          [来自逐样本表]")


if __name__ == "__main__":
    main()
