"""从 full 模式预测结果里，按少数类每类随机抽取「实例」，生成抽检样本 id JSON。

一个「实例」= inst 邻接矩阵里的一个连通分量（同一特征的所有 face）。
少数类（hole=1 / boss=2 / chamfer=3）每类随机抽 N 个实例，
不足则全取。同时兼容 labels 模式（无 inst）作为降级：此时把 label 相同且
在面顺序上相邻的 face 当作一个近似实例。

输出 inspection_sample.json，供 copy_inspection_samples.py 使用。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MINORITY_LABELS = {1: "hole", 2: "boss", 3: "chamfer"}


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def parse_pred_file(path: Path) -> tuple[str, dict, list[list[int]] | None]:
    """返回 (sample_id, seg_dict, inst_matrix_or_None)。

    兼容两种格式：
    - full 模式: [[sample_id, {"seg": {...}, "inst": [[...]]}]]
    - labels 模式: [1, 1, 0, ...]  → 降级，inst 为 None
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("空文件或格式异常")
    if isinstance(data[0], int):
        # labels 模式：seg = {face_idx: label}
        seg = {str(i): int(v) for i, v in enumerate(data)}
        return path.stem, seg, None
    sample_id, payload = data[0]
    payload = payload or {}
    seg = payload.get("seg", {})
    inst = payload.get("inst")
    return sample_id, seg, inst


def instances_from_inst(seg: dict, inst: list[list[int]]) -> list[tuple[int, list[int]]]:
    """用并查集从 inst 邻接矩阵还原实例，返回 [(label, [face_idx, ...]), ...]。"""
    if not inst:
        return []
    n = len(inst)
    dsu = DSU(n)
    for i in range(n):
        row = inst[i]
        for j in range(i + 1, n):
            if row[j]:
                dsu.union(i, j)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[dsu.find(i)].append(i)
    instances: list[tuple[int, list[int]]] = []
    for faces in comps.values():
        labels = {int(seg.get(str(f), 0)) for f in faces}
        # 一个实例内 label 应一致；若不一致取众数（防御性，正常不会发生）。
        label = max(labels, key=lambda l: sum(1 for f in faces if int(seg.get(str(f), 0)) == l))
        instances.append((label, sorted(faces)))
    return instances


def instances_from_labels_flat(seg: dict) -> list[tuple[int, list[int]]]:
    """降级：把 label 相同且面序号相邻的 face 当成一个近似实例。"""
    faces = sorted(int(k) for k in seg)
    instances: list[tuple[int, list[int]]] = []
    cur_label: int | None = None
    cur_faces: list[int] = []
    for f in faces:
        lab = int(seg[str(f)])
        if cur_label is None or lab != cur_label:
            if cur_faces:
                instances.append((cur_label, cur_faces))
            cur_label, cur_faces = lab, [f]
        else:
            cur_faces.append(f)
    if cur_faces:
        instances.append((cur_label, cur_faces))
    return instances


def collect_instance_pool(items):
    """按 label 分组所有少数类实例：{label: [(sample_id, pred_file, faces), ...]}。"""
    pool: dict[int, list[tuple[str, Path, list[int]]]] = defaultdict(list)
    for sample_id, pred_file, seg, inst in items:
        instances = (
            instances_from_inst(seg, inst)
            if inst is not None
            else instances_from_labels_flat(seg)
        )
        for label, faces in instances:
            if label in MINORITY_LABELS:
                pool[label].append((sample_id, pred_file, faces))
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="从 full 模式预测结果按实例抽样")
    parser.add_argument(
        "--pred-dir",
        default=r"E:\dataset\MFR\MFR\pred_label",
        help="预测结果目录（full 或 labels 模式均可）",
    )
    parser.add_argument("--per-class", type=int, default=100, help="每个少数类抽样实例数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--out", default="inspection_sample.json", help="输出清单 JSON 路径")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"预测目录不存在: {pred_dir}")

    items = []
    full_count = 0
    labels_count = 0
    for path in sorted(pred_dir.glob("*.json")):
        try:
            sample_id, seg, inst = parse_pred_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[skip] 无法解析 {path.name}: {exc}")
            continue
        items.append((sample_id, path, seg, inst))
        if inst is not None:
            full_count += 1
        else:
            labels_count += 1
    print(f"载入预测文件 {len(items)} 个（full 模式 {full_count}，labels 模式降级 {labels_count}）")
    if labels_count:
        print("提示：检测到 labels 模式文件，按相邻同标签近似实例；建议用 --mode full 重新预测以获得准确实例。")

    pool = collect_instance_pool(items)
    rng = random.Random(args.seed)

    sampled: list[dict] = []
    summary: dict[str, dict] = {}
    for label, name in MINORITY_LABELS.items():
        candidates = pool.get(label, [])
        n_take = min(args.per_class, len(candidates))
        picks = rng.sample(candidates, n_take) if n_take else []
        for sample_id, pred_file, faces in picks:
            sampled.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "label_name": name,
                    "faces": faces,
                    "pred_file": pred_file.name,
                }
            )
        summary[name] = {"label": label, "available": len(candidates), "sampled": n_take}

    sample_ids = sorted({s["sample_id"] for s in sampled})

    out = {
        "seed": args.seed,
        "per_class": args.per_class,
        "unit": "instance",
        "pred_dir": str(pred_dir),
        "summary": summary,
        "total_instances_sampled": len(sampled),
        "total_sample_ids": len(sample_ids),
        "sample_ids": sample_ids,
        "samples": sorted(sampled, key=lambda s: (s["label"], s["sample_id"])),
    }

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("抽样汇总（按实例）：")
    for name, info in summary.items():
        print(f"  {name:8s}(label={info['label']}): 可选 {info['available']:6d}，抽样 {info['sampled']}")
    print(f"共抽样实例 {len(sampled)} 个，涉及样本 {len(sample_ids)} 个")
    print(f"清单已写出 -> {args.out}")


if __name__ == "__main__":
    main()
