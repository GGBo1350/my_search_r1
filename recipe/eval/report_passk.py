#!/usr/bin/env python3
"""汇总未训练基线 / 训练后的 pass@1 与 pass@k，输出 markdown 对比表。

四份输入是四个评测脚本分别写出的验证 JSONL（文件名可能是 0.jsonl 或
global_step_{n}.jsonl，均可作为目录或文件传入，脚本会自动定位）：

- 未训练基线 greedy（run_pretrained_baseline.sh VAL_K=1 温度 0）→ pass@1
- 未训练基线 采样（run_pretrained_baseline.sh VAL_K=k 温度 0.7）→ pass@k
- 训练后    greedy（run_fixed200_after_training.sh 固定 n=1）→ pass@1
- 训练后    采样（run_passk_after_training.sh VAL_K=k 温度 0.7）→ pass@k

用法：:

    python recipe/eval/report_passk.py \
        --baseline-greedy /path/baseline_greedy \
        --baseline-sample /path/baseline_passk5 \
        --trained-greedy  /path/step50_greedy \
        --trained-sample  /path/step50_passk5 \
        --n 5

可选：``--f1-threshold 0.5`` 额外输出 F1 口径的 pass@k；``--output table.md``
把 markdown 写入文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _to_float(value: Any) -> float:
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return float(value or 0.0)


def _resolve_dump(path: Path) -> Path:
    """接受文件或目录；目录时自动找其中的 *.jsonl（验证 dump 可能叫 0.jsonl
    或 global_step_{n}.jsonl，取决于是否加载了 checkpoint）。"""
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"目录中没有 *.jsonl：{path}")
        return candidates[0]
    return path


def _load(dump_path: Path) -> dict[str, list[dict[str, Any]]]:
    dump_path = _resolve_dump(dump_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with dump_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            groups[str(record.get("input", ""))].append(record)
    return groups


def pass_at_k(
    groups: dict[str, list[dict[str, Any]]],
    k: int,
    metric: str,
    f1_threshold: float,
) -> float | None:
    """返回 pass@k；用每组的前 k 条判定，至少一条通过即通过。"""
    total = 0
    passed = 0
    for records in groups.values():
        sample = records[:k]
        if not sample:
            continue
        if metric == "answer_exact":
            ok = any(_to_bool(r.get("answer_exact")) for r in sample)
        else:
            ok = any(_to_float(r.get("answer_f1")) >= f1_threshold for r in sample)
        total += 1
        passed += int(ok)
    return passed / total if total else None


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def pass_at_k_by_strategy(
    groups: dict[str, list[dict[str, Any]]],
    k: int,
    strategy: str,
    metric: str,
    f1_threshold: float,
) -> float | None:
    """指定策略（parallel/sequential）题目的答案 pass@k。"""
    total = 0
    passed = 0
    for records in groups.values():
        if not records:
            continue
        if str(records[0].get("expected_strategy") or "") != strategy:
            continue
        sample = records[:k]
        if not sample:
            continue
        if metric == "answer_exact":
            ok = any(_to_bool(r.get("answer_exact")) for r in sample)
        else:
            ok = any(_to_float(r.get("answer_f1")) >= f1_threshold for r in sample)
        total += 1
        passed += int(ok)
    return passed / total if total else None


def strategy_rate(groups: dict[str, list[dict[str, Any]]], k: int | None = None) -> float | None:
    """策略遵循率：全部轨迹（k=None）或每组前 k 条中 strategy_correct 的比例。"""
    total = 0
    passed = 0
    for records in groups.values():
        sample = records[:k] if k else records
        for record in sample:
            total += 1
            passed += int(_to_bool(record.get("strategy_correct")))
    return passed / total if total else None


def strategy_pass_at_k(
    groups: dict[str, list[dict[str, Any]]],
    k: int,
    also_answer: bool = False,
    metric: str = "answer_exact",
    f1_threshold: float = 0.5,
) -> float | None:
    """策略 pass@k：每题 k 条中至少一条遵循（also_answer 时还需答案正确）。"""
    total = 0
    passed = 0
    for records in groups.values():
        sample = records[:k]
        if not sample:
            continue
        ok = False
        for record in sample:
            strategy_ok = _to_bool(record.get("strategy_correct"))
            if not also_answer:
                ok = ok or strategy_ok
                continue
            if metric == "answer_exact":
                answer_ok = _to_bool(record.get("answer_exact"))
            else:
                answer_ok = _to_float(record.get("answer_f1")) >= f1_threshold
            ok = ok or (strategy_ok and answer_ok)
        total += 1
        passed += int(ok)
    return passed / total if total else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-greedy", type=Path, required=True)
    parser.add_argument("--baseline-sample", type=Path, required=True)
    parser.add_argument("--trained-greedy", type=Path, required=True)
    parser.add_argument("--trained-sample", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5, help="采样评测的 k")
    parser.add_argument("--f1-threshold", type=float, default=0.5, help="F1 口径判定阈值")
    parser.add_argument("--output", type=Path, help="可选：把 markdown 写入文件")
    parser.add_argument(
        "--extra-pair",
        action="append",
        default=[],
        metavar="LABEL:GREEDY_PATH:SAMPLE_PATH",
        help="额外对比行（可多次指定），例如 --extra-pair 训练100step:/path/greedy:/path/sample",
    )
    args = parser.parse_args()

    bg = _load(args.baseline_greedy)
    bs = _load(args.baseline_sample)
    tg = _load(args.trained_greedy)
    ts = _load(args.trained_sample)

    rows = [
        ("未训练 Qwen3-4B", bg, bs),
        ("训练 checkpoint", tg, ts),
    ]
    for spec in args.extra_pair:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            parser.error(f"--extra-pair 需要 LABEL:GREEDY_PATH:SAMPLE_PATH，得到：{spec}")
        label, greedy_path, sample_path = parts
        rows.append((label, _load(Path(greedy_path)), _load(Path(sample_path))))

    # 区分两种 pass@1 口径：
    #   greedy pass@1 —— 温度 0 确定性解码（greedy 文件里每条问题的第 1 条轨迹）
    #   采样   pass@1 —— 温度 >0 采样（采样文件里每条问题的第 1 条轨迹）
    # 两者数值通常不同，必须分别报告、分别对比。
    lines = [
        "| 模型 | greedy pass@1 | 采样 pass@1 | "
        f"采样 pass@{args.n} (exact) | 采样 pass@{args.n} (F1≥{args.f1_threshold}) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, greedy_groups, sample_groups in rows:
        p1_greedy = pass_at_k(greedy_groups, 1, "answer_exact", args.f1_threshold)
        p1_sample = pass_at_k(sample_groups, 1, "answer_exact", args.f1_threshold)
        pk_exact = pass_at_k(sample_groups, args.n, "answer_exact", args.f1_threshold)
        pk_f1 = pass_at_k(sample_groups, args.n, "answer_f1", args.f1_threshold)
        lines.append(
            f"| {name} | {_fmt(p1_greedy)} | {_fmt(p1_sample)} | "
            f"{_fmt(pk_exact)} | {_fmt(pk_f1)} |"
        )

    table = "\n".join(lines)
    print(table)

    lines2 = [
        "| 模型 | greedy 策略遵循率 | 采样策略遵循率 | "
        f"采样 pass@{args.n} 策略遵循 | 采样 pass@{args.n} 策略+答对 | "
        f"采样 pass@{args.n} 并行答对 | 采样 pass@{args.n} 串行答对 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, greedy_groups, sample_groups in rows:
        g_rate = strategy_rate(greedy_groups, k=1)
        s_rate = strategy_rate(sample_groups)
        spk = strategy_pass_at_k(sample_groups, args.n, also_answer=False, metric="answer_exact", f1_threshold=args.f1_threshold)
        spk_answer = strategy_pass_at_k(sample_groups, args.n, also_answer=True, metric="answer_f1", f1_threshold=args.f1_threshold)
        parallel_answer = pass_at_k_by_strategy(sample_groups, args.n, "parallel", "answer_exact", args.f1_threshold)
        sequential_answer = pass_at_k_by_strategy(sample_groups, args.n, "sequential", "answer_exact", args.f1_threshold)
        lines2.append(
            f"| {name} | {_fmt(g_rate)} | {_fmt(s_rate)} | "
            f"{_fmt(spk)} | {_fmt(spk_answer)} | {_fmt(parallel_answer)} | {_fmt(sequential_answer)} |"
        )
    table2 = "\n".join(lines2)
    print()
    print(table2)

    # 顺便输出样本量，便于核对口径。
    for name, greedy_groups, sample_groups in rows:
        print(
            f"{name}: greedy 题数 {len(greedy_groups)}，"
            f"采样题数 {len(sample_groups)}",
            file=sys.stderr,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n\n" + table2 + "\n", encoding="utf-8")
        print(f"已写出：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
