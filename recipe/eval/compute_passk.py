#!/usr/bin/env python3
"""从 pass@k 评测的验证 JSONL 中计算 pass@k。

输入是 run_passk_after_training.sh 输出到 ``validation_data_dir`` 的
验证 JSONL。文件名由 verl 的 global_steps 决定：未加载 checkpoint 时是
``0.jsonl``；加载 global_step_50 后是 ``50.jsonl``。本脚本的 ``--dump``
既接受具体文件路径，也接受目录路径（目录会自动找到其中唯一的 ``*.jsonl``）。
每行对应一条采样轨迹，同一问题（按 ``input`` 分组）有连续 ``n`` 行，其中
``n`` 是验证时设置的 ``val_kwargs.n``。

用法：:

    python recipe/eval/compute_passk.py \
        --dump /root/autodl-tmp/validation/qwen3_4b_vllm_passk \
        --n 5

可选：加上 ``--write-eval-json <path>`` 会输出与 RepExp 画图脚本兼容的
``eval.json``（扁平 dict，键形如 ``reward/pass@{k}/mean``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_float(value: Any) -> float:
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return float(value or 0.0)


def _pass_values(records: list[dict[str, Any]], metric: str, f1_threshold: float) -> list[int]:
    """把每题 n 条轨迹折叠成 0/1：至少一条通过则为 1。"""
    if metric == "answer_exact":
        hits = [int(_as_bool(r.get("answer_exact"))) for r in records]
    else:
        hits = [int(_as_float(r.get("answer_f1")) >= f1_threshold) for r in records]
    # 顺序无关紧要，同一问题只要有一个成功即通过。
    return [1 if any(hits) else 0]


def _strategy_pass_values(records: list[dict[str, Any]]) -> list[int]:
    """At least one record follows the expected strategy (strategy_correct)."""
    return [1 if any(bool(r.get("strategy_correct")) for r in records) else 0]


def _strategy_and_answer_pass_values(
    records: list[dict[str, Any]], metric: str, f1_threshold: float
) -> list[int]:
    """At least one record follows the strategy AND answers correctly."""
    hits = []
    for record in records:
        strategy_ok = bool(record.get("strategy_correct"))
        if metric == "answer_exact":
            answer_ok = _as_bool(record.get("answer_exact"))
        else:
            answer_ok = _as_float(record.get("answer_f1")) >= f1_threshold
        hits.append(int(strategy_ok and answer_ok))
    return [1 if any(hits) else 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True, help="验证 JSONL（validation_data_dir/0.jsonl）")
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="每条问题采样的轨迹数（= 训练时的 val_kwargs.n）",
    )
    parser.add_argument(
        "--metric",
        choices=["answer_exact", "answer_f1"],
        default="answer_exact",
        help="判定“通过”的指标（默认答案完全匹配）",
    )
    parser.add_argument("--f1-threshold", type=float, default=1.0, help="用 answer_f1 时的阈值")
    parser.add_argument("--write-eval-json", type=Path, help="可选：写出与 RepExp 画图脚本兼容的 eval.json")
    args = parser.parse_args()

    dump_path = args.dump
    if dump_path.is_dir():
        candidates = sorted(dump_path.glob("*.jsonl"))
        if not candidates:
            print(f"错误：{dump_path} 目录中没有 *.jsonl", file=sys.stderr)
            return 1
        dump_path = candidates[0]
        print(f"使用目录中的验证 dump：{dump_path}", file=sys.stderr)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with dump_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            groups[str(record.get("input", ""))].append(record)

    if not groups:
        print(f"错误：{args.dump} 中没有可解析的轨迹", file=sys.stderr)
        return 1

    total = len(groups)
    if args.n < 1:
        print("错误：--n 必须为正整数", file=sys.stderr)
        return 1

    # 除常用的 1/2/4/8... 外，始终包含用户实际指定的 k。
    # 例如 n=5 时必须汇报 pass@5，而不能只停在 pass@4。
    report_ks = {k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= args.n}
    report_ks.add(args.n)
    pass_counts = {k: 0 for k in sorted(report_ks)}
    strategy_pass_counts = {k: 0 for k in sorted(report_ks)}
    strategy_answer_pass_counts = {k: 0 for k in sorted(report_ks)}
    strategy_hits = 0
    strategy_total = 0
    strategy_hits_by_type: dict[str, int] = defaultdict(int)
    strategy_total_by_type: dict[str, int] = defaultdict(int)
    parallel_total = 0
    sequential_total = 0
    parallel_pass_counts = {k: 0 for k in sorted(report_ks)}
    sequential_pass_counts = {k: 0 for k in sorted(report_ks)}
    valid_total = 0
    for prompt, records in groups.items():
        if len(records) != args.n:
            print(
                f"警告：题目 {prompt[:60]!r} 有 {len(records)} 条轨迹，不等于 --n {args.n}，跳过该题",
                file=sys.stderr,
            )
            continue
        valid_total += 1
        for k in pass_counts:
            # pass@k：前 k 条中至少一条通过（interleave 顺序即采样顺序）。
            prefix = records[:k]
            if _pass_values(prefix, args.metric, args.f1_threshold)[0]:
                pass_counts[k] += 1
        for record in records:
            strategy_total += 1
            strategy_type = str(record.get("expected_strategy") or "unlabeled")
            strategy_total_by_type[strategy_type] += 1
            if bool(record.get("strategy_correct")):
                strategy_hits += 1
                strategy_hits_by_type[strategy_type] += 1
        for k in strategy_pass_counts:
            if _strategy_pass_values(records[:k])[0]:
                strategy_pass_counts[k] += 1
            if _strategy_and_answer_pass_values(records[:k], args.metric, args.f1_threshold)[0]:
                strategy_answer_pass_counts[k] += 1
        question_strategy = str(records[0].get("expected_strategy") or "")
        if question_strategy == "parallel":
            parallel_total += 1
            target_counts = parallel_pass_counts
        elif question_strategy == "sequential":
            sequential_total += 1
            target_counts = sequential_pass_counts
        else:
            target_counts = None
        if target_counts is not None:
            for k in target_counts:
                if _pass_values(records[:k], args.metric, args.f1_threshold)[0]:
                    target_counts[k] += 1

    if valid_total == 0:
        print(
            f"错误：没有题目的轨迹数等于 --n {args.n}；请确认 --n 与验证时的 val_kwargs.n 一致。",
            file=sys.stderr,
        )
        return 1

    print(f"题目数：{total}（有效 {valid_total}，指标 {args.metric}）")
    for k in sorted(pass_counts):
        rate = pass_counts[k] / valid_total
        print(f"pass@{k:<4} = {pass_counts[k]:>4}/{valid_total} = {rate:.4f}")

    if strategy_total:
        print(
            f"策略遵循率（全部轨迹）: {strategy_hits}/{strategy_total} = "
            f"{strategy_hits / strategy_total:.4f}"
        )
        for strategy_type in sorted(strategy_total_by_type):
            if strategy_total_by_type[strategy_type]:
                rate = strategy_hits_by_type[strategy_type] / strategy_total_by_type[strategy_type]
                print(
                    f"  {strategy_type}: {strategy_hits_by_type[strategy_type]}/"
                    f"{strategy_total_by_type[strategy_type]} = {rate:.4f}"
                )
    for k in sorted(strategy_pass_counts):
        rate = strategy_pass_counts[k] / valid_total
        print(f"策略pass@{k:<4} = {strategy_pass_counts[k]:>4}/{valid_total} = {rate:.4f}")
    for k in sorted(strategy_answer_pass_counts):
        rate = strategy_answer_pass_counts[k] / valid_total
        print(f"策略+答案pass@{k:<4} = {strategy_answer_pass_counts[k]:>4}/{valid_total} = {rate:.4f}")

    if parallel_total:
        for k in sorted(parallel_pass_counts):
            rate = parallel_pass_counts[k] / parallel_total
            print(f"并行pass@{k:<4} = {parallel_pass_counts[k]:>4}/{parallel_total} = {rate:.4f}")
    if sequential_total:
        for k in sorted(sequential_pass_counts):
            rate = sequential_pass_counts[k] / sequential_total
            print(f"串行pass@{k:<4} = {sequential_pass_counts[k]:>4}/{sequential_total} = {rate:.4f}")

    if args.write_eval_json is not None:
        eval_data = {f"reward/pass@{k}/mean": pass_counts[k] / valid_total for k in pass_counts}
        if strategy_total:
            eval_data["strategy/rate"] = strategy_hits / strategy_total
        eval_data.update({f"strategy/pass@{k}/mean": strategy_pass_counts[k] / valid_total for k in strategy_pass_counts})
        eval_data.update(
            {
                f"strategy_and_answer/pass@{k}/mean": strategy_answer_pass_counts[k] / valid_total
                for k in strategy_answer_pass_counts
            }
        )
        if parallel_total:
            eval_data.update({f"answer/parallel/pass@{k}/mean": parallel_pass_counts[k] / parallel_total for k in parallel_pass_counts})
        if sequential_total:
            eval_data.update({f"answer/sequential/pass@{k}/mean": sequential_pass_counts[k] / sequential_total for k in sequential_pass_counts})
        args.write_eval_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_eval_json.write_text(
            json.dumps(eval_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已写出 eval.json：{args.write_eval_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
