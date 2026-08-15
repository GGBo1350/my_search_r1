# Search-R1 实现说明

本代码树基于 veRL v0.8，面向 HotpotQA distractor hard 的多跳搜索 Agent。公开实验将 Phase 1（GRPO）和 Phase 2（OPD）作为从同一 Base 模型出发的两条对照路线，不把 Phase 2 描述成 Phase 1 的继续训练。

## 任务定义

- Bridge：第二跳实体依赖第一跳结果，正确工具调用分组为 `[1,1]`。
- Comparison：两个查询对象由题面独立给出，正确分组为同轮 `[2]`。
- 模型最多执行三次搜索，最终答案限制为简洁实体或短句。

题型、gold 标题和期望拓扑只保存在 `extra_info` 中供奖励与评测使用，不泄漏到模型 prompt。

## 数据

`recipe/data/deepseek_clean_hard_data.py` 对 hard 候选执行双 gold 文档审核，并剔除题面直答、单文档足够、并非恰好双搜索，以及题型与依赖拓扑不一致的样本。审核结果增量缓存到 JSONL，支持断点续跑；API 密钥只从环境变量读取。

默认数据规模为：

- Train：1200 Bridge + 400 Comparison。
- Validation：100 Bridge + 100 Comparison。

检索使用本地 SQLite/FTS5 + BM25 候选、FAISS dense 候选及 reranker。原始数据、Parquet、索引、API 缓存与模型权重不进入 Git。

## Phase 1：GRPO 对照路线

Qwen3-4B Base 使用 LoRA rank 32 / alpha 64，目标模块覆盖每个 Transformer block 的：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

优化信号同时包含答案 F1/Exact、检索召回、搜索策略、格式和调用约束。固定 200 条 greedy 评测中，125-step 实验的 `global_step_100` 最佳：Exact 0.590、Mean F1 0.720、Strategy 0.915。

## Phase 2：OPD 对照路线

OPD student 从独立的 Qwen3-4B Base 初始化，在自身 on-policy 搜索轨迹上学习 teacher token 分布。已实现：

- 全 7 投影 LoRA 的 Bridge s75 + Comparison s25 双静态 teacher，按 `teacher_route` 路由。
- Phase 1 s100 单静态 teacher，sample-token `k=3` 蒸馏。
- 已完成的单卡路由双 teacher MOPD：混合 Bridge/Comparison 数据，sample-token `k=3`，100 step；teacher 最大并发降为 2，并在 teacher 打分后、actor backward 前释放 KV cache。Forward-KL Top-32 长跑曾在 actor backward 阶段 OOM。
- 已完成单卡串行 Sample-K3 消融：独立 Base student 先接受 Bridge teacher 75 step，再接受 Comparison teacher 25 step；Bridge 保留 s25/s50/s75，最终 student 为 s100。
- teacher 与 student adapter 的模块、rank、文件完整性在训练前 fail-closed 校验。
- 训练时记录 teacher probability mass、token overlap、蒸馏 loss 与工具执行指标到 SwanLab。

最终路由 MOPD 的答案最佳点为 s50：Exact 0.585 / Mean F1 0.712 / Strategy 0.830；策略最佳点为 s100：Exact 0.575 / Mean F1 0.701 / Strategy 0.890。当前最佳综合单模型仍来自 Phase 1 GRPO s100（0.590 / 0.720 / 0.915），MOPD 则验证了不依赖任务 reward 的独立多策略学习路径。

串行 Sample-K3 的答案最佳点是 s50（Exact 0.565 / Strategy 0.530），策略最佳点是 s100（Exact 0.560 / Mean F1 0.698 / Strategy 0.825）。Comparison 阶段将 Comparison Strategy 从 s75 的 0.150 提升到 0.960，但 Bridge Strategy 从 0.810 回落到 0.690，显示简单的单向串行蒸馏存在顺序干扰。路由 MOPD s100 将整体 Strategy 提高到 0.890、Bridge Strategy 提高到 0.830，分别比串行 s100 高 6.5/14 pp。

当前专项 teacher 在固定 200 条 greedy 评测中的完整交叉结果为：Bridge s75 在 Bridge 上 Exact/Strategy 为 `0.540/0.810`，在 Comparison 上为 `0.710/0.070`；Compare s25 在 Bridge 上为 `0.430/0.600`，在 Comparison 上为 `0.700/0.980`。Bridge teacher 虽然在 Comparison 上答案分高，但没有学会 `[2]` 并行拓扑，因此必须按行为路由。两者都从 Qwen3-4B Base 独立训练，LoRA 覆盖 `q/k/v/o_proj` 与 `gate/up/down_proj`。

## veRL 关键扩展

- SGLang AgentLoop 的多工具调用、同轮并发执行和工具响应拼接。
- 静态 PEFT LoRA teacher 加载及与 student rollout 的资源共置。
- 多 teacher 题型路由与 teacher adapter 生命周期管理。
- Top-k forward KL、sample-token OPD 及相应监控指标。
- rollout 完成后释放 KV cache，降低 teacher forward 与 actor backward 的峰值显存冲突。

## 复现原则

- 模型间对比统一使用固定的 100 Bridge + 100 Comparison 验证集。
- 主要指标由 JSONL 轨迹离线严格复算，不以训练内 reward 或 LLM judge 代替。
- Checkpoint 同时报答案最佳与策略最佳：答案点优先按 Strict Exact、再按 Mean F1；策略点按 Strategy 选择，并并列保留答案、召回和格式指标。
- 机器编号只用于历史脚本名，不影响代码逻辑；路径和运行参数均可通过环境变量覆盖。

入口、命令和完整结果见 [`README.md`](README.md)，各实验脚本说明见 [`recipe/README.md`](recipe/README.md)。
