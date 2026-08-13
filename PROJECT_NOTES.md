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

- 全 7 投影 LoRA 的 Bridge s75 + Comparison s25 双静态 teacher，按 `teacher_route` 路由，Top-k forward KL。
- Phase 1 s100 单静态 teacher，sample-token `k=3` 蒸馏。
- teacher 与 student adapter 的模块、rank、文件完整性在训练前 fail-closed 校验。
- 训练时记录 teacher probability mass、token overlap、蒸馏 loss 与工具执行指标到 SwanLab。

双 teacher OPD s100 的固定集结果为 Exact 0.545 / Strategy 0.855；sample-token OPD s25 为 0.585 / 0.780，s100 为 0.580 / 0.865。当前最佳综合结果仍来自 Phase 1 GRPO s100，OPD 则验证了不依赖任务 reward 的独立策略学习路径。

当前专项 teacher 在固定 200 条 greedy 评测中的路由内结果为：Bridge s75 的 Bridge Exact/Strategy `0.540/0.810`，Comparison s25 的 Comparison Exact/Strategy `0.700/0.980`。两者都从 Qwen3-4B Base 独立训练，LoRA 覆盖 `q/k/v/o_proj` 与 `gate/up/down_proj`。已完成的双 teacher OPD s100 属于此前独立运行，不能把更新后的 teacher 指标追溯解释为该 student 的训练配置。

## veRL 关键扩展

- SGLang AgentLoop 的多工具调用、同轮并发执行和工具响应拼接。
- 静态 PEFT LoRA teacher 加载及与 student rollout 的资源共置。
- 多 teacher 题型路由与 teacher adapter 生命周期管理。
- Top-k forward KL、sample-token OPD 及相应监控指标。
- rollout 完成后释放 KV cache，降低 teacher forward 与 actor backward 的峰值显存冲突。

## 复现原则

- 模型间对比统一使用固定的 100 Bridge + 100 Comparison 验证集。
- 主要指标由 JSONL 轨迹离线严格复算，不以训练内 reward 或 LLM judge 代替。
- 所有 checkpoint 选择均同时参考答案、策略、召回和格式，避免单指标挑选。
- 机器编号只用于历史脚本名，不影响代码逻辑；路径和运行参数均可通过环境变量覆盖。

入口、命令和完整结果见 [`README.md`](README.md)，各实验脚本说明见 [`recipe/README.md`](recipe/README.md)。
