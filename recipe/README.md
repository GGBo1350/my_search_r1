# Search-R1 实验入口

`recipe/` 保存本项目相对 veRL 上游新增的完整实验链路。Phase 1 与 Phase 2 是从同一 Qwen3-4B Base 出发的两条对照路线，而不是先后继承权重的连续训练阶段。

所有 Qwen3-4B LoRA 训练、蒸馏和评测入口统一使用 rank 32、alpha 64 的全 7 投影配置：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。公共启动器会拒绝更窄的 target 列表；历史两投影 checkpoint 不能通过这些入口续训或评测。

## 目录

- `core/`：搜索工具、工具调用解析器、混合奖励与工具配置。
- `data/`：HotpotQA 下载、双证据样本清洗、Parquet 构建、答案变体与混合索引。
- `train_lora/`：LoRA + GRPO 公共训练入口。
- `eval/`：固定 200 条 greedy、pass@k、逐题指标复算与汇总。
- `phase1/`：使用任务奖励和过程奖励训练通用搜索策略。
- `phase2/`：静态 LoRA teacher、双 teacher OPD 与 sample-token OPD。

## 数据与检索

公开仓库不包含 HotpotQA 原始文件、生成的 Parquet、API 审核缓存、SQLite/FAISS 索引、模型权重或 rollout。

```bash
python recipe/data/download_hotpotqa.py --output-dir data/hotpot_qa_distractor

# 只统计候选量，不调用 API
python recipe/data/deepseek_clean_hard_data.py --dry-run

# 完整参数见 --help；密钥仅从环境变量读取
export DEEPSEEK_API_KEY=your_key
python recipe/data/deepseek_clean_hard_data.py \
  --output-dir data/processed/search_r1_clean

# 可选：基于 gold evidence 生成同一答案的表面变体；建议先 --max-rows 20
python recipe/data/enrich_answer_variants.py \
  --train-file data/processed/search_r1_clean/train.parquet \
  --validation-file data/processed/search_r1_clean/validation.parquet \
  --output-dir data/processed/search_r1_clean_variants \
  --max-rows 20

python recipe/data/build_hybrid_index.py \
  --train data/processed/search_r1_clean/train.parquet \
  --validation data/processed/search_r1_clean/validation.parquet \
  --output-dir data/search_index
```

`deepseek_clean_hard_data.py` 只审核 HotpotQA hard 候选，并要求答案真正依赖两个 gold 文档：

- Bridge：第二跳目标必须由第一跳证据发现，期望工具拓扑为 `[1,1]`。
- Comparison：两个查询目标从题面即可独立识别，期望工具拓扑为 `[2]`。
- 排除题面可直接回答、任一单文档足够回答、无需恰好两次 focused search，以及题型/拓扑不一致的样本。

API 审核逐条追加到 JSONL 缓存，可中断续跑；默认目标为 1200 Bridge + 400 Comparison 训练样本，以及各 100 条的验证样本。

`enrich_answer_variants.py` 保留原 gold，并只依据 gold evidence 构造同一实体的全名、昵称、简称等表面形式；最多 8 个，写入 `reward_model.ground_truth` 和 `extra_info.answer_variants`。完整数据构建与奖励说明见 [`docs/search_r1_reward_and_data.md`](../docs/search_r1_reward_and_data.md)。

## Phase 1：GRPO 路线

Phase 1 用答案、召回、格式和搜索拓扑奖励直接训练一个通用学生模型：Bridge 学习串行两跳，Comparison 学习同轮并行双查。

主要入口：

- `phase1/run_sglang_train_4b_full_lora_125step_939.sh`：Qwen3-4B、全 7 类线性投影 LoRA，训练 125 step，每 25 step 保存。
- `phase1/run_eval_full_lora_checkpoints_805.sh`：对五个 checkpoint 串行执行固定 200 条 greedy 评测。
- `phase1/analyze_full_lora_checkpoints.py`：复算整体、Bridge 和 Comparison 指标。
- `data/data_preprocess_no_strategy.py` + `core/my_reward_exact_only.py` + `phase1/run_exact_answer_only_50step.sh`：严格 Exact Answer-only 消融。

固定集结果的最佳 checkpoint 为 `global_step_100`：Exact 0.590、Mean F1 0.720、Strategy 0.915；继续至 step 125 后出现回落。

复合奖励权重为 Answer F1/Exact `0.35/0.15`、gold-title Recall `0.30`、Strategy `0.15`、Format `0.05`；答案分要求至少一次成功搜索。`ANSWER_LLM_JUDGE=1` 可在部分匹配时调用 LLM 判断语义等价，但固定集比较统一设置为 0。严格 Exact-only 对照则只保留搜索门控后的 `EM` 与超次调用惩罚。

## Phase 2：OPD 路线

Phase 2 从另一份 Base student 出发，用 student 自身 on-policy 轨迹上的 teacher token 分布进行蒸馏。GRPO checkpoint 或专项 checkpoint只作为静态 teacher，不作为 student 初始化权重。

实现了两组对照：

1. Bridge `global_step_75` + Comparison `global_step_25` 双 teacher，按题型路由并计算 Top-k forward KL。
2. Phase 1 `global_step_100` 单 teacher，使用 sample-token `k=3` OPD；student 与 teacher adapter 都覆盖 `q/k/v/o_proj` 和 `gate/up/down_proj`。
3. 双 teacher Reverse Top-k：在 teacher Top-k token 与一个剩余词表 `other` 桶组成的共享支持集上，直接优化 `KL(student || teacher)`。

当前全 7 投影专项 teacher 的固定集路由内结果：Bridge s75 为 Exact `0.540` / Strategy `0.810`，Comparison s25 为 Exact `0.700` / Strategy `0.980`。默认仍按 `bridge→s75`、`compare→s25` 路由；已完成的旧 OPD student 结果不因 teacher 基准更新而改变。

主要入口：

- `phase2/add_teacher_route.py`：为旧 Parquet 非覆盖式增加 `teacher_route`。
- `phase2/verify_opd_routes.py`：逐条验证路由与题型一致性。
- `phase2/extract_teacher_lora.py`：从 veRL actor checkpoint 导出 PEFT adapter。
- `phase2/verify_teacher_adapters.py`：校验 adapter 完整性、rank 和目标模块。
- `phase2/prepare_full_lora_teacher_pair_931.sh`：从 931 保留的 Bridge s75 / Comparison s25 checkpoint 导出并校验全 7 投影 teacher adapter；完整产物可安全复用。
- `phase2/run_mopd_bridge_compare_2gpu_931.sh`：双 teacher 路由的 Forward-KL Top-32 OPD。
- `phase2/run_single_teacher_sample_k3_2gpu_931.sh`：串行消融内部复用的双卡单 teacher Sample-K3 阶段入口。
- `phase2/run_serial_bridge_then_compare_sample_k3_2gpu_931.sh`：同一 Base student 先接受 Bridge teacher 75 step、再接受 Comparison teacher 25 step 的 Sample-K3 串行消融；Bridge 保留 s25/s50/s75，Comparison 只保留最终 s100。
- `phase2/run_mopd_bridge_compare_reverse_top32_2gpu.sh`：双 teacher Reverse Top-32 OPD（teacher Top-k + other 桶）。
- `phase2/run_opd_phase1_s100_sample_token_1gpu_805.sh`：单 teacher sample-token OPD。
- `phase2/run_eval_mopd_student_931.sh` 与 `phase2/run_eval_opd_phase1_s100_sample_token_checkpoints_805.sh`：固定集评测。

## 统一评测口径

- 200 条固定验证集：100 Bridge + 100 Comparison。
- greedy 单次 rollout；答案用 HotpotQA 归一化后严格复算 Exact/F1。
- Strategy 检查 `[1,1]` 或 `[2]`，同时报告 Recall、Format 和平均工具调用数。
- 训练内 reward 只用于优化和诊断，模型间结论以离线固定集为准。

完整结果、运行命令和实现说明见仓库根目录 [`README.md`](../README.md)。
