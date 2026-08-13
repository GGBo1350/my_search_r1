# 实验日志（Search-R1 V3）

> 按时间倒序记录各阶段实验：配置、结果、结论。配套总览见 [README.md](README.md)。

## 2026-08-13 ｜ 全 7 投影 LoRA 专项 Teacher 定稿

**统一口径**：Bridge 与 Comparison 专项 teacher 均从同一个 Qwen3-4B Base 独立初始化，LoRA 覆盖 `q/k/v/o_proj` 与 `gate/up/down_proj`，rank 32、alpha 64。固定 200 条验证集（100 Bridge + 100 Comparison），greedy 单次 rollout，关闭 LLM judge，并离线严格复算 Exact/F1、Strategy、Recall、Format 与工具调用数。

**Bridge teacher checkpoint 扫描**

| checkpoint | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | 0.590 | 0.735 | 0.709 | 0.475 | 0.858 | 0.945 | 1.98 |
| s50 | 0.575 | 0.750 | 0.711 | 0.545 | **0.885** | **0.960** | 2.04 |
| **s75** | **0.625** | **0.780** | **0.745** | 0.440 | 0.877 | 0.955 | 2.06 |

| checkpoint | 题型 | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | Bridge | 0.500 | 0.650 | 0.626 | 0.730 | 0.775 | 0.940 | 2.00 |
| s50 | Bridge | 0.490 | 0.670 | 0.639 | **0.850** | **0.820** | **0.950** | 2.08 |
| **s75** | Bridge | **0.540** | **0.700** | **0.663** | 0.810 | 0.810 | 0.920 | 2.14 |
| s25 | Comparison | 0.680 | 0.820 | 0.792 | 0.220 | 0.940 | 0.950 | 1.96 |
| s50 | Comparison | 0.660 | 0.830 | 0.783 | 0.240 | **0.950** | 0.970 | 1.99 |
| **s75** | Comparison | **0.710** | **0.860** | **0.827** | 0.070 | 0.945 | **0.990** | 1.98 |

Bridge 路由选择 `global_step_75`：它在目标题型上取得最高 Exact、F1≥0.5 与 Mean F1，同时保持 0.810 的串行策略正确率。`global_step_50` 的串行策略与召回更高，可作为强调纯策略轨迹时的备选，但不是当前默认 teacher。

**Comparison teacher `global_step_25`**

| 题型 | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.565 | 0.730 | 0.696 | 0.790 | 0.830 | 0.955 | 1.99 |
| Bridge | 0.430 | 0.600 | 0.568 | 0.600 | 0.725 | 0.920 | 1.96 |
| **Comparison** | **0.700** | **0.860** | **0.825** | **0.980** | **0.935** | **0.990** | 2.02 |

**当前 Teacher 能力矩阵**

| 模型 | Bridge Exact | Bridge Strategy | Compare Exact | Compare Strategy |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B Base | 0.440 | 0.630 | 0.680 | 0.180 |
| GRPO s100 | 0.510 | **0.870** | 0.670 | 0.960 |
| **Bridge teacher s75** | **0.540** | 0.810 | **0.710** | 0.070 |
| **Compare teacher s25** | 0.430 | 0.600 | 0.700 | **0.980** |

**结论**

1. 后续双 teacher OPD 固定使用 Bridge `global_step_75` 与 Comparison `global_step_25`，两者都为全 7 投影 LoRA。
2. Bridge teacher 提供读取第一跳真实结果后再生成第二跳查询的 `[1,1]` 轨迹；Comparison teacher 提供同一轮生成两个独立查询的 `[2]` 轨迹。
3. 两位 teacher 按样本顶层 `teacher_route=bridge|compare` 路由，不能跨题型混用。
4. 已完成的双 teacher OPD s100 是此前独立运行的历史结果；更新 teacher 基准不会追溯改变该 student 的训练权重或评测结果，使用当前 teacher 的 OPD student 需要重新训练后单独记录。

## 2026-08-11 ｜ 全 7 投影 LoRA GRPO 与 sample-token OPD 最终对照

**统一口径**：固定 200 条验证（100 Bridge + 100 Comparison），greedy 单次 rollout，严格 HotpotQA 归一化答案指标；同时复算 Strategy、Recall、Format 与工具调用数。两条路线都从 Qwen3-4B Base 出发，OPD student 不继承 GRPO 权重。

**Phase 1 全 7 投影 LoRA checkpoint 扫描**

| checkpoint | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | 0.550 | 0.715 | 0.671 | 0.860 | 0.853 | 0.925 |
| s50 | 0.550 | 0.715 | 0.679 | 0.865 | 0.880 | 0.935 |
| s75 | 0.580 | 0.755 | 0.719 | 0.900 | 0.860 | **0.960** |
| **s100** | **0.590** | **0.755** | **0.720** | **0.915** | 0.877 | 0.955 |
| s125 | 0.570 | 0.740 | 0.706 | 0.895 | 0.870 | 0.950 |

`global_step_100` 综合最佳；s125 答案和策略同时回落，因此在固定集上早停于 s100。按题型拆解：Bridge Exact/Strategy 为 `0.510/0.870`，Comparison 为 `0.670/0.960`。

**两条路线最终对照**

| 模型 | Exact | Mean F1 | Strategy | Recall | Format |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Base | 0.560 | 0.687 | 0.405 | 0.840 | 0.930 |
| Exact-only 50-step | 0.535 | 0.657 | 0.235 | 0.753 | 0.910 |
| **GRPO s100** | **0.590** | **0.720** | **0.915** | **0.878** | 0.955 |
| Dual-teacher OPD s100 | 0.545 | 0.664 | 0.855 | 0.848 | 0.915 |
| Sample-token OPD s25 | 0.585 | 0.711 | 0.780 | 0.863 | **0.965** |
| Sample-token OPD s100 | 0.580 | 0.704 | 0.865 | 0.873 | 0.945 |

**结论**

1. 混合任务/过程奖励的 GRPO s100 在答案和策略上综合最佳；纯 Exact 0/1 信号反而同时损伤答案与搜索拓扑。
2. sample-token OPD s25 偏答案，继续训练到 s100 后偏策略：Strategy `0.780→0.865`，Exact `0.585→0.580`。
3. OPD 已在不混入任务 reward 的情况下学到明显策略信号，但当前仍未超过 GRPO s100；因此二者是训练范式对照，而非阶段式升级关系。

## 2026-08-10 ｜ 双 Teacher OPD 单卡端到端 Smoke Test

**目标**：在 931 的单张 RTX PRO 6000 Blackwell 上验证完整链路，而不修改 bridge/compare 原始 checkpoint：student rollout、bridge s75 teacher、compare s25 teacher 三个 SGLang 实例共置；按样本顶层 `teacher_route` 选择 teacher；执行真实搜索轨迹、top-k teacher logprob、forward KL 和一次 student 更新。

**配置**

- 数据：2 条平衡 smoke 样本（bridge 1 / compare 1），由 routed 训练集派生，不覆盖原 Parquet。
- Teacher：bridge `global_step_75`、compare `global_step_25` 导出的静态 LoRA adapter；共同复用 Qwen3-4B base。
- 蒸馏：`forward_kl_topk`、top-k 32、系数 1.0、`use_policy_gradient=False`、`use_task_rewards=False`。
- 单卡共置：student rollout 显存比例 0.25，两位 teacher 各 0.14；FlashInfer；最大模型长度 4096。
- 训练：batch 2、mini batch 2、micro batch 1、每题 1 条轨迹、1 step。

**结果**

| 指标 | 数值 |
| --- | ---: |
| distillation loss | 0.011327 |
| teacher top-k probability mass | 0.999997 |
| student mass on teacher top-k | 0.999759 |
| teacher/student top-k overlap | 0.981742 |
| grad norm | 0.336377 |
| search execution success rate | 1.000 |
| retrieval recall | 1.000 |
| mean tool calls | 2.000 |
| step wall time | 42.78 s |

**结论**

1. 端到端 1 step 成功，蒸馏 loss、概率质量和梯度均为有限值，无 NaN；双 teacher 静态 LoRA、样本路由、真实工具调用和 student 更新链路可运行。
2. actor 最终 loss 与 distillation loss 一致（约 `0.011327`）；任务 reward 虽作为观测指标计算，但没有混入本次 actor 优化。
3. 训练结束后 GPU 显存已释放为 0 MiB。
4. verl 在 `trainer.save_freq > 0` 时会强制保存最后一步，即使步数不是保存频率的整数倍。本次因此生成了约 8.5 GB 的 smoke `global_step_1`；后续纯 smoke 使用 `SAVE_FREQ=-1`。该 smoke checkpoint 已在用户确认后于同日删除。

**存储清理（用户确认）**：已删除 smoke `qwen3_4b_mopd_smoke_20260810_1616/global_step_1` 和非最优 compare teacher `global_step_20`，合计约 17 GB；compare 最优 `global_step_25` 保留且复核完整。

## 2026-08-09 ｜ Ablation：纯 exact 答案奖励 + 无策略 prompt（50 步）

**背景**：验证"去掉策略引导和策略/检索奖励、只用严格答案奖励"的效果（907 机对照实验）。

**配置**

- 奖励 `recipe/core/my_reward_exact_only.py`（旧实验文件名 `my_reward_answer_only.py`）：答案仅严格 exact（对=1，不完全一样=0，无 F1 奖励、无 LLM 判定）；保留超次调用惩罚（整条轨迹 >4 次起每个扣 0.05）和“至少一次有效检索才有答案分”门控；retrieval/strategy/query/format 权重全 0。
- Prompt `recipe/data/data_preprocess_no_strategy.py`：删掉策略要求（比较题并行双查、桥接题串行、3 次搜索上限）和并行查询说明，只保留“可用 search 工具查询”的基本协议 + grounding + 答案格式。
- 数据：`data/hotpotqa_v3_no_strategy/`（1600 训练 / 200 验证，system 提示策略关键词 0 残留）。
- 训练：50 步，bs16×n8、mini 4、max_response 4096，其余同 Phase 1。

**结果：严格口径答案指标（无大模型）**

| 模型 | greedy exact@1 | greedy F1≥0.5 | 采样 exact@1 | pass@5 exact | pass@5 F1≥0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.560 | 0.725 | 0.560 | 0.670 | 0.835 |
| 50step（Phase 1） | 0.590 | 0.740 | 0.575 | 0.670 | 0.835 |
| 100step（Phase 1） | 0.580 | 0.745 | 0.585 | 0.650 | 0.840 |
| ansonly50 | 0.535 | 0.690 | 0.530 | 0.620 | 0.785 |

**行为拆解（greedy）**

| 模型 | bridge exact | compare exact | strategy_correct | 平均调用数 |
| --- | ---: | ---: | ---: | ---: |
| base | 0.440 | 0.680 | 0.405 | 1.88 |
| 50step（Phase 1） | 0.500 | 0.680 | 0.845 | 2.04 |
| ansonly50 | 0.400 | 0.670 | 0.235 | 1.68 |

**结论（greedy）**

1. 纯 exact 答案奖励为负贡献：全部指标低于 base（greedy exact 0.535 vs 0.560），50 步内学不动 0/1 稀疏信号，还破坏了 base 已有能力。
2. 策略完全丢失：compare 并行率 base 0.18 → 50step 0.93 → ansonly 0.000；bridge 串行多查退化（调用数 1.88→1.54）。
3. 损失集中在 bridge（0.400 < base 0.440 < 50step 0.500）；compare 三者持平（0.67-0.68，base 本就能答好）。
4. 反向验证：Phase 1 的策略 prompt + 策略/检索/答案混合奖励是必要的，去掉后 50 步连 base 都不如。
## 2026-08-09 ｜ Phase 1 收尾：base vs 50step vs 100step（50→100 续跑）

**背景**：第一阶段单模型 RL 已完成 50 步（checkpoint `qwen3_4b_sglang_16x8_50step_llmjudge_20260808_173324`），本次从 global_step_50 续跑 50 步到 100。

**训练配置（与前半程一致，除奖励权重）**

- Qwen3-4B + LoRA（rank 32 / alpha 64），GRPO，bs16 × n8 = 128 轨迹/步，mini batch 4
- `MAX_RESPONSE_LENGTH=4096`、prompt 2048、max_model_len 12288
- SGLang + flashinfer、hybrid 检索 + reranker、LLM 语义判定开启、分块熵
- ⚠️ 奖励权重变化：前 50 步 exact=0.10/query=0.05；续跑的 50 步 exact=0.15/query=0.0（答案 F1 0.35 不变）

**评测口径**：固定 200 条验证；greedy + 采样 ×5；两套打分（严格匹配 / LLM 语义判定）。

**结果：答案指标（严格匹配）**

| 模型 | greedy exact@1 | greedy F1≥0.5 | 采样 exact@1 | pass@5 exact | pass@5 F1≥0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.560 | 0.725 | 0.560 | 0.670 | 0.835 |
| 50step | 0.590 | 0.740 | 0.575 | 0.670 | 0.835 |
| 100step（50→100） | 0.580 | 0.745 | 0.585 | 0.650 | 0.840 |

**结果：答案指标（LLM 判定口径）**

| 模型 | greedy pass@1 | 采样 pass@1 | pass@5 exact | pass@5 F1≥0.5 |
| --- | ---: | ---: | ---: | ---: |
| base | 0.755 | 0.750 | 0.860 | 0.865 |
| 50step | 0.775 | 0.770 | 0.855 | 0.860 |
| 100step（50→100） | 0.770 | 0.760 | 0.865 | 0.870 |

**结果：策略指标（LLM 判定口径）**

| 模型 | greedy 遵循 | 采样遵循 | pass@5 遵循 | 策略+答对 | 并行答对 | 串行答对 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.405 | 0.467 | 0.725 | 0.610 | 0.900 | 0.820 |
| 50step | 0.845 | 0.815 | 0.960 | 0.835 | 0.910 | 0.800 |
| 100step（50→100） | 0.830 | 0.847 | 0.980 | 0.840 | 0.920 | 0.810 |

**结论（greedy）**

1. 续跑 50 步对答案基本无提升：严格口径 greedy exact 0.590→0.580、pass@5 exact 0.670→0.650（持平/微降）；F1≥0.5 微升（0.740→0.745）。50 步附近已接近答案收敛。
2. 续跑的主要收益在策略稳定性：pass@5 策略遵循 0.960→0.980、采样遵循 0.815→0.847、并行答对 0.910→0.920。
3. 该 100 步模型前后 50 步奖励权重不一致（exact 0.10→0.15），答案未涨部分受此影响；若要验证权重调整，需从 base 重跑 50 步。
4. Phase 1 定位结论：单模型 RL 在 50 步内把策略遵循率从 0.41 拉到 0.85、严格答案从 0.56 提到 0.59；答案天花板受限于"策略对但表达不匹配"的样本 → Part 2（bridge/compare 专精 teacher + OPD）的目标。
