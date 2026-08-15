# 实验日志（Search-R1 V3）

> 按时间倒序记录各阶段实验：配置、结果、结论。配套总览见 [README.md](README.md)。

## 2026-08-15 ｜ 双 Teacher 路由 MOPD Sample-K3 完整训练与评测

**配置**：运行 `20260814_200626` 使用单张 96 GB GPU、独立 Qwen3-4B Base student、全 7 投影 LoRA（rank 32、alpha 64），在 1200 Bridge + 400 Comparison 混合数据上训练 100 step。每条样本按 `teacher_route` 选择 Bridge s75 或 Compare s25 teacher，关闭 task reward 与 policy gradient，使用 sample-token `k=3` 蒸馏。训练 batch 为 16、mini batch 为 4、micro batch 为 1、每题 1 条 rollout；teacher 最大并发为 2，并在 teacher 打分结束后、actor backward 前释放静态 teacher KV cache。

训练总耗时 2 h 32 min 56 s，平均约 91.75 s/step；保存 s25/s50/s75/s100 四个 checkpoint。统一评测仍为固定 200 题、greedy 单次 rollout、关闭 LLM judge。

| checkpoint | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | 0.580 | 0.730 | 0.692 | 0.810 | 0.853 | 0.930 | 2.04 |
| **s50** | **0.585** | **0.750** | **0.712** | 0.830 | **0.873** | **0.950** | 2.11 |
| s75 | 0.575 | 0.725 | 0.690 | 0.875 | 0.860 | **0.950** | 2.00 |
| **s100** | 0.575 | 0.725 | 0.701 | **0.890** | 0.863 | 0.945 | 2.05 |

| checkpoint | 题型 | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | Bridge | 0.480 | 0.650 | 0.607 | 0.780 | 0.780 | 0.910 | 2.10 |
| s50 | Bridge | **0.480** | **0.650** | **0.613** | 0.760 | **0.805** | 0.900 | 2.17 |
| s75 | Bridge | 0.470 | 0.630 | 0.596 | **0.830** | 0.785 | **0.920** | 1.98 |
| s100 | Bridge | 0.450 | 0.600 | 0.583 | **0.830** | 0.800 | **0.920** | 2.07 |
| s25 | Comparison | 0.680 | 0.810 | 0.777 | 0.840 | 0.925 | 0.950 | 1.98 |
| s50 | Comparison | 0.690 | **0.850** | 0.810 | 0.900 | **0.940** | **1.000** | 2.04 |
| s75 | Comparison | 0.680 | 0.820 | 0.784 | 0.920 | 0.935 | 0.980 | 2.01 |
| s100 | Comparison | **0.700** | **0.850** | **0.819** | **0.950** | 0.925 | 0.970 | 2.02 |

**Checkpoint 选择**

1. 按 Strict Exact 优先、Mean F1 次优的答案口径，MOPD 主分数取 s50：`0.585 Exact / 0.712 Mean F1 / 0.830 Strategy`。
2. 按搜索行为选择，MOPD 策略点取 s100：`0.575 Exact / 0.701 Mean F1 / 0.890 Strategy`。不能用 s100 覆盖 s50 的答案峰值，也不能只报 s50 忽略后续策略收益。
3. 相比串行 OPD s100，MOPD s100 的整体 Strategy 高 `6.5 pp`，Bridge Strategy 高 `14 pp`，说明混合题型路由缓解了 Bridge→Comparison 顺序蒸馏的最后教师偏置。
4. 与 GRPO s100 相比，MOPD s50 的 Exact 只低 `0.5 pp`、Mean F1 低约 `0.84 pp`；GRPO 的 Strategy 仍高 `8.5 pp`。

**错误状态摘要**：下列四个故障列可重叠，不能相加为 Strict 错误总数。

| checkpoint | 题型 | Strict 错误 | 拓扑错误 | gold-title 召回不全 | 完整召回但答案错 | 格式错误 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| s50 | Bridge | 52 | 17 | 26 | 26 | 10 |
| s50 | Comparison | 31 | 5 | 5 | 26 | 0 |
| s100 | Bridge | 55 | 14 | 29 | 26 | 8 |
| s100 | Comparison | 30 | 3 | 7 | 23 | 3 |

MOPD 从 s50 到 s100 减少了拓扑错误，但答案分略回落；Bridge 剩余瓶颈集中在第二跳检索/query rewrite 与完整证据后的答案抽取，Comparison 则主要是完整召回后的比较逻辑和严格答案边界。完整模型矩阵与逐题案例见 [`docs/final_model_comparison.md`](docs/final_model_comparison.md)。

## 2026-08-14 ｜ 双 Teacher Forward-KL Top-32 OOM

运行 `20260814_123655` 使用双卡路由、全 7 投影 LoRA student、Bridge s75 / Comparison s25 teacher、`forward_kl_topk/topk=32`。训练在 actor backward 阶段因 CUDA OOM 终止：GPU 0 当时仅余约 268 MiB，反向传播仍需申请 1.46 GiB，因此未产出可用 checkpoint。后续默认入口改为 sample-token `k=3`（`loss_mode=k3`、`topk=null`），仅请求 student 实际采样 token 的 teacher log-prob，继续保持 teacher 路由、数据、student 初始化和保存策略不变。K3 重跑 `20260814_132111` 成功完成前 4 个 actor update，但在 FSDP 导出 LoRA 并同步 rollout 权重时再次 OOM：仅需追加 24 MiB，而 GPU 0 只剩 17.94 MiB。日志确认同卡有两个 AgentLoop 检索进程分别占约 6.88/5.33 GiB，因此后续入口将工具 worker 从 4 个减为 2 个，并按 `[0,1]` 每卡各放一个；损失、batch 和 teacher 配置保持不变。

## 2026-08-14 ｜ Bridge→Comparison 串行 Sample-K3 OPD

**配置**：OPD student 从独立的 Qwen3-4B Base 初始化，使用全 7 投影 LoRA（rank 32、alpha 64）。先在 1200 条 Bridge 数据上接受 Bridge teacher 的 sample-token `k=3` 蒸馏 75 step，再保留 student 模型、优化器、scheduler 与 RNG 状态，在 400 条 Comparison 数据上接受 Comparison teacher 蒸馏 25 step。GRPO/专项 checkpoint 只提供 teacher token 分布，不作为 student 初始化。训练运行 ID 为 `20260813_220646`。

**统一评测口径**：固定 200 条验证集（100 Bridge + 100 Comparison），greedy 单次 rollout，关闭 LLM judge，使用既有 checkpoint 固定集入口与相同的 `2048/4096/3072` 长度、`3/4` 工具轮数设置。`s25/s50/s75` 来自 Bridge 阶段，`s100` 是继续完成 Comparison 阶段后的最终 student。

| checkpoint | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | 0.560 | 0.725 | 0.686 | 0.525 | 0.858 | 0.900 | 2.02 |
| **s50** | **0.565** | 0.725 | 0.690 | 0.530 | **0.870** | **0.935** | 2.09 |
| s75 | 0.560 | 0.730 | 0.692 | 0.480 | **0.870** | 0.930 | 2.12 |
| **s100** | 0.560 | **0.735** | **0.698** | **0.825** | 0.833 | 0.925 | 1.97 |

| checkpoint | 题型 | Exact | F1≥0.5 | Mean F1 | Strategy | Recall | Format | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s25 | Bridge | **0.460** | **0.640** | **0.597** | **0.820** | 0.785 | 0.890 | 2.06 |
| s50 | Bridge | 0.430 | 0.620 | 0.581 | 0.810 | **0.795** | **0.910** | 2.16 |
| s75 | Bridge | 0.440 | 0.630 | 0.593 | 0.810 | 0.790 | 0.890 | 2.20 |
| s100 | Bridge | 0.450 | 0.630 | 0.597 | 0.690 | 0.730 | 0.870 | 1.88 |
| s25 | Comparison | 0.660 | 0.810 | 0.775 | 0.230 | 0.930 | 0.910 | 1.97 |
| **s50** | Comparison | **0.700** | 0.830 | 0.799 | 0.250 | 0.945 | 0.960 | 2.02 |
| s75 | Comparison | 0.680 | 0.830 | 0.790 | 0.150 | **0.950** | 0.970 | 2.05 |
| **s100** | Comparison | 0.670 | **0.840** | **0.800** | **0.960** | 0.935 | **0.980** | 2.06 |

**结论**

1. Bridge 阶段继续从 s25 训练到 s75 没有带来单调收益：Bridge Exact 从 `0.460` 变为 `0.440`，Strategy 基本持平于 `0.810–0.820`。若只选 Bridge 专项状态，s25 的答案与策略组合最好。
2. Comparison 阶段产生了明确的目标行为迁移：s75→s100 时 Comparison Strategy 从 `0.150` 跃升到 `0.960`，同时 Comparison Exact 仅变化 `-1 pp`、Mean F1 提升约 `1 pp`。
3. 该迁移伴随顺序干扰：Bridge Strategy 从 `0.810` 降至 `0.690`，Bridge Recall 从 `0.790` 降至 `0.730`。因此简单的 Bridge→Comparison 串行蒸馏没有无损合并两位 teacher 的策略。
4. 串行结果同时保留两个点：s50 是答案最佳（Exact `0.565`），但尚未学会 Comparison 并行策略；s100 是策略最佳，具有最高整体 Mean F1 `0.698`、F1≥0.5 `0.735` 与 Strategy `0.825`。
5. 相比 Qwen3-4B Base，s100 的 Exact 持平于 `0.560`，Strategy 提升 `42 pp`；相比 GRPO s100，整体 Exact/Strategy 仍低 `3/9 pp`。分题型看，s100 已匹配 GRPO 的 Comparison `0.670/0.960`，差距主要来自 Bridge（`0.450/0.690` vs. `0.510/0.870`）。这支持后续采用题型混合、Bridge replay 或路由双 teacher，而不是继续单向串行训练。

**产物**：评测轨迹位于 `/root/autodl-tmp/rollouts/serial_k3_ckpt_eval_20260813_220646_20260814_112541/`，汇总报告位于 `/root/autodl-tmp/eval_reports/serial_k3_ckpt_eval_20260813_220646_20260814_112541/`。

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
