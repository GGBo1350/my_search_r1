# Search-R1：Base、GRPO、双教师、串行 OPD 与 MOPD 最终对比

> 整理日期：2026-08-15
> 评测口径：固定 200 题（Bridge 100 + Comparison 100）、greedy 单次 rollout、统一答案与过程指标
> 模型范围：Qwen3-4B Base、通用 GRPO s100、两位专项 teacher、Bridge→Comparison 串行 OPD s50/s100、双教师路由 MOPD s50/s100

## 1. 最终结论

这组实验得到的不是一个简单的“后一个模型全面超过前一个模型”，而是五种训练方式在答案、检索拓扑、证据召回和多策略保持上的清晰分工：

1. **Base 已有问答知识，但不会稳定组织搜索。** Overall Exact 为 56%，Comparison Exact 已有 68%，但整体 Strategy 只有 40.5%，Comparison 正确单轮并行只有 18%。
2. **GRPO 是综合表现最好的单 student checkpoint。** s100 达到 59% Exact、72.01% mean F1、91.5% Strategy；相对 Base 最大收益不是 Exact 的 +3 pp，而是 Strategy 的 +51 pp。
3. **路由双教师是能力参考上限，不是单一 student。** Bridge teacher s75 与 Compare teacher s25 按题型组合后达到 62% Exact、74.41% F1、89.5% Strategy。它说明专项策略轨迹具有互补性，但部署时需要保留两套 adapter/推理实例。
4. **串行 OPD 学会了最后一个教师，却遗忘了前一个教师。** 从 Bridge 切换到 Comparison 后，Compare Strategy 从 15% 升至 96%，Bridge Strategy 却从 81% 降至 69%，呈现明显的顺序干扰。
5. **MOPD 必须同时报告答案峰值和策略峰值。** 以 Strict Exact 为主选择标准时，MOPD s50 最好：58.5% Exact、71.17% mean F1、83% Strategy；继续到 s100 后 Strategy 升至 89%，但 Exact/F1 回落到 57.5%/70.09%。因此主分数采用 s50，行为保持采用 s100。
6. **串行 OPD 也存在同样的 checkpoint 权衡。** s50 的 Exact 最高，为 56.5%，但整体 Strategy 只有 53%，Comparison Strategy 仅 25%；s100 的 Exact 略降至 56%，Strategy 升至 82.5%，同时 Bridge Strategy 因教师切换降至 69%。
7. **MOPD s100 把 Bridge 错误从上游移到了下游。** 串行 OPD s100 有 23 个 Bridge 错误同时表现为“拓扑错误+召回不全”，MOPD s100 降到 14 个；MOPD s100 有 26 个错误已经“拓扑正确+完整召回”，说明搜索链更可靠，剩余瓶颈转向证据聚合和答案抽取。
8. **Comparison 已接近策略饱和。** GRPO、Compare teacher、串行 OPD s100、MOPD s100 的 Comparison Strategy 分别为 96%/98%/96%/95%；继续加大并行拓扑奖励的边际收益很小，后续更应优化比较推理与短答案边界。

## 2. 实验关系：不是阶段继承

```text
同一个 Qwen3-4B Base、同一搜索工具与固定评测集
│
├─ Base：不训练，直接评测
│
├─ GRPO generalist：任务奖励 + 过程奖励 → 通用 GRPO s100
│
├─ GRPO specialists
│  ├─ 独立训练 Bridge teacher s75
│  └─ 独立训练 Compare teacher s25
│
├─ Serial OPD student：从 Base 独立初始化
│  └─ Bridge teacher 75 step → 同一 student 切换 Compare teacher 25 step
│
└─ MOPD student：从 Base 独立初始化
   └─ Bridge/Compare 混合数据，按题型同时路由到两个 teacher，共 100 step
```

GRPO 与 OPD/MOPD 是两条并列训练范式：

- GRPO 用答案、证据召回、调用拓扑、查询质量和格式等标量奖励；
- OPD/MOPD 不使用任务奖励反传，让 student 在真实工具环境中生成 on-policy 轨迹，再由 teacher 对 student 实际访问的 token 前缀提供分布监督；
- 所有 student 都从 Qwen3-4B Base 独立初始化；GRPO checkpoint 在 OPD 中只提供 teacher token 分布，不是 student 初始化权重。

## 3. 训练配置与成本

| 模型 | 训练信号 | 调度 | 每 step prompt / rollout | 训练步数 | 平均秒/step | 约计训练时间 |
|---|---|---|---:|---:|---:|---:|
| Base | 无 | 无 | — | 0 | — | — |
| 通用 GRPO | 任务/过程 reward | Bridge/Compare 混合 | 16 / 128（`n=8`） | 100（最佳点） | 约 271.6 s | 约 7 h 33 min |
| Bridge + Compare teachers | 任务/过程 reward | 两个专家分别训练 | 16 / 128（`n=8`） | 75 + 25 | 288.3 / 253.5 s | 串行约 7 h 49 min |
| 串行 OPD | sample-token K=3 | Bridge 75 → Compare 25 | 16 / 16（`n=1`） | 100 | 加权约 95.24 s | 约 2 h 43 min |
| MOPD | sample-token K=3 | 两类样本约 3:1 混合路由 | 16 / 16（`n=1`） | 100 | 91.75 s | 2 h 32 min 56 s |

四组训练均使用全 7 投影 LoRA（`q/k/v/o_proj` 与 `gate/up/down_proj`，rank 32、alpha 64）。GRPO 每个 prompt 生成 8 条候选轨迹，而 OPD/MOPD 每个 prompt 只有 1 条 student 轨迹，因此不能把秒/step 直接解释为算法本身 3 倍更快；两者每 step 的生成工作量不同。

训练过程还有三个重要现象：

- 通用 GRPO 从 s100 继续到 s125 后，Exact 从 59% 回落到 57%、Strategy 从 91.5% 回落到 89.5%，因此 s100 是固定集早停结果。
- 串行 OPD 在 Bridge→Compare 切换时，distill loss 放大 8.94 倍、grad norm 放大 2.59 倍；Comparison loss 随后在 25 step 内下降 83.73%，说明模型快速适配新教师，同时覆盖已有行为。
- MOPD loss 从 0.04827 降至 0.01311，下降 72.84%。工程上将 teacher 最大并发降为 2，并在 teacher 完成打分后、Actor backward 前释放静态 teacher KV cache，使双教师与 student 在单卡稳定完成训练。

## 4. Checkpoint 选择与固定 200 题总体对比

本文明确区分两种选择标准：

- **答案最佳**：优先选择 Strict Exact 最高的 checkpoint；若 Exact 相同，再比较 mean F1。
- **策略最佳**：选择 Strategy 最高的 checkpoint，同时保留该点的答案指标，不用策略分替代答案分。

### 4.1 各训练族的答案峰值与策略峰值

| 训练族 | 答案最佳 checkpoint | Exact / Mean F1 / Strategy | 策略最佳 checkpoint | Exact / Mean F1 / Strategy |
|---|---|---:|---|---:|
| 通用 GRPO | s100 | **0.590 / 0.720 / 0.915** | s100 | **0.590 / 0.720 / 0.915** |
| Bridge teacher | **s75** | **0.540 / 0.663 / 0.810** | **s50** | 0.490 / 0.639 / **0.850** |
| Compare teacher | **s25** | **0.700 / 0.825 / 0.980** | **s25** | **0.700 / 0.825 / 0.980** |
| 串行 OPD | **s50** | **0.565 / 0.690 / 0.530** | **s100** | 0.560 / **0.698** / **0.825** |
| MOPD | **s50** | **0.585 / 0.712 / 0.830** | **s100** | 0.575 / 0.701 / **0.890** |

Bridge/Compare teacher 的数值均为各自路由题型上的成绩，不是两位教师在完整 200 题上的 Overall。Bridge teacher s75 更适合作为答案 teacher，s50 更适合作为纯策略轨迹 teacher；当前 MOPD 使用的仍是 s75，因为实验目标同时重视答案 token。

### 4.2 两位默认 Teacher 的完整交叉评测

| Teacher | 评测分组 | Exact | Mean F1 | Strategy | Recall | Format | Calls |
|---|---|---:|---:|---:|---:|---:|---:|
| Bridge teacher s75 | Overall | 0.625 | 0.745 | 0.440 | 0.878 | 0.955 | 2.06 |
| Bridge teacher s75 | **Bridge 目标路由** | **0.540** | 0.663 | **0.810** | 0.810 | 0.920 | 2.14 |
| Bridge teacher s75 | Comparison 交叉路由 | **0.710** | **0.827** | **0.070** | 0.945 | 0.990 | 1.98 |
| Compare teacher s25 | Overall | 0.565 | 0.696 | 0.790 | 0.830 | 0.955 | 1.99 |
| Compare teacher s25 | Bridge 交叉路由 | 0.430 | 0.568 | 0.600 | 0.725 | 0.920 | 1.96 |
| Compare teacher s25 | **Comparison 目标路由** | **0.700** | **0.825** | **0.980** | 0.935 | 0.990 | 2.02 |

Bridge teacher s75 在 Comparison 上有全表最高的 71% Exact，却只有 7% 正确 `[2]` 并行策略；它很可能依靠已有知识或串行路径答对。Compare teacher s25 在 Bridge 上也只有 43% Exact / 60% Strategy。两位 teacher 必须按题型路由，不能因为某个交叉题型 Exact 较高就跨路由使用。

### 4.3 总体结果

“路由双教师”表示 Bridge 题使用答案最佳的 Bridge teacher s75、Comparison 题使用 Compare teacher s25，是两个专家按题型组合的参考结果，不是一个 student checkpoint。

| 模型 | 定位 | Exact | Mean F1 | Strategy | Recall | Format | Calls | Exact∩Strategy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 基线 | 56.0% | 68.74% | 40.5% | 84.00% | 93.0% | 1.88 | 24.5% |
| **GRPO s100** | 答案/策略均最佳 | **59.0%** | **72.01%** | **91.5%** | **87.75%** | 95.5% | 2.07 | **56.0%** |
| 路由双教师 | 专家参考上限 | **62.0%** | **74.41%** | 89.5% | 87.25% | **95.5%** | 2.08 | **59.0%** |
| Serial OPD s50 | **Serial 答案最佳** | **56.5%** | 68.97% | 53.0% | 87.00% | 93.5% | 2.09 | 31.0% |
| Serial OPD s100 | Serial 策略最佳 | 56.0% | **69.83%** | **82.5%** | 83.25% | 92.5% | 1.97 | **51.0%** |
| MOPD s50 | **MOPD 答案最佳** | **58.5%** | **71.17%** | 83.0% | **87.25%** | **95.0%** | 2.11 | 52.5% |
| MOPD s100 | MOPD 策略最佳 | 57.5% | 70.09% | **89.0%** | 86.25% | 94.5% | 2.05 | **55.0%** |

从答案分数看，MOPD s50 只比 GRPO s100 低 0.5 pp Exact、0.84 pp mean F1，是 OPD/MOPD 中最接近 GRPO 的 checkpoint；从行为看，MOPD s100 的 Strategy 比 s50 高 6 pp，`Exact∩Strategy` 也由 52.5% 升到 55%。两者都应保留，不能用 s100 取代 s50 的答案峰值。

路由双教师比 MOPD s50 的 Exact 高 3.5 pp、F1 高 3.24 pp；与 MOPD s100 比，Strategy 只高 0.5 pp、Recall 只高 1 pp。说明 student 已较完整迁移行为，答案生成仍有差距。

## 5. 按题型对比

### 5.1 Bridge

| 模型 | checkpoint 定位 | Exact | Mean F1 | Strategy `[1,1]` | Recall | Format | Calls | Exact∩Strategy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 基线 | 44% | 59.04% | 63% | 74.5% | 94% | 1.86 | 34% |
| **GRPO s100** | 答案/策略均最佳 | 51% | 65.34% | **87%** | **82.5%** | 94% | 2.09 | 47% |
| Bridge teacher s75 | **Teacher 答案最佳** | **54%** | **66.33%** | 81% | 81% | 92% | 2.14 | **48%** |
| Bridge teacher s50 | Teacher 策略最佳 | 49% | 63.87% | **85%** | **82%** | **95%** | 2.08 | — |
| Serial OPD s50 | Serial 总体 Exact 最佳 | 43% | 58.06% | 81% | 79.5% | 91% | 2.16 | 42% |
| Serial OPD s100 | Serial 策略终点 | 45% | 59.66% | 69% | 73% | 87% | 1.88 | 37% |
| MOPD s50 | **MOPD 答案最佳** | **48%** | **61.31%** | 76% | **80.5%** | 90% | 2.17 | 41% |
| MOPD s100 | MOPD 策略最佳 | 45% | 58.26% | **83%** | 80% | 92% | 2.07 | **42%** |

Bridge 是最能区分几种训练方法的题型：

- GRPO 的直接过程奖励把 Strategy 推到最高的 87%，也是综合最好的单 student Bridge checkpoint。
- Bridge teacher 的 Exact/F1 最高，但 Strategy 低于通用 GRPO，说明专用教师的选择更偏答案质量。
- 串行 OPD 从 s50 到 s100 完成 Comparison 阶段后，Bridge Strategy 从 81% 降到 69%、Recall 从 79.5% 降到 73%，出现第二跳消失或错误并行。
- MOPD s50 的 Bridge Exact 最高，为 48%；继续到 s100 后 Bridge Strategy 从 76% 升到 83%，但 Exact 从 48% 回落到 45%，体现答案与策略权衡。

### 5.2 Comparison

| 模型 | checkpoint 定位 | Exact | Mean F1 | Strategy `[2]` | Recall | Format | Calls | Exact∩Strategy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 基线 | 68% | 78.44% | 18% | 93.5% | 92% | 1.89 | 15% |
| GRPO s100 | 答案/策略均最佳 | 67% | 78.68% | 96% | 93.0% | 97% | 2.04 | 65% |
| **Compare teacher s25** | **Teacher 答案/策略最佳** | **70%** | **82.48%** | **98%** | **93.5%** | **99%** | 2.02 | **70%** |
| Serial OPD s50 | Serial 总体 Exact 最佳 | **70%** | 79.88% | 25% | **94.5%** | 96% | 2.02 | 20% |
| Serial OPD s100 | Serial 策略终点 | 67% | 80.01% | **96%** | 93.5% | 98% | 2.06 | **65%** |
| MOPD s50 | MOPD 答案最佳 | 69% | 81.03% | 90% | **94.0%** | **100%** | 2.04 | 64% |
| MOPD s100 | MOPD 策略最佳 | **70%** | **81.93%** | **95%** | 92.5% | 97% | 2.02 | **68%** |

Base 的 Comparison Exact 已有 68%，但真正同轮双查只有 18%。Serial s50 也展示同一问题：Comparison Exact 已达 70%，Strategy 只有 25%。当训练推进到策略峰值后，GRPO、Compare teacher、Serial s100、MOPD s100 的 Strategy 都在 95% 以上，而 Exact 仍为 67%–70%，说明并行行为已经学会，新的瓶颈是两侧证据的属性比较、实体消歧和最终短答案规范化。

## 6. 错误归因方法

对每个 Strict Exact 错误，不直接凭最终答案猜原因，而是先检查两个可复算过程维度：

- **Strategy**：Bridge 是否严格为依赖式 `[1,1]`；Comparison 是否严格为单轮 `[2]`；
- **Recall**：两个 gold supporting titles 是否都出现在工具返回中。

由此得到四种互斥状态，四项之和等于该模型的 Strict Exact 错误总数：

| 状态 | 含义 |
|---|---|
| 拓扑错 + 召回不全 | 上游搜索链整体失败：漏调用、错误并行/串行、错误实体或检索失败可能同时存在 |
| 拓扑错 + 完整召回 | 找到了证据，但调用组织方式不符合目标；Base Comparison 中最常见 |
| 拓扑对 + 召回不全 | 调用轮次正确，但第一跳实体、第二跳 query rewrite、同名页或检索排序仍失败 |
| 拓扑对 + 完整召回 | 搜索链基本完成，错误主要位于比较推理、证据聚合、答案抽取或严格边界 |

该矩阵是“过程状态”而不是强行声称唯一因果。gold-title Recall 只是字符串指标：有时非 gold 页面也包含答案，有时完整召回后模型仍没有读取正确句子。因此本文再用真实输出案例区分 query rewrite、检索器和答案侧错误。

## 7. Bridge 错误类型对比

### 7.1 互斥过程状态

| 模型 | checkpoint 定位 | Strict 错误 | 拓扑错+召回不全 | 拓扑错+完整召回 | 拓扑对+召回不全 | 拓扑对+完整召回 |
|---|---|---:|---:|---:|---:|---:|
| Base | 基线 | 56 | 26 | 1 | 12 | 17 |
| GRPO s100 | 答案/策略均最佳 | **49** | 9 | 0 | 14 | **26** |
| Bridge teacher s75 | Teacher 答案最佳 | **46** | 13 | 0 | 14 | 19 |
| Serial OPD s50 | Serial 答案最佳 | 57 | 17 | 1 | 12 | 27 |
| Serial OPD s100 | Serial 策略最佳 | 55 | **23** | 0 | 13 | 19 |
| MOPD s50 | **MOPD 答案最佳** | **52** | 15 | 2 | **11** | 24 |
| MOPD s100 | MOPD 策略最佳 | 55 | **14** | 0 | 15 | **26** |

这张表同时解释了 checkpoint 的分数/策略权衡：

- MOPD s50 的 Bridge Strict 错误最少，为 52；这与它 48% Bridge Exact 的答案峰值一致，应作为答案主结果；
- MOPD s100 相比 s50 多错 3 题，但“拓扑错+召回不全”从 15 降到 14、Strategy 从 76% 升到 83%，是行为峰值；
- 在策略最佳点比较，Serial s100 与 MOPD s100 都错 55 题，但前者有 23 题“拓扑错+召回不全”，后者只有 14 题；MOPD 把更多错误推进到答案下游。

### 7.2 可重叠过程故障计数

以下列只回答“Strict 错误样本中是否出现该现象”，同一题可以同时计入多列：

| 模型 | checkpoint 定位 | 拓扑错误 | gold-title 召回不全 | 完整召回但答案错 | 格式错误 |
|---|---|---:|---:|---:|---:|
| Base | 基线 | 27 | 38 | 18 | 6 |
| GRPO s100 | 答案/策略均最佳 | 9 | 23 | 26 | 6 |
| Bridge teacher s75 | Teacher 答案最佳 | 13 | 27 | 19 | 8 |
| Serial OPD s50 | Serial 答案最佳 | 18 | 29 | 28 | 9 |
| Serial OPD s100 | Serial 策略最佳 | **23** | **36** | 19 | **13** |
| MOPD s50 | MOPD 答案最佳 | 17 | **26** | 26 | 10 |
| MOPD s100 | MOPD 策略最佳 | 14 | 29 | **26** | 8 |

串行 OPD s100 的 Bridge 错误同时具有最高的拓扑错误、召回不全和格式错误，符合切换到 Compare teacher 后的行为遗忘。MOPD s50 的召回不全最少且 Exact 最高；MOPD s100 的拓扑错误进一步减少，但答案峰值已经过去。两点同时说明：s50 应作为分数主模型，s100 应作为多策略保持模型。

### 7.3 GRPO Bridge 的人工精细复核

GRPO s100 的 49 个 Bridge Strict 错误已经逐题人工检查，可进一步拆为：

| 互斥类型 | 数量 |
|---|---:|
| 第一/第二跳实质检索错误 | 10 |
| 中间实体到第二跳 query 的改写错误 | 5 |
| 证据齐全后的真正抽取/推理错误 | 10 |
| 语义接近但答案边界/别名错误 | 24 |

这里的“实质检索错误 10”小于过程表中的“gold-title 召回不全 23”，因为 title 字符串未命中不一定阻断答案：同名页、非 gold 页面或第一篇 passage 有时已包含足够事实。对其他模型本文保留可复算过程矩阵和人工案例，不伪造未经逐题语义复核的精确 query/retriever 分界数。

## 8. Comparison 错误类型对比

### 8.1 互斥过程状态

| 模型 | checkpoint 定位 | Strict 错误 | 拓扑错+召回不全 | 拓扑错+完整召回 | 拓扑对+召回不全 | 拓扑对+完整召回 |
|---|---|---:|---:|---:|---:|---:|
| Base | 基线 | 32 | 7 | **22** | 0 | 3 |
| GRPO s100 | 答案/策略均最佳 | 33 | 1 | 1 | 6 | 25 |
| Compare teacher s25 | Teacher 答案/策略最佳 | **30** | 1 | 1 | 5 | 23 |
| Serial OPD s50 | Serial 答案最佳 | **30** | 5 | **20** | 0 | 5 |
| Serial OPD s100 | Serial 策略最佳 | 33 | 1 | 1 | 5 | **26** |
| MOPD s50 | MOPD 答案最佳 | 31 | 2 | 3 | **3** | 23 |
| MOPD s100 | MOPD 策略最佳 | **30** | 2 | 1 | 5 | 22 |

Base 的错误主要是“证据已经召回，但调用拓扑仍是串行”：32 个错误中有 22 个属于拓扑错+完整召回。Serial s50 虽有 70% Comparison Exact，但 30 个错误中仍有 20 个属于同类状态，证明它尚未接受 Compare teacher。到 Serial s100 或 MOPD s100，这类错误几乎消失，错误集中到“正确 `[2]` + 完整召回后仍答错”。

### 8.2 可重叠过程故障计数

| 模型 | checkpoint 定位 | 拓扑错误 | gold-title 召回不全 | 完整召回但答案错 | 格式错误 |
|---|---|---:|---:|---:|---:|
| Base | 基线 | **29** | 7 | 25 | 8 |
| GRPO s100 | 答案/策略均最佳 | 2 | 7 | 26 | 3 |
| Compare teacher s25 | Teacher 答案/策略最佳 | 2 | 6 | 24 | 1 |
| Serial OPD s50 | Serial 答案最佳 | **25** | 5 | 25 | 4 |
| Serial OPD s100 | Serial 策略最佳 | 2 | 6 | **27** | 2 |
| MOPD s50 | MOPD 答案最佳 | 5 | **5** | 26 | **0** |
| MOPD s100 | MOPD 策略最佳 | 3 | 7 | **23** | 3 |

答案峰值不等于行为峰值：Serial s50 有 70% Comparison Exact，但错误子集中仍有 25 个拓扑错误；MOPD s50 已把该数降到 5。到策略最佳点后，GRPO、Teacher、Serial s100、MOPD s100 的 Comparison 拓扑错误都只剩 2–3 个。MOPD s100 与 Compare teacher 同为 70% Exact，完整召回后答错数也最少；Serial s100 虽有 96% Strategy，但完整召回后仍答错 27 题，说明最后教师的调用模式学得快，比较结论并没有逐题完全复制。

GRPO s100 的 33 个 Comparison 错误经人工复核后可分为：3 个上游规划/检索失败、9 个真正比较推理或答案抽取错误、21 个语义正确但严格边界/别名失败。因此 Comparison 后续最值得优化的是比较算子与 answer token，而不是继续强化 `[2]`。

## 9. 真实案例

### 9.1 Comparison：答案正确不代表并行策略正确

问题：`Are Grant Nicholas and Danny Shirley both American singers?`，Gold：`no`。

| 模型 | 调用 | 召回 | 答案 | 结论 |
|---|---|---:|---|---|
| Base | `[1,1]` | 1.0 | `no` ✓ | 会答，但把两个独立查询串行执行 |
| GRPO | `[2]` | 1.0 | `no` ✓ | 策略和答案都正确 |
| Compare teacher | `[2]` | 1.0 | `no` ✓ | 专项策略正确 |
| 串行 OPD | `[2]` | 1.0 | `no` ✓ | 已学会最后一个教师 |
| MOPD | `[2]` | 1.0 | `no` ✓ | 同时保持并行策略 |

固定集上有 50 个 Comparison 样本符合“Base 答案对但策略错，MOPD 答案与策略都对”。这正是 Strategy 必须与 Exact 分开报告的原因。

### 9.2 Comparison：同名页检索与答案完成

问题：`A Silent Film` 与 `The Frames` 谁成员更多？Gold：`The Frames`。

| 模型 | 调用与召回 | 输出 |
|---|---|---|
| Base | `[1,1,1]`；先命中同名 album，第三次才补查 band | `The Frames` ✓，策略错 |
| GRPO | `[2]`；`A Silent Film members` 命中同名 album，Recall 0.5 | 空答案 ✗ |
| Compare teacher | `[2]`；使用 `members count`，两侧标题完整召回 | `The Frames` ✓ |
| 串行 OPD | `[2]`；仍命中 album，Recall 0.5 | `The Frames` ✓ |
| MOPD | `[2]`；复现 teacher 的 `members count` 查询，Recall 1.0 | `The Frames` ✓ |

这个案例同时包含拓扑、查询消歧、检索结果和答案完成四层差异。MOPD 不只是把两次调用放在同一轮，还学到了更能避开同名专辑的 query 表达。

### 9.3 Bridge：MOPD 修复串行 OPD 的行为遗忘

问题：参加 1986 年世界杯的人在 1982 Scottish League Cup Final 为哪支队伍效力？Gold：`Celtic`。

| 模型 | 轨迹 | 结果 |
|---|---|---|
| Base | 两次宽泛查询，只召回 `Charlie Nicholas` | Recall 0.5，但碰巧答 `Celtic` ✓ |
| GRPO | 连续四次宽泛搜索，最后才接近 Charlie Nicholas | 无合法答案，策略错 |
| Bridge teacher | 连续四次停留在 World Cup 宽泛页面 | Recall 0，失败 |
| 串行 OPD | 没有形成有效工具调用 | 无答案，失败 |
| MOPD | `1986 FIFA World Cup players` → `Charlie Nicholas 1982 Scottish League Cup` | `[1,1]`、Recall 1.0、答 `Celtic` ✓ |

这是 student 超过 teacher 的典型例子。OPD/MOPD 不是离线复制 teacher 的固定答案；student 先生成自己的 on-policy 轨迹，teacher 只在这些前缀上提供 token 分布，因此 student 可以形成 teacher 本次 greedy 轨迹没有出现的正确路径。

### 9.4 Bridge 查询改写：保留中间实体还不够，还要保留约束

问题：距《Brief Encounter》使用车站 3 英里的车站？Gold：`Borwick railway station`。

| 模型 | 第二跳 query | 第二跳结果 | 答案 |
|---|---|---|---|
| GRPO | `stations 3 miles from Carnforth railway station` | `Borwick railway station` | 正确 |
| Bridge teacher | `stations near Carnforth railway station` | 再次返回 Carnforth | `Paddington` |
| 串行 OPD | `stations near Carnforth railway station` | 再次返回 Carnforth | `Liverpool Street Station` |
| MOPD | `stations near Carnforth railway station` | 再次返回 Carnforth | `Oxford Circus` |

四个模型都获得了第一跳中间实体 `Carnforth railway station`，但只有 GRPO 把问题中的数字约束 `3 miles` 写入第二跳 query。该错误应归为 **query rewrite 约束丢失**，不是缺少串行拓扑。

### 9.5 Bridge：query 已改善，但检索排序仍可能失败

问题：Shrewsbury & Newport Canals Trust 推动修复哪条 navigable canal？Gold：`Shropshire Union Canal`。

- 首跳 passage 同时出现 `Shrewsbury Canal` 和 `the Newport Arm of the Shropshire Union Canal`。
- GRPO/Bridge teacher 把第二跳错误改写成 `Shrewsbury Canal`，返回同名错误页面。
- 串行 OPD 只完成第一跳便停止。
- MOPD 的第二跳已经写成 `Shrewsbury Canal Shropshire Union Canal`，说明正确实体进入 query，但检索器仍返回 `Shrewsbury Canal`，最终也选错。

该案例展示 query rewrite 与 retrieval ranker 的边界：MOPD 已部分修复 query token，却仍需要检索器或证据选择模块消除相近实体竞争。

### 9.6 完整证据后的答案类型错误

问题：Robert Gould Shaw 指挥的步兵由什么授权？Gold：`Emancipation Proclamation`。

| 模型 | Strategy / Recall | 答案 |
|---|---|---|
| Base | 单次调用，策略错 / 0.5 | `Emancipation Proclamation` ✓ |
| GRPO | `[1,1]` / 1.0 | `Abraham Lincoln` ✗ |
| Bridge teacher | `[1,1]` / 1.0 | `Emancipation Proclamation` ✓ |
| 串行 OPD | 单次调用，策略错 / 0.5 | `Emancipation Proclamation` ✓ |
| MOPD | `[1,1]` / 1.0 | `Abraham Lincoln` ✗ |

GRPO/MOPD 已召回完整证据，却把“授权文件”回答成“签署人”。这不是检索问题，而是关系类型和答案 span 选择错误。

### 9.7 Comparison：完整召回后的逻辑判断错误

问题：Chien Français Blanc et Orange 与 Cretan Hound 是否都用于 pack hunting？Gold：`no`。

- GRPO、串行 OPD、MOPD 都使用正确 `[2]` 并完整召回两篇证据，却回答 `yes`；
- Compare teacher 使用同样 `[2]` 与完整证据，回答 `no`；
- 错误发生在对两个布尔属性做逻辑合取，而不是检索或调用拓扑。

这类错误需要显式的比较算子监督，例如 `both = property(A) AND property(B)`，而不是继续增加并行调用奖励。

### 9.8 严格答案边界错误

问题：Adena Friedman 曾任 CFO 的公司有哪四个业务领域？

- Gold：`corporate private equity, real assets, global market strategies, and investment solutions`；
- Bridge teacher / 串行 OPD：完整输出，Strict Exact 正确；
- GRPO / MOPD：输出中漏掉连接词 `and`，F1 为 95.2%，但 Strict Exact 判错；
- 所有模型均使用正确 `[1,1]` 并完整召回两篇 gold 文档。

这类样本不应增加检索 reward，应在 `<answer>` token 上加强边界、连接词、单位、全名和别名监督，并同时保留 mean F1 与语义等价评测。

## 10. 每个模型到底学到了什么

### Base

- 优点：已有较强事实知识和 Comparison 答案能力；部分题即使 Recall 不完整也能答对。
- 缺点：把独立 Comparison 查询串行化，Bridge 常单跳停止；答案正确经常依赖记忆或偶然路径。
- 定位：能力下限与“只看答案会误判策略”的关键基线。

### 通用 GRPO s100

- 优点：单 student 综合最佳；过程奖励直接把 Bridge/Compare 两种拓扑都训练到较高水平。
- 缺点：仍有较多答案边界和证据后抽取错误；继续训练到 s125 出现局部轨迹漂移。
- 定位：最终综合主模型、MOPD 的强单模型对照。

### 路由双教师

- 优点：按目标题型组合后答案/F1 最高；Compare teacher 的并行行为几乎饱和，Bridge teacher 提供更干净的专项答案轨迹。
- 缺点：它是两个模型组合，不是压缩后的统一 student；Bridge teacher 在少数宽泛查询上也会失败。
- 交叉表现：Bridge teacher s75 在 Comparison 上有 71% Exact、却只有 7% Strategy；Compare teacher s25 在 Bridge 上为 43%/60%。这证明 teacher 必须按行为路由，不能只按答案分挑选。
- 定位：MOPD 的行为/答案参考上限和 token 分布来源。

### 串行 OPD

- 优点：s50 的 Overall Exact 达到 56.5%；随后 25 个 Comparison step 将 Compare Strategy 从 25% 提升到 96%，证明 token-level 监督学习专项行为很快。
- 缺点：答案最佳与策略最佳分离；s100 出现最后教师偏置，Bridge 拓扑、召回、格式同时退化，Exact 回到 Base 水平。
- 定位：证明顺序蒸馏存在灾难性干扰的关键消融。

### MOPD

- 优点：s50 达到 58.5% Exact / 71.17% F1，是蒸馏路线答案最佳；s100 在单 student 中把 Strategy 提升到 89%，并显著缓解串行 OPD 的 Bridge 行为遗忘。
- 缺点：s50→s100 期间答案分回落；Bridge 答案抽取没有与策略同步提升，双教师并存还带来 KV cache 和显存生命周期管理问题。
- 定位：s50 代表答案效率，s100 代表多策略保持；两者共同构成 MOPD 结果，不能只挑其中一个概括全程。

## 11. 最终模型与 checkpoint 选择

- **综合单模型：GRPO s100**。Exact、F1、Strategy 和联合指标最均衡，也是该训练族的答案/策略共同峰值。
- **MOPD 主分数：s50**。按 Strict Exact 优先的选择标准，s50 的 58.5% Exact / 71.17% F1 是 MOPD 正式答案结果。
- **MOPD 行为结果：s100**。Strategy 89%、`Exact∩Strategy` 55%，用于证明多教师路由缓解顺序干扰；不能用它覆盖 s50 的答案峰值。
- **Serial 主分数：s50**。Exact 56.5%；但 Strategy 只有 53%，必须与 s100 行为点一起呈现。
- **Serial 行为结果：s100**。Strategy 82.5%，用于展示学会 Comparison 后的 Bridge 遗忘。
- **Bridge teacher 答案点：s75**。Bridge 54% Exact / 81% Strategy；**策略点：s50**，49% Exact / 85% Strategy。
- **Compare teacher：s25**。Comparison 70% Exact / 98% Strategy，答案与策略峰值一致。
- **路由 teacher 参考：Bridge s75 + Compare s25**。总体 62% Exact / 74.41% F1 / 89.5% Strategy，但它是两个专家的组合，不包装成单 student。

## 12. 对下一轮 reward / 蒸馏权重的启示

1. Bridge 的策略并未饱和，但 MOPD 已把更多错误推进到答案侧；应优先提高第二跳关键约束 token 和最终 `<answer>` token 权重。
2. 区分“中间实体已出现但改写错误”与“query 正确但检索返回错误页”：前者修 student，后者修检索 ranker 或加入 hard-negative entity disambiguation。
3. Comparison `[2]` 已接近饱和，不再大幅提高 topology reward；增加大小、时间先后、属性交集、布尔合取和共同实体抽取监督。
4. checkpoint 选择同时报告 Exact、mean F1、Strategy、Recall 和 `Exact∩Strategy`，避免只奖励答案或只奖励调用形式。
5. 对常用名/全名、数值单位、连接词和异常 gold 建立 alias/语义等价口径，Strict Exact 继续保留但不单独承担模型诊断。

## 13. 数据与复核文件

- [总体与分题型指标](results/model_overall_comparison.csv)
- [错误状态矩阵](results/model_error_state_matrix.csv)
- [两位 Teacher 完整交叉评测](results/teacher_cross_route_metrics.csv)

## 14. 数据口径与限制

- 五组输出的问题集合、题型和顺序已经对齐；最终指标来自固定 200 题 greedy 评测。
- Base 归档来自较早评测实现，本文按当前项目的 Strict/alias 规则重新核验 prediction 与 gold，使其与后续 summary 口径一致。
- “路由双教师”是两组专项 checkpoint 按题型拼接的统计参考，不等同于单模型参数量和推理成本；文中同时给出两位 teacher 的交叉路由成绩。
- 第 9 节真实案例主要使用行为峰值 MOPD s100 / Serial s100，以解释多策略保持；答案主结果仍采用各自 s50。
- 状态矩阵中的 Recall 是 gold-title 字符串召回，不等同于证据可抽取性的人工判断；因此同时保留人工案例与 GRPO 逐题语义复核。
- 训练耗时来自不同日期和运行环境，适合说明量级与 rollout 工作量，不是严格的硬件吞吐 benchmark。

所有公开汇总指标均可由固定评测集的 JSONL/summary.csv 重新计算，不依赖 SwanLab 页面继续在线可用。
