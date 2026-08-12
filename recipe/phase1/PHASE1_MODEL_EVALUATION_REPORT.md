# Phase 1 模型评测对比与案例分析

## 1. 报告范围

本报告比较 Phase 1 的四个模型：

| 简称 | 模型 / checkpoint | 作用 |
| --- | --- | --- |
| Base | Qwen3-4B 未训练模型 | 训练前基线 |
| 50step | `qwen3_4b_sglang_16x8_50step_llmjudge_20260808_173324/global_step_50` | 策略 prompt + 检索/策略/答案混合奖励 |
| 100step | 同一实验续跑到 `global_step_100` | 检查继续训练的收益和退化 |
| Answer-only 50 | `qwen3_4b_answeronly_50step_907_20260809_140947/global_step_50` | 去掉策略提示和策略奖励、只保留稀疏 exact 的消融 |

评测集是固定 200 条 HotpotQA distractor 验证数据，bridge 与 compare 各 100 条。四份 greedy 文件严格逐题对齐；sampled 评测每题 5 条，共 1000 条轨迹/模型。

本文的 exact/F1 均用当前 `recipe.core.my_reward._answer_metrics` 从 `pred` 与多 gold 重新计算。不能直接采用 JSONL 内的 `answer_exact`：启用 LLM judge 的运行中，该字段可能包含语义判定，不等于严格字符串口径。

## 2. 总体结果

### 2.1 Greedy：严格答案与行为指标

| 模型 | Exact | F1≥0.5 | Mean F1 | 策略正确 | 检索召回 | Format OK | 平均调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.560 | 0.725 | 0.687 | 0.405 | 0.840 | 0.930 | 1.88 |
| 50step | **0.590** | 0.740 | **0.711** | **0.845** | **0.868** | 0.950 | 2.05 |
| 100step | 0.580 | **0.745** | 0.709 | 0.830 | 0.855 | **0.965** | 2.09 |
| Answer-only 50 | 0.535 | 0.690 | 0.657 | 0.235 | 0.753 | 0.910 | 1.68 |

50step 是最均衡的 Phase 1 checkpoint：相对 Base，严格 exact 提升 3 个百分点，策略正确率提升 44 个百分点。继续到 100step 后，F1≥0.5 和格式略升，但 exact、策略和召回均轻微回落。Answer-only 全面低于 Base，说明 0/1 exact 稀疏奖励不足以稳定学习搜索策略。

### 2.2 Greedy：按题型拆分

| 模型 | Bridge exact | Bridge 策略 | Bridge recall | Compare exact | Compare 策略 | Compare recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.440 | 0.630 | 0.745 | 0.680 | 0.180 | **0.935** |
| 50step | **0.500** | 0.760 | **0.805** | 0.680 | **0.930** | 0.930 |
| 100step | 0.460 | **0.790** | 0.790 | **0.700** | 0.870 | 0.920 |
| Answer-only 50 | 0.400 | 0.470 | 0.615 | 0.670 | 0.000 | 0.890 |

主要收益来自两个不同方向：

1. Bridge 的答案收益集中在前 50 步：exact `0.44 → 0.50`，随后回落到 `0.46`。
2. Compare 的核心收益是执行方式，而不是答案：Base 已有 `0.68` exact，但仅 `0.18` 真正同轮双查；50step 把并行率提高到 `0.93`，exact 仍为 `0.68`。
3. Answer-only 的 compare exact 仍有 `0.67`，但并行率为 `0.00`。模型可以靠已有知识或串行搜索答对，却完全没有学会目标策略。

### 2.3 Sampled ×5：严格口径与策略可恢复性

| 模型 | Sample exact@1 | Pass@5 exact | Pass@5 F1≥0.5 | 单轨迹策略正确 | Pass@5 策略正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.560 | **0.670** | 0.835 | 0.467 | 0.725 |
| 50step | 0.575 | **0.670** | 0.835 | 0.815 | 0.960 |
| 100step | **0.585** | 0.650 | **0.840** | **0.847** | **0.980** |
| Answer-only 50 | 0.530 | 0.620 | 0.785 | 0.253 | 0.385 |

采样进一步说明“策略学会”和“答案上限提高”不是一回事：50/100step 的策略 pass@5 大幅上升，但严格答案 pass@5 没超过 Base。按题型看，bridge pass@5 exact 为 Base `0.62`、50step `0.60`、100step `0.58`；compare 分别为 `0.72`、`0.74`、`0.72`。继续训练主要稳定了工具行为，没有突破 bridge 的答案瓶颈。

## 3. 逐题迁移矩阵

### 3.1 严格 exact 的变化

| 对比 | 一直错 | 错→对 | 对→错 | 一直对 | 净变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base → 50step | 75 | 13 | 7 | 105 | +6 |
| 50step → 100step | 78 | 4 | 6 | 112 | -2 |
| Base → Answer-only | 79 | 9 | 14 | 98 | -5 |

Base→50step 的 13 个新增正确中，bridge 有 10 个、compare 只有 3 个；7 个新增错误中 bridge 4 个、compare 3 个。50→100step 的 6 个退化样本全部来自 bridge；compare 有 2 个恢复、没有 exact 退化。

### 3.2 策略正确性的变化

| 对比 | 一直错 | 错→对 | 对→错 | 一直对 | 净变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base → 50step | 23 | 96 | 8 | 73 | +88 |
| 50step → 100step | 17 | 14 | 17 | 152 | -3 |
| Base → Answer-only | 111 | 8 | 42 | 39 | -34 |

Base→50step 的最大变化是 compare：76 题从非并行变为并行，仅 1 题反向退化。50→100step 则出现分化：bridge 净增 3 个策略正确，compare 净减 6 个；100step 更容易追加不必要的第二批查询。Answer-only 在 compare 上从 Base 的 18 个并行成功降为 0。

## 4. 具体案例：训练前错，训练后对

### 4.1 Bridge：真正学会根据第一跳实体构造第二跳

| 问题 | Gold | Base | 50step | 变化 |
| --- | --- | --- | --- | --- |
| Peter Daou 的网站口号是什么？ | `media for the 65.8 million` | 只查一次 `Peter Daou website slogan`，把网站名 `Verrit` 当答案 | `[1,1]`：先查 Peter Daou 的网站，再查 `Verrit website slogan`，答对口号 | 从“查到中间实体”变成真正完成第二跳 |
| 哪位荷兰画家擅长中产阶级室内场景，并有黄披肩女子画作？ | `Johannes Vermeer` | 单查后答 `Vermeer`，严格 exact 不通过 | `[1,1]`，定位画家后用第二跳核对画作，答完整姓名 | recall `0.5→1.0`，策略与答案同时改善 |
| Melanie Griffith 女儿主演的恐怖片改编叫什么？ | `Suspiria` | `[1,1]` 但第二跳召回不足，答 `None` | `[1,1]`，第二跳 `Dakota Johnson upcoming horror films`，答 `Suspiria` | 相同调用结构下，查询表达和召回改善 |
| Peter Daou 案例之外，Pollution Prevention Act 扩展的是哪类数据库？ | `publicly available` | 答 `public` | 先定位法案关联数据库，再核对 TRI，答 `publicly available` | 答案边界更精确 |

### 4.2 Compare：从串行或不搜索变成同轮双查

| 问题 | Gold | Base | 50step | 变化 |
| --- | --- | --- | --- | --- |
| Fitz and The Tantrums 与 The Contortionist 谁先成立？ | `The Contortionist` | `[1,1]` 串行查两队，最终没有合法答案 | `[2]` 同轮查两队成立年份，答对 | 并行策略、格式和答案同时修复 |
| Beau Bokan 与 Jason Scheff 共同属于哪类音乐人？ | `singer` | 没有工具调用，也没有答案 | `[2]` 同轮查询两人的音乐角色，答 `singer` | 从零检索到完整并行比较 |
| Indiana University 与 University of Pennsylvania 谁更偏东？ | `University of Pennsylvania` | `[1,1]` 检索到两校位置但无最终答案 | `[2]` 同轮双查并答对 | 说明格式/决策失败也能被策略训练修复 |

## 5. 具体案例：训练前对，训练后错

这些案例说明策略正确并不自动保证语义正确。

| 问题 | Gold | 训练前 / 较早 checkpoint | 训练后 / 较晚 checkpoint | 原因 |
| --- | --- | --- | --- | --- |
| 谁授权了 Robert Gould Shaw 指挥的这支步兵？ | `Emancipation Proclamation` | Base 单查后答对 | 50/100step 都执行 `[1,1]`，却把第二跳“谁制定公告”误答为 `Abraham Lincoln` | 串行策略正确，但第二跳改变了问题语义 |
| Alfred Santell 与 David Fincher 的共同职业？ | `director` | Base 串行查后答 `director` | 50/100step 并行正确，但答 `director and producer`，strict exact 失败、F1=0.5 | 过度回答，不是检索失败 |
| Melanie Griffith 女儿主演的恐怖片改编？ | `Suspiria` | 50step 答对 | 100step 同样 `[1,1]`，召回下降并答 `None` | 继续训练后的 bridge 回退 |
| Jal Pari 对应什么传奇生物？ | `mermaid` | 50step `[1,1]` 答 `Mermaid` | 100step 做 3 次串行查询，答 `Undine` | 额外查询引入干扰，策略和答案一起退化 |
| 英国首本 Linux 专门杂志如何评价 Qvwm？ | `an unusually impressive imposter` | 50step 两跳检索并答对评价 | 100step 检索相同信息，却把来源 `Linux Format` 当答案 | 证据选择/最终抽取退化 |

## 6. 50step 与 100step：继续训练既有恢复也有遗忘

### 6.1 100step 恢复的题

| 问题 | Gold | 50step | 100step |
| --- | --- | --- | --- |
| `Southern Child` 演唱者的姓氏？ | `Penniman` | 找到 Little Richard，但答名字 `Richard` | 第二跳仍查 Little Richard 的姓，正确抽取 `Penniman` |
| A Silent Film 与 The Frames 谁成员更多？ | `The Frames` | `[2]` 检索正确但没有输出合法答案 | `[2]` 后输出 `The Frames` |
| Mastodon 与 Hole 谁成员更多？ | `Mastodon` | `[2]` 后误答数字 `4` | `[2]` 后正确选择 `Mastodon` |

### 6.2 100step 新增的行为退化

| 问题 | 50step | 100step | 观察 |
| --- | --- | --- | --- |
| 哪所大学更偏东？ | `[2]` 一轮双查并答对 | `[2,2]` 又查一轮城市经度，答案仍对 | 答案没受损，但因过度搜索而策略判错 |
| Rimo I 与 Passu Sar 属于哪个山脉？ | `[2]`，答 `Karakoram`（F1=0.5） | 完全不调用工具、无答案 | compare 策略从正确变为失败 |
| Ralph Hefferline 任教大学位于哪座城市？ | `New York City` | `New York` | 知识基本正确，但严格答案边界退化 |

100step 的变化不是简单的“训练更久更好”：bridge 策略略稳，但 bridge exact 减少 4 题；compare exact 增加 2 题，却有更多过度查询导致策略率从 `0.93` 降到 `0.87`。

## 7. Answer-only 消融：多数退化与少量反例

### 7.1 典型退化

| 题型与问题 | Gold | Base / 50step | Answer-only 50 | 退化类型 |
| --- | --- | --- | --- | --- |
| Bridge：Arizona SR 51 以哪个部族女性命名？ | `Hopi` | Base、50、100 均 `[1,1]` 答对 | 只查一次且没有最终答案 | 第二跳消失，格式失败 |
| Bridge：`The Black Belly of the Tarantula` 女演员嫁给哪位 Beatles 前成员？ | `Ringo Starr` | Base、50、100 均答对 | 零调用、零答案 | 已有能力被训练破坏 |
| Bridge：发行 `Said and Done` 的男团截至 2013 卖出多少唱片？ | `25 million` | Base、50、100 均答对 | 错把团体定位为 98 Degrees，答 `2.2 million` | 第一跳实体识别错误 |
| Compare：两部动画电影共同在哪年上映？ | `2009` | Base 串行答对；50/100 并行答对 | 零调用、零答案 | 去掉策略信号后直接放弃检索 |
| Compare：Nicholas Hytner 与 Maya Deren 是否从事同一电影角色？ | `no` | Base 串行答对；50/100 并行答对 | 串行两查后答 `yes` | 有召回但比较推理错误 |

### 7.2 必须保留的反例：Answer-only 并非每题都更差

| 问题 | Gold | Base / 50 / 100 | Answer-only 50 | 含义 |
| --- | --- | --- | --- | --- |
| 距《Brief Encounter》使用车站 3 英里的车站？ | `Borwick railway station` | 分别为空、`King's Cross Station`、`Carnforth railway station` | `[1,1]` 后唯一答对 `Borwick railway station` | 个别样本的查询改写碰巧更精确 |
| Chien Français Blanc et Orange 与 Cretan Hound 是否都用于群猎？ | `no` | Base/100 答 `yes`，50 无合法答案 | Answer-only 串行多查后唯一答 `no` | 稀疏奖励仍可能在个别比较题上改善答案，但没有学会并行 |

因此结论应是“Answer-only 的期望效果显著更差”，而不是“每个样本必然退化”。其净变化为 9 题错→对、14 题对→错。

## 8. 策略改善但答案不变：Phase 1 最稳定的收益

| 问题 | Base | 50/100step | Answer-only | 结论 |
| --- | --- | --- | --- | --- |
| 两部电影谁先上映？ | `[1,1]`，答案正确 | `[2]`，答案仍正确 | `[1,1]`，答案正确 | 训练主要改变执行效率，不改变已有知识 |
| 两部动画电影是否同年上映？ | `[1,1]`，答 `2009` | `[2]`，仍答 `2009` | 零调用、无答案 | 混合奖励把已有答案能力绑定到正确工具策略 |
| The Silent Historian 与 The Betrayal 属于什么电影？ | Base `[2]`，答 `documentary`（F1 部分匹配） | 50/100 也 `[2]`，答案边界仍未达到 `documentary film` | Answer-only 退化为 `[1,1]` | 策略正确不能自动修复答案表达 |

Compare exact 在 Base 和 50step 都是 `0.68`，但并行率从 `0.18` 升到 `0.93`，正是这类样本在总体指标中的体现。

## 9. 严格 exact 的边界案例

报告必须区分真正知识错误与表达差异：

| 问题 | Gold | 预测 | Strict | F1 / 说明 |
| --- | --- | --- | ---: | --- |
| `Laura Warholic` 作者最著名的小说？ | `Darconville’s Cat` | Base：`Darconville's Cat` | 错 | 仅弯/直撇号不同；F1=0.5，50/100 使用 gold 同款字符后 exact |
| Yi Guan 帮助防止哪种呼吸道疾病暴发？ | `severe acute respiratory syndrome` | 100step：`Severe Acute Respiratory Syndrome (SARS)` | 错 | 语义正确，但追加缩写导致 exact 失败；F1=0.889 |
| 两位导演的共同职业？ | `director` | `director and producer` | 错 | 包含 gold 但过度回答；F1=0.5 |
| Columbia University 所在城市？ | `New York City` | `New York` | 错 | 地理语义接近，但严格边界不一致 |

这解释了为什么 100step 的 strict exact 下降，而 F1≥0.5 和格式指标仍略升；评估时应同时保留 strict、F1 和 LLM 语义判定，但模型选择仍需事先固定主口径。

## 10. Greedy 与采样的具体差异

| 模型与问题 | Greedy | 5 次采样 | 观察 |
| --- | --- | --- | --- |
| Base：Fitz vs The Contortionist | 无合法答案 | 4/5 次答 `The Contortionist` | Base 有潜在知识，但 greedy 解码不稳定 |
| Base：Alfred Santell 与 David Fincher 的共同职业 | `director`，正确 | 5/5 都答 `director and producer`，strict 全错 | 采样可能稳定地产生同一种过度回答 |
| 50step：`Suspiria` | Greedy 正确 | 5/5 均错误或 `None` | 单次 greedy 成功不等于采样鲁棒 |
| 100step：2009 NASCAR Nationwide Series | Greedy 答 `Carl Edwards` | 3/5 答 `Kyle Busch` | 采样能恢复部分 bridge 错误 |
| Answer-only：Borwick railway station | Greedy 唯一答对 | 5/5 全错 | 消融模型的偶然 greedy 命中不可泛化 |

因此 pass@5 既测“能力支持集”也暴露随机性。50/100step 的策略 pass@5 接近饱和，但答案 pass@5 不升，说明当前限制更接近证据抽取和最终表达，而不是“完全不会搜索”。

## 11. 最终结论

1. **Phase 1 的主要成功是策略塑形。** Base→50step 把 greedy 策略正确率从 `0.405` 提高到 `0.845`，尤其将 compare 并行率从 `0.18` 提高到 `0.93`。
2. **50step 是最佳均衡点。** 它取得最高 strict exact `0.590` 和最高 mean F1 `0.711`；100step 的额外训练主要提高采样策略稳定性和格式，不再提高答案。
3. **Bridge 是答案瓶颈。** 前 50 步 bridge exact 上升 6 点，但继续训练回落 4 点；sampled pass@5 也没有提高。失败集中在第二跳查询、证据选择和最终答案边界。
4. **Compare 的答案与并行策略明显解耦。** Base 即使串行也能答到 `0.68`，因此必须单独报告策略正确率，不能只看答案 exact。
5. **纯 exact 奖励是负向消融。** Answer-only 的 exact、F1、召回、格式和策略都下降，compare 并行率归零；但仍有 9 个错→对反例，说明结论是统计性的而非逐题绝对。
6. **继续训练存在遗忘与过度搜索。** 100step 在 compare 上会追加第二批查询，在 bridge 上出现已会题目回退。后续训练应保存中间 checkpoint，并以固定 200 条同时评估答案、策略和调用结构。
7. **Phase 2/OPD 的目标合理。** 两位专精 teacher 应优先提供互补的干净策略分布；蒸馏后仍需重点验证 bridge 的答案抽取，而不能假设策略 KL 降低就必然带来 exact 提升。

## 12. 数据来源与复现

本报告使用 907 上以下只读评测产物：

```text
/root/autodl-tmp/rollouts/eval_base_v2_greedy/0.jsonl
/root/autodl-tmp/rollouts/eval_50_llmjudge_greedy/50.jsonl
/root/autodl-tmp/rollouts/eval_100_v2_greedy/100.jsonl
/root/autodl-tmp/rollouts/eval_ansonly_greedy/50.jsonl
/root/autodl-tmp/rollouts/eval_base_v2_sample/0.jsonl
/root/autodl-tmp/rollouts/eval_50_llmjudge_sample/50.jsonl
/root/autodl-tmp/rollouts/eval_100_v2_sample/100.jsonl
/root/autodl-tmp/rollouts/eval_ansonly_sample/50.jsonl
```

可用 `recipe/phase1/analyze_phase1_comparison.py` 重新生成严格指标、迁移矩阵与候选案例。分析脚本要求四份 greedy 严格逐题对齐，并拒绝混用问题集合不同的文件。
