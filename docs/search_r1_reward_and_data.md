# Search-R1 奖励函数与答案数据

本文说明 GRPO 路线的复合奖励、可选 LLM 答案裁判、答案表面变体扩增，以及严格 Exact Answer-only 消融。模型间的正式结果仍以固定 200 条验证集上的离线 Strict Exact/F1 为准，训练 reward 只用于优化和诊断。

## 1. 复合奖励

实现位于 [`recipe/core/my_reward.py`](../recipe/core/my_reward.py)。对一条完整工具轨迹，先抽取最终 `<answer>`、工具调用分组、搜索结果标题和格式状态，再计算：

\[
R=\operatorname{clip}_{[-0.2,1]}
\left[
g_{search}(0.35F1+0.15EM)
+0.30Recall
+0.15Strategy
+0.05Format
-P_{extra}
-P_{duplicate}
\right].
\]

| 分量 | 权重 | 定义 |
| --- | ---: | --- |
| Answer F1 | 0.35 | HotpotQA 归一化后的 token F1，对多个 gold 取最大值 |
| Answer Exact | 0.15 | 归一化预测与任一 gold 完全一致 |
| Retrieval Recall | 0.30 | 搜索结果覆盖两个 gold title 的比例 |
| Strategy | 0.15 | Bridge 为 `[1,1]` 且两次调用间确实收到证据；Comparison 为同轮 `[2]` |
| Format | 0.05 | 有最终 `<answer>`、工具 XML/JSON 可解析、所有工具调用发生在答案之前 |
| Query | 0.00 | 仍记录 query 质量用于诊断，但当前不参与总奖励 |

答案奖励带有搜索门控：

\[
g_{search}=\mathbb{1}[\text{至少一次工具调用可解析且成功返回}].
\]

因此模型即使依靠参数记忆直接答对，也拿不到 Answer F1/Exact 分。Recall、Strategy 和 Format 仍独立计算，便于定位“答对但没有按要求搜索”和“搜索正确但答案抽取失败”两类问题。

两项惩罚为：

- 第 5 次起，每多一次工具调用扣 `0.05`；
- 成功执行的归一化重复 query 每次扣 `0.10`，最多扣 `0.20`；失败后重试不按重复 query 处罚。

`think_tokens`、单轮/总思考长度越界率、工具解析率和工具执行成功率都会上传为诊断指标，但思考长度本身不参与 reward，避免模型为了规避长度惩罚过早停止推理。

## 2. 可选 LLM 答案裁判

规则 Exact/F1 对别名可能过严，例如 `Steven Williams` 与 `Steve Williams`。`my_reward.py` 可以选择 LLM 做语义等价裁判：

```bash
export ANSWER_LLM_JUDGE=1
export DEEPSEEK_API_KEY=your_key
export ANSWER_LLM_MODEL=deepseek-v4-flash
export ANSWER_LLM_BASE_URL=https://api.deepseek.com
export ANSWER_LLM_TIMEOUT=30
```

`my_reward.py` 的模块级默认值是启用判定，但没有 API key 时会立即回退规则分数；本项目正式 Phase 1 与所有固定集评测脚本都显式设置 `ANSWER_LLM_JUDGE=0`。需要进行裁判增强实验时，应同时显式设置为 1 并提供 API key。

裁判只在“预测非空、规则 F1 大于 0、但尚未 Exact 命中”时触发。输入只包含问题、最多五条已检索文档、预测答案和 gold 列表；裁判只能返回 JSON `{"match": true|false}`。调用失败会重试一次，仍失败则回退到原规则分数，不会让训练任务因 API 故障中断。

判定成功时，语义等价样本的 Answer F1/Exact 都置为 1，不等价则都置为 0。因此它会改变训练奖励，而不仅是增加日志指标。需要注意：

- 训练中启用会带来 API 成本、延迟和裁判噪声；
- 固定集模型选择与公开主表必须设置 `ANSWER_LLM_JUDGE=0`，重新从原始 gold 计算 Strict Exact/F1；
- LLM judge 适合作为训练增强或辅助分析，不能替代可复现的主评测口径。

主 Phase 1、OPD checkpoint 评测脚本已显式关闭 LLM judge。教师专项训练可以按实验需要通过环境变量选择开启或关闭。

## 3. 答案表面变体扩增

[`recipe/data/enrich_answer_variants.py`](../recipe/data/enrich_answer_variants.py) 用“问题 + 原始 gold + 两条 gold evidence”构造受约束请求，让 LLM 只生成同一答案实体的全名、简称、昵称、别名或常见书写形式。

脚本的约束包括：

- 原 gold 永远位于列表首位；
- 只能依据给定 gold 和 evidence，不允许补充外部知识或更宽泛概念；
- 按归一化形式去重，每题最多保留 8 个变体；
- 每条结果增量写入 JSONL 缓存，支持并发、失败重试和中断续跑；
- 同时写入 `reward_model.ground_truth` 与 `extra_info.answer_variants`。

建议先小批量人工检查：

```bash
export DEEPSEEK_API_KEY=your_key

python recipe/data/enrich_answer_variants.py \
  --train-file data/hotpotqa_v3_hard_1600/train.parquet \
  --validation-file data/hotpotqa_v3_hard_1600/validation.parquet \
  --output-dir data/hotpotqa_v3_hard_1600_variants_smoke \
  --max-rows 20 \
  --concurrency 4
```

检查缓存后再跑完整数据：

```bash
python recipe/data/enrich_answer_variants.py \
  --train-file data/hotpotqa_v3_hard_1600/train.parquet \
  --validation-file data/hotpotqa_v3_hard_1600/validation.parquet \
  --output-dir data/hotpotqa_v3_hard_1600_variants \
  --concurrency 8
```

奖励函数会对 `ground_truth` 中任一变体计算 Exact/F1。为了保持公开指标严格可比，推荐只把扩增数据用于训练；正式评测继续使用未扩增的原始 validation gold。

## 4. 严格 Exact Answer-only 与 Base 对照

该消融同时移除显式搜索策略提示和过程奖励，用来回答：“只告诉模型最终答案是否完全正确，能否自然学会 Bridge 串行与 Comparison 并行？”

### 4.1 构造无策略提示数据

[`recipe/data/data_preprocess_no_strategy.py`](../recipe/data/data_preprocess_no_strategy.py) 接收已经构建好的 Search-R1 Parquet，只替换 system prompt：

- 删除 Comparison 并行、Bridge 串行和最多三次搜索等显式策略提示；
- 保留工具调用协议、证据 grounding 和最短精确答案格式；
- 不改变问题、gold、工具上下文、题型/路由及诊断 metadata。

```bash
python recipe/data/data_preprocess_no_strategy.py \
  --train-file data/hotpotqa_v3_hard_1600/train.parquet \
  --validation-file data/hotpotqa_v3_hard_1600/validation.parquet \
  --output-dir data/hotpotqa_v3_no_strategy
```

### 4.2 Strict Exact-only reward

[`recipe/core/my_reward_exact_only.py`](../recipe/core/my_reward_exact_only.py) 的训练分数为：

\[
R_{exact}=\operatorname{clip}_{[-0.2,1]}
\left[
g_{search}\cdot EM-0.05\max(0,N_{calls}-4)
\right].
\]

- F1 只记录、不提供部分分；
- Recall、Strategy、Query、Format 和重复 query 都只记录、不参与总分；
- LLM judge 强制关闭；
- Exact 使用 HotpotQA 归一化规则，训练数据保持原始单 gold，不做答案变体扩增。

训练入口：

```bash
MODEL_PATH=/path/to/Qwen3-4B \
PROJECT_ROOT=$PWD \
ARTIFACT_ROOT=/path/to/artifacts \
bash recipe/phase1/run_exact_answer_only_50step.sh
```

Base 模型不进行训练。公平基线是在同一份无策略 validation 上直接评测，并关闭 LLM judge：

```bash
ANSWER_LLM_JUDGE=0 \
REWARD_PATH=recipe/core/my_reward_exact_only.py \
MODEL_PATH=/path/to/Qwen3-4B \
TEST_FILE=./data/hotpotqa_v3_no_strategy/validation.parquet \
VALIDATION_LOG=/path/to/base_greedy.log \
VALIDATION_OUTPUT_DIR=/path/to/base_greedy \
VAL_K=1 VAL_TEMPERATURE=0 \
bash recipe/eval/run_pretrained_baseline.sh
```

固定 200 条结果中，Base 的 Exact/Strategy 为 `0.560/0.405`，Exact-only 50-step 为 `0.535/0.235`；而复合奖励 GRPO s100 达到 `0.590/0.915`。这说明稀疏的最终答案信号不足以稳定发现两种不同的搜索拓扑，过程奖励的主要贡献是策略学习，而不只是答案拟合。
