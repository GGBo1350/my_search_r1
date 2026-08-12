# HotpotQA 数据难度分析：easy / medium / hard × bridge / comparison

> 基于原始 `hotpot_train_v1.1.json`（90,447 条训练样本）的统计与示例。
> 本文档用于理解本项目数据筛选（只保留 hard）、检索策略设计（bridge→串行、comparison→并行）以及 OPD 阶段 teacher 数据构造的依据。

## 1. 背景与目的

Search-R1 V3 使用 HotpotQA 作为多跳问答检索强化学习的数据源。原始 HotpotQA 中每个问题带两个独立维度：

- **level**：`easy` / `medium` / `hard` —— 回答难度；
- **type**：`bridge` / `comparison` —— 检索拓扑类型（bridge = 桥接两跳，comparison = 比较两个实体）。

理解这两个维度的关系，才能解释我们为什么：

1. 清洗后只保留 **hard** 子集（easy 检索即答，不需要检索策略）；
2. 训练/评测按 **bridge（串行）与 comparison（并行）** 拆分并配比；
3. OPD 阶段按类型训练两个专精 teacher。

## 2. 数据来源与字段

- 文件：`hotpot_train_v1.1.json`（本地 `D:\@DevCode\RAGProgram\data\HotpotQA\raw\`）
- 字段：`level`、`type`、`question`、`answer`、`context`（段落列表）、`supporting_facts`（支撑句坐标）、`_id`
- 本项目清洗后的数据：`hotpotqa_v3_hard_1600`（train 1600 = bridge 1200 + comparison 400；validation 200 = 各 100）

## 3. 分布（level × type）

| | bridge | comparison | 合计 |
| --- | ---: | ---: | ---: |
| easy | 14,466 | 3,506 | 17,972（19.9%） |
| medium | 46,074 | 10,740 | 56,814（62.8%） |
| hard | 12,451 | 3,210 | 15,661（17.3%） |
| 合计 | 72,991 | 17,456 | 90,447 |

要点：bridge 数量远多于 comparison（约 4.2:1）；难度上 medium 最多，easy 与 hard 相当。这与我们训练集"bridge 1200 / comparison 400"的配比逻辑一致——比较题天然少。

## 4. 定量特征对比

以下统计基于本地原始数据的全部 90,447 条：

| level / type | 支撑事实数(均值) | 涉及 gold 文档数 | 答案显式出现在 gold 文档(均值) | 答案字面出现在问题中 | 问题长度(字符) |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy / bridge | 2.28 | 2 | **1.37** | 3.2% | 178 |
| easy / comparison | 2.16 | 2 | **1.35** | 22.7% | 76 |
| medium / bridge | 2.44 | 2 | 1.27 | 0.8% | 98 |
| medium / comparison | 2.30 | 2 | 0.97 | 45.3% | 69 |
| hard / bridge | 2.45 | 2 | **1.20** | 1.7% | 98 |
| hard / comparison | 2.29 | 2 | **0.95** | 41.9% | 70 |

解读：

- **hop 数不是难度区分点**：所有难度基本都是 2 个支撑事实、2 个 gold 文档。区别在"证据的直接性"和"推理量"。
- **答案显式出现率是核心信号**：easy 的答案平均出现在 ~1.35-1.37 个 gold 文档中（几乎明写）；hard 只有 0.95-1.20（常常要跨文档推断/验证）。难度越高，答案越"不直接"。
- **答案泄漏**：bridge 题泄漏率很低（0.8%-3.2%）；comparison 题较高（22.7%-45.3%），因为问题本身常含实体名或 yes/no 结构，但这不是难度的主要来源。
- **问题长度**：bridge 题通常更长（98-178 字符），comparison 题更短（69-76 字符，直接列出两个实体）。

## 5. 三个难度的定性区别

| 难度 | 定义性特征 | 对检索策略的要求 |
| --- | --- | --- |
| easy | 答案/连接在支撑句里几乎直接写明；检索到正确文档即可作答 | 基本不需要多跳策略，单次检索即可 |
| medium | 两跳，但桥接实体直白（中间实体在两个文档中都出现）；比较题需提取信息但无陷阱 | 需要按顺序查两次，但不需要跨文档"验证" |
| hard | bridge：桥接实体需要跨文档确认（别名/间接关系）；comparison：存在陷阱（国籍、日期比较、否定） | 必须串行多跳（bridge）或并行双查再仔细比对（comparison） |

## 6. 十二条例子（每难度 4 条：bridge 2 + comparison 2，含中文翻译）

### 6.1 easy（4 条）

#### easy / bridge

**例 1**

> **Q**：In which American football game was Malcolm Smith named Most Valuable player?
> **Q 中译**：马尔科姆·史密斯（Malcolm Smith）在哪一场美式橄榄球比赛中被评为最有价值球员（MVP）？
> **A**：Super Bowl XLVIII（第四十八届超级碗）
> **gold**：Malcolm Smith (American football)、Super Bowl XLVIII
> - [Malcolm Smith] "Smith was named the Most Valuable Player of Super Bowl XLVIII after they defeated the Denver Broncos."
>   - 中译：在击败丹佛野马队之后，史密斯被评为第四十八届超级碗的最有价值球员。
> - [Super Bowl XLVIII] "Super Bowl XLVIII was an American football game between the ... Denver Broncos and ... Seattle Seahawks..."
>   - 中译：第四十八届超级碗是美联冠军丹佛野马队与国联冠军西雅图海鹰队之间的一场美式橄榄球比赛……

**为什么 easy**：答案被第一句支撑句直接点名，第二句只是确认它是一场比赛。

**例 2**

> **Q**：Dua Lipa, an English singer, songwriter and model, the album spawned the number-one single "New Rules" is a song by English singer Dua Lipa from her eponymous debut studio album, released in what year?
> **Q 中译**：英国歌手、词曲作者兼模特杜阿·利帕的同名首张录音室专辑（其中诞生了冠军单曲《New Rules》）是在哪一年发行的？
> **A**：2017（2017 年）
> **gold**：Dua Lipa、New Rules (song)
> - [Dua Lipa] "Her self-titled debut studio album was released on 2 June 2017."
>   - 中译：她的同名首张录音室专辑于 2017 年 6 月 2 日发行。
> - [New Rules (song)] ""New Rules" is a song by English singer Dua Lipa from her eponymous debut studio album (2017)."
>   - 中译：《New Rules》是英国歌手杜阿·利帕收录于其同名首张录音室专辑（2017）中的一首歌曲。

**为什么 easy**：两个文档都直接给出年份 2017，检索即得。

#### easy / comparison

**例 3**

> **Q**：750 7th Avenue and 101 Park Avenue, are located in which city?
> **Q 中译**：750 第七大道和 101 公园大道位于哪座城市？
> **A**：New York City（纽约市）
> **gold**：750 7th Avenue、101 Park Avenue
> - [750 7th Avenue] "750 Seventh Avenue is a 615 ft (187m) tall Class-A office skyscraper in New York City."
>   - 中译：750 第七大道是纽约市一栋 615 英尺（187 米）高的 A 级写字楼。
> - [101 Park Avenue] "101 Park Avenue is a 629 ft tall skyscraper in New York City, New York."
>   - 中译：101 公园大道是纽约州纽约市一栋 629 英尺高的摩天大楼。

**为什么 easy**：两个文档都直接给出同一答案，取交集即可。

**例 4**

> **Q**：Which private research university is located in Chestnut Hill, Massachusetts Boston College or Stanford University?
> **Q 中译**：波士顿学院和斯坦福大学，哪所私立研究型大学位于马萨诸塞州的栗树山（Chestnut Hill）？
> **A**：Boston College（波士顿学院）
> **gold**：Boston College、Stanford University
> - [Boston College] "Boston College ... is a private Jesuit Catholic research university located in ... Chestnut Hill, Massachusetts..."
>   - 中译：波士顿学院是一所私立耶稣会天主教研究型大学，位于马萨诸塞州栗树山……
> - [Stanford University] "Stanford University ... is a private research university in Stanford, California..."
>   - 中译：斯坦福大学是一所位于加利福尼亚州斯坦福市的私立研究型大学……

**为什么 easy**：只需确认哪个大学在 Chestnut Hill（波士顿学院），另一个是干扰项，证据直接。

### 6.2 medium（4 条）

#### medium / bridge

**例 5**

> **Q**：The Oberoi family is part of a hotel company that has a head office in what city?
> **Q 中译**：奥贝罗伊家族（Oberoi family）所属的酒店公司总部位于哪座城市？
> **A**：Delhi（德里）
> **gold**：Oberoi family、The Oberoi Group
> - [Oberoi family] "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group."
>   - 中译：奥贝罗伊家族是一个以经营酒店闻名的印度家族，主要通过奥贝罗伊集团（The Oberoi Group）从事酒店业务。
> - [The Oberoi Group] "The Oberoi Group is a hotel company with its head office in Delhi."
>   - 中译：奥贝罗伊集团是一家总部位于德里的酒店公司。

**为什么 medium**：需要两跳（家族 → 公司 → 总部城市），但桥接实体 The Oberoi Group 在两个文档中都明确出现，无需猜测或验证。

**例 6**

> **Q**：Cadmium Chloride is slightly soluble in this chemical, it is also called what?
> **Q 中译**：氯化镉（Cadmium Chloride）微溶于这种化学物质，这种物质又叫什么？
> **A**：alcohol（酒精）
> **gold**：Cadmium chloride、Ethanol
> - [Cadmium chloride] "It is a hygroscopic solid that is highly soluble in water and slightly soluble in alcohol."
>   - 中译：它是一种易吸湿的固体，易溶于水，微溶于酒精。
> - [Ethanol] "Ethanol, also called alcohol, ethyl alcohol, and drinking alcohol, is a compound and simple alcohol..."
>   - 中译：乙醇，又称酒精、乙醇、饮用酒精，是一种化合物和简单的醇……

**为什么 medium**：需要把"这种化学物质（alcohol）"和"又叫什么（alcohol=ethanol）"连起来，两跳但连接词明显。

#### medium / comparison

**例 7**

> **Q**：Which magazine was started first Arthur's Magazine or First for Women?
> **Q 中译**：亚瑟杂志（Arthur's Magazine）和《First for Women》杂志，哪一本创办得更早？
> **A**：Arthur's Magazine（亚瑟杂志）
> **gold**：Arthur's Magazine、First for Women
> - [Arthur's Magazine] "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia..."
>   - 中译：亚瑟杂志（1844–1846）是一本美国文学期刊，19 世纪在费城出版。
> - [First for Women] "First for Women is a woman's magazine published by Bauer Media Group in the USA."
>   - 中译：《First for Women》是由鲍尔媒体集团在美国出版的一本女性杂志。

**为什么 medium**：需要提取两个创立时间并比较，但时间信息直白、无陷阱。

**例 8**

> **Q**：Were Pavel Urysohn and Leonid Levin known for the same type of work?
> **Q 中译**：帕维尔·乌雷松（Pavel Urysohn）和列昂尼德·莱文（Leonid Levin）是否以同类型的工作著称？
> **A**：no（否）
> **gold**：Pavel Urysohn、Leonid Levin
> - [Pavel Urysohn] "Pavel Samuilovich Urysohn ... was a Soviet mathematician ... best known for his contributions in dimension theory, ... topology."
>   - 中译：帕维尔·乌雷松是一位苏联数学家，以维数理论以及乌雷松度量化定理、乌雷松引理等拓扑学基础成果著称。
> - [Leonid Levin] "Leonid Anatolievich Levin ... is a Soviet-American computer scientist."
>   - 中译：列昂尼德·莱文是一位苏裔美国计算机科学家。

**为什么 medium**：需要判断两人职业是否相同（数学家 vs 计算机科学家），信息直白，只需读完两句。

### 6.3 hard（4 条）

#### hard / bridge

**例 9**

> **Q**：What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
> **Q 中译**：哪条美国国道通往齐尔波路（Zilpo Road），同时又被称作"米德兰小道"（Midland Trail）？
> **A**：US 60（美国 60 号国道）
> **gold**：Zilpo Road、Morehead, Kentucky
> - [Zilpo Road] "The nine mile byway starts south of Morehead, Kentucky and can be accessed by U.S. Highway 60."
>   - 中译：这条九英里长的小路始于肯塔基州莫尔黑德（Morehead）以南，可通过美国 60 号国道到达。
> - [Morehead, Kentucky] "Morehead is a home rule-class city located along US 60 (the historic Midland Trail)..."
>   - 中译：莫尔黑德是一座位于美国 60 号国道（即历史上的米德兰小道）沿线的城市……

**为什么 hard**：桥接链为 Zilpo Road →（Morehead）→ US 60 = Midland Trail。"Midland Trail" 别名只出现在第二个文档，必须跨文档确认；且桥接实体 Morehead 不是问题里的词，要靠检索发现。

**例 10**

> **Q**：Guitars for Wounded Warriors is an album that was recorded in the village in which New York county?
> **Q 中译**：专辑《Guitars for Wounded Warriors》是在纽约州哪个县的村庄里录制的？
> **A**：Ulster County（阿尔斯特县）
> **gold**：Guitars for Wounded Warriors、New Paltz (village), New York
> - [Guitars for Wounded Warriors] "All tracks were recorded at Tarquin's Jungle Room Studios in New Paltz (village), New York."
>   - 中译：所有曲目都是在纽约州新帕尔茨（村庄）的 Tarquin's Jungle Room 录音室录制的。
> - [New Paltz (village), New York] "New Paltz is a village in Ulster County located in the U.S. state of New York."
>   - 中译：新帕尔茨是纽约州阿尔斯特县的一个村庄。

**为什么 hard**：需要"专辑 → 录音地（New Paltz）→ 所在县（Ulster County）"两级推断，第一跳结果（New Paltz）是第二跳的查询词，必须按顺序检索。

#### hard / comparison

**例 11**

> **Q**：Are both The New Pornographers and Kings of Leon American rock bands?
> **Q 中译**：新色情狂乐队（The New Pornographers）和莱昂国王乐队（Kings of Leon）都是美国摇滚乐队吗？
> **A**：no（否）
> **gold**：The New Pornographers、Kings of Leon
> - [The New Pornographers] "The New Pornographers is a Canadian indie rock band formed in 1997 in Vancouver..."
>   - 中译：新色情狂是一支加拿大独立摇滚乐队，1997 年成立于温哥华。
> - [Kings of Leon] "Kings of Leon is an American rock band that formed in Nashville, Tennessee..."
>   - 中译：莱昂国王是一支美国摇滚乐队，1999 年成立于田纳西州纳什维尔。

**为什么 hard**：陷阱题——一个是加拿大乐队，"both American" 为假；必须把两个文档都读全才能答对。

**例 12**

> **Q**：Who was born first, Pablo Trapero or Aleksander Ford?
> **Q 中译**：巴勃罗·特拉佩罗（Pablo Trapero）和亚历山大·福特（Aleksander Ford），谁更早出生？
> **A**：Aleksander Ford（亚历山大·福特）
> **gold**：Pablo Trapero、Aleksander Ford
> - [Pablo Trapero] "Pablo Trapero (Born 4 October 1971) is an Argentine film producer, editor and director."
>   - 中译：巴勃罗·特拉佩罗（生于 1971 年 10 月 4 日）是阿根廷电影制片人、剪辑师和导演。
> - [Aleksander Ford] "Aleksander Ford (born Mosze Lifszyc; 24 November 1908 in Kiev, Russian Empire – 4 April 1980...)"
>   - 中译：亚历山大·福特（本名 Mosze Lifszyc，1908 年 11 月 24 日生于基辅……1980 年 4 月 4 日……）是波兰犹太裔电影导演。

**为什么 hard**：需要分别提取两个出生日期（1971 vs 1908）再比较，任一文档单独都不够，必须精确读取。
## 7. 对项目（Search-R1 V3）的意义

1. **只用 hard 子集**：easy 检索即答、medium 桥接直白，都不需要真正的多跳检索策略；只有 hard 才能区分"会组织检索"和"不会组织检索"的模型。
2. **按 type 设计策略**：bridge → 串行多跳（先查 A 找桥接实体，再查 B）；comparison → 单轮并行双查。这也对应项目"复现 Search-R1 + 单轮多查改进"的核心设定。
3. **训练配比**：hard 子集 train 1600 = bridge 1200 + comparison 400（比较题天然少，按比例少配）；validation 200 = 各 100，保证两类评测均衡。
4. **OPD 阶段**：按 type 拆分后的 `train_bridge_1200.parquet` / `train_compare_400.parquet` 直接用于训练两个专精 teacher；最终模型用统一 prompt 在通用数据上蒸馏。

## 8. 附：hard 子集额外统计（`hotpotqa_hardness_stats.json`）

- hard 的 gold 支撑文档数全部为 2（bridge 12,451 / comparison 3,210）。
- hard 中答案字面出现在问题里的比例：bridge 222/12,451 ≈ 1.8%；comparison 1,307/3,210 ≈ 40.7%（比较题多为实体名/是否类问题，泄漏率天然高）。
- hard 中 gold 文档包含答案的分布：bridge 集中在 1 个文档（10,541/12,451），comparison 有 770 条（24%）**没有任何** gold 文档直接包含答案——必须靠推断。