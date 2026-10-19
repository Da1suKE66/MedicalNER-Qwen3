# 迁移后项目审核与关系抽取恢复计划（2026-10-12）

本轮结论：关系低分是真实的预测退化，不能靠修改 F1 公式消除。最强的训练侧解释是 V2 将未标注候选改为 UNKNOWN 后，base 没有补充受监督的可靠“不连边”样例，同时保留大量待模型生成的候选决策；实体覆盖不足、银标缺陷和训练/推理输入分布差异进一步限制结果。当前证据尚不能把全部退化归因于一个因素。

本轮完成了迁移完整性校验、代码与测试审核、历史指标独立核算、数据分布重算及已确认工程问题修复。没有启动新一轮 GPU 质量实验，也没有将 CPU 测试或历史五篇预测的复算写成新的模型改进结果。

## 1. 审核范围与可复现依据

- 封存包：`medicalner-migration-20261012-v1`，679 个封存文件、2,349,709,500 字节 SHA256 校验通过；195 个 JSON 目录项严格解析通过；1345 条原始记录保留，train/tune 来源组重叠为 0。
- 深层检查：14,596 个 mention 映射、8,702 行 repaired partial 数据、两组各 1,845 行 V3 的真实 Qwen tokenizer、chat 模板、UNKNOWN 字符/Token 掩码检查通过。V3 最大长度 4,932 tokens，小于历史 6,144 上限。完整性不等于医学标签正确。
- 修复前基线：封存代码的 agent、原项目 tests、迁移 portable 测试共 **785 passed**。
- 工作分支从公开的 `codex/relation-abstention-v3-20260928` 恢复；封存代码原本未跟踪的 `evaluation_recovery.py` 及其测试也纳入记录。封存包不修改，新代码位于独立工作仓库。
- 公开证据目录：[audits/20261012](audits/20261012/README.md)。仅包含聚合计数与合成测试；医学原文、原始预测、数据包、权重、账号连接信息不随本次提交上传。

项目有三套不同历史路径，不能混用其提示词、schema、数据或 checkpoint：

| 路径 | 用途及审核结论 |
|---|---|
| `src/kg_lora`、根目录 LF 配置 | 早期一阶段 CoT/schema_v2 与 858 系列；数据清洗、转换、结构测试纳入测试集。其 schema 和 1345/两阶段实验不等价，旧 README 不是当前 relation 运行手册。 |
| `agent/kg_agent`、`train_sft.py`、classifier | 旧两阶段推理/训练与分类器历史路线；发现并修复候选前缀截断、入口绕过监督门禁等问题。保留旧模板以便历史回放。 |
| `agent/generative`、`data_repair`、`robustness` | 当前主线；重点审核多关系协议、UNKNOWN mask、负例、训练保存、身份对齐、图评分、规则与 RL。 |
| `history`、冻结数据、迁移 `run.py` | 证据及历史执行入口。`run.py` 仍引用封存代码，不会自动使用本分支修复；新实验应使用本仓库的 `agent/`。 |

没有逐条完成人工医学仲裁，也没有在新服务器重跑旧 GPU 作业。旧 adapter/checkpoint 不在迁移包中；只有底座模型存在不能完成旧模型零步回放。

## 2. 低分的定位与归因

### 2.1 V2 出现“几乎每对都连边”，不是算术误差

以下均是 **81 条 free_text 路由样本、638 条去重有向银标关系**（其中含 4 条裸 JSON 名称列表）的历史指标；67 篇 ICD 的确定性解析单独报告。fixed 表示固定历史 entity cache，不能称实时端到端。

| 历史条件 | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| N0，fixed | 74 / 560 / 564 | 11.67% | 11.60% | 11.64% |
| N0，oracle entity | 352 / 683 / 286 | 34.01% | 55.17% | 42.08% |
| V2 base，fixed | 85 / 7949 / 553 | 1.06% | 13.32% | 1.96% |
| V2 base，oracle entity | 440 / 13731 / 198 | 3.10% | 68.97% | 5.94% |

本次独立核算了 20 份历史报告的逐样本 TP/FP/FN → overall/free_text/ICD 汇总 → P/R/F1，全部一致。四组 V2 完整原始 predictions/generations 没有迁移，因此不能说已经用新 parser 从原始输出重算全部历史实验。

另外，迁移保留的 **同五篇、同 V2 prompt** 原始预测可独立复算：V2 为 21/113/2，F1=26.75%；历史 N0 为 12/5/11，F1=60.00%。134 个候选决策中，V2 空关系 0 个，N0 空关系 117 个；都没有输出截断。这是定向诊断子集，不是新测试集平均性能。

V2 四组存在实际权重更新、保存/重载验收及相应训练步数。现代入口共用非 thinking chat 模板，prompt/pad/UNKNOWN mask、长度拒绝与梯度累计转发未发现对应的确定性错误。因此“模型没训练到”“只因输出截断”“全是 F1 算错”都不符合现有证据。

### 2.2 拒绝错误连边的监督丢失：强假设，需单变量实验确认

| 数据版本 | 训练行数 | 正例 pair | 受监督负例 pair | UNKNOWN pair | 行级特征 |
|---|---:|---:|---:|---:|---|
| V2 base | 1333 | 4765 | 0 | 14049 | 每行均含正例 |
| V2 noise10 | 1333 | 4765 | 270 | 14049 | 仅 135 行混入合成负例 |
| V3 missing-only | 1845 | 5789 | 2484 | 16685 | 1431 行仅正例参与监督、414 行正负混合；均仍含 UNKNOWN |
| V3 grounded-none | 1845 | 5789 | 3900 | 15269 | 1333 行仅正例参与监督、512 行正负混合；均仍含 UNKNOWN |

两组 V3 **都没有已确认的整批无关系训练行**。V2 的 UNKNOWN mask 避免了“漏标=负例”，这是必要保护；但它没有提供相反的学习信号。训练时 UNKNOWN 空占位还会作为 teacher-forcing 上下文，推理时模型须自己生成这些 decision，可能产生额外分布偏移。不能用移除 mask 的方式修复，否则又把未知关系当负例。

V3 虽补充了可靠构造负例，但尚未训练，且两个版本的受监督 completion tokens 分别为 224,478 / 243,200，不能宣称完全等预算。

### 2.3 Entity 与 grounding 是第二个独立瓶颈

本次用当前代码重新构造候选：fixed entity F1=46.38%，可触达银标关系 **120/638=18.81%**；oracle entity 可触达 **546/638=85.58%**。oracle 仍漏的 92 条关系均涉及 24 个无法原文定位的端点。

18.81% 是当前候选下的 recall 上限，不是 F1 上限。即使 precision=100%，对应最高 F1 也只有约 31.66%。改 relation 的学习率无法凭空恢复缺失端点。gold 实体也可能是规范名/别名而非原文表面形式，需要保留 canonical name 与 mention 的映射，不能把定位失败自动宣布为假实体。

### 2.4 数据质量和分布

- 1345 条源记录：739 自由文本、606 ICD。自由文本含 7164 实体、6165 关系；251 实体无法原文定位，牵连 895 关系（14.52%）。
- 原待审 264 项来自 256 条记录：251 实体定位、2 证据、11 语义/原文问题；目前仍需仲裁。
- 197 条关系引用有多个 exact 命中，其中 188 条进入 base 正监督（train 150、tune 38）。重复句并不一定错，但当前引用没有区分疾病/段落，不能把所有命中同时当证据。
- 多关系实体对只有 **30 个**，三个稀有关系类别各只有 1 条。只能做描述性结果，无法从单条验证稳定泛化。
- 已有 train/tune 规范化同文交叉数为 0；一组同文冲突涉及的两条记录已隔离。
- 旧 `train_entities_v4` 含内部 tune 的 **119 行/69 条记录**；旧 relation stage 含内部 tune 的 **89 行/49 条记录**。重训 entity 必须按冻结 source/group 删去这些 tune 数据。这里不是指外层 148 回归集泄漏。
- 旧 missing-entity 注入发生在 grounding 之后，而正常在线构造器会先剔除原文不存在的实体。这种容易负例可测试协议容错，但不能替代真实 NER 错误、错疾病绑定、否定范围和多关系的挑战集。

## 3. 已确认代码问题及本轮处理

| 严重度 | 定位 | 后果与处理 |
|---|---|---|
| P1 | `agent/generative/protocol.py::parse_decisions` | 重复 pair_id 的后续新关系被丢弃，会漏计 FP。修复为保留独特关系参与评分，同时报告 duplicate_pair；相同三元组仍去重。历史仅发现少量重复，不能解释 V2 大幅崩塌。 |
| P1 | `agent/evaluate.py`、`evaluate_graphs.py` | None/空名实体被静默忽略，重复 ID 可覆盖端点。坏实体逐条计 FP，歧义 ID 不绑定关系，坏 gold 报错；closure 明确只是顶层 JSON 成功率。 |
| P1 | `agent/train_sft.py` | 旧 CLI 绕过新 partial/zero-negative/split 门禁并使用裸 prompt。现在拒绝新生成协议与 partial 数据、非 train 显式标签、stage 错配和非空输出，指向 `generative.train`。旧无 split stage 回放仍可用，但不认证来源隔离。 |
| P1 | `data_repair/supervision_v2.py` | 实体类型与 allowed 表不同步仍可通过校验。新增 ID、类型、候选端点和 ontology 表一致性检查。 |
| P1 | `data_repair/release_v2.py`、`supervision_v2.py` | 多位置证据错误地合并为正监督。新构造默认版本化 `unique-evidence-v1`，不明确整对变 UNKNOWN 并进入待审；显式 legacy 策略可精确重建旧数据。 |
| P2 | `agent/robustness/noise_evaluation.py` | 只按 zip 位置评分。现在强制生成 ID 一致；新输出附 prompt hash，已提供的 hash 必须一致。旧无 hash 产物不能宣称完整身份已验证。 |
| P1 | `agent/kg_agent/pipeline.py` | 旧主入口先截前 128 对再分批，后半候选永久丢失。现在默认所有候选分批，显式 cap 仍可回放并在 trace 报遗漏量。现代 generative 路线原已无此 cap，不能用此解释 V2。 |
| P2 | `agent/kg_agent/prompts.py` | 旧 prompt 要求同对选一个关系。新增 `--relation-multilabel` 显式模板选项；默认旧 prompt 保持字节兼容。现代 generative 本来支持多关系。 |
| P1 | `agent/generative/build_data.py` | 旧构造器默认把银标未标注对填空列表。CLI 现在必须显式 `--assume-closed-world-silver` 才能历史重建；新数据使用 UNKNOWN-aware release。底层旧 supervision 保留供历史/穷尽合成样例使用。 |

证据策略变更是标签可用性变化，必须视为新数据版本。实际只读对比：train 1333→1323 行，4785→4635 个正关系标签；tune 325→297 行，482→444 个标签。这里统计的是关系标签，不是上表的正 pair。没有新增负标签，也没有改完整图 gold 或删除评测分母。历史 12,392 行 V2/V3 仍可读取验证；legacy 重建与原 base 行完全一致。

保留的显式限制：旧 `run_experiment.py` 和恢复脚本有原服务器路径/GPU0 锁、旧多 epoch checkpoint 保留假设，不应直接作为新单卡调度器。旧默认 128 对/768 输出 tokens 仍可能解码截断；本轮消除候选丢失不等于解决输出预算。`contracts.py` 的 ontology 明确采用 source-type-only，target-type 较宽；不能未经 schema 审定直接换成上方旧 signature 表。

## 4. 用户所说的 negative 应怎样实现

旧 relation stage 确实有 **8017 个字面 NONE**。V2/V3 改成 `relations: []` 后，只换了表示形式；这不是核心区别。核心是哪些输入被确认不应产生边，以及 loss 是否教会模型省略这些边。

推荐新建 `sparse-positive-v1`，与当前 dense-decision-v1 做同数据对照：

1. 输入保留原文、entity stage 得到的候选，以及有明确来源的干扰实体；干扰项不进入正确 entity 金标，也不给它编造 relation 输出。
2. 输出仅包含有证据的有向关系。同一实体对允许多条不同关系。可靠的全负批输出 `{"relations":[]}`，不输出名为 NONE 的关系。
3. 从 train-only 的按来源组 OOF entity 输出获得真实错误候选，覆盖“实体在原文中但与该疾病无关”等困难情况；原文缺失的简单干扰只是一个分层。
4. 普通银标缺失仍是 UNKNOWN。未知对不能仅因没有记录就当成可监督省略。第一轮 sparse 仅用于候选范围已经穷尽审定的小批或可证明的构造文本；不把整个未审定文档所有漏标关系作为负监督。
5. 对 uncertain 的旧语料，可继续 dense partial-mask 路线；不要直接从现有 completion 删除负 decision，因为旧 parser 将报 missing_pair_decisions。新协议必须同时实现 builder/prompt/parser/loss validator/evaluator 适配及版本检查。
6. 负例标签表示“这段输入没有支持这条关系”，不是“医学世界不存在此关系”。来源、构造证据、适用候选范围和人工状态存入训练外元数据，不泄漏进 prompt。

本轮没有伪装为已经实现或训练了 sparse 协议。它是下一阶段首个需验收的改动。也不能把干扰 pair 的 loss 全 mask 后声称完成了负例训练。

## 5. 之前做了什么、缺少什么

| 方向 | 已有实验与结果 | 仍缺的关键对照 |
|---|---|---|
| 短输出 | G2 fixed/oracle F1=10.30/30.36%；G6 组合=12.03/43.98% | G2 监督 pair 曝光 17702 vs N0 59088；G6 933 vs 279 steps，不能单因果归给短输出。 |
| 短输入 | V2 full_common vs local_common 同样本监督；prompt tokens -20.03%、耗时 -24.75%，质量仍低 | 在监督正常的新基线上复做；必须单报窗口丢证据比例、全图 FN。 |
| 顺序 | G3 source-order=8.75/28.43%，并非未做；有截断 | 固定训练覆盖、曝光与解码预算，比较首次出现顺序和输入驱动固定随机顺序；禁用标签/正例优先排序。 |
| 多关系 | 现代输出已支持集合；G4 重采样=11.01/41.16% | 小规模真实多关系审定集、每对集合准确率/标签召回；数据只有 30 对，需补充。 |
| 规则 | 历史有过滤；V2 oracle 过滤后约 6.27–6.71%，且删 TP | 同一原始输出上逐条报告删 TP/FP、FN 增量、候选覆盖；语义规则在 tune 锁定。 |
| N10/N25 | fixed/oracle 约 10.99/40.89 与 11.17/42.06%；人工噪声上有部分改善 | 真实 OOF NER 错误、原文内干扰、全负批；clean/noisy 必须用共同 41 条未受干扰的关系比较保留率，不能把 47 与 41 个关系标签混分母。 |
| DPO | beta .05/.1 已训练，fixed/oracle=8.86/29.42、7.75/23.99%，误边增加 | 同 loader 零步 SFT、偏好审定、真实当前策略错例、预算/种子匹配。no-hard 训练完成但 tune 超时，不算完整结果。 |
| GRPO / PPO | GRPO 两组已各 384 steps/6144 rollouts，未超过基线；未见标准 PPO 完成记录 | partial-safe 奖励、同 loader 零步控制、奖励作弊验证；PPO 不作为首要任务。 |
| 全流程/统计 | fixed、oracle、合成压力集与单 seed 居多 | 实时 entity→relation、可信 cache 身份、新盲测人工 gold、3 seeds、按来源组 bootstrap。 |

上表 fixed/oracle F1 数值以百分数表示，均为 relation micro F1；步数、曝光量和 beta 等仍按各自单位。这些数据不是新服务器重跑结果。

## 6. 新一轮训练与评测计划

### 阶段 A：先冻结数据、协议和评测

- 完成 264 待审和新增证据歧义队列的 train/tune 分离仲裁；保存原标签、新标签、理由、证据位置和审阅者，保留 UNKNOWN。不让 tune 仲裁内容成为训练负例词库。
- 对旧 stage 按 source_index/group 重新生成 entity train。新盲测建议先收集不少于 100 个新来源组，覆盖长短、否定、跨句、多疾病、多关系；分层记录样本量，无法达到则明确试点不足。先冻结后训练。
- 在候选穷尽审定的小批中实现 sparse-positive-v1 与 dense 决策双构造。相同候选、正边集合、可靠负例、来源组、模板语言和模型起点；全负序列比例试点 0/10/25%，不是从未标注文档随机制造全负。
- 精确版本独立环境：历史 relation 主线 torch 2.8.0 / transformers 4.56.2 / tokenizers 0.22.2 / peft 0.17.1；RL 另需 trl 0.23.1。现有 LLaMA-Factory 环境版本不同，不能不验证就混用。普通 LF SFT 尚未实现这里的 UNKNOWN span mask，不可直接导入当成等价实验。

### 阶段 B：先证明拒绝连边恢复，再讨论超参

同一 Qwen3-8B revision、BF16 LoRA r16/alpha32/dropout .05，非 thinking 模板。48GB 单卡从 batch1×accum16、max_length6144、lr5e-5 起步；历史 BF16 配置峰值约 35.6GiB 仅供参考，新卡仍先测。开启2-step前后向/保存重载和长度审计，不静默截断。

| ID | 实验 | 预算与选择 |
|---|---|---|
| B0 | 裸底座零步；若找回 N0 adapter，同 loader 零步重放 | 同一冻结 tune，固定 seed、temperature0、候选、decode、规则；保存原始输出。 |
| B1/B2 | 相同 dense 输入与正例，可靠负例 loss off/on | off 仅显式诊断；匹配正例曝光与 optimizer updates，分别记录负例/token 暴露差异，不能称总监督相同。 |
| B3/B4 | 可靠全负序列比例 10% / 25% | 两臂都用正确负例，不用错标替代；固定正例曝光，披露新增总算力。 |
| B5 | dense vs 用户要求的 sparse 输出 | 相同已审定候选集合和正/负信息；输出 tokens 必然不同，同时报告质量与资源。 |
| B6 | 恢复不连边后的 LR=1e-5/2e-5/5e-5 | 先各64-step有限试点；选不退化的两组到1 epoch，再由完整 tune 决定是否到3 epochs。 |

64 steps 只排除塌缩，不用其 loss 宣称最终收益。每个实验保存全部阶段 checkpoint 或只评测明确存在的 checkpoint；不使用“训练 4 epochs 却只保留3个”旧调度假设。若新模型几乎全连边/全空、有效输出坏、输出截断明显增加，停止扩展并检查原始预测、loss mask 和数据配比，不靠改 gold 或过滤测试样本修分。

### 阶段 C：长度、顺序、规则、实体与多关系

1. 在 B 阶段可信基线上比较 16/8/4 对每次决策。按监督正/负 pair 的真实曝光匹配主预算；尽量用累计梯度对齐 updates，不能兼得时提供等曝光和等算力两张表。记录 context tokens、supervised tokens、updates 和 GPU 秒。
2. 相同 pair 和可监督正例比较完整原文/局部窗口。窗口只由输入实体/原文决定，不能偷看 gold evidence；上下文缺证据作为独立分层而非删出评测。
3. 首次出现顺序 vs 固定随机顺序。实体重编号不暴露 gold，canonical output 顺序稳定；另做无标签 ID/输入排列扰动测试，检查位置捷径。
4. 每条规则单独开关；有效 ID、出处定位、重复边是结构层，source+target 类型、否定、疾病范围属于待审语义层。先估删 TP/FP 再组成规则集，原始图和最终图并列。
5. 重新训练清洁 entity，并在 train-only 三折来源组 OOF 预测上训练 relation。报告实体识别、grounding、候选生成、relation 四段损失；最终只以实时全流程评价用户体验。
6. 多关系保留集合，补足审定案例。重采样上限、每类支持数、单/多关系分层全部显式；只改排序无法弥补监督稀缺。

### 阶段 D：仅在 SFT 稳定后再做偏好优化

先收集选中 SFT 在 train-only 的真实错误，平衡错疾病/无支持边/否定/遗漏偏好，人工确认 chosen/rejected 不改变输入任务。未审定银标 FP 不直接当 rejected。DPO 从16/32/64-step、lr≤1e-6、beta .05/.1 小试开始，并配完全相同 loader 的0-step对照。偏好长度/答案边数分布必须审计，防止学成一律多连或一律不连。

UNKNOWN 数据不能直接进入现有 RL reward；先实现 partial-safe reward 并检查奖励作弊。GRPO 需记录 KL、语义分项、采样数、全空/全连边率，PPO 在单卡资源和价值证据不足时排最后。DPO/GRPO 历史失败意味着此阶段不是默认优先路线。

算法边界参考：[DPO 原论文](https://arxiv.org/abs/2305.18290)、[PPO 原论文](https://arxiv.org/abs/1707.06347)、[本项目固定版本的 TRL GRPO 文档](https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer)。这些方法本身不保证本项目的医学抽取质量，是否有效以冻结数据上的实际对照为准。

基线与最终候选重复 seeds 17/42/73；按来源组做配对 bootstrap（建议2000次），报告 ΔP/ΔR/ΔF1 区间。选择只使用 tune：延用 P≥50%、R≥10% 的预注册资格门槛后最大化 F0.5，同时报告 F1；全不合格就写“无合格模型”，不能把回退最优包装成达标。

## 7. 指标契约：防止“准确率/F1 看着正常却错了”

- 主图指标：每文档用 `(source类型+规范名, relation, target类型+规范名)` 的**有向集合**匹配。规范化仅 NFKC、大小写、空白；不做无证据同义词合并。同对不同关系分别计数，相同三元组去重。
- 先按样本算 TP/FP/FN，再全量加总：`P=TP/(TP+FP)`、`R=TP/(TP+FN)`、`F1=2TP/(2TP+FP+FN)`。不要平均各批次 F1 当 micro F1；比例与百分数显示明确区分。
- **Precision（精确率）不是 Accuracy（准确率）**。开放图抽取没有自然 TN，主报告不造 accuracy。只在候选全集冻结且正/负穷尽审定的闭集上报告 pair accuracy/exact-set accuracy，UNKNOWN 排除并显示排除量；同时给 always-empty 基线防类不平衡误导。
- 每篇所有 gold 关系始终进入分母。没有实体、没进候选、窗口外、解析失败或截断造成的遗漏均为 FN；不能只在触达候选上算主 F1。非法边计 FP，坏实体计 FP，缺行/身份错位报错。
- fixed cache、oracle、实时 E2E 分表；自由文本与 ICD 分表；严格原始 gold 与仲裁后的新版本分表，禁止静默替换。
- 每次报告样本数、gold/predicted 数、TP/FP/FN、P/R/F1/F0.5、各类 support 与 F1、macro 支持类别集合、候选覆盖、evidence exact/compatible、无效/截断/空输出率、延迟/tokens/显存。evidence compatible 只是文本片段兼容，不等于语义正确。
- 每行保存输入 hash、原始 sample_index、模型/adapter hash、prompt/protocol/代码版本、原始生成和组装输出。新正式评测用 strict identity；旧 cache 保留 unverified，不能补个新 hash 后升级成已验证历史来源。
- 所谓“幻觉”至少拆成 silver 不匹配 FP、原文不支持率、错疾病绑定率、无效 ID/证据率。后两类需要可追踪的裁定；银标 FP 不是医学上错误的同义词。

当 F1 很低，按“身份与分母 → 原始输出/解码 → 实体候选上限 → pair 拒绝分布 → label/方向 → 证据/标注”顺序定位。当前 V2 正是：算术无误、无截断、oracle 也差、FP 暴涨、不拒绝连边；因此先修监督与输入分布，再调参。

## 8. 本地复现与交接

从仓库根目录，在独立 CPU 环境安装 [requirements-audit.txt](../requirements-audit.txt)，然后运行：

```bash
python -m pytest -q -p no:cacheprovider agent/tests tests
python scripts/audit_migration_metrics.py --migration-root ../medicalner-migration-20261012-v1 --output ../audit_work/recomputed-metrics.json
python scripts/audit_relation_supervision.py --migration-root ../medicalner-migration-20261012-v1 --output ../audit_work/recomputed-supervision.json --compare-evidence-policy
```

私有集成测试可用 `REVIEW_TRIAGE_REAL_CANDIDATE=../medicalner-migration-20261012-v1/data/repaired_v2` 显式开启。迁移 `verify.py --deep --tokenizer` 从封存包运行用于验证原包；代码修改的测试则从本仓库运行。

新训练必须先冻结本计划的数据和协议版本。V3 历史臂可直接用本仓库 `PYTHONPATH=agent python -m generative.train --mode sft --data <冻结V3路径> --base-model <本地Qwen3-8B> --output-dir <新目录> --max-steps 64 --save-steps 64 --verify-save` 进行**历史 dense 试点**；它不是 sparse 新方案，也不会自动修复 V3 中旧证据歧义。完整审核后的新实验应重新发布版本并记录差异。

最终状态：工程缺陷已修复并附回归测试；实验缺口和依赖已明确。人工仲裁、sparse 新协议训练、实时 E2E、新盲测以及重复种子仍是后续工作，未声称已完成。
