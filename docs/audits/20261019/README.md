# 语义文本流水线：数据与训练准入审核

本目录记录 2026-10-19 的数据发布和训练代码检查，只包含聚合计数、标签、相对路径和哈希，不包含医学原文、凭据或模型文件。实体基线已完成两轮正式训练，每条样本均访问两次，关系训练和完整评测由持久队列继续执行。实际运行状态见 `run-status.json`；`trainer-review.json` 保留初审时的状态快照。训练 loss 不代表抽取质量，本页不提供尚未完成的效果分数。具体参数见 [实验配置](../../../configs/experiments/semantic_full_20261019.json)。

| 文件 | 内容 |
|---|---|
| [semantic-data-release.json](semantic-data-release.json) | 冻结拆分、旧训练集泄漏、监督准入、完整评测分母、三种视图的 token 与更新预算。 |
| [label-distribution.json](label-distribution.json) | 训练及 tune 的标签计数、来源覆盖、零关系标注来源比例；不是模型结果。 |
| [trainer-review.json](trainer-review.json) | 掩码、EOS、累计梯度、曝光和保存核验；记录已修复的失败状态与文件 SHA 门禁。 |
| [augmentation-audit.json](augmentation-audit.json) | DeepSeek Flash 小批扩增与逐例复核的聚合记录；候选仍有语义错误，实际新增训练数据为零。 |
| [original-silver-semantic-audit.json](original-silver-semantic-audit.json) | 原银标的定向复核：同义词被写成上下位关系等问题；抽查不是总体错误率，也没有修改冻结 gold。 |
| [relation-distractor-design.json](relation-distractor-design.json) | 初版输入干扰实体方案，正确关系目标保持不变；存在明显位置捷径，仅作压力测试设计。 |
| [relation-distractor-canonical-design.json](relation-distractor-canonical-design.json) | 与 short_canonical 配对的干扰实体方案，保存真实位置及类型偏置统计；已编译和独立复验，尚未训练。 |
| [run-status.json](run-status.json) | 有时间戳的正式训练、完整评测及样本曝光状态。 |

## 数据来源与隔离

冻结来源共 1345 条：train 1215 条，包含 670 条自由文本和 545 条结构化记录；tune 130 条，包含 69 条自由文本和 61 条结构化记录。来源组分别为 545 和 61，无跨拆分来源组交集。606 个组的标题全部来自同组结构化输入，保存原记录 SHA 与原 user 字符切片；标题不从 gold 规范名补造。

旧实体训练文件的 1563 行中有 119 行来自当前 tune 的 69 条来源；旧关系训练文件的 1135 行中有 89 行来自当前 tune 的 49 条来源。本轮从冻结 train 来源重建，没有复用这两份训练文件。备用 858 来源拆分与当前拆分不能直接拼接，其训练部分也覆盖当前 tune 与历史评测来源，必须按当前来源身份重新隔离。

模型训练使用自由文本。结构化记录完整处理并单列确定性基线，避免大量简单结构样本掩盖自由文本结果。670 条自由文本全部经过编译审计，其中 2 条原有隔离来源不进入监督，得到 668 个实体训练来源和 542 个关系训练来源。

实体原有 6516 条标注，扣除隔离来源的 17 条、无来源支持规范名 9 条、精确 mention 不可用 4 条，并合并 2 个重复类型/规范名 ID，准入 6484 条。关系原有 5592 条标注，扣除隔离来源的 15 条、端点不可用 22 条、原有 pair 阻断 2 条和目标类型冲突 2 条，准入 5551 个证据监督原子。跳过项不是负例。

## 三种训练视图

| 视图 | 实体行 / tokens | 关系行 / tokens | 每轮更新数，实体 / 关系 |
|---|---:|---:|---:|
| full_source | 668 / 741104 | 542 / 1201053 | 84 / 68 |
| short_source | 1092 / 803065 | 951 / 1536807 | 137 / 119 |
| short_canonical | 1092 / 805249 | 951 / 1536807 | 137 / 119 |

更新数按单 GPU、batch size 1、梯度累计 8 计算，尾批保留。三种视图逐原子 Counter 相同，每轮均覆盖 6484 个实体原子和 5551 个关系证据原子。短窗口只由来源文本划分，3 个无法完整放入短窗的证据原子显式使用全上下文回退，统计包含 train 与 tune。关系窗口引用完整且已定位的实体表，证据必须存在于当前窗口。

完整视图最大实体序列为 3851 tokens、最大关系序列为 5815 tokens；短视图最大值分别不超过 1664 和 5140。所有训练行在 8192 内，无 prompt 或 target 截断。

相同 epoch 代表相同标注原子的遍历次数，**不代表相同计算量或优化步数**。short_source 每轮总 tokens 比 full_source 多约 20.5%，来自窗口增加后重复的提示与完整实体表。实体和关系 completion 分别只增加 424 和 409 个 tokens，恰好等于多出的 EOS 数；监督标签内容未增加。短窗口效果包含上下文划分、更新次数与序列终止次数的联合变化，不能单独归因为长度。short_source 与 short_canonical 的行数、原子数、EOS 数和更新数一致，适合比较该短窗口条件下的顺序变化。

当前正式基线设计为实体、关系各 2 个完整 epoch，从相同 base 分别训练 LoRA，固定取最终 checkpoint。预计更新数为实体 168、关系 136；实际以训练 receipt 的逐行访问次数为准。当前 runtime 的实体阶段使用完整来源，后续短实体训练 arm 必须披露或解决训练与推理上下文差异。

## 监督与标签覆盖限制

训练目标是语义 TAB 行，由程序组装 JSON。prompt 和 padding 标签为 `-100`，完整 completion 编码一次并附加一个 EOS。EOS 表示所提供标注序列结束，不证明未标注关系均为负类。没有把 UNKNOWN 转成 `NONE`、空银标监督或 DPO 负例。

实体中 Symptom、Disease、Diagnostic Criterion 合计 6377/6484，约 98.35%；Assessment、Medication、Examination、Assessment Scale 分别只有 1、2、3、4 条。27 种训练关系中前 5 种占 4436/5551，约 79.91%，8 种关系不超过 9 条。训练关系中未出现的 `suggests` 在 tune 有 2 条；这是覆盖缺口，不能据此从 tune 造训练样例。

原 train 自由文本中 124/670 条没有关系标注，tune 中为 18/69 条。由于银标不完整，这些来源不自动视作可靠“无关系”任务，因此当前关系 SFT 目标全部非空。模型可能仍有偏向输出关系的风险，需以完整评测验证。后续扩增应从允许的训练来源出发，按预先确定的标签覆盖计划补充有明确证据的少数类、方向/限定词对照和经审核的无关系任务；不要大量复制单例银标或把未标注 pair 当可靠负例。

## 评测与训练代码核验

tune 的 130 条原始记录与全部 gold 保留：自由文本 648 个实体、573 条关系，结构化部分单独分层。只有 572 个自由文本关系证据原子可用于同协议的监督编译，缺少定位的 1 条仍留在评测分母。历史 148 条属于已经查看过的回归集，不作为新的盲测集；标题增强必须另有原始来源映射证据。

正式质量指标使用实际实体输出进入关系阶段后的 `live.filtered.free_text.relation`：按类型、规范名、方向及关系标签严格比较去重集合，报告 TP、FP、FN、precision、recall 和 F1。对应 `live.raw` 的纯集合指标必须同时列出，以观察规则删除了多少正确与错误关系。`strict_raw` 额外把无效输出行计入 FP，是无效输出惩罚诊断，不与纯集合 F1 混称。旧评测快照若将该诊断字段列为 `primary_metric`，解读时仍以原始 `filtered`/`raw` 集合字段为准，不重算或修改任何分子、分母。

历史评测发布保留全部 148 条、979 个实体和 704 条关系。81 条自由文本中，57 条通过原文精确匹配及同源结构化记录验证标题，24 条无法验证标题的样本保留原文与全部 gold，不猜标题、不移除样本。其标题输入条件与过去实验不同，不能把分数差异全部归为模型提升。

训练入口逐行拒绝 tune、混合 stage 和重复身份；正式运行还必须提供审核发布清单中的数据 SHA。记录实际完成的 microstep 与每行曝光；正式完成要求每行至少达到整数 epoch 次数。LoRA 初末指纹必须变化，保存后的 safetensors 与内存参数逐项逐位比较。异常处理清除 `completed` 并标记失败，runtime 拒绝失败或 smoke adapter。

对安装的 Transformers 4.56.2 源码核对显示，Qwen3 的 loss 使用累计 microbatch 中非掩码目标 token 总数作分母，`ObservedTrainer` 没有再次除以梯度累计数。低 token 平均 loss 不能替代实体/关系精确率或 F1。代码核验与数据长度核验均不构成训练质量结论。

复现数据发布使用 `PYTHONPATH=agent python -m semantic.data --source-root <私有 repaired_v2> --output-dir <新目录> --tokenizer <本地 Qwen3 tokenizer> --max-length 8192`。发布器拒绝覆盖现有目录；完整私有输入用于重验来源切片与标注守恒。不要将派生窗口行数相加当成独立文档数量。
