# 六小时方向筛选：2026-10-12 14:20—20:20（北京时间）

用户要求先在五到六小时内形成多个方向的可比较证据，明确合理方向和仍有问题。该轮是对 [完整交付路线](relation_delivery_plan_20261012.md) 的优先级调整；不把子集筛选、1遍小数据训练或16步DPO当最终模型。

## 实际运行状态

- 原P0 `grounded_s42` 已完成64步，权重变化、保存与重载验收通过。adapter位于 `/workspace/medicalner/20261012/pilot/grounded_s42/train/final`。
- 原P0的 `base-only-tune` 因用户新时限要求主动中断；原队列status为failed/signal15，另有 `pilot/replan_for_six_hour_screen.json` 说明。这不是模型训练失败，也不是完整tune结果。完整130条和压力评测仍待后续补齐；已生成raw_stream保留。
- `missing64` 使用原冻结数据、相同64步和全部训练参数完成训练，权重更新、保存与重载验收通过；耗时1673.8秒，loss0.07989（不是评测成绩）。输出 `/workspace/medicalner/20261012/rapid/missing64`，adapter指纹 `4ad1271d47f3093f9e4278edd3258467c988939c8721c55dc473a0a5da1dc084`。
- 原快速快照v1漏带 `schemas/v2.0.0/schema.json`，三组医疗评测在生成预测前失败。队列已主动停止；DPO零步导出完成，刚开始的DPO16中断。失败日志、状态与产物保留在 `rapid/screen_v1`，不能计为有效评测。
- 补齐schema并单独校验SHA后的配置为 [rapid_screen_20261012_v2.json](../configs/experiments/rapid_screen_20261012_v2.json)，冻结快照 `rapid/snapshot_v2`，结果 `rapid/screen_v2`。增加从实际解压快照隔离导入的CPU预检，覆盖四种医疗条件、机械任务和训练数据。实验数据、模型、顺序、超参及截止时间不变；v1配置继续留存用于复现失败。
- v2本机及服务器预检均通过，队列已重启并写出首条真实生成记录。预检脚本 [validate_rapid_snapshot.py](../scripts/validate_rapid_snapshot.py) 有4项回归测试，收据 [rapid-snapshot-v2-preflight.json](audits/20261012/rapid-snapshot-v2-preflight.json) 仅验证CPU数据与解码契约，不是模型效果。统计报告脚本的12项测试与预检4项测试联合通过。
- 队列硬截止19:35，留45分钟给下载、指标复算和结论整理；20:20前提供完成/失败/未完成清单。阶段超时保留数据、不排名；依赖失败则明确跳过，独立阶段继续。没有数据支撑的方向写证据不足，不赶时间造结论。
- 本任务15分钟一次的有限自动跟进已创建（id `ner`），只处理这轮筛选，截止后报告并暂停；不是无限期P0—P6自动训练。本机Codex调度需可用，服务器tmux队列独立运行。

## 比较内容与边界

截至15:29已有六个完整条件，经逐条预测独立复算通过；使用同12篇oracle实体子集，全部40条既有gold（银标评测标签）计入。以下是关系micro precision/recall/F1，不是Accuracy，不是完整流水线或最终模型表现。

| 条件 | TP / FP / FN | Precision | Recall | F1 | 加相同规则后F1 |
|---|---|---|---|---|---|
| grounded64 原顺序16对 | 9 / 175 / 31 | 4.89% | 22.50% | 8.04% | 15.00% |
| grounded64 实体顺序16对 | 14 / 207 / 26 | 6.33% | 35.00% | 10.73% | 18.67% |
| grounded64 短输出4对 | 10 / 176 / 30 | 5.38% | 25.00% | 8.85% | 15.75% |
| grounded64 短输入16对 | 10 / 160 / 30 | 5.88% | 25.00% | 9.52% | 15.75% |
| missing64 原顺序16对 | 10 / 183 / 30 | 5.18% | 25.00% | 8.58% | 16.26% |
| 未训练底座 原顺序16对 | 10 / 327 / 30 | 2.97% | 25.00% | 5.31% | 7.87% |

六条件均未触及2048生成上限。规则删除83—123个相对gold的FP、0个TP，全部来自症状谓词目标类型规则；另两条规则均无效果。原始逐阶段结果与DPO训练凭据：[rapid-progress-1529.json](audits/20261012/rapid-progress-1529.json)。15:08的前两条件记录仍保留在 [rapid-initial-results.json](audits/20261012/rapid-initial-results.json)。

目前的方向判断：规则与实体顺序有继续验证价值，但顺序提高召回的同时增加FP；短输入和短输出只有小幅质量变化，不能宣称已解决准确度问题。短输出耗时160.9秒，比原顺序16对的236.3秒低约31.9%，但无效JSON为16/64批，原条件4/22批，因此不能将加速等同于格式更稳。短输入输入token总量仅降低约6.3%，该局部窗口对本子集的长度干预较弱。missing负例对照仅增加1个TP和8个FP，没有显示出明显抑制误报的效果。

相对裸底座，grounded64把FP从327降至175，但TP从10变9；SFT有抑制过度连边的信号，也出现底座没有的JSON完整性问题（底座0/22无效JSON，grounded64原条件4/22）。已安排对训练掩码和解码停止路径的针对性复查，尚未归因于具体代码缺陷。

DPO16已完成64条合成偏好的一遍训练，用时75.0秒；权重有更新、保存重载一致、reference保持不变、初始policy等于零步对照。当前在做医学/机械对照评测，训练完成本身不能证明RL有效。matched256三臂专门训练仍按原冻结队列推进。

针对首组8.04%的低F1，另一次标准库独立计数确认184个预测三元组、40个gold、9TP/175FP/31FN；原始生成重建与图输出、身份、SHA、22批prompt均一致。31FN中14条候选不可达，7条位于4个无效JSON整批，9条同向pair选错谓词，1条为空或缺决策。4个无效JSON均不足500tokens，末尾不完整，未触及2048上限；不能将其误报为长度预算截断。

175FP可互斥分解为104条症状谓词目标类型不符、11条非法谓词占位、10条已有gold有向pair上的错谓词、16条仅反向pair有标注、34条两端有实体但该有向pair未标注。后两类合计50条是当前严格指标FP，未经审定不能全部称为医学幻觉。这说明需要同时处理候选/实体定位、类型约束、输出完整性和关系判别，不能只调一个训练参数。诊断：[rapid-initial-error-analysis.json](audits/20261012/rapid-initial-error-analysis.json)。

首组可解析输出中的190个唯一候选对有183个声明非空关系、仅7个为空，显示明显的过度连边倾向。实际ontology仅限制这类谓词的源类型，因此104条症状谓词错目标类型仍能通过允许列表；规则是额外语义约束。忽略11条非法占位后F1也仅8.45%，低分不能主要归因于占位计分。无效JSON的现有记录不足以区分EOS和其他停止原因，应继续保留这一不确定性。

| 方向 | 本轮实际设计 | 可以回答 / 不能回答 |
|---|---|---|
| 负例监督 | grounded64与missing64相同训练预算，同子集full/hash/16，并有裸底座 | 初步误连/漏连差异；不是完整医学质量结论 |
| 规则 | 在每个相同原始输出上逐条规则及规则合并，列删除TP/FP | 是否以大量召回换精确率；不把过滤后precision当全部效果 |
| 顺序 | grounded64仅改推理source顺序；另做matched256控制/全局source训练 | 区分推理格式改变和专门训练的信号；小样本单seed不能证明稳定收益 |
| 短输出 | grounded64把每批16对改4对；另做matched256 short4训练 | 输出长度/拒答/截断与成本；曝光相同仍不等计算量或相同序列loss权重 |
| 短输入 | 同一grounded64采用输入端点驱动local句窗 | 推理端窗口的方向信号；本轮尚无local专用重训，不宣称短输入训练已验证 |
| DPO | 同loader零步控制、64条纯合成偏好、16真实更新、机械与医学子集分开评测 | 训练工程是否可行及初步行为变化；不是医学偏好已审定、不是在线GRPO/PPO |

15阶段顺序：grounded四条件→missing与底座各一条件→DPO零步导出/16步训练→零步与DPO医学各一条件→两组机械holdout→matched256三组训练→三组匹配条件评测。有限队列不根据中途分数偷偷换样本、解码预算、负例或训练学习率。

## 冻结筛选子集

从原130条tune中的69篇free_text按输入字符数、oracle提供实体的候选数分层，seed42固定抽12篇，总240候选。选择不读relation标签或模型表现。所有条件使用同一12篇、原sample_indices、输入SHA与全部40条严格gold三元组，统一2048生成上限、batch4、每条件1500秒。原正式评测16000上限不变。

排除20篇零候选、22篇超过96候选的文档，另15篇合格但未选中；文本长度164—3903字符。该子集不代表高密度长文、零候选病例或完整130条。hash16/source16/local16各22个生成批次，hash4共64批。所有条件都保存raw生成、图、错误和截断率，整条件完成后才给主指标。

重要的预先归因：40条gold仅26条在候选内，14条全部因3篇中的3个Disease实体不能在原文定位（source端2条、target端12条）。类型/方向ontology和同名ID映射排除均为0。候选recall上限65%，若precision=1则F1上限78.79%；这不是模型实测成绩。四条件覆盖相同，顺序/长度不能恢复这些缺失端点。规范名/别名需要另外审定，不能直接改gold抬分。证据：[rapid-coverage.json](audits/20261012/rapid-coverage.json)。

## matched256完整一遍的训练对照

按metadata哈希从冻结V4中选择192条基础＋64条构造父行，127个source groups。先合并同source的选中候选，再按输入驱动顺序重分，不使用relation标签排序。控制/source16/hash4分别256/253/679行；第四个source4组合已发布但本轮不额外训练。

三臂保留完全相同的818正例pair（821条边）＋488负例pair及证据、来源和权重，原子Counter SHA一致。只跳过无任何有效监督的全UNKNOWN块，不能把UNKNOWN变负例。每臂从同一底座开始、seed42、lr5e-5、batch1×accum16、训练完整1epoch、max_steps=-1；预计16/16/43步，实测manifest为准。已核对固定Transformers4.56.2使用ceil累积步数，末尾余数不会被简单floor丢弃。最终还须核对actual observed_exposure。

各臂prompt一次曝光约665k/655k/1083k tokens，supervised completion tokens也不完全相同。不得称等算力、等语法/EOS监督或完全相同优化轨迹；报告更新数、监督token、实际GPU时长。短输出专用模型同时测hash4/hash16，source模型测source16/hash16，control模型测全部三种，避免只挑对某模型有利的输入格式。

## RL快速检查

64条偏好完全合成，8类机械错误；chosen边数更多/更少/相同为16/16/32，token更长/更短/相同为24/16/24。16条机械holdout仅名字不同且共享模板，不代表医学泛化。没有把UNKNOWN银标缺失当负例。

实际冻结配置：从grounded64初始化，DPO16步、batch1×accum4、lr1e-6、beta0.1、max_length2048。发布器生成的proposed recipe建议5e-6仅为候选，本轮实际执行以冻结queue的1e-6为准，不能混记。零步使用相同DPO policy/reference初始化后导出；DPO初始policy指纹必须等于零步指纹，reference不变、训练权重有更新且fresh reload验证通过。

医学子集用与其他模型相同的冻结matrix与2048上限；机械holdout使用对应source模板、相同1024上限，两个分母分表。没有运行在线GRPO或标准PPO，当前RL信号仍受合成偏好、小预算、单seed限制。

## 验证、收集与报告

新增数据重构、固定分层评测、DPO初始导出及截止队列；932项测试通过。额外复审修复了“最后一个生成批次超时仍标complete”的漏洞，收尾也复查deadline。运行快照代码SHA：`edd30c04c82d687f2ec7730f46c78fd4f2f0f567baaf0afd303e1e8161e59c3c`。数据和脚本单独哈希，旧正在运行的代码不被修改。

最终收集每个完整condition的12条预测、原始生成与metrics，严格身份核验后从预测重新计算TP/FP/FN，再计算micro P/R/F1；缺行、错位或指标不一致则不能发布为有效成绩。规则删除TP/FP独立复算，训练暴露计数与冻结清单核对。精确率不称Accuracy；不完整condition保留全部分母，但不参与比较。

独立复算工具为 [report_rapid_screen.py](../scripts/report_rapid_screen.py)，另有12项专项测试通过。医疗预测按原始身份和全部gold逐样本集合计数，同时检查条件完成状态、原始生成数量、文件SHA、解码预算、截断数和规则删除TP/FP。机械偏好单独成表。训练报告把记录到的实际曝光与预期单遍计数分开；运行时未分别记录的正负类别实际曝光保持空值。

将服务器 `rapid/screen_v2` 的状态、运行清单和预测结果同步至本机私有工作区后，在仓库根目录执行（不要把逐条预测或原文提交到GitHub）：

```bash
../.venv-audit/bin/python scripts/report_rapid_screen.py \
  --root ../audit_work/rapid_20261012/collected/screen_v2 \
  --screen-dir ../audit_work/rapid_20261012/evaluation \
  --queue-config ../audit_work/rapid_20261012/queue_v2.json \
  --mechanical-gold ../audit_work/rapid_20261012/rl/mechanical_holdout_records.json \
  --training-manifest ../audit_work/rapid_20261012/data/manifest.json \
  --output-prefix ../audit_work/rapid_20261012/report
```

输出同名前缀的聚合JSON与Markdown。某个完整条件可以先验收，同任务中尚未完成的其他条件仍标为missing/incomplete；缺失或无效条件没有有效P/R/F1，不能用于方向排名。

截止报告按方向给“值得进入正式训练 / 当前不支持 / 证据不足”，同时列推理改动与专门训练的证据层次、失败、耗时和未完成原因。多方向结果用于后续完整实验排序，最终模型、全流程和独立测试的要求继续有效。
