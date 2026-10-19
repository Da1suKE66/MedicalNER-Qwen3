# 数据修复v2执行结果

2026-09-15。此报告是数据修复与验收结果，不是新模型效果报告。

## 交付与不变项

- 最终候选：工作区`work/data_repair_20260915_v2/release_candidate_02`；`candidate_01`是补回上游句首前的中间版本，不用于后续训练。
- 原始1345条教师记录、858条原始/Schema v2数据、既有训练版本、148条历史回归标签未覆盖。
- 保留1345条原记录、全部原始实体关系和来源哈希；新的模型监督另存，不把原银标静默改成金标。
- 本轮没有训练或切换远程实验；没有将医疗数据、权重或密钥上传GitHub。

## 逐项修复结果

| 问题 | 处理结果 | 剩余边界 |
|---|---|---|
| nan词污染/Definition遗漏 | 修复图谱已知坏词0；描述性图字段59处变更，源文本145处变更（含Unicode，不是全为坏词） | ID、代码、URI、合法null和原记录不改；字典不保证发现全部错字 |
| 单位NFKC不对称 | 499/504/511三例实体、证据均恢复 | 未定位实体254→251，剩余不能一律判假实体 |
| 证据句点边界 | 658实际证据匹配恢复 | 不猜证据、不回退首句 |
| 原文引用问题 | 自由文本证据定位失败6→2 | 39/904两例需证据来源/绑定审定，不判为两条已确认医学幻觉 |
| 正常现象当诊断标准 | 14的D1→C6不参与正例loss | 保留原标签和复核理由，不冒充人工已审定 |
| 关系目标类型/粒度 | 47的D1→S11鉴别关系隔离 | 不改全局ontology，不反转或直接判NONE |
| 相同输入、不同分型标签 | 904/909两条整记录隔离 | 需补主病种上下文，不能合并疾病分型 |
| 47词中截块 | 同一ICD release URI及代码匹配，4241字符完整残段唯一定位原字段，补回原句首 | 保留旧user、新有效user、上游字段哈希及字符位置；不是语言模型猜词 |
| ICD整组过滤 | 355的4项全部保留，其中3项有有效引用语法、1项没有 | 未命名列表的语义仍待审；不自动改成排除/子型关系 |
| UNKNOWN拖掉整批 | 逐pair监督、整段UNKNOWN token屏蔽；同批其他关系保留 | 全UNKNOWN行不参加SFT；上下文仍保留候选表 |
| 银标缺失被当NONE | 未标注pair不受空关系目标loss | 真实负例池尚未经人工审定；正标签也可能不完整 |
| 错误entity负例 | 独立135条辅助序列、270个明确合成的错误实体关联负例；哈希排序不把负例固定放末尾 | 容易的合成负例，不等同真实实体错误分布；未与主数据自动混训 |
| 多关系丢失 | 合并同标签证据，保留一对多标签；不再用单值字典覆盖 | 不能由结构保留证明教师没有漏标 |
| 训练/RL误用 | 新SFT支持显式掩码；现有DPO/GRPO和偏好构造入口拒绝UNKNOWN数据 | 需审定监督/负例或另实现并验证适配奖励，才能宣称RL-ready |
| 历史评测标签污染 | 148条仅做独立审计，原始评分标签SHA不变 | 没有回改旧标签抬高F1，也没有产生新模型准确率 |

## 新版训练监督数量

下表“标签”是保留的去重有向关系标签，不是准确率。

| 版本 | 训练序列 | 训练标签 | 内部验证序列 | 内部验证标签 |
|---|---:|---:|---:|---:|
| base（完整上下文，16对/批） | 1333 | 4785 | 325 | 482 |
| short（完整上下文，4对/批） | 3344 | 4785 | 435 | 482 |
| source_order（完整上下文，原文顺序） | 1172 | 4785 | 315 | 482 |
| local（局部窗口） | 1322 | 4606 | 321 | 472 |

base相对旧G1的4779条训练标签：恢复8条监督、隔离2条疑点，净增加6条；这是数据覆盖变化，不是模型提升。904/909的原疾病端点本已无法落地，不能把整记录隔离数再次当成删除同样数量的有效监督。

base/short/source_order的标签集合逐条一致，消除了旧整批UNKNOWN造成的意外差异；序列/token/未知上下文曝光仍不同，训练比较必须报告。local仍缺179条训练、10条内部验证标签，不能冒充纯长度实验；已提供四版本共同监督交集清单。

## 文件用途

- `records.jsonl`：原记录、修复图、原/有效源文本、双坐标mention、证据及审计状态。不是可直接当金标使用的聊天训练文件。
- `changes.jsonl`、`upstream_restorations.jsonl`：字段修改、来源恢复及哈希证据。
- `review_queue.jsonl`：264项主要复核入口，包括未定位实体、语义/引用疑点；ICD各列表项另在`source_view.labelled_lists`保留独立语义待审标记。
- `base/short/source_order/local/{train,tune}.jsonl`：生成式关系部分监督候选，必须使用带掩码的新训练读取器；不是新的实体阶段训练导出。
- `synthetic_negative_auxiliary/train.jsonl`：独立合成负例对照，不将漏标pair当真实负例。
- `ablation_common_supervision.jsonl`：共同标签集合；不是声称已经做过模型消融训练。
- `historical_regression_audit_only.jsonl`：只读审计视图，不替换原评分gold。
- `manifest.json`：源/输出/代码哈希、冻结划分、计数和明确未完成的发布门禁。

## 验收边界

单元/集成测试覆盖JSONL往返、输入完整性、词/Unicode和原始位置映射、UNKNOWN token、跨BPE边界、prompt/pad/EOS、真/假证据、多关系、拆批顺序、RL拒绝、来源恢复及防覆盖。真实分词使用从远程实际Qwen3-8B读取的分词文件，在本地验证，不占GPU；具体计数以工作区`validation_remote_qwen_02_final.json`为准。

最终独立代码复核补上SFT输入门禁：prompt必须等于实体表、候选表、span表构造出的唯一允许输入，或其明确Qwen非思考模板；不能塞入额外gold/来源字段。直接内部execute调用也执行数据门禁。该缺口可合成复现，但未在本次生成数据中发现污染。最终验收sidecar记录加固后源码哈希与构建时源码差异，不覆盖旧manifest。

本地旧adapter分词核心tokenizer.json与远程相同，但tokenizer_config/chat_template有差异。因此最终验收改用远程原配置，不把旧adapter模板相同长度当作配置一致。

尚未完成：251个实体别名/上下文审定、2条局部证据来源审定、2条冲突输入上下文补全、正常性/类型语义人工裁定、真实负例审定、ICD未命名字段语义审定、独立医学金标、新模型训练与盲测。新候选不自动接入旧实验，不宣称医学准确率。

## 复现入口

从repo使用`PYTHONPATH=agent`运行：

```sh
python -m data_repair.release_v2 --input ORIGINAL_TEACHER_JSON --frozen-split FROZEN_G1_MANIFEST --expected-sha256 SOURCE_SHA --upstream-raw ORIGINAL_858_JSON --expected-upstream-sha256 UPSTREAM_SHA --historical-dev FROZEN_DEV_JSON --output-dir NEW_PRIVATE_VERSION
python -m data_repair.verify_v2 --directory NEW_PRIVATE_VERSION --tokenizer-dir LOCAL_REMOTE_QWEN_TOKENIZER --max-length 6144 --report NEW_VALIDATION_JSON
```

分词检查需要transformers 4.56.2、tokenizers及jinja2；不需要加载模型。输出目录和验收报告存在即拒绝覆盖。人工审定完成前，manifest的医学/真实负例/RL/模型回归门禁保持false，验收sidecar只证明机械和token合同。
