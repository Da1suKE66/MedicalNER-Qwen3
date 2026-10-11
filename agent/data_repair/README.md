# 数据修复流水线：保留原始版本、机械修复、分流人工审定

本模块处理当前 DeepSeek `medical-kg-extraction-v3.1.1` 教师银标，不把另一个
858条 Schema v2/Gemini 清洗产物直接混入。复用既有 v2 `text_corrections`
白名单及已修复的分句器；不直接套用 v2 关系重写、主疾病注入或实体合并。

## 处理顺序

1. **冻结数据版本和来源划分**：输入SHA256必须与登记值及545/61来源组清单一致；
   检查每条system中的实际教师协议。输出只能放入全新目录，不覆盖任何输入。
2. **确定性修复文本**：按既有完整词白名单同步修复原文、实体名、证据描述；
   不做全局null/nan替换，不修改实体ID、关系标签或URL。所有变更保留路径及前后值。
3. **重建span与原文定位**：使用已修复的缩写/小数切句；实体保留原有概念名及ID，
   新增精确mention和修复前后偏移。大小写/空白可定位；别名、症状拆合不猜测合并。
4. **保留全部关系**：按有向实体对保留关系集合和所有原标注索引；
   只按同一关系聚合证据，不用单值dict覆盖其他关系。
5. **核验证据**：所有证据片段必须能唯一定位。禁止首句回退、只检查第一片段，
   或把“同句出现”当自动修复出的标注证据。未定位、歧义、类型不明进入review。
6. **分流语义风险**：证据未明确包含端点、否定范围、并列/层级混淆等仅是
   待审风险信号，不自动判为假关系。一个pair内任一标签待审，整对暂缓监督。
7. **防泄漏与发布门禁**：冻结原来源组；检查修复后内部train/tune同文重复。
   全部记录和标注仍保存。未标注pair默认UNKNOWN，不自动当NONE。
8. **后续人工/受控语义审定**：审定队列需给出accept/reject/relabel/alias及证据、
   审定人/版本；不把所有模型FP当负例，不把关系文本包含率当语义正确率。
9. **训练前复检与公平对照**：补历史dev重叠检查和独立盲测；从审定结果生成同一
   生成式模型的训练输入，固定曝光量/更新预算对照旧数据、新数据及去项数据。
   最终同时测标准实体、预测实体、真实全流程；不覆盖旧严格评测和原始银标。

## 第一阶段已实现的边界

CLI可运行步骤1–7并产出私有候选版本。机械可用标注依然是银标，**不是人工金标**。
当前不会启动新训练、调用外部LLM、消耗远程GPU或修改运行中的G0/G1/G6。
步骤8–9是后续发布门禁；不能把首轮机械清洗完成写成全部数据问题已解决。

在 `agent` 目录运行：

```sh
python -m data_repair.pipeline \
  --input /absolute/path/to/original_train.json \
  --frozen-split /absolute/path/to/G1_data/manifest.json \
  --expected-sha256 REGISTERED_SOURCE_SHA256 \
  --output-dir /absolute/path/to/new_private_release

python -m data_repair.verify /absolute/path/to/new_private_release
```

## 输出与隐私

- `records.jsonl`：每条原始记录、修复视图、实体定位、所有关系和待审状态。
- `changes.jsonl`：逐项改动及字符偏移；原始JSON字节另由输入SHA256冻结。
- `review_queue.jsonl`：引用record ID/标注索引的待审项，不会自动变成负例。
- `pair_candidates.jsonl`：通过机械检查的正关系集合，明确禁止自动当作训练发布。
- `frozen_split.jsonl`：原始索引/来源组/内部划分，不重排样本重新切分。
- `manifest.json`：源/代码/词表/输出哈希、修复数量、保留数量、待审原因、门禁。

数据文件仅写本地私有工作目录，目录权限700、文件600；stdout只打印聚合manifest。
Git仅提交程序、合成测试、方案和聚合结果，不提交上述原始或修复医学记录。

## 2026-10-12：证据定位策略的版本边界

`release_v2` 新生成的候选版本默认使用 `unique-evidence-v1`，版本名为
`teacher-silver-repair-v2-partial-supervision-unique-evidence-v1`。同一证据文本在
完整原文中多次命中时，该实体对的全部关系标记为 `UNKNOWN_EVIDENCE`，原因是
`evidence_occurrence_ambiguous`，并进入待审清单。局部窗口只保留一个命中不能
消除完整原文的歧义；明确指定的有效 span ID 可以消除定位歧义。这只是定位门禁，
不表示已认证医学语义。

历史 V2/V3 文件保持原样，读取、结构验证不会重写旧标签、掩码或版本。显式添加
`--evidence-policy legacy-all-occurrences-v1` 可重建历史的多位置合并策略；只能
输出到新目录。新旧策略的训练曝光量不同，必须在实验 manifest 中区分，不能作为
相同训练数据的精确复现。历史银标评测标签也不随训练门禁改写。

Partial-row 验证现在同时核对实体 ID、规范类型、有向候选对和当前 ontology 的
allowed-relation 表，拒绝输入类型变化后遗留的旧 allowed 列表。该检查沿用当前
source-only ontology，不引入新的 target-type 规则，也不把 UNKNOWN 变成 NONE。
