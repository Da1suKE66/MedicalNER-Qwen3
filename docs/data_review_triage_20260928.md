# 2026-09-28：264 项 proposal-only 复核分流包

审阅子任务仅新增独立工具、测试、本文和本地输出。没有更改原清单、实体／关系标签、隔离或 UNKNOWN 状态；审阅工具不使用远程，也不修改训练。主线程后续代码提交与有限训练状态见[本轮进度](progress_20260928.md)。

**264 项全部保持 PENDING，human-review 仍需 264 项；自动关闭 0、标签修改 0。不是金标，不是经确认的医学别名字典，不是可直接应用的训练数据。**

## 实现范围与计划落实

1. 先阅读 `data_repair_v2_plan.md`、`data_repair_v2_results.md`、相关修复实现及测试，确认最终输入是 `release_candidate_02`。
2. 只读取候选中的 `manifest.json`、`records.jsonl`、`frozen_split.jsonl`、`review_queue.jsonl` 四个文件；核对发布哈希、原记录／原文哈希、冻结 split、来源组互斥和问题引用。
3. 按原队列顺序一项对应一项输出，重复问题不去重。旧清单没有独立 issue ID 时使用 `rq-{完整队列SHA256}-L{六位行号}`；已有 `issue_id` 或 `id` 原样保留，另设唯一 `triage_id`。
4. 只按可重现的局部字符串观察分组；不做医学推断、词干化、模糊匹配、疾病分型合并、关系反转或负例生成。
5. 合成边界测试与真实 264 项独立双构建验证通过后，输出全新目录。输出目录（包括空目录或悬空软链接）已存在即拒绝，输入改变或验证失败时不发布成功 manifest。

实现：`agent/data_repair/review_triage.py`。只依赖标准库与既有 `text_normalization.py`；新工具没有导入训练／评测入口。

## 真实候选结果

输入 1345 条记录、264 项旧问题；输出仍为 264 项，涉及的 **256 条完整记录**单独快照。train 234 项、tune 30 项。旧队列构成为实体 251、关系引用 2、记录／语义 11。

| 严格 proposal-only 细类 | train | tune | 合计 |
|---|---:|---:|---:|
| 仅逗号分隔的表面字符串候选 | 10 | 3 | 13 |
| 连字符／空格分词边界候选 | 5 | 0 | 5 |
| 仅词内命中，不作为实体定位 | 7 | 0 | 7 |
| 仅最长字面前缀可定位 | 192 | 25 | 217 |
| 名称／上下文仍无定位 | 8 | 1 | 9 |
| 引用来源／范围／绑定待审 | 1 | 1 | 2 |
| 正常性与诊断标准语义 | 1 | 0 | 1 |
| 端点类型／粒度 | 1 | 0 | 1 |
| 正常性章节原有 hint | 4 | 0 | 4 |
| 未解析 JSON 源块 | 2 | 0 | 2 |
| 原有句界空格提示 | 1 | 0 | 1 |
| 同输入冲突 | 2 | 0 | 2 |
| **合计** | **234** | **30** | **264** |

当前队列的完整词边界精确匹配、Unicode、单纯大小写／空白、混合分隔、非白名单符号完整词序候选及其他类均为 0；不是不支持这些检查，相关边界有合成回归测试。

18 个表面字符串候选 + 233 个仍未完整定位的实体问题 = 251 个实体问题；再加 13 个非实体问题 = 264。**18 不是已修复数或可释放监督数**，其余 246 也不是全部 human-review 数：两类都待审。

## 字符串证据与语义边界

- 完整匹配沿用对称 NFKC、大小写／空白比较和严格词边界；原始字符偏移保留，单位为 Python Unicode code point，区间为 `[start, end)`。
- 分隔候选要求完整词序一致，不增加、删除、重排字母／数字／否定词。额外规则只考虑字母词之间单个逗号或 `-`、U+2010、U+2011 与空白的变化，不拼接两个词。
- 数字分隔、小数、斜杠、加减号、撇号、括号、句界、重复标点、逗号直接换连字符均不接受为机械分隔候选。NFKC／大小写比较依然只是字符规则，不是医学语义证明。
- **纯标点也可能改变修饰范围与语义。**13 个逗号候选及 5 个连字符候选均设置 `separator_scope_requires_human_review=true`；`semantic_equivalence_confirmed=false`、`automatic_training_use_allowed=false`。
- `Manic Episode` 在 `Hypomanic Episode` 中、`EEG` 在 `EEGs` 中，以及实体末尾黏连 `and/or`，只提供词内命中提示。单复数、缩略词、缺失空格不自动裁定。
- 217 个前缀提示只是最长字面前缀的原文锚点；可能短至一个普通词。它们不证明后缀可删除、不证明缺失的是某种分型，更不等于别名。
- 原来的 normality hint 仍是 hint，不新增隔离。904／909 的冲突和原 blocked 状态保留；39／904 的引用问题不自动认定为医学幻觉。
- 本包只覆盖旧 264 项。ICD 未命名列表等其他问题仍在原候选中保留；本包不借“计数守恒”声称全量医学／来源审计已完成。

## train-only 候选表与隔离

`build_train_surface_candidates()` 是无文件访问的纯函数：检查所有输入项的 split 后，才访问任何名称、标签或证据；出现 tune/test/其他 split 即拒绝。主流程先完成 train 项分类和候选表构建，再分类其余审阅项。

本轮从 15 个 train 机械候选问题生成 **11 行去重的训练表面形式提案**，保留每次原文命中的 source_index、entity_id、偏移与来源哈希。它不是语义别名映射，不将标签跨记录转移，也不跨 split 应用。

为审阅原 264 项，整体输入 `records.jsonl` 本身包含 tune 记录，程序会校验并输出相应原文；**这些记录不传给 train-only 候选函数，也不参与训练候选表的学习**。不读取任何 `base/short/source_order/local/{train,tune}.jsonl`、test 文件或历史评测文件。测试检查读取文件白名单、holdout 提前拒绝，以及仅改变 tune 内容时训练候选表逐字节不变。

## 输出与来源

工作区输出：`work/round_20260928/review_triage`。

| 文件 | 用途 |
|---|---|
| `README.md`、`groups/*.md` | 分组计数和每项原问题、理由、完整关联证据、字符定位、审阅留空栏 |
| `triage.jsonl` | 264 项一一对应，原问题与 ID、不丢失的证据、记录行号／哈希、proposal-only 门禁 |
| `original_review_queue.jsonl` | 原队列的逐字节副本 |
| `source_records.jsonl` | 256 条涉及记录的完整快照，包括 original_record、修复图、源文本、原始审核状态及来源字段 |
| `train_surface_candidates.jsonl` | 11 行仅训练内的、尚未审定的表面字符串提案 |
| `manifest.json` | 输入／输出／代码 SHA256、运行时版本、守恒计数与全 false 发布门禁 |

`record_ref.line` 是快照 JSONL 的一基行号，`source_index` 保持候选原值。原始记录逐项可通过行号和 `prepared_record_sha256`、`original_record_sha256` 回查；不是只留摘要或实体名。源格式／冲突类 Markdown 包含完整记录图与原证据审核内容。

固定输入 SHA256：

| 输入 | SHA256 |
|---|---|
| `manifest.json` | `cce07aa6add954dae08ca9dcaa16048468e84466cf167eb39e5ee1b1655d8c28` |
| `records.jsonl` | `d7d3449322c935b890b302f379eb6f21844b90120ac44ea0537649eb43166eee` |
| `review_queue.jsonl` | `4f5505c02a3a0827a18e34adc1a338bc6825c329a977946c9e12c254edab5102` |
| `frozen_split.jsonl` | `691e97fff38aa46f4716cf4f844905287acb28e264cf9a0512a8d6dfa29eb2bd` |

输出目录 0700、文件 0600；stdout 只输出聚合计数，不打印医学原文。相同输入、代码和 Python／Unicode 运行时生成相同字节；无时间戳和输出绝对路径导致的重现差异。

## 运行与测试

从仓库目录执行，`python` 指已有本地 Python 运行时：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=agent python -m data_repair.review_triage \
  --candidate-dir ../work/data_repair_20260915_v2/release_candidate_02 \
  --output-dir ../work/round_20260928/review_triage \
  --expected-items 264 \
  --expected-manifest-sha256 cce07aa6add954dae08ca9dcaa16048468e84466cf167eb39e5ee1b1655d8c28
```

已交付路径存在，再执行此命令会按设计拒绝；复现应选新的独立空路径，不删除或覆盖旧包。

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=agent:../work/audit_20260907/tooling \
REVIEW_TRIAGE_REAL_CANDIDATE=../work/data_repair_20260915_v2/release_candidate_02 \
python -m pytest -p no:cacheprovider \
  agent/tests/test_review_triage.py \
  agent/tests/test_text_normalization_v2.py \
  agent/tests/test_review_policy_v2.py \
  agent/tests/test_release_v2.py -q
```

本轮 **269 passed**：新增专项 80 项（包含实际 264 项集成），相关既有回归 189 项。使用已有本地 pytest 8.4.2，没有下载依赖。真实数据测试包括独立双构建全部产物逐字节一致、每项原问题与完整记录快照一致、字符偏移回查、原文件哈希不变、全部 PENDING、train-only 来源及已有目录拒绝覆盖。

测试不等于医学审定或模型效果；主线程训练／实验状态不在本任务验收范围。

## 安全的后续建议（未执行）

1. 优先人工核验 18 项有完整词序与字符坐标的表面形式提案；先审逗号／连字符的修饰范围，再决定是否进入另一个带版本、审阅人和证据坐标的补丁。不可直接把这 18 项恢复为监督。
2. 保持 7 项词内命中的严格边界，逐项检查单复数、词缀和黏连问题；不要为修复漏检放宽全局匹配，否则 `Manic`／`Hypomanic` 等会误合并。
3. 使用 217 项前缀锚点帮助定位原文，但为 233 个未完整定位实体补充同源上下文或可审计的人工别名证据；不得当负例。
4. 先处理 2 个冲突输入、2 个引用疑问及正常性／类型疑问，继续保留隔离和原标签。4 个章节 hint、源格式提示分别审阅，不混同医学错误。
5. 审定结果独立留存；任何训练接入、标签重写、金标发布均需另外授权与验证，本工具不提供自动应用入口。
