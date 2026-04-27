# Qwen3-8B / LoRA 对比分析报告

## 1. 分析对象与方法

本报告比较三组输出效果：

- `base_qwen`
- `specific_adapter`
- `standard_adapter`

评测样本共 5 条，来自当前 `sample_eval_cases_expanded.json` 对应的 5 个疾病条目。  
分析分两部分：

1. 推理结果对比：格式稳定性、实体/关系覆盖度、典型错误模式、`specific` 与 `standard` 的差异。
2. 训练数据分析：`cot`/`thinking` 长度分布、`thinking` 相对 `output` 的比例、潜在的数据问题。

需要先说明一个重要现象：

- 三组结果文件里的原始输出都不是“纯 JSON”。
- `base_qwen` 输出是“长篇 `<think>` + JSON”。
- 两个 LoRA 输出是“空 `<think></think>` + JSON”。

这意味着：

- LoRA 之后，模型的结构化输出明显更稳定。
- 但显式 CoT 并没有真正保留下来，至少在当前评测 prompt 下没有保留下来。

## 2. 总体结论

### 2.1 LoRA 微调是否有改进

有，而且改进很明显，主要体现在两点：

1. `JSON` 结构稳定性显著提升。
2. 实体与关系的覆盖度明显提升。

量化结果如下：

| 模型 | 原样可直接 JSON 解析 | 去掉 `<think>` 后可解析 | 平均实体数 | 平均关系数 | think 内容情况 |
|---|---:|---:|---:|---:|---|
| `base_qwen` | 0 / 5 | 4 / 5 | 7.0 | 6.25 | 5 / 5 非空，平均约 10.3k 字符 |
| `specific_adapter` | 0 / 5 | 5 / 5 | 13.0 | 12.2 | 5 / 5 为空 |
| `standard_adapter` | 0 / 5 | 5 / 5 | 15.2 | 14.0 | 5 / 5 为空 |

此外，`base_qwen` 的 schema 也不稳定：

- `entities` 有时是对象，有时是列表。
- `relations` 有时是对象，有时是列表。

两个 LoRA 则全部统一成：

- `entities: list`
- `relations: list`

所以从“结构化图谱抽取”的角度看，LoRA 是有效的。

### 2.2 LoRA 后的主要缺点

LoRA 后的主要问题也很清楚：

1. 可见 CoT 基本消失，只剩空壳。
2. `specific` 偏保守，但仍存在关系语义错误。
3. `standard` 覆盖更高，但更容易过度结构化、抽取过多。

## 3. 直接证据：输出原话

### 3.1 Base 模型保留了长篇显式思考

以 ASD 样本为例，`base_qwen` 开头直接输出了长篇推理：

> `<think>`
>
> `Okay, let's tackle this. The user wants me to extract entities and relations from the given medical text about Autism Spectrum Disorder (ASD) without intellectual disability and with mild or no language impairment.`
>
> `First, I need to identify all the key entities. The main disease here is Autism Spectrum Disorder (ASD) as per DSM-5.`

同样，抑郁样本和 ADHD 样本也都是这种风格：

> `Okay, let's tackle this. The user wants me to extract entities and relations from the given medical text and output a JSON.`

> `Okay, let's tackle this. The user wants me to extract entities and relations from the provided medical text about ADHD.`

这说明 base model 在当前 prompt 下依然强烈倾向于显式推理。

### 3.2 两个 LoRA 都只剩空 think

`specific_adapter` 的 ASD case 开头是：

> `<think>`
>
> ``
>
> `</think>`

`standard_adapter` 的 ASD case 也是：

> `<think>`
>
> ``
>
> `</think>`

抑郁 case 和 ADHD case 也一样。  
也就是说，不是脚本删掉了 think，而是模型自己输出了“空 think + 正文 JSON”。

## 4. 微调后的改进点

### 4.1 格式稳定性显著改善

`base_qwen` 里存在明显的结构不稳定现象。以 ASD case 为例，它最后给出的 JSON 片段是：

> `"entities": {`
>
> `"Disease": {`
>
> `"name": "Autism Spectrum Disorder"`

也就是说，`entities` 被写成了对象，而不是实体列表。

而 `specific_adapter` 对同一 case 的开头已经稳定成：

> `"entities": [`
>
> `{`
>
> `"id": "D1",`
>
> `"label": "Disease",`
>
> `"name": "Autism spectrum disorder without disorder of intellectual development and with mild or no impairment of functional language"`

`standard_adapter` 也是同样的列表 schema。

这说明微调后模型已经更稳定地学会了目标输出格式。

### 4.2 覆盖度提升明显

在 5 个评测样本上：

- `base_qwen` 平均实体数 `7.0`
- `specific_adapter` 平均实体数 `13.0`
- `standard_adapter` 平均实体数 `15.2`

关系数也是同样趋势：

- `base_qwen` 平均关系数 `6.25`
- `specific_adapter` 平均关系数 `12.2`
- `standard_adapter` 平均关系数 `14.0`

这说明两个 LoRA 都明显提升了抽取覆盖率。

## 5. 主要问题与具体案例

## 5.1 `base_qwen`：思考很多，但结构化产出不稳定

`base_qwen` 的问题不只是“think 太长”，而是：

- 结构经常不一致
- 有一条样本即使去掉 `<think>` 后也仍然不是合法 JSON
- 实体和关系的 schema 不是统一的

换句话说，`base_qwen` 更像“懂任务，但不适合作为稳定图谱抽取器”。

## 5.2 `specific_adapter`：更克制，但仍有语义错误

### 案例 A：抑郁样本中过度把排除项当成疾病节点

在 `specific_adapter` 的抑郁样本中，出现了：

> `"name": "Delusions"`

以及：

> `"name": "Hallucinations"`

并且把它们作为差异化疾病处理：

> `"relation_name": "Differentiates From"`

原文实际上只是：

> `There are no delusions or hallucinations during the Depressive Episode.`

这里的“delusions / hallucinations”更像是排除条件，而不是应该单独建成 `Disease` 节点的鉴别诊断对象。  
这说明 `specific_adapter` 虽然更保守，但仍会在“否定条件 -> disease node”这一步发生硬解释。

### 案例 B：ADHD 样本中关系语义有明显错误

`specific_adapter` 在 ADHD 样本中出现了：

> `"source": "D1", "target": "S1", "relation_name": "Excludes If Present"`

同类关系还有：

> `"source": "D1", "target": "S2", "relation_name": "Excludes If Present"`

> `"source": "D1", "target": "S3", "relation_name": "Excludes If Present"`

但 `S1/S2/S3` 对应的是：

- `Inattention`
- `Hyperactivity`
- `Impulsivity`

这三个恰恰是 ADHD 的核心症状。  
把核心症状连成 `Excludes If Present` 明显是关系语义错了。

## 5.3 `standard_adapter`：覆盖更高，但更容易过度结构化

### 案例 C：ASD 样本中把 qualifier 过度抽成 symptom

`standard_adapter` 在 ASD 样本中新增了：

> `"name": "Intellectual functioning at least within the average range"`

> `"name": "Adaptive behaviour at least within the average range"`

> `"name": "Mild or no impairment in functional language"`

这些文本当然都来源于原文，但它们更像“诊断条件/限定词”，而不是标准意义上的 `Symptom` 节点。  
相比之下，`specific_adapter` 对这一条更倾向于保留为 `Diagnostic Criteria`：

> `"name": "Intellectual functioning and adaptive behaviour criteria"`

> `"name": "Functional language capacity criteria"`

这一点说明 `specific` 的 schema 选择更克制，`standard` 更倾向于把一切显式信息都拆成实体。

### 案例 D：ADHD 样本中 subtype 和附加节点扩展更多

`standard_adapter` 在 ADHD 样本中比 `specific_adapter` 多出了一批节点，例如：

> `"name": "Attention Deficit Hyperactivity Disorder, predominantly inattentive presentation"`

> `"name": "Attention Deficit Hyperactivity Disorder, predominantly hyperactive-impulsive presentation"`

> `"name": "Attention Deficit Hyperactivity Disorder, combined presentation"`

还加入了：

> `"name": "Teacher and Parent Reports"`

> `"name": "Significant Other Reports"`

这些内容确实都能在原文中找到依据，因此不能简单说它错；但它体现了 `standard` 的明显倾向：

- 更高覆盖
- 更细粒度拆分
- 更容易过度扩张知识图谱

## 6. `specific` 和 `standard` 的本质区别

从这 5 个样本看，二者不是简单的“谁更好”，而是风格差异明显。

### `specific`

特点：

- 更偏向围绕主疾病、核心 criteria、关键症状做收敛式抽取
- 更少新增 subtype / interview tool / patient info 的扩展节点
- 更像“抓重点”

代价：

- 覆盖不如 `standard`
- 某些地方仍然会错配关系或把排除项抽成疾病

### `standard`

特点：

- 更偏向模板式高覆盖抽取
- 更愿意展开 subtype、patient info、risk、interview tool
- 实体和关系数量普遍更高

代价：

- 更容易把 qualifier / supporting detail 抽成额外节点
- 更容易把图谱做得“很满”，但语义边界变松

简化地说：

- `specific` 更像“重点版”
- `standard` 更像“全量版”

## 7. 训练数据分析

## 7.1 训练集中并不是“没有 CoT”

原始训练数据统计如下：

### `specific` 数据

- 总记录数：`614`
- 非空 `cot`：`568`
- 空 `cot`：`46`
- 空 `cot` 比例：`7.5%`

### `standard` 数据

- 总记录数：`635`
- 非空 `cot`：`616`
- 空 `cot`：`19`
- 空 `cot` 比例：`3.0%`

所以不能说“训练数据没把 CoT 喂进去”。  
CoT 是存在的，而且大多数样本都有。

## 7.2 真正的问题：thinking 相对 output 太弱

### 原始数据长度分布

#### `specific`

- `cot` 中位长度：`587.5`
- `output JSON` 中位长度：`4198.5`
- `cot / output` 中位比值：`0.15`

#### `standard`

- `cot` 中位长度：`256`
- `output JSON` 中位长度：`4359`
- `cot / output` 中位比值：`0.07`

### 转换后 assistant 长度分布

#### `kg_cot_specific_614.json`

- `thinking` 中位长度：`600`
- `output` 中位长度：`5245`
- `thinking / output` 中位比值：`0.1219`

#### `kg_cot_standard_635.json`

- `thinking` 中位长度：`258`
- `output` 中位长度：`5520`
- `thinking / output` 中位比值：`0.0603`

这意味着：

- 模型在训练时看到的主要监督压力仍然来自 `output JSON`
- `thinking` 在 loss 中虽然存在，但相对弱很多
- 尤其 `standard`，其 CoT 强度明显更弱

这和当前推理结果“空 think + 强 JSON”高度一致。

## 7.3 训练样本原话也能看出这种差异

### `specific` 的 CoT 更像样本特异的抽取决策

原始 `specific` 样本里，CoT 是这种风格：

> `1) Extracted "Disorders of intellectual development" as the primary disease, including the "Moderate" subtype mentioned as a severity example.`
>
> `2) Identified core symptoms (limitations in intellectual functioning and adaptive behavior) and specific domains (conceptual, social, practical skills) along with diagnostic thresholds (2+ standard deviations below mean).`

另一条也是类似风格：

> `1) Extracted "Disorder of intellectual development, mild" as the primary disease, noting its specific severity (2-3 standard deviations below the mean).`
>
> `2) Identified core symptoms (intellectual functioning, adaptive behavior) and associated symptoms (language/academic difficulties, self-care mastery).`

这类 CoT 确实比 `standard` 更贴近样本本身。

### `standard` 的 CoT 更模板化、且整体更短

原始 `standard` 样本里常见这种风格：

> `1. 识别核心疾病“发育期智力障碍”及其严重程度亚型（轻度、中度）。`
>
> `2. 提取核心症状：智力功能受限（推理、记忆等）与适应行为受限（概念、社交、实用技能）。`
>
> `3. 映射诊断标准：低于均值2个标准差、发育期起病、需综合评估。`

另一条也是高度模板化：

> `1. 识别核心疾病：轻度智力发育障碍 (Disorder of intellectual development, mild)。`
>
> `2. 提取核心症状：智力功能显著低于平均水平、适应性行为显著低于平均水平（均低于均值2-3个标准差）。`
>
> `3. 提取相关症状：复杂语言概念理解困难、学术技能习得困难。`

这类 CoT 有信息，但更像“抽取清单”，同时长度又更短，所以更难在模型里保留成稳定的可见思维文本。

## 7.4 还有一个数据问题：`standard` 的输入长度不一致

进一步检查发现：

- `specific` 数据里 `input_used > 12000` 的样本数：`0`
- `standard` 数据里 `input_used > 12000` 的样本数：`68 / 635`

也就是说，`standard` 这批数据并没有严格保持统一的输入截断上限。  
这会带来两个问题：

1. 训练分布不一致
2. 长输入样本更容易稀释 `thinking` 的监督信号

## 8. 为什么会出现“空 think + 正确 JSON”

综合结果看，更合理的解释不是“模型把 CoT 学丢了”，而是：

1. 训练时 assistant 确实写入了 `<thinking> ... </thinking>` 与 `<output> ... </output>`。
2. 但 `thinking` 相对 `output` 太短，监督权重天然偏弱。
3. 当前评测 prompt 又明确要求：

> `Output the JSON object directly, with no markdown code blocks.`

所以模型最容易学出的折中策略就是：

- 保留一个 `think` 外壳
- 但把主要能力全部用在后面的 JSON 上
- 最后 `think` 退化为空

## 9. 后续数据改进方向

### 9.1 先统一目标：你到底要“可见 CoT”还是“最好 JSON”

如果你未来真正关心的是：

- 图谱抽取质量

那没必要强求显式 CoT 一定可见，可以把训练直接改成更稳的 JSON-only 风格。

如果你未来真正关心的是：

- thinking 本身也要拿出来比较

那就必须让训练目标和推理目标一致，而不能现在这样：

- 训练：`<thinking> + <output>`
- 推理：要求“直接 JSON”

### 9.2 提升 thinking 的监督强度

核心不是简单“把 CoT 变长”，而是：

- 增加 sample-specific 的决策性内容
- 减少机械模板
- 让 `thinking / output` 比值上升

当前两个集合的中位比值：

- `specific`: `0.1219`
- `standard`: `0.0603`

这都偏低，尤其 `standard` 太低。

### 9.3 对 `standard` 数据优先做三件事

1. 统一 `input_used` 上限，消除 `>12000` 的不一致样本。
2. 提升 `thinking` 长度和信息密度。
3. 减少模板化 CoT，增加更具体的“为什么保留/为什么不保留”。

### 9.4 对 `specific` 数据优先做两件事

1. 保留其“样本特异”的 CoT 风格。
2. 增加反例，压制以下错误：
   - 把否定条件硬转成 disease node
   - 把核心症状错误连成 `Excludes If Present`

### 9.5 如果你的目标是最终构图质量，我更建议下一版走混合策略

从这 5 条样本看，一个更可能有效的方向是：

- 以 `specific` 的收敛式风格为主
- 有选择地吸收 `standard` 在 subtype / interview tool / patient info 上的高覆盖能力

不要直接把 `standard` 的“多抽一切”照搬过来，否则图谱会更满，但噪声也会更大。

## 10. 最后的判断

如果只看当前这 5 个样本：

- **LoRA 是有效的。**
- **`specific` 和 `standard` 都比 `base_qwen` 更适合做结构化图谱抽取。**
- **`standard` 覆盖更高，但更激进。**
- **`specific` 更保守，但还没完全解决语义误配。**
- **两个 LoRA 都没有真正保住“可见 thinking 内容”。**

因此下一步最值得做的不是先继续调推理，而是先改数据：

1. 统一训练和推理目标。
2. 提高 `thinking` 监督强度。
3. 修复 `standard` 的输入一致性问题。
4. 用反例压制“过度结构化”和“关系语义错配”。

