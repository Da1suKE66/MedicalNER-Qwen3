# Train / heldout 欠拟合与过拟合诊断（2026-08-19）

## 实验设计

- 数据：完整 `deepseek_watermark_20260804_182312_output_only_full.json`（1493 条）。
- 划分：冻结的 group-disjoint manifest；train 1345 条、heldout 148 条，group overlap=0。
- 样本：从 heldout 中按相同 primary source field、相近输入/标签长度匹配 8 对，共 16 条：
  `170/316 (parent)`、`675/943 (exclusions)`、`276/310 (ancestor)`、`402/28 (diagnosticCriteria)`、`1424/1434 (definition)`、`264/239 (synonyms)`、`301/251 (indexTerms)`、`149/165 (ancestor)`。
- teacher-forced loss：base Qwen、checkpoint-350、500、507，mask 掉 prompt，只计算 output token。
- 生成：checkpoint-500，`enable_thinking=False`、greedy、`max_new_tokens=4096`、batch=1；训练/heldout 使用相同提示和解码参数。
- 解释边界：DeepSeek target 是当前冻结 teacher reference，不是人工 gold；指标用于 fit-gap 诊断，不代表医学正确率。

## Token-level fit gap

| 模型 | train mean loss | heldout mean loss | heldout - train | heldout/train |
|---|---:|---:|---:|---:|
| base Qwen | 0.262750 | 0.287322 | 0.024573 | 1.09x |
| checkpoint-350 | 0.004680 | 0.006683 | 0.002003 | 1.43x |
| checkpoint-500 | 0.004544 | 0.007025 | 0.002482 | 1.55x |
| checkpoint-507 | 0.004505 | 0.007150 | 0.002645 | 1.59x |

结论：微调后 train 与 heldout 的绝对 loss 都很低，不符合“整体都学不会”的欠拟合形态；但 checkpoint-350 到 507，heldout/train 比从 1.43x 增到 1.59x，存在轻度且逐步增大的泛化差距，不能再把 checkpoint-507 的更低 train loss 当成更好泛化的证据。

## 实际生成差距（checkpoint-500）

| split | 条数 | schema valid | exact | entity F1 | relation F1 | relation hallucination |
|---|---:|---:|---:|---:|---:|---:|
| train | 8 | 8/8 | 5/8 | 0.7188 | 0.5766 | 47.5% |
| heldout | 8 | 8/8 | 5/8 | 0.7634 | 0.0351 | 96.8% |

最有区分度的一对是 `indexTerms`：train `id=301` 为 33 entities / 32 relations，生成 exact match；heldout `id=251` 同样为 33 / 32，生成数量看似一致，但 relation TP=0/32，说明模型学到了长 JSON/关系链模板，却没有把新疾病的关系端点正确泛化。`diagnosticCriteria` 的 train `id=402` 与 heldout `id=28` 也都出现大量关系不匹配，说明 relation supervision/teacher consistency 本身仍是主要瓶颈。

## cutoff 检查

- 16 条 teacher-forced 样本全部 `skipped=0`。
- prompt token 范围 2462–4083，target token 范围 77–3905，最大总长 7988，低于训练 `cutoff_len=16384`。
- 生成最大 3904 tokens，`hit_max_new_tokens=0`；本轮没有证据表明训练标签被 16384 截断或生成被 4096 硬截断。

## 判断

当前主要问题不是简单欠拟合，也不是已经发生灾难性过拟合，而是：

1. token-level loss 已经很低，继续训练只带来很小的 train loss 改善，同时扩大 heldout gap；
2. 结构化生成在训练关系链上可以“背下来”，换到 group-disjoint heldout 后 relation 端点几乎全错；
3. 因此更像“轻度过拟合 + JSON/关系模板记忆 + teacher relation 标签不稳定”，而不是单纯增大模型或 cutoff 就能解决。

下一轮应优先以 heldout relation F1、relation hallucination、按 source field 的 paired gap 作为 early-stop/选 checkpoint 指标，并先清理 relation annotation policy；不建议按更低 train/eval token loss 继续把 checkpoint-507 当默认模型。
