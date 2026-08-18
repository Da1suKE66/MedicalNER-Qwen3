# Output-only checkpoint structured evaluation (16-case canary)

本报告比较现有 output-only `checkpoint-350/400/450/500`。每个 checkpoint 使用相同的 16 条 group-disjoint 配对样本（train 8、heldout 8），`enable_thinking=false`、greedy、`max_new_tokens=4096`、batch 1。DeepSeek target 仅作为 noisy diagnostic reference。

| checkpoint | split | schema | exact | entity F1 | relation F1 | relation hallucination | cap hits |
|---|---|---:|---:|---:|---:|---:|---:|
| 350 | train | 7/8 | 4/8 | 0.731 | 0.000 | 100.0% | 1 |
| 350 | heldout | 7/8 | 4/8 | 0.698 | 0.000 | 100.0% | 1 |
| 400 | train | 6/8 | 4/8 | 0.133 | 0.000 | 100.0% | 2 |
| 400 | heldout | 6/8 | 5/8 | 0.184 | 0.065 | 80.0% | 2 |
| 450 | train | 7/8 | 5/8 | 0.731 | 0.719 | 17.9% | 1 |
| 450 | heldout | 8/8 | 5/8 | 0.750 | 0.036 | 96.6% | 0 |
| 500 | train | 8/8 | 5/8 | 0.719 | 0.577 | 47.5% | 0 |
| 500 | heldout | 8/8 | 5/8 | 0.763 | 0.035 | 96.8% | 0 |

## 快速判断

- `checkpoint-350` 和 `400` 有明显生成截断/格式失败，不能作为当前默认 checkpoint。
- `checkpoint-450` 在 train 上关系 F1 高，但 heldout 关系 F1 仍只有 `0.036`，是最直观的关系过拟合信号。
- `checkpoint-500` 结构完整性最好，但 heldout relation F1 与 450 基本相同；继续训练没有带来关系泛化改善。
- 这个 16-case canary 不能决定最终医学模型，但足以说明“继续训练到 500/507”不是当前关系问题的解法。下一轮应先清洗 relation target、加入结构化评估选 checkpoint，再调整训练参数。
