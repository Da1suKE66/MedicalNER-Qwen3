# relations_v2 output-only 训练与结构化评估摘要

## 训练状态

- 远程主机：`lsh-temp31030`
- 训练输出：`/cache/liluchen/medicalner_relations_v2/output/lora_output_only_20260825`
- LLaMA-Factory 配置：`qwen3_8b_lora_deepseek_relations_v2_output_only_20260825.yaml`
- `train_exit_code=0`，819/819 steps，3 epochs
- 最佳 checkpoint：`checkpoint-750`
- 最佳 eval loss：`0.03482325002551079`
- 最终 train loss：`0.034307218758305204`
- 完成时间：`2026-08-25T07:11:22+08:00`
- 训练后评估完成：`2026-08-25T07:58:50+08:00`

本地已复制最终 root adapter（其 SHA256 与 checkpoint-750 相同）以及审计、评估、训练日志和 loss 图；没有把权重上传 GitHub。

## cutoff 审计

训练配置和推理评估均使用 `cutoff_len/max_new_tokens=16384`。训练数据 2183 条、dev 148 条，训练/验证组合 token 长度超过 16384 的记录均为 0；最大组合长度分别为 8657 和 8951。

## JSON 闭合

开发 probe 共 8 条，DeepSeek target 8/8 闭合；output-only 6/8 闭合。`id=4`（heldout diagnosticCriteria）和 `id=15`（train diagnosticCriteria）均生成满 16384 token，括号统计分别为 1016/1014、986/984，最终 `invalid_json`。这是模型在 `entities` 中持续过生成后触达上限，不是训练数据超过 cutoff。

训练 probe 共 6 条，output-only 6/6 闭合，最大生成 2844 token；没有训练集 probe 的截断。

## relation scorer 聚合结果

分数是相对于 DeepSeek teacher 的一致性，不是人工医学 gold。五套结果如下。

### dev（8 cases，target 68 relations，prediction 30）

| 视图 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| raw-ID strict | 11 | 19 | 57 | 0.2245 |
| entity-aligned strict | 3 | 27 | 65 | 0.0612 |
| core triple only | 3 | 27 | 65 | 0.0612 |
| inverse-normalized | 3 | 27 | 65 | 0.0612 |
| source endpoint | 3 | 27 | 65 | 0.0612 |
| target endpoint | 10 | 20 | 58 | 0.2041 |
| relation type | 15 | 15 | 53 | 0.3061 |

方向反转计数为 0，两个端点正确但类型错误计数为 0，审计候选 inverse rescue 为 0。

### train（6 cases，target 54 relations，prediction 91）

| 视图 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| raw-ID strict | 41 | 50 | 13 | 0.5655 |
| entity-aligned strict | 10 | 81 | 44 | 0.1379 |
| core triple only | 42 | 49 | 12 | 0.5793 |
| inverse-normalized | 42 | 49 | 12 | 0.5793 |
| source endpoint | 42 | 49 | 12 | 0.5793 |
| target endpoint | 47 | 44 | 7 | 0.6483 |
| relation type | 45 | 46 | 9 | 0.6207 |

方向反转、端点正确但类型错误、inverse rescue 均为 0。

## 评分器/模型问题定位证据

- train `id=327`：raw-ID F1=1.0、core triple F1=1.0，但 entity-aligned F1=0；唯一 code/span 冲突是中心实体 `6A60.9` vs `6A60.8`。这复现了“实体 code 纳入 identity 会把关系整体连带判错”的评分器效应。
- dev `id=40`：模型生成 33 个实体但 0 条 relation，且有 12 个同 ID 的实体名称/type 冲突；这里不是单纯评分器问题，关系确实没有生成。
- train `id=251`：target 21 entities、prediction 54 entities，7 个 core entity 冲突，prediction 有明显过抽；其低分主要是模型实体绑定/过抽问题，而非仅 code mismatch。
- dev `id=4`、`id=15`：首先是生成截断，不能把空 relation 直接解释成关系学习失败。

因此，当前结论是：评分器的 code/span identity 确实会造成关系级联误判，core triple 视图必须保留；但 dev 上 relation 学习不足和 diagnosticCriteria 过生成仍是真实模型问题，不能仅靠改评分器解决。

## 产物

- `comparison_dev_probe.json` / `comparison_train_probe.json`：完整 raw target/prediction 与 generation metadata
- `raw_dev.json` / `raw_train.json`：原始字符串
- `closure_dev.json` / `closure_train.json`：闭合、截断和括号审计
- `relation_dev.json` / `relation_train.json`：五套 relation scorer 与 endpoint/type 分解
- `trainer_state.json`、`training_loss.png`、`training_eval_loss.png`：LLaMA-Factory 训练证据
- 最终权重、图像、日志和完整 retained checkpoints：远程 `/temp/liluchen/medicalner_relations_v2/snapshots/final/`（约 6.5 GiB）；本地最终 adapter 和图像位于 `reports/remote_relations_v2_20260825/final/`。
