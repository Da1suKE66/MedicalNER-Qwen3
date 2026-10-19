# 审核证据索引

这些文件由 2026-10-12 对私有迁移包的只读检查产生，仅包含计数、哈希、相对路径和分析，不包含医学原文。完整解释与实验计划见 [审核报告](../../relation_audit_20261012.md)。

| 文件 | 含义 |
|---|---|
| [migration-integrity.json](migration-integrity.json) | 封存文件、拆分、真实 tokenizer/mask 校验；不认证医学正确性。 |
| [metrics.json](metrics.json) | 独立脚本核对20份历史指标和五篇原始预测；没有完整原始预测的实验明确标注核验层级。 |
| [candidate-coverage.json](candidate-coverage.json) | 当前候选重建的 fixed/oracle 上限及不可触达原因；旧 cache 来源仍未验证。 |
| [supervision.json](supervision.json) | 19个派生文件的正/负/UNKNOWN、行级分布、待审、旧 stage 内部 tune 暴露，以及新旧证据策略对比。 |
| [evidence-policy-impact.json](evidence-policy-impact.json) | 12392行历史格式兼容验证及新策略暂缓的标签数。 |
| [experiments.json](experiments.json) | 历史训练清单、有效曝光、保存重载观察、未证实的因果假设及后续矩阵。 |
| [validation.json](validation.json) | 修复后仓库全量843项测试，含私有 review 集成；本次未跑 GPU 质量实验。 |

复现 `metrics.json` 使用 `scripts/audit_migration_metrics.py`；复现 `supervision.json` 使用 `scripts/audit_relation_supervision.py --compare-evidence-policy`。两者必须传入私有迁移包路径和一个新的、包外输出文件。脚本拒绝覆盖旧报告。哈希只用于绑定版本，不能将历史未验证实体缓存转为 verified。

模型、数据及预测未上传至本目录；需要相同私有输入才可复核全部数字。不得把这些派生数据集的行数相加当成独立病例样本量。
