# `relations.csv` schema and relation-scoring audit

审计对象：`/Users/doc66/Downloads/relations.csv`

- SHA-256：`c35a284f2a30e43f177dfb1cf05553751ab34ac13e697769341e0e95116f6c84`
- 82 条数据行、10 列；`Relation` 有 78 个唯一值。
- 建议状态分布：46 条可直接加入（含建议命名/规范化的可加入项）、28 条需要复核或先修正 source、8 条删除/不建边或改为节点属性。
- 当前仓库的关系表在 `src/kg_lora/extraction_schema.py:217-302`，逐行对比后只有 4 个名称不一致：

  | CSV 新名称 | 仓库当前名称 | CSV 行 | 当前训练数据实例数 |
  |---|---|---:|---:|
  | `aggravated_by` | `worsens` | 29 | 0 / 1 |
  | `has_subsymptom` | `has_manifestation` | 33 | 0 / 76 |
  | `increased_b` | `increases_risk_of` | 65 | 0 / 10 |
  | `reduced_by` | `protective_against` | 66 | 0 / 0 |

  因此，若直接把 CSV 作为新 closed vocabulary，而不同时转换/重生成 teacher target，现有数据中至少 87 条关系会出现标签协议不一致（`worsens` 1、`has_manifestation` 76、`increases_risk_of` 10）。本轮 output-only 训练不能只替换 prompt 表格；必须确定采用新名称后同步更新 target、validator、评估器和样例。

## CSV 的协议风险

1. CSV 给出了 `Target（最终推断）`，但当前 prompt 明确写着“workbook does not define target-label constraints”，只校验 relation 的 source module（见 `src/kg_lora/extraction_schema.py:379-381` 和 `validate_generated_graph`）。这会允许 `Diagnostic Criterion` 被模型输出成 `Symptom`，也无法在训练/验证时拒绝错误 target 类型。若采纳 CSV 的 target 列，应把它编译成按 `(source module, relation)` 的 target-label contract，而不是仅把列展示给模型。
2. 9 行有 `Source修正建议`，其中 6 行是 `TreatmentPlan` source、2 行是 `Communication` source、1 行是 `Diagnostic Criterion` source；这些需先决定是否将 module 改成实际 node label，或保留当前的 `MODULE_TO_NODE_LABEL` 映射。
3. 5 行有重命名建议，需在训练前冻结：
   - `causes`（Disease）→ `causes_disease`
   - `associated_with_poor_prognosis` → `associated_with_poor_prognosis_in`
   - `required_for` → `required_for_diagnosis_of`
   - `supports_Diagnostic Criterion_of` → `supports_diagnostic_criterion`
   - `escalates_to` → `requires_escalated_management`
4. 同一 relation 名称跨 module 复用（`causes`、`supports_diagnosis_of`、`recommended_for`、`recommended_by`）。这不是必然错误，但 target 类型和 source label 必须纳入 schema/validator；仅按 relation 字符串做 loss 或评估会混淆语义。
5. 新 CSV 自身并非全量可直接启用的协议：8 条明确不建边/改属性，28 条仍需医学或结构复核。建议先生成一个“active frozen subset”和一个“review queue”，不要把 82 行无条件塞入训练 prompt。

## 评分器结论

旧评分器 `scripts/analyze_comparison_20260811.py:94-112` 用 `(label, name, ICD-11 Code)` 作为实体和关系端点身份，会把 code 属性错误传播到所有 incident relations。对 heldout `id=251` 的直接复算：

- target/prediction 都是 33 entities、32 relations；
- 唯一实体差异是中心疾病 code：`6A60.4` vs `6A60.11`；
- 旧 scorer 因此得到 relation TP=0、FP=32、FN=32；
- 去掉 code 的 `(normalized label, normalized name)` relation key 后，32/32 relations 全部匹配。

这已经由新审计器 `scripts/audit_relation_scorer_20260819.py` 和回归测试覆盖（`tests/test_relation_scorer_audit.py`，4 tests pass）。生产评估应至少拆成：

1. entity core identity（name + label）；
2. entity metadata（ICD code/span）单独评分；
3. core relation triple F1（entity-aligned endpoint）；
4. raw-ID strict 只作调试；
5. evidence/span grounding 单独评分。

不能再用 code-sensitive relation F1 选择 checkpoint。

## `diagnosticCriteria` 的实际证据

现有 8-case raw artifact：`reports/remote_20260819/relation_scorer_raw_ck500_8cases_20260819.json`。

- `id=402`（train）：teacher 有 6 个 `Diagnostic Criterion` 节点和 6 条 `Disease --has_diagnostic_criterion--> Diagnostic Criterion`；prediction 把多个长 criterion 文本输出成 `Symptom`，并改成 `is_core_symptom_of`/`is_associated_symptom_of`，另有 5 条 `supports_Diagnostic Criterion_of`。因此 target endpoint 尚可对到主疾病，但 source label/type 与 relation type 同时错。
- `id=28`（heldout）：teacher 有 4 个 `Diagnostic Criterion` 节点和 4 条 `required_for`；prediction 没有 `Diagnostic Criterion` 节点，把多个内容片段当作 `Symptom`，并把 `Dysphonia`、`Dysarthria`、`Selective Mutism` 等 target-side Disease 输出成 Symptom。这里不是 ID scorer 问题，而是长标准文本的节点类型/边类型决策没有学会。
- 已有 8-case semantic audit 的 `diagnosticCriteria` core triple 为 34 target / 40 prediction，P/R/F1 = `0.225 / 0.265 / 0.243`；source endpoint F1 `0.270`、target endpoint F1 `0.811`、relation type F1 `0.378`。所以“疾病 target 能找到，但 source/type 不稳定”的判断有直接证据。

建议的可执行改法：

1. 在 prompt 和 validator 中明确 `has_diagnostic_criterion` / `required_for` 的 target 必须为 `Diagnostic Criterion` 或 `Disease`（按 CSV 冻结的方向），并明确 criterion 长段落只能建 `Diagnostic Criterion`，不能因为文本长而自动归为 `Symptom`。
2. 在训练数据 audit 中按 `(source label, target label, relation)` 统计并拒绝/修复非法关系；目前 `validate_generated_graph` 只检查 source module，不检查 target label。
3. 对 diagnosticCriteria 单独做 field-bucket 评估，报告 criterion-node F1、疾病/症状 endpoint F1、relation-type F1；不要让 indexTerms 的 32-edge hub 样本掩盖该子任务。
4. 先冻结关系命名和 target policy，再生成 output-only 数据；否则新旧 relation 名称混在一个 SFT target 中，会让关系 loss 进一步稀释。

## Verification

```text
python3 -m unittest tests.test_relation_scorer_audit -v
Ran 4 tests ... OK
```

