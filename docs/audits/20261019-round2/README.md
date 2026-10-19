# 2026-10-19 afternoon experiments

The user authorized another 3–4 hours of work. This round began at15:05 Beijing
time and has a19:00 deadline. The previous experiment artifacts remain frozen.
One GPU queue runs the jobs; CPU data preparation proceeds alongside training.

The fixed starting point is entity and canonical relation SFT at learning
rate1e-4, rank16, two epochs. On the69 free-text cases in the130-case frozen
validation set, entity F1 is68.7822%; live relation TP/FP/FN is238/183/335,
precision56.5321%, recall41.5358%, F1 47.8873%. These are reused validation
results, not an independent test or human adjudication. The70–80% target has
not been achieved.

Initial schedule:

| Experiment | Scope | Status at this record |
|---|---|---|
| Entity three-epoch control | All1361 admitted entity windows; same LR and data | Training on GPU |
| Targeted DPO | 208 train preferences; 32 mechanical checks excluded | Queued after real GPU smoke |
| DPO live pipeline evaluation | All130 original cases; fixed actual entity output | Queued |
| Shorter input screening | Same6 source-selected long cases,2400 vs1600 characters | Both arms queued |
| Final full-data training and validation | Choose a data/context change using pilot evidence | Pending pilot results |

DPO preferences contain64 outside-table references,64 omissions of known
positive relations,32 ontology-invalid direction reversals,32 repetitions,
and16 locally reviewed semantic contradictions (13 explicit-negation label
flips and3 same-type hierarchy reversals). Chosen completions preserve the
original SFT rows. Unannotated relations are not negatives. Formal parameters
are one epoch, learning rate5e-6, beta0.1, accumulation4:52 optimizer updates.
Both policy and reference start from the currently selected relation SFT.
Training file SHA256:
`1191b1e6ffa56a352539e86614179e57f89efc760fe580ddb582a1c200bea3b9`.

Prepared data candidates include1600-character entity windows:2042 rows from
the same668 admitted sources, preserving all6484 unique entity atoms. This
changes overlap exposure and update count, so it is not a pure length
ablation. A separate recipe repeats148 complete rows containing long
diagnostic criteria, producing1509 entity rows without altering their targets.
These are resampling/context changes, not new annotated source documents.

A completed CPU replay tested retaining valid relation records from otherwise
invalid, EOS-complete windows. On the currently selected full130 outputs it
added2 true positives and28 false positives; free-text relation F1 fell from
47.8873% to46.8750%. This rule is not adopted. The original predictions, gold,
failure traces and published baseline remain unchanged.

An attempted12-source DeepSeek review was rejected by automatic approval
review because disclosure of the specific medical source text had not been
explicitly authorized. No API request succeeded. The16 semantic preferences
above were locally reviewed and rebuilt from resident training text; they are
not attributed to DeepSeek. No blocked historical data was uploaded.

Code: [semantic experiment programs](../../../scripts/semantic_round2/README.md).
Only code and aggregate evidence are published. Original texts, completions,
credentials, weights and private preference records remain outside the repo.
