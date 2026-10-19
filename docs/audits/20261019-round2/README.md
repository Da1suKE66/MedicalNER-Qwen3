# 2026-10-19 afternoon experiments

The user authorized another 3–4 hours of work. This round began at 15:05 Beijing
time and has a 19:00 deadline. The previous experiment artifacts remain frozen.
One GPU queue runs the jobs; CPU data preparation proceeds alongside training.

The fixed starting point is entity and canonical relation SFT at learning
rate 1e-4, rank 16, two epochs. On the 69 free-text cases in the 130-case frozen
validation set, entity F1 is 68.7822%; live relation TP/FP/FN is 238/183/335,
precision 56.5321%, recall 41.5358%, F1 47.8873%. These are reused validation
results, not an independent test or human adjudication. The 70–80% target has
not been achieved.

Progress at 16:08 Beijing:

| Experiment | Scope | Status at this record |
|---|---|---|
| Entity three-epoch control | All 1361 admitted entity windows; same LR and data | Trained: 513 updates; full evaluation queued |
| Targeted DPO | 208 train preferences; 32 mechanical checks excluded | Trained: 52 updates; each pair used once |
| DPO live pipeline evaluation | All 130 original cases; fixed actual entity output | Running; quality result pending |
| Shorter input screening | Same 6 source-selected long cases, 2400 vs 1600 characters | Both arms queued |
| Final full-data training and validation | 1509-row long-criterion replay, or a context change justified by the pilot | Full training and 130-case evaluation queued |

Both completed training runs include saved-weight verification. DPO took
466.2 seconds after its separate GPU smoke check. Its frozen reference stayed
identical to the selected SFT parent, while policy weights changed. A separate
CPU review reconciled data/code hashes, token targets, and all 208 successful
preference visits against the saved receipts. This review did not rerun CUDA
training or independently reload the remote weights. Training completion is
not evidence of improved extraction quality; that requires the complete live
pipeline comparison. See [training evidence](dpo-training.json) and
[independent evidence review](dpo-training-review.json).

DPO preferences contain 64 outside-table references, 64 omissions of known
positive relations, 32 ontology-invalid direction reversals, 32 repetitions,
and 16 locally reviewed semantic contradictions (13 explicit-negation label
flips and 3 same-type hierarchy reversals). Chosen completions preserve the
original SFT rows. Unannotated relations are not negatives. Formal parameters
are one epoch, learning rate 5e-6, beta 0.1, accumulation 4: 52 optimizer updates.
Both policy and reference start from the currently selected relation SFT.
Training file SHA256:
`1191b1e6ffa56a352539e86614179e57f89efc760fe580ddb582a1c200bea3b9`.

Prepared data candidates include 1600-character entity windows: 2042 rows from
the same 668 admitted sources, preserving all 6484 unique entity atoms. This
changes overlap exposure and update count, so it is not a pure length
ablation. A separate recipe repeats 148 complete rows containing long
diagnostic criteria, producing 1509 entity rows without altering their targets.
These are resampling/context changes, not new annotated source documents.

A completed CPU replay tested retaining valid relation records from otherwise
invalid, EOS-complete windows. On the currently selected full 130 outputs it
added 2 true positives and 28 false positives; free-text relation F1 fell from
47.8873% to 46.8750%. This rule is not adopted. The original predictions, gold,
failure traces and published baseline remain unchanged.

Review of 38 training relations with modal language retained 28 supported and
9 uncertain cases. One source incorrectly used an optional usage instruction
as a diagnostic criterion. A separate, untrained candidate removes its two
unsupported diagnostic relations while retaining the other 14 targets.
Uncertainty was not converted into a negative label. Existing training parents
and all evaluation labels remain frozen.

An attempted 12-source DeepSeek review was rejected by automatic approval
review because disclosure of the specific medical source text had not been
explicitly authorized. No API request succeeded. The 16 semantic preferences
above were locally reviewed and rebuilt from resident training text; they are
not attributed to DeepSeek. No blocked historical data was uploaded.

Code: [semantic experiment programs](../../../scripts/semantic_round2/README.md).
Only code and aggregate evidence are published. Original texts, completions,
credentials, weights and private preference records remain outside the repo.
