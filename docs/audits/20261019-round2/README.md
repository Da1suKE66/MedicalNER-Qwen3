# 2026-10-19 afternoon experiments

**Final round result:** all four new trained candidates completed the same full130 validation. The original E2+relation SFT remains selected; free-text relation F1 is47.8873%. DPO52 finished at46.5808%, with235TP/201FP/338FN. Label-weighted training and the GPU batch probe were not run because their remaining-time guards failed. See [Chinese final summary](SUMMARY.zh.md), [DPO52 result](dpo52-results.json), [DPO52 attribution](dpo52-error-attribution.json) and [closeout status](round-closeout.json). Earlier progress statements below are historical snapshots.

The user authorized another 3–4 hours of work. This round began at 15:05 Beijing
time and has a 19:00 deadline. The previous experiment artifacts remain frozen.
One GPU queue runs the jobs; CPU data preparation proceeds alongside training.

The fixed starting point is entity and canonical relation SFT at learning
rate 1e-4, rank 16, two epochs. On the 69 free-text cases in the 130-case frozen
validation set, entity F1 is 68.7822%; live relation TP/FP/FN is 238/183/335,
precision 56.5321%, recall 41.5358%, F1 47.8873%. These are reused validation
results, not an independent test or human adjudication. The 70–80% target has
not been achieved.

The first DPO run did not improve the complete live pipeline evaluation. All
130 cases finished; the primary 69-case free-text comparison is:

| Relation model | TP / FP / FN | Precision | Recall | Micro-F1 |
|---|---|---|---|---|
| Fixed SFT parent | 238 / 183 / 335 | 56.5321% | 41.5358% | 47.8873% |
| DPO, 208 pairs, one epoch | 234 / 194 / 339 | 54.6729% | 40.8377% | 46.7532% |

The entity outputs and all 125 relation prompts were verified equal between
the runs. Independent CPU recount retained every original gold item and
reproduced the metrics. DPO gained 5 true positives but lost 9; it removed
11 false positives and added 22. Relation window failures increased from
18 to 20. Both runs reached EOS without token-limit truncation. The relation
model remains the SFT parent; this DPO checkpoint is not selected. See the
[complete aggregate result and limitations](dpo-results.json).

Eight of the nine lost true positives came from normally parsed semantic
changes, and one from literal evidence matching after a quotation-mark change.
All 22 new false positives came from normally parsed outputs. New failed
windows did not account for the filtered true-positive loss. See the
[failure attribution](dpo-error-attribution.json).

The matched short-input screen is also complete. This table describes only
the same six long source texts, with 125 entity and 105 relation gold items:

| Input characters | Entity TP / FP / FN | Entity recall | Relation TP / FP / FN | Relation precision | Relation F1 |
|---|---|---|---|---|---|
| 2400 | 76 / 28 / 49 | 60.80% | 46 / 37 / 59 | 55.42% | 48.94% |
| 1600 | 84 / 45 / 41 | 67.20% | 42 / 67 / 63 | 38.53% | 39.25% |

Neither arm recovered any of the six long diagnostic criteria. The 1600 arm
produced two relation windows with repeated records and no EOS at the 4096-token
limit; the 2400 arm had no failed windows. Total generated tokens increased
from 9079 to 22243, and elapsed time from 274 to 783 seconds. Keep 2400 for the
remaining trials. This inference-only screen does not establish that dedicated
1600-context training could never help. See [pilot metrics](short-input-pilot.json)
and [repetition evidence without original text](short-input-failures.json).

Both training and inference deliberately supply the full source entity table
to each relation window. The observed difference is annotated tables in SFT
versus imperfect predicted tables in the live pipeline. Shortening reduced the
average body tokens by 31.3%, but total relation prompt tokens by only 5.25%,
because predicted entity tables grew. This screen changes both entity and
relation windowing; it cannot isolate the relation-context effect. A simple
local-only table rule would also remove endpoints needed by 74 admitted training
relation evidence atoms. No such rule was deployed. See the
[context diagnosis and protected cross-window supervision](context-diagnosis.json).

The entity three-epoch control has now completed the full 130-case evaluation.
On the 69 free-text cases, entity TP/FP/FN changes from 401/117/247 to
406/102/242: precision 79.9213%, recall 62.6543%, F1 **70.2422%**, up 1.4601
percentage points. With the same relation checkpoint, downstream relation
TP/FP/FN is 238/196/335 and F1 **47.2691%**, down 0.6182 points. This is a
standalone entity gain, not an improvement in the selected final pipeline.
The current best complete pipeline remains the original two-epoch pair.

Long diagnostic criteria remain weak: exact matches increase from 3/53 to
4/53. Relation false negatives with an absent entity endpoint decrease from
262 to 253, but those with both endpoints available increase from 73 to 82.
Neither arm has token-limit truncation. One source has an identical entity
table and relation prompt but changed relation output; its batch composition
and padding also changed. This limits per-source causal attribution to the
entity changes alone. A static inference review found no definite implementation bug; a bounded
GPU reproduction is still pending. See the
[complete entity-control result and limitations](entity-e3-results.json).

Historical progress snapshot at 17:58 Beijing (superseded by the final record above):

| Experiment | Scope | Status at this record |
|---|---|---|
| Entity three-epoch control | All 1361 admitted entity windows; same LR and data | Full130 complete: entity F1 +1.4601 points; downstream relation F1 -0.6182 points |
| Targeted DPO | 208 train preferences; 32 mechanical checks excluded | Trained: 52 updates; each pair used once |
| DPO live pipeline evaluation | All 130 original cases; fixed actual entity output | Complete: F1 decreased 1.1341 percentage points |
| Shorter input screening | Same 6 source-selected long cases, 2400 vs 1600 characters | Complete: keep 2400 after negative screen |
| Final full-data training and validation | All 1361 base windows plus 148 complete long-criterion replays | Full130 complete: entity F1 68.2292%, relation F1 43.6105%; not selected |
| Semantic-focused DPO52 | 36 reviewed semantic contrasts plus 16 table-reference contrasts | Trained:52 pairs once,13 updates; full130 fixed-E2 evaluation running |
| Relation-label weighted SFT | All1000 original relation rows, one-epoch continuation, label-token weight4 | CPU admission and independent review passed; queued after original10jobs, conditional on50minutes remaining |

The long-criterion replay completed both training and the full130 evaluation.
All1509 training windows were used exactly twice, for378 updates. On the69
free-text cases, entity TP/FP/FN is393/111/255 (F1 **68.2292%**) and relation
TP/FP/FN is215/198/358 (precision52.0581%, recall37.5218%, F1 **43.6105%**).
This loses0.5530 entity and4.2768 relation F1 points against the starting pair,
so the replay checkpoint is not selected. Long criteria improve only from3/53
to4/53; all matches remain within one source, while15 other sources with long
criteria still have no exact match. Disease TP falls198 to193 and symptom
TP189 to185.

The replay loses40 previously correct relations and gains17. Of the40 losses,
21 now lack a predicted entity endpoint; the other19 have changed outputs in
normally parsed former-support windows. Its one4096-token/no-EOS window
contains repeated records (one record59 times), but all3 previously correct
relations supported by that window survive through other successful windows.
It directly accounts for none of the lost old true positives. See the
[complete scores and failure attribution](entity-replay-results.json) and
[training evidence](entity-replay-training.json).

On this frozen69-case validation set, the starting entity table makes only
311 of573 gold relations reachable under the current endpoint constraint.
Even perfect extraction of those311 with zero false positives would yield
recall54.2757% and F1 70.3620%. This is an optimistic diagnostic bound for that
fixed table and protocol, not a measured score or a ceiling for future improved
entity models. It shows why relation training alone cannot reach80% F1 here.

The new52-pair preference model has completed training with LR2e-6, beta0.1
and one epoch, for13 optimizer updates; every pair was used exactly once. It excludes the known source188 preference issue and removes
the generic omission, repetition and ontology-invalid-direction pairs from this
new mixture. Both its smoke and formal run started afresh from the SFT parent.
The queue required45minutes remaining before launch, reserving time for its
complete evaluation. Its full130 fixed-E2 evaluation is now running. Changes to composition, learning rate and update count are
combined, so this is not a single-factor ablation. The 52 preferences are a
separate DPO dataset, not a reduction of the full SFT training set. See
[preparation evidence](dpo52-preparation.json),
[completed training](dpo52-training.json),
[independent192-check review](dpo52-training-review.json) and
[versioned code and source-free recipes](../../../scripts/semantic_round2/dpo52_v2/README.md).

The optional relation continuation uses LR2e-5 and125 updates. Its full1000-row
release contains399973 target tokens including EOS; relation-label tokens are
6.94% and entity names plus evidence are84.70%. These counts do not measure
actual loss or gradient contributions. All original tokens remain supervised;
only relation-label-field token weights change from1 to4. The real CPU
HF Trainer regression matched an explicit grouped-loss reference, including
unequal microbatches and a final partial accumulation group. A separate GPU
smoke must still pass before formal training. The main10jobs retain priority.

This candidate combines an extra epoch, a new LR and label weighting; without
a matched unweighted continuation it cannot establish the weighting's isolated
effect. Its runtime estimate and remaining-time gate do not guarantee an
improvement or completed evaluation. No result is claimed yet. See
[preparation and limits](label-weight-preparation.json),
[independent code review](label-weight-code-review.json) and
[versioned candidate](../../../scripts/semantic_round2/label_weight_sft/README.md).
The code/configuration's prepared status is a frozen preparation record;
run-status.json reports the current queue state.

All completed training runs include saved-weight verification. The208-pair DPO took
466.2 seconds after its separate GPU smoke check. Its frozen reference stayed
identical to the selected SFT parent, while policy weights changed. A separate
CPU review reconciled data/code hashes, token targets, and all 208 successful
preference visits against the saved receipts. This review did not rerun CUDA
training or independently reload the remote weights. The completed live
pipeline comparison above establishes that this run did not improve the
measured quality. See [training evidence](dpo-training.json) and
[independent evidence review](dpo-training-review.json).

The earlier208-pair DPO preferences contain64 outside-table references, 64 omissions of known
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

One of the 208 trained preference pairs contains those two questionable
diagnostic relations in both chosen and rejected completions. That pair's
preference difference is the omission of a different known positive relation.
This is a retained label-quality limitation; it does not establish the cause
of the measured DPO regression. The52-pair follow-up quarantined this source and used more directly
reviewed semantic contradictions. The already
trained data and its result remain immutable.

An attempted 12-source DeepSeek review was rejected by automatic approval
review because disclosure of the specific medical source text had not been
explicitly authorized. No API request succeeded. The 16 semantic preferences
above were locally reviewed and rebuilt from resident training text; they are
not attributed to DeepSeek. No blocked historical data was uploaded.

Code: [semantic experiment programs](../../../scripts/semantic_round2/README.md).
Only code and aggregate evidence are published. Original texts, completions,
credentials, weights and private preference records remain outside the repo.
