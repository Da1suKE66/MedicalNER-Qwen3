# Relation abstention repair — 2026-09-28

## Scope and status

The generative architecture is unchanged: source routing → entity extraction → deterministic normalization → generative relation decisions → validation → JSON assembly. No classifier, wider scoring rule, test-label editing, or production-model replacement is introduced.

Local implementation and verification are complete. Remote deployment was blocked by the upload safety review; explicit confirmation of this private training-data upload is pending. **Neither pilot has trained or produced new F1 results.**

## Failure addressed

The frozen V2 base data has 4,765 supervised positive pairs and 14,049 masked UNKNOWN pairs, but zero supervised NONE pairs. Masking UNKNOWN correctly avoids treating missing silver annotations as false relations; removing all NONE supervision also leaves an important abstention-learning gap.

In the prior full oracle-entity evaluation, V2 base emitted nonempty relations for every one of 14,204 parseable candidate decisions. Strict silver-label F1 was 5.94%, versus historical N0's 42.08%. Relaxing direction matching raised V2 only to 6.04%. This supports overprediction as a major failure, not merely ID or evidence-string mismatch. The matched pilots below test a targeted intervention; the historical comparison alone is not a causal ablation.

## Implemented changes

1. `agent/data_repair/abstention_repair.py` constructs bounded, train-only abstention examples with explicit provenance and negative-construction checks. Original training rows and their UNKNOWN masks remain unchanged.
2. `agent/generative/supervision_guard.py` rejects partial SFT with zero supervised NONE pairs before model loading. A diagnostic override requires explicit `--allow-positive-only-ablation` and a positive finite `--max-steps`. Invalid masks, mixed legacy/partial rows, zero/invalid weights and held-out data are not bypassed. The guard checks admission, not clinical correctness or an optimal negative ratio.
3. `agent/evaluation_identity.py` binds predictions to original `sample_index` and SHA256 of the exact user text. Contradictory identities fail before scoring; legacy artifacts remain `unverified`, including downstream results using an unverified entity cache. `--strict-identity` rejects them. Explicit subset indices are required; they are never inferred from predictions.
4. Generation, graph scoring and rule filtering preserve identity. Existing relation scoring is unchanged. A local compatibility adapter was added to the pre-existing untracked evaluation-recovery script; that user-owned file is not included in this code-only publication.

## Matched data intervention

Both arms contain 1,333 unchanged original rows plus 512 constructed two-statement excerpts derived only from existing training positives, spanning 93 training source groups. The missing-name donors contain 296 distinct natural symptom names from training data, not fixed dummy-name tokens. Constructed excerpts use three paraphrase templates and hash-varied order.

| Per dataset pass | Missing-only control | Grounded-NONE treatment |
|---|---:|---:|
| Training rows | 1,845 | 1,845 |
| Supervised positive pairs | 5,789 | 5,789 |
| Supervised NONE pairs | 2,484 | 3,900 |
| Original masked UNKNOWN pairs | 14,049 | 14,049 |
| Additional constructed UNKNOWN pairs | 2,636 | 1,220 |
| Supervised completion tokens | 224,478 | 243,200 |
| Maximum total sequence length | 4,932 | 4,932 |

The two arms have identical input prompts, completions and positive targets. The only intervention is unmasking 1,416 constructed grounded-NONE decisions. This is **not** an equal-supervised-token-budget comparison. The counts are per full dataset pass, not actual exposure under the fixed-step sampler.

Missing-only teaches no relation for candidate entities absent from the controlled context. Grounded-NONE also teaches no relation between co-listed symptoms, across unrelated disease scopes, or between two unconnected diseases in that context. Reverse Disease→Symptom decisions remain UNKNOWN because some schemas allow alternate interpretations.

Toy example: a constructed excerpt states “Feature Alpha is a symptom of Condition Kappa” and “Feature Beta is a symptom of Condition Lambda.” Alpha→Kappa retains the known positive label; Alpha→Lambda and Alpha→Beta receive NONE in the treatment only. This is evidence-grounded extraction supervision, **not** a claim that the concepts can never be clinically associated.

Synthetic excerpts are explicitly marked as synthetic, not original medical source text or expert-adjudicated gold. Negative proof fields and supervision status are not exposed in model prompts. No held-out vocabulary or relation labels are consumed by the builder; train/tune group overlap is rejected. Original unannotated pairs are never automatically relabeled NONE.

## Frozen pilot protocol

- Two jobs: `V3_missing_only_s42` and `V3_grounded_none_s42`.
- Same original Qwen3-8B initialization, LoRA rank 16 / alpha 32, learning rate 5e-5, batch 2 × accumulation 8, seed 42, maximum training length 6,144.
- Each job: independent 2-step save/reload smoke check, then 64 optimizer steps. Total budget: 132 steps across at most two GPUs, no automatic retries or expansion.
- Only checkpoint 64 is evaluated; the unchanged internal tune set has 130 records. Regression uses the frozen 148-record set with both cached predicted entities and oracle entities, followed by the existing 64-pair clean/noisy stress test. Generation limit: 16,000 tokens.
- The whole 148-record set is preserved, with free-text metrics separately reported over its 81 routed free-text records. Four raw JSON-list inputs remain an explicitly identified source-quality issue, not silently removed to raise scores.
- Compare both arms against each other and against historical N0 and V2. N0 is historical, not a new matched-initialization control. Reject a “win” based solely on beating the collapsed V2 model.
- No DPO/PPO/GRPO is launched: partial silver UNKNOWNs do not provide trustworthy complete preference/reward labels. The existing RL data-admission rejection remains enabled.

## Scoring and acceptance

Primary relation matching is the unique directed tuple `(source type, normalized source name, relation label, target type, normalized target name)`. Entity IDs, array order, properties and evidence wording are not part of primary relation matching. Evidence has separate metrics. Micro P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).

Report raw TP/FP/FN, P/R/F1, NONE/nonempty decisions, invalid or missing decisions, truncation, candidate coverage, actual positive/negative exposure and runtime. A silver-label FP is not automatically a medically false hallucination. The fixed historical entity cache cannot be retroactively certified by adding a new digest.

Engineering validation: **778 tests passed; 1 offline private-corpus integration test skipped** because `REVIEW_TRIAGE_REAL_CANDIDATE` was not supplied. Both real-tokenizer data audits pass with no truncation, and 512 construction proofs are checked in each arm. Model quality is still pending actual pilot training and evaluation.

## Remaining limitations

- The 512 constructed samples do not close the 264 historical human-review items.
- Controlled short contexts do not fully reproduce real long-context ambiguity, coreference, negation or implicit medical relations.
- Entity omissions still impose a relation-recall ceiling; this repair does not recover missing entities.
- One seed and 64 steps are diagnostic pilots, not a robust final benchmark or resume headline.
- No raw medical corpus, prompts, model predictions, weights, credentials or private deployment bundle are included in GitHub publication.
