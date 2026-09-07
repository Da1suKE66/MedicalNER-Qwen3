# Relation accuracy audit and controlled improvement — 2026-09-07

Scope: retain SourceRouter → entity selection → normalization → relation decisions → validation → deterministic assembly. Keep the historical 2026-09-01 outputs immutable. Published baseline import: `20ccab3` (same tree as original local `75e660c`).

## Audit findings verified before editing

- v4 relation: 1,135 records; 13,246 decisions; 8,017 NONE. All 920 mixed batches have positives before NONE, unlike inference's entity ordering. v6 changed this, but does not solve the other mismatches.
- Only 4/1,135 training sequences exceed 4,096 tokens. Length alone cannot explain the failure. Generation truncation is a separate inference problem.
- Original training split: 1,345 records (606 structured ICD, 739 free text); historical dev: 148 (67 ICD, 81 free text). Exact normalized source overlap is zero; the provenance manifest reports disjoint source groups. These labels are teacher outputs, not independently adjudicated human gold.
- Among 6,165 free-text training relations, 910 lose an unaligned endpoint, 13 are excluded by hand-written signatures, 911 use an unsafe first-span evidence fallback. 2,485 retained relations need gold-dependent cross-chunk endpoint injection. Counts refer to the legacy converter audit, not independent causal experiments.
- 31 raw training endpoint pairs have repeated annotations; 30 have distinct multiple relation labels. A dictionary keyed only by endpoints can discard labels. The source schema defines source modules, not the target-type restrictions added in the agent.
- The splitter creates 1,846 one-character spans in the training text (17,268 spans), including broken abbreviations and decimal numbers.
- The scorer skips invalid predicted graphs, ignores invalid edge endpoints, and omits evidence from triple F1. JSON closure is not semantic validity. The 16,000-token problem sample has TP=0, FP=32, FN=12; TP=1 was the aggregate of three samples, not this sample.

## Ordered experiment plan

1. Version the baseline and input hashes; correct metrics and add regression tests for failure accounting, entity ordering, abbreviations, unsupported evidence and multiple labels.
2. Share the same candidate/context builder in train and inference. Recover all exact mentions; choose bounded local context by input text only; retain endpoint names/types; log unavailable candidates. Never use gold to supply missing inference inputs or first-span evidence.
3. Keep the Qwen entity adapter fixed for a relation-only comparison. Cache its historical predictions. Test narrow postprocessing rules on cached predictions and report both rejected FP and rejected TP.
4. Train a Qwen3-8B LoRA sequence classifier on individual named pairs and short source context (no relation JSON generation). Independent sigmoid labels permit several relations per pair. Invalid/uncovered annotations are excluded from negative supervision, not relabelled NONE. This changes only the relation decision module.
5. Reserve 10% of original training source groups for tuning; keep all chunks of each source in one group. Train up to two epochs, select checkpoint and rejection threshold on tuning only; report threshold sweep and precision/recall/F1/F0.5. Compare predicted-entity and oracle-entity evaluation separately.
6. Evaluate on all 148 historical dev records with stratified ICD/free-text triple scores, per-class metrics, evidence-aware scores, candidate recall ceiling, invalid counts, and the original failure sample. Historical dev has already been inspected repeatedly; label it a regression set, not a fresh blind test.
7. Retain the new relation module only if regression precision and F1 improve without relying on the ICD branch. A failure remains a documented experiment. DPO/PPO are deferred until supervised targets and evidence judgments are reliable; prefer DPO over PPO for a later small, adjudicated preference set.

## Git and artifacts

Work on `codex/relation-audit-20260907` in MedicalNER-Qwen3 (default pending user preference). Commit source, tests, aggregate metrics and this log. Full corpora, model weights, raw generations and local connection settings remain in ignored/private artifact directories. Record exact run commands, seed, software versions, input hashes and checkpoint paths. Do not replace production weights automatically.

## Completed experiment and next priorities

Two epochs completed; epoch 2 selected on tuning. Full cached-entity regression,
oracle diagnosis, same-entity 16k failure reproduction, threshold comparison,
synthetic probes and real pipeline smoke tests are complete. See
`reports/20260907/RESULTS.md` for all results, including failures.

The type-specific threshold policy was fitted from tuning sweeps before reading
the new regression scores. The later narrow target-type guard and orthography
diagnostic were added after inspecting regression failures and are marked post-hoc.
They do not replace the strict primary comparison.

Next work must prioritize source corruption repair, a canonical entity/alias and
span-granularity contract, independently reviewed evaluation, and training relation
on out-of-fold predicted entities. The current prototype is not deployment-ready.
