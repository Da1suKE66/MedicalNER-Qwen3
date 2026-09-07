# Reliable two-stage medical KG agent

The existing ICD parser, Qwen entity adapter and deterministic JSON assembly
remain. An opt-in, non-generative relation head replaces long relation-JSON
generation. Read [the audit](reports/20260907/AUDIT.md) and
[the experiment plan](PLAN.md) before treating scores as production accuracy.

## What is trained

Qwen3-8B LoRA sequence classification on an ordered pair's **names, types and
local source text**. Independent sigmoid outputs allow multiple relation labels;
NONE means no allowed label exceeds the tuning-selected threshold. The model
does not invent entity IDs or evidence text. `kg_agent/pairwise.py` constructs
identical candidate inputs for training, evaluation and the live agent.

The active schema is `kg_agent/ontology.json`, derived from the actual legacy
teacher system prompt. It is not interchangeable with the repository's schema
v2 migration pipeline. The historical converter `build_stage_data.py` requires
explicit `--allow-legacy-relations` and is retained only for reproduction;
new relation data uses `build_pair_data.py`.

## Reproduce (run from this directory)

Use the recorded CUDA environment in `requirements.txt`; do not replace an
already-working CUDA stack unnecessarily. Deterministic unit tests do not need
PyTorch. Set the following task-specific paths yourself: `KG_BASE`, `KG_TRAIN`,
`KG_DEV`, `KG_ENTITY_ADAPTER`, `KG_CACHED_ENTITIES`, `KG_RUN`.

```bash
python3 build_pair_data.py --train "$KG_TRAIN" --dev "$KG_DEV" \
  --output-dir data/pair_v8 --seed 42

python3 train_relation_classifier.py --base-model "$KG_BASE" \
  --data-dir data/pair_v8 --output-dir "$KG_RUN/classifier_lr5e5" \
  --learning-rate 0.00005 --epochs 2 --batch-size 4 \
  --gradient-accumulation 4 --max-length 768 --seed 42

python3 evaluate_classifier.py --gold "$KG_DEV" \
  --entity-predictions "$KG_CACHED_ENTITIES" --base-model "$KG_BASE" \
  --selection "$KG_RUN/classifier_lr5e5/selection.json" \
  --output-dir "$KG_RUN/eval_predicted"

python3 evaluate_classifier.py --gold "$KG_DEV" \
  --entity-predictions "$KG_CACHED_ENTITIES" --base-model "$KG_BASE" \
  --selection "$KG_RUN/classifier_lr5e5/selection.json" \
  --output-dir "$KG_RUN/eval_oracle" --oracle

python3 evaluate_graphs.py --gold "$KG_DEV" \
  --predictions "$KG_RUN/eval_predicted/predictions.jsonl" \
  --output "$KG_RUN/rules_metrics.json" \
  --filter-output "$KG_RUN/predictions_rules.jsonl"

python3 evaluate_challenges.py --base-model "$KG_BASE" \
  --selection "$KG_RUN/classifier_lr5e5/selection.json" \
  --output "$KG_RUN/challenges.json"

python3 -m pytest tests -q
```

`KG_CACHED_ENTITIES` must contain one graph per dev record in the same order.
The primary comparison keeps these entity outputs byte-for-byte logically
unchanged and measures all gold edges, including unavailable candidates. Oracle
metrics use gold entities and are diagnostic only. The 148 historical dev rows
are a repeatedly inspected regression set, not an untouched blind test.

The original grouped export starts each source group with one structured ICD
record. The builder relies on that validated ordering; do not pass shuffled
records or a different provenance format without adapting group reconstruction.
The run manifest records input hashes, source groups, excluded/retained counts,
seed, hyperparameters and package versions. Full data and weights remain private.

## Live pipeline

```bash
python3 run_pipeline.py --input data/input.jsonl --output artifacts/predictions.jsonl \
  --base-model "$KG_BASE" --entity-adapter "$KG_ENTITY_ADAPTER" \
  --entity-mention-copy --max-spans-per-chunk 32 --max-chars-per-chunk 12000 \
  --relation-classifier-selection "$KG_RUN/classifier_lr5e5/selection.json"
```

`--relation-rules` optionally enables narrow postprocessing. Do not pass the
old `--relation-adapter` with the classifier. Entity generation still has its
own token limit; the classifier has an **input** budget and no relation output
token budget. The two model heads currently load separate bases, so plan for
their combined memory. For ICD-only input, no model arguments are needed.

After moving a checkpoint, update the `checkpoint` path in a copied
`selection.json`; keep the original run manifest immutable. No new checkpoint
is automatically substituted for a production model.

## Limits that remain

Source copying ensures provenance, not semantic entailment. Silver teacher
labels may omit valid relations. Rare/multiple relations, negation, implicit
main-disease references and distant evidence require more targeted reviewed
data. Local candidate retrieval intentionally trades coverage for shorter
inputs; its recall ceiling must be reported. Never describe `1-precision` as a
human-verified hallucination rate.
