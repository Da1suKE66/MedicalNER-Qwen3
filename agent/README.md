# Reliable two-stage medical KG agent

**Recommended entry, 2026-10-19: semantic text for both stages.**
[`run_semantic_pipeline.py`](run_semantic_pipeline.py) runs the actual causal-LM
entity stage followed by the relation stage, applies source/semantic rules, and
assembles graph JSON in Python. Entity completions are
`type<TAB>source mention<TAB>full canonical name`; relation completions are
`source type<TAB>source name<TAB>full relation label<TAB>target type<TAB>target name<TAB>source evidence`.
Neither model generates graph JSON, numeric label indices or hash references.
This entry requires completed, non-smoke `semantic-text-v1` training receipts
with matching stage, protocol, base configuration/tokenizer and adapter weights.
Historical JSON, numeric-choice and four-field relation-label adapters are
rejected. A working entry point does not establish clinical accuracy.

Run from the repository root with local, approved semantic adapters:

```bash
PYTHONPATH=agent python agent/run_semantic_pipeline.py \
  --source-file /private/path/original-source.txt \
  --base-model /private/path/Qwen3-8B \
  --entity-adapter /private/path/semantic-entities \
  --relation-adapter /private/path/semantic-relations \
  --variant full_source --output /private/path/prediction.json
```

An original structured ICD JSON/header input uses `SourceRouter` and
`parse_icd_graph` as a separate deterministic path and needs no model arguments
or GPU. It must contain an explicit nonempty title/name. Pass original source
files, not chat records, gold or previously extracted graph JSON.

For a free-text passage whose subject title comes from a separate original
header in the same source group, explicitly add:

```bash
  --title-source-file /private/path/original-structured-header.json \
  --source-group-id declared-source-group
```

The CLI verifies the header type, literal title slice, original input hashes
and receipt using the shared runtime verifier. The group ID records the caller's
same-group declaration; it does not independently prove a cross-file provenance
relationship. Conflicting embedded group IDs, absent literal titles and
unverifiable headers fail. Without an explicit title source, no title is guessed
or read from gold. Unicode-escaped JSON title text without a literal source
slice is rejected for this title-context path.

The output is an envelope with `status`, `graph`, input hashes and a compact
summary. Complete raw inputs, prompts, generations, parsing and failure traces
are stored alongside it as `prediction.json.trace.json`; both files use mode
`0600` and existing files are never overwritten. A failed generation exits
nonzero with `graph: null` and any `partial_graph` explicitly marked. A completed
empty sequence is distinguished from failed generation and from predictions
removed by rules. Stdout prints only paths, counts and status. No oracle-entity
mode is exposed by this live CLI.

`--variant full_source|short_source|short_canonical` controls the relation stage;
the latter two use source-derived relation windows and the complete validated
entity table. The entity stage defaults to `--entity-variant full_source`.
For an entity adapter trained on complete source windows, add
`--entity-variant short_source --short-chars 2400`; `--variant full_source` can
still use the existing full-source relation adapter. The CLI explicitly accepts
the entity training variants `short_source` and `short_source_complete_windows`
for that input view and rejects other mismatches. The output and private trace
retain the actual training variant, input view and training data SHA; matching
input views alone does not establish complete-window supervision. Successful
entity windows merge by type/name with global body offsets. Any failed window
keeps the result failed, with its successful portion only in `partial_graph`.

In frozen data release v2, canonical relation targets
sort by `(source type, normalized source name, target type, normalized target
name, relation label, evidence start)`. Source-ordered relation targets sort by
`(source entity position, target entity position, relation label, evidence start)`;
they are neither opaque-ID order nor pure evidence-position order. The relation
parser accepts arbitrary record order, and the prompt's `order` selects entity
table presentation. The legacy paths below remain for historical reproduction.

**2026-10-12 user correction: literal relation labels.** The selected relation
output is now readable records such as
`E001<TAB>is_core_symptom_of<TAB>E002<TAB>SENT_004`, using full semantic labels
and existing source references. Python validates records, merges duplicates,
copies evidence and assembles JSON. There is no dynamic relation-label index,
numeric state machine or ascending-output requirement in this relation path.
Qwen remains a causal LM with LoRA, without a classification head.

[`train_label_relations.py`](train_label_relations.py) uses the shared SFT
trainer with literal-label targets and a distinct training receipt.
[`evaluate_label_relations.py`](evaluate_label_relations.py) and
[`evaluate_label_medical.py`](evaluate_label_medical.py) retain complete gold
denominators and explicitly account for parse/termination failures.
[`run_label_pipeline.py`](run_label_pipeline.py) connects that relation adapter
to a separately verified entity adapter. The entity stage still has its own
existing output contract; this correction does not claim new entity training.
Numeric/JSON relation checkpoints are not interchangeable with a label-trained
checkpoint. The new code is prepared; no new trained label checkpoint or quality
improvement is claimed by the code change. See the correction in
[the delivery plan](../docs/relation_delivery_plan_20261012.md).

The earlier [`run_choice_pipeline.py`](run_choice_pipeline.py) and its integer
training/evaluation entries remain available for historical reproduction.

The current literal-label compiler admits only recipe-verified artificial training worlds for
engineering verification. Medical silver annotation completeness, clean entity
splits and UNKNOWN loss migration still require a separate release. Passing a
decoder test or a short GPU run does not establish improved medical F1.

The classifier documentation below describes the historical implementation.

The existing ICD parser, Qwen entity adapter and deterministic JSON assembly
remain. An opt-in, non-generative relation head replaces long relation-JSON
generation. Read [the audit](reports/20260907/AUDIT.md) and
[the experiment plan](PLAN.md) and [measured results](reports/20260907/RESULTS.md)
before treating scores as production accuracy. The relation head improves the
regression baseline, but end-to-end quality remains insufficient for deployment.

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

### Optional per-type thresholds

`calibrate_relation_thresholds.py` fits a second policy from the selected epoch's
saved tuning sweep only. `rethreshold_predictions.py` compares it on cached
scores without rerunning the model. The resulting `selection_labelwise.json`
can be passed to the same live pipeline. It abstains on two unreliable
diagnostic-relation types; inspect the report before enabling it. An optional
type guard added after regression inspection is separately marked post-hoc.

`evaluate_identity_diagnostic.py` only quantifies spelling/punctuation effects.
It does not change primary scoring, model selection or production identities.
The audit also reports source-word corruption: the frozen experiment corpus
still contains known bad words, so a future training run must first connect the
existing reviewed correction dictionary to this data path.
