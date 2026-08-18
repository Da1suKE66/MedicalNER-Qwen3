# Relation scoring correction and revised diagnosis

## Confirmed scoring artifact

The previous relation scorer identified each endpoint by `(label, name, ICD-11 Code)`. On heldout `id=251`, checkpoint-450/500 generated all 32 relation triples and all 32 evidence strings exactly, but predicted the hub D1 code as `6A60.3` / `6A60.11` instead of target `6A60.4`. Because every relation connects to D1, the property mismatch incorrectly changed relation TP from 32 to 0.

Using `(label, normalized name)` for relation endpoint identity while scoring properties separately gives:

| checkpoint | split | strict relation F1 | name-aligned relation F1 | target-positive macro F1 |
|---|---|---:|---:|---:|
| 350 | train | 0.000 | 0.000 | 0.000 |
| 350 | heldout | 0.000 | 0.000 | 0.000 |
| 400 | train | 0.000 | 0.000 | 0.000 |
| 400 | heldout | 0.065 | 0.065 | 0.333 |
| 450 | train | 0.719 | 0.719 | 0.333 |
| 450 | heldout | 0.036 | 0.739 | 0.796 |
| 500 | train | 0.577 | 0.595 | 0.352 |
| 500 | heldout | 0.035 | 0.737 | 0.803 |

This 16-case curve does not demonstrate general overfitting. Relation-list generation emerges around checkpoint-450 on both train and heldout. The sample remains too small and is dominated by `id=301/251`.

## Confirmed teacher/student context mismatch

- In 702/1493 records, the target D1 contains an ICD code that is absent from the student user prompt.
- The canonical title is absent from 267/1493 prompts; title and code are both absent in 256 records.
- In the 16-case checkpoint-500 probe, all 10 codes present in the prompt were predicted correctly. Of the six codes absent from the prompt, only one was correct; that one is train `id=301`.
- `id=251` provenance supplies `source_code=6A60.4`, and the DeepSeek reasoning/target uses it, but the converted student prompt includes only the indexTerms text. The teacher therefore had privileged metadata that the student did not receive.

This is a data-conversion/input-identifiability issue, not evidence that DeepSeek output quality is globally poor.

## Remaining genuine relation weakness

After the same endpoint correction, the broader 25-case output-only probe improves only from relation F1 `0.151` to `0.177`. Therefore endpoint scoring does not explain all errors. On checkpoint-500 in the paired probe:

- indexTerms: `id=301` and `id=251` both reproduce 32/32 relation structures;
- diagnosticCriteria: `id=402` matches 1/16 and `id=28` matches 8/18 by name-aligned endpoints;
- exclusions: `id=675` matches 0/2 while `id=943` matches 2/2;
- definition: `id=1424/1434` emit additional source-grounded relations absent from the teacher target, which requires a human policy audit before calling them hallucinations.

The revised diagnosis is field-dependent relation instability plus student-input/target misalignment and evaluator coupling. Current evidence does not prove general overfitting or globally poor teacher quality.
