# Semantic-focused DPO52 candidate

This immutable second candidate contains 36 locally reviewed semantic contrasts
(19 explicit-negation errors and 17 reversed subtype relations), plus 16
outside-entity-table references selected deterministically from the prior data.
It excludes source188 as a source or donor. No new clinical facts or exhaustive
negative annotations are generated. These files contain code and recipes using
source IDs and hashes; private source text and preference records are excluded.

`prepare_release.py` rebuilds the dataset from already-resident TRAIN files and
checks both prior16 and new20 reviewed preference outputs byte for byte.
`launch_plan.json` records explicit reconstruction and training arguments.
`run_smoke_then_train.py` starts both smoke and formal training independently
from the original selected relation SFT. The smoke covers the globally longest
pair and all three actual error categories.

The planned run uses one epoch, LR2e-6, beta0.1, accumulation4, and13 updates.
Composition, learning rate and update count change together: this is an
exploratory quality comparison, not a single-factor ablation. A complete130-case
live pipeline evaluation is required before selecting this checkpoint.

See [preparation and deployment status](../../../docs/audits/20261019-round2/dpo52-preparation.json).
