DPO24 independent continuation runtime; prepared only, not deployed or launched.

The root task plans a finite follow-up after delivering the label4 full130 result. The13:20UTC deadline is the root arrangement, not an exact time newly asserted as explicit user authorization. Start by12:40UTC only when the existing label4 queue has finished and the GPU is idle. Otherwise skip this continuation.

Use dpo24_config.json and launch_plan.prepared.json only after root checks those conditions. The wrapper requires --label4-result-delivered and validates the completed label4 full130 manifest, actual label4 adapter fingerprint and metrics digest. The prepared plan contains that flag; invoking it is the root acknowledgement and must occur only after delivery. Do not append or replace the active label4 plan before it completes.

24 reviewed TRAIN preferences,13positive-label,8negation-label,3subtype-direction; no source text in this runtime directory. Formal training is exactly one visit per pair, LR2e-6,beta0.1,accum4,6updates. The separate4pair smoke covers all3subclasses and the global3273token longest sample. Original SFT parent is reloaded independently for smoke and formal; the label4 or smoke adapter is never a parent. Parent/reference/data/encoding/causal-mask/EOS/save-reload validation and DPO math are unchanged from frozen round3/trainer.

CPU admission verifies24 train and32 excluded holdout pairs, source/group isolation, full chosen target and EOS, no truncation with maxlength4096. This is not a GPU smoke and does not establish any quality gain. The normal runtime performs its existing real-GPU reference/policy/save checks before formal completion.

Total smoke+formal subprocess budget600s. Full130 fixedE2 inference reserve1800s; evaluation duration remains an estimate, not a promise. Failure/timeout does not yield a successful formal receipt. New output directories are required.

Training count/review composition/update count changed together relative to previous experiments; results cannot isolate a single cause. Preferences remain silver-derived, not clinician-verified gold. Full fixedE2 full130 comparison uses unchanged labels, prompts, scorer and strict parser.

Deploy only source-free .py/config/plan/readme files. The root separately reconstructs and transfers the authorized reviewed24 release to resident paths. Never publish candidate_train.jsonl, raw API review bodies or private examples.
