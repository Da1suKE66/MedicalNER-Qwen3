# Relations v2 output-only remote run

This is an engineering progress record for the run on `lsh-temp31030`.

- Run root: `/cache/liluchen/medicalner_relations_v2`
- Temporary recovery snapshots: `/temp/liluchen/medicalner_relations_v2/snapshots`
- Base model: Qwen3-8B ModelScope snapshot under `/cache/liluchen/model_cache/models/Qwen--Qwen3-8B/snapshots/master`
- SFT config: `configs/llamafactory/qwen3_8b_lora_deepseek_relations_v2_output_only_20260825.yaml`
- Data: revised `relations.csv`, compact output-only targets, relation-bearing train rows oversampled once
- Training: LoRA rank 8, all linear modules, bf16, gradient checkpointing, batch 1, accumulation 8, 3 epochs, `cutoff_len=16384`
- Preflight cutoff audit: train max total 8657 tokens, dev max 8951, 0 rows over 16384
- Step-50 checkpoint: `checkpoint-50/adapter_model.safetensors` (83 MiB), local recovery copy in `reports/remote_relations_v2_20260825/checkpoint-50/`
- Step-50 metrics: `eval_loss=0.09095559269189835`; training loss near the checkpoint was 0.0482--0.1283
- Step-100 checkpoint: `checkpoint-100/adapter_model.safetensors` (83 MiB), local recovery copy in `reports/remote_relations_v2_20260825/checkpoint-100/`
- Step-100 metric: `eval_loss=0.059865765273571014`
- Step-150 checkpoint: `checkpoint-150/adapter_model.safetensors` (83 MiB), local recovery copy in `reports/remote_relations_v2_20260825/checkpoint-150/`
- Step-150 metric: `eval_loss=0.04945255443453789`
- Step-200 checkpoint: `checkpoint-200/adapter_model.safetensors` (83 MiB), local recovery copy in `reports/remote_relations_v2_20260825/checkpoint-200/`; the same checkpoint was copied to both `snapshots/latest/output/checkpoint-200/` and `snapshots/periodic/checkpoint-200_manual/`.
- Step-200 metric: `eval_loss=0.04599086940288544`
- Step-201 observation: training remains active, GPU utilization 100%, memory about 68.9 GiB; no OOM, NaN, or interruption observed.
- Step-250 checkpoint: `checkpoint-250/adapter_model.safetensors` (83 MiB), copied to local recovery and to both `snapshots/latest/output/checkpoint-250/` and `snapshots/periodic/checkpoint-250_manual/`.
- Step-250 metric: `eval_loss=0.04204870015382767`
- Step-300 checkpoint: `checkpoint-300/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-300 metric: `eval_loss=0.040372252464294434`
- Step-350 checkpoint: `checkpoint-350/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-350 metric: `eval_loss=0.03768041357398033`
- Step-400 checkpoint: `checkpoint-400/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-400 metric: `eval_loss=0.03874474763870239`; this is above step-350, so the current best checkpoint remains step-350.
- Step-450 checkpoint: `checkpoint-450/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-450 metric: `eval_loss=0.03674810007214546`; this is the current best validation checkpoint.
- Step-500 checkpoint: `checkpoint-500/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-500 metric: `eval_loss=0.03637446463108063`; this is the current best validation checkpoint.
- Step-550 checkpoint: `checkpoint-550/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-550 metric: `eval_loss=0.03678682819008827`; step-500 remains best.
- Step-600 checkpoint: `checkpoint-600/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-600 metric: `eval_loss=0.035197049379348755`; this is the current best validation checkpoint.
- Step-650 checkpoint: `checkpoint-650/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-650 metric: `eval_loss=0.035780396312475204`; step-600 remains best.
- Step-700 checkpoint: `checkpoint-700/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-700 metric: `eval_loss=0.034858740866184235`; this is the current best validation checkpoint.
- Step-750 checkpoint: `checkpoint-750/adapter_model.safetensors` (83 MiB), copied to local recovery and both `/temp` snapshot tiers.
- Step-750 metric: `eval_loss=0.03482325002551079`; this is the current best validation checkpoint before final evaluation.
- Automatic post-train probe: `scripts/post_train_relations_v2_eval_remote.sh`; it uses `max_new_tokens=16384`, structured JSON stopping, closure audit, raw-output export, and the five-view relation scorer.

This is not a semantic-quality conclusion.  The relation metrics are intentionally deferred until the post-training free-generation audit.
