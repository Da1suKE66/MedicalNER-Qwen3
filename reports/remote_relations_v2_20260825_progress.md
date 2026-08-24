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
- Automatic post-train probe: `scripts/post_train_relations_v2_eval_remote.sh`; it uses `max_new_tokens=16384`, structured JSON stopping, closure audit, raw-output export, and the five-view relation scorer.

This is not a semantic-quality conclusion.  The relation metrics are intentionally deferred until the post-training free-generation audit.
