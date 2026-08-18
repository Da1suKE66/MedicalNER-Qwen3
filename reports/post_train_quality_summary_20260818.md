# 2026-08-18 post-training quality audit

## Result

The problem is not solved by increasing the generation cap. The formal output-only and output-priority runs both used `cutoff_len=16384`, and converted training examples fit within that limit. Long-output inference still produced non-closing JSON on several diagnostic-criteria samples.

## Training

- Frozen group-disjoint split: 1,345 train / 148 held out, overlap 0.
- 3 epochs, 507 optimization steps, no OOM or interruption.
- Output-only best: `checkpoint-500`, eval loss `0.0229976`.
- Output-priority best: `checkpoint-500`, eval loss `0.3103762`.

## 25-case max-8192 comparison

Teacher-relative scores are diagnostic only because DeepSeek targets are noisy.

| model | schema-valid | exact | entity F1 | relation F1 | relation hallucination | cap hits |
|---|---:|---:|---:|---:|---:|---:|
| base Qwen | 13/25 | 1/25 | 0.1942 | 0.0000 | 100.0% | 2 |
| output-only | 23/25 | 6/25 | 0.5206 | 0.1508 | 85.7% | 2 |
| output-priority (completed partial probe) | 4/6 | 0/6 | 0.0526 | 0.0238 | 98.4% | 2/6 |

The priority row is explicitly a six-case timeout-bounded diagnostic, not a full-25 benchmark. Its completed cases already show that long thinking/relation hallucination is a serious regression.

## Cutoff and early-stop evidence

- Original CoT audit: 109/1493 examples exceed 16,384 tokens; 141 are near the cutoff.
- Converted output-only and output-priority audits: 0 examples exceed 16,384.
- Output-only IDs 4 and 320: 8,192-token run hit the cap; 16,384-token retry also hit the cap and remained invalid JSON (brace counts 271/268 and 357/354 in the structured-stop run).
- Priority ID 4: 8,192-token run hit the cap and had 179/176 braces with no `</output>`.
- A new opt-in stopper (`</output>` or first complete JSON object) did not stop ID4, ID11, or output-only IDs 4/320 early. This demonstrates that the model did not emit a complete object for the cap cases; the issue is not merely an output-limit setting.

## Decision

Do not choose the next checkpoint by loss alone. The 20-example overfit canary fit (train loss `0.0254289`, eval loss `0.075863`), so this is not a simple capacity/underfitting problem. The next training iteration should clean and shorten teacher thinking, keep output-only as the main objective, and add explicit JSON-completion/stop supervision or constrained decoding before another full run.

Machine-readable details are in [post_train_quality_summary_20260818.json](post_train_quality_summary_20260818.json).
