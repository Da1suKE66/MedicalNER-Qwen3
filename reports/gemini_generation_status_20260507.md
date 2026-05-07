# Gemini Generation Status - 2026-05-07

This repository snapshot archives the previous generated datasets under `version1/` and adds the latest Gemini generation outputs that are currently available.

## Current complete outputs

| Area | File | Status |
| --- | --- | --- |
| Pro COT long_text | `data/generated/gemini_split/pro_cot_001_858_complete_schema0413.json` | Complete, 858 records, global_idx 0-857 covered |
| LLaMAFactory Pro COT long_text | `data/llamafactory/pro_cot_001_858_complete_llamafactory.json` | Complete, 858 ShareGPT samples |
| Flash structure | `data/generated/gemini_split/flash3_structure_001_858_complete_schema0413.json` | Complete, 858 records, 0 failures |
| Expanded eval samples | `data/samples/sample_eval_cases_expanded.json` | Complete expanded source records for the 5 eval cases |

## Current in-progress outputs

| Area | Status file | Progress |
| --- | --- | --- |
| Pro structure | `data/generated/gemini_split/pro_structure_resume_state_schema0413.json` | 449 / 858 successful, 409 missing |

The Pro structure run is still blocked by Gemini Pro daily quota. The latest recorded quota retry text in the state file is `21h9m29`.

## Archived version1 data

Previous generated and LLaMAFactory files were moved into:

- `data/generated/version1/`
- `data/llamafactory/version1/`

Those files are kept for reproducibility and comparison with the newer complete Pro COT and Flash structure outputs.

## Notes

- `data/llamafactory/dataset_info.json` now registers `pro_cot_001_858_complete_schema0413`.
- Pro structure chunk outputs are retained in `data/generated/gemini_split/pro_structure_*.json` so generation can resume by recomputing coverage from `source_id` / global index.
- Pro COT long_text is represented as a deduplicated complete file in this repository snapshot rather than every individual retry shard.
