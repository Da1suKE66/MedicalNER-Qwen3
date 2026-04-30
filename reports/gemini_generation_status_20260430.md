# Gemini Split/COT Generation Status

Date: 2026-04-30

## Pro COT Progress

Model: `gemini-3-pro-preview`

Current combined files:

- `data/generated/gemini_split/pro_cot_001_471_partial_schema0413.json`
- `data/generated/gemini_split/pro_cot_001_471_partial_schema0413.metadata.json`
- `data/llamafactory/pro_cot_001_471_partial_llamafactory.json`

Current successful records: 456.

Covered range: 1-471, partial.

Complete ranges:

- 1-410 complete.
- 411-454 complete.
- 456 complete.
- 471 complete.

Missing records inside 1-471:

- 455
- 457-470

Records after 471 are not generated yet.

Validation for `pro_cot_001_471_partial_schema0413.json`:

- Duplicate `global_idx`: none.
- Empty `cot`: none.
- Empty output: none.
- Invalid relation endpoints: none.
- Chinese character hits: 0.
- LLaMA-Factory conversion: 456 records, 0 skipped.

Known blocker:

- Gemini returned `429 RESOURCE_EXHAUSTED`.
- Quota metric: `generativelanguage.googleapis.com/generate_requests_per_model_per_day`.
- Limit: 250 requests/day for `gemini-3.1-pro`.
- The failed region should be resumed from the missing records above after quota reset.

Recommended resume order:

1. Retry missing records 455 and 457-470.
2. Continue 472-510.
3. Continue in checked chunks: 511-610, 611-710, 711-810, 811-858.
4. After every chunk, validate missing indices, empty COT/output, invalid relation endpoints, and Chinese character hits.

## Flash Progress

Current Flash artifacts in this repository:

- `data/generated/gemini_split/flash_structure_011_110_schema0413_partial.json`
- `data/generated/gemini_split/flash_structure_011_110_schema0413.json`
- `data/generated/gemini_split/flash_long_text_011_110_schema0413.json`
- `data/generated/gemini_split/flash_merged_011_110_schema0413.json`

Current status:

- Flash 11-110 is complete.
- Structure pass: 100 records, 0 failures.
- Long-text pass: 100 records, 0 failures.
- Merged output: 858 manifest records, with generated KG content for records 11-110.
- Non-empty merged records: 100.
- Invalid relation endpoints: 0.
- Chinese character hits: 0.

Repair note:

- The Flash long-text output for `6A05` was truncated at a final incomplete relation.
- The raw output contained complete entities and 185 complete relations before the truncated tail.
- The final incomplete trailing relation was dropped and the JSON was closed before merging.
