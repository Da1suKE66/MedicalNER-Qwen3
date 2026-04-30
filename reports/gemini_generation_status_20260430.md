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

Current Flash artifact in this repository:

- `data/generated/gemini_split/flash_structure_011_110_schema0413_partial.json`

Known status before continuing:

- The Flash structure run for 11-110 was interrupted earlier after 90 records.
- It covers 11-100 and still needs 101-110 for structure.
- The Flash long-text pass for 11-110 still needs to be generated before a merged Flash 11-110 output can be produced.
