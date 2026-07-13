# Schema regression set

`schema_regression_20.json` contains 20 records selected from the same 858 records used
to build the original training dataset. It is intentionally retained to reproduce known
structural failures such as description-derived duplicate nodes, relation direction, and
strict JSON formatting.

It is **not an independent test set** and must not be used to report model generalization.
Schema v2 generalization metrics belong to `data/schema_v2/splits/test_v2_heldout.json`,
whose source records are excluded from v2 training and grouped by ICD parent/child
components.

Files:

- `schema_regression_20.json`: review metadata and gold graphs.
- `schema_regression_20.jsonl`: line-oriented form for review tooling.
- `schema_regression_viewer.html`: local review UI.
- `scripts/build_schema_regression_dataset.py`: rebuilds the LLaMA-Factory representation.
