# Semantic extraction follow-up experiments

These programs continue the semantic entity-name and relation-label protocol.
They do not train JSON generation or numeric candidate selection. Run with
`PYTHONPATH=agent` and the pinned server environment used by `semantic.train`.

- `build_pairs.py` reconstructs preferences from the admitted resident TRAIN
  release. The chosen completion is the original complete SFT completion. A
  rejected completion introduces one verified error; absent annotations are
  never treated as negatives. Source groups separate preference training and
  mechanical checks. The mechanical checks are not an unseen medical test.
- `semantic_dpo.py` initializes policy and frozen reference from the exact
  selected canonical SFT adapter. It scores completion tokens including EOS,
  masks the prompt, verifies reference stability and saved/reloaded weights,
  and rejects incomplete, timed-out or smoke-only runs for normal evaluation.
- `run_smoke_then_train.py` checks the explicit candidate SHA and count, runs
  four longest-category GPU smoke pairs, then starts formal training afresh
  from the original SFT. Its private JSON configuration supplies literal
  command arguments, output paths, expected pair count and training SHA.
- `short_window_screen.py` compares source-only character windows with fixed
  adapters. A pilot selects long inputs by source-text SHA, without inspecting
  labels. `--sample-count 0 --selection-stratum all` scores all original130
  cases; pilot results must not be presented as full-population scores.
- `build_relation1600.py` reconstructs the shorter relation supervision using
  resident training rows and the existing anchor receipt. Cross-window atoms
  retain explicit full-source auxiliary training rows. Inference never uses
  gold to choose windows.
- `rebuild_quality_data.py` applies source-free, hash-checked recipes to
  resident TRAIN rows: reviewed single-relation preference mutations or
  complete-row replay of long diagnostic criteria. It does not create new
  original documents or repair evaluation labels.

Runtime artifacts, datasets, model outputs, API credentials and weights belong
outside this repository. These scripts have explicit input/parent locks for
this experiment; do not remove those checks to mix releases.

The primary outcome is live pipeline typed directed relation set precision,
recall and F1 against every original gold item. Report failed samples and the
raw graph companion metric. Structured deterministic cases are a separate
stratum. Oracle-entity results and invalid-attempt penalties are diagnostics.

The isolated [relation-label weighted SFT candidate](label_weight_sft/README.md)
continues the original relation adapter on all1000 admitted rows. It preserves
every target token and EOS while increasing only semantic relation-label token
weights. CPU math and data checks are preparation evidence, not quality results;
its GPU smoke, complete training and fixed-entity full130 evaluation must each
succeed before it can be selected.
