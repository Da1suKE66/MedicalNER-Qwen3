import copy
import json

import pytest

from data_repair.controlled_experiments import build, context_pair, noise_replacement, positive_signatures, audit_rows
from data_repair.supervision_v2 import inject_contract_negative, validate_partial_row
from robustness.finite_queue import evaluation_command
from test_partial_supervision_v2 import example, Tokenizer


def test_noise_replaces_parent_and_preserves_every_positive():
    row, _, doc = example()
    original = copy.deepcopy(row)
    noisy = inject_contract_negative(row, doc.raw)
    rows = noise_replacement([row], [noisy])
    assert len(rows) == 1 and rows[0]['id'] == noisy['id']
    assert positive_signatures(rows) == positive_signatures([row])
    assert sum(v == 'SYNTHETIC_NEGATIVE' for v in rows[0]['pair_supervision'].values()) == 2
    assert row == original


@pytest.mark.parametrize('field,value', [('group_id', 'wrong'), ('split', 'tune'), ('sample_weight', .5)])
def test_noise_metadata_drift_rejected(field, value):
    row, _, doc = example()
    noisy = inject_contract_negative(row, doc.raw)
    noisy[field] = value
    with pytest.raises(ValueError):
        noise_replacement([row], [noisy])


def test_duplicate_and_wrong_parent_rejected():
    row, _, doc = example()
    noisy = inject_contract_negative(row, doc.raw)
    with pytest.raises(ValueError, match='duplicate'):
        noise_replacement([row], [noisy, noisy])
    noisy['entity_noise']['source_row_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='provenance'):
        noise_replacement([row], [noisy])


def test_context_pair_preserves_complete_targets_masks_and_input_tables():
    full, _, _ = example()
    local, _, _ = example(context='local')
    originals = copy.deepcopy([full, local])
    controls, treatments = context_pair([full], [local])
    control, treatment = controls[0], treatments[0]
    for field in ('entities', 'pairs', 'completion', 'loss_mask_spans', 'pair_supervision'):
        assert control[field] == treatment[field]
    assert control['prompt'] == full['prompt']
    assert validate_partial_row(control)
    assert [full, local] == originals


def test_context_missing_parent_or_id_drift_rejected():
    full, _, _ = example()
    local = copy.deepcopy(full)
    local['id'] = 'unknown'
    with pytest.raises(ValueError, match='parent'):
        context_pair([full], [local])
    local['id'] = full['id']
    local['group_id'] = 'wrong'
    with pytest.raises(ValueError, match='group_id'):
        context_pair([full], [local])


def test_context_changed_text_is_not_shorter_context():
    row, _, _ = example()
    local = copy.deepcopy(row)
    local['spans'][0]['text'] = 'Unrelated text.'
    with pytest.raises(ValueError, match='literal subset'):
        context_pair([row], [local])


def test_counts_keep_unknown_distinct_from_synthetic_negatives():
    row, _, doc = example()
    noisy = inject_contract_negative(row, doc.raw)
    counts = audit_rows([noisy])
    assert counts['UNKNOWN_UNANNOTATED'] == 5
    assert counts['SUPPORTED_SILVER'] == 1
    assert counts['SYNTHETIC_NEGATIVE'] == 2


def test_new_matrix_forwarded_without_changing_output_budget():
    job = dict(mode='sft', tune_gold='tune', regression_gold='test', entity_predictions='cache',
               base_model='base', variant='V2_base', matrix='frozen/matrix.json')
    command = evaluation_command(job, 'adapter', 'output', tune=True)
    assert command[command.index('--matrix') + 1] == 'frozen/matrix.json'
    assert command[command.index('--max-new-tokens') + 1] == '16000'
    assert '--entity-predictions' not in command


def test_real_synthetic_release_and_bundle_reject_overwrite_and_drift(tmp_path):
    from test_release_v2 import fixture
    from data_repair.release_v2 import run, sha
    source, split = fixture(tmp_path)
    release = tmp_path / 'release'
    run(source, split, release, sha(source.read_bytes()))
    # Small fixtures may select no 10% synthetic rows. This bundle requires an
    # actual noise treatment, not an empty "experiment" silently called noise.
    base_path = release / 'base/train.jsonl'
    base = [json.loads(line) for line in base_path.read_text().splitlines()]
    auxiliary = inject_contract_negative(base[0], ' '.join(s['text'] for s in base[0]['spans']))
    aux_path = release / 'synthetic_negative_auxiliary/train.jsonl'
    aux_path.write_text(json.dumps(auxiliary) + '\n')
    manifest_path = release / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['output_sha256']['synthetic_negative_auxiliary/train.jsonl'] = sha(aux_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    result = build(release, tmp_path / 'bundle')
    assert result['context_rows_targets_masks_matched']
    assert result['noise_positive_evidence_exposure_matched']
    assert result['eligible_for_existing_rl'] is False
    assert result['real_tokenizer_checked'] is False
    with pytest.raises(ValueError, match='overwrite'):
        build(release, tmp_path / 'bundle')
    base_path.write_text('tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        build(release, tmp_path / 'other')


def test_v2_evaluator_retains_original_unicode_evidence():
    from generative.evaluate import run_records
    from generative.protocol import ProtocolConfig, dumps
    from test_release_v2 import message
    text = 'Feature A at 5㎎ is a core symptom of Condition B.'
    graph = {'entities': [{'id': 'S', 'name': 'Feature A', 'label': 'Symptom'},
                          {'id': 'D', 'name': 'Condition B', 'label': 'Disease'}],
             'relations': []}
    def generate(batch):
        return {'text': dumps({'decisions': [
            {'pair_id': p['pair_id'], 'relations': ([{'relation': 'is_core_symptom_of',
             'evidence_span_ids': [batch['spans'][0]['span_id']]}] if p['source'] == 'S' else [])}
            for p in batch['pairs']]})}
    predictions, _, _ = run_records([message(text, graph)], None,
                                    ProtocolConfig(matching='nfkc_v2'), generate, oracle=True)
    assert predictions[0]['output']['relations'][0]['evidence'] == text
