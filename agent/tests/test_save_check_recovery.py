from copy import deepcopy

import pytest

from robustness.recover_save_check import validate_recovery


def fixture(tmp_path):
    job = dict(mode='sft', data='train.jsonl', base_model='base', max_steps=279, save_steps=93,
               seed=42, batch_size=2, gradient_accumulation=8, max_length=6144,
               inputs={'training_data': {'sha256': 'data-sha'}}, reuse_training={'path': str(tmp_path / 'train')})
    report = dict(status='failed', arguments={k: v for k, v in job.items() if k not in ('inputs', 'reuse_training')},
                  error='Saved adapter reload differs from trained policy: 0.5', global_step=279,
                  train_metrics={'train_loss': .04}, policy_weights_changed=True,
                  policy_adapter_sha256_after='actual-trained-sha', data_sha256='data-sha',
                  save_reload_max_logit_difference=.5, final_adapter=str(tmp_path / 'train/final'))
    return report, job


def test_recovery_preserves_source_and_requires_exact_job(tmp_path):
    report, job = fixture(tmp_path)
    original = deepcopy(report)
    assert validate_recovery(report, job) == tmp_path / 'train'
    assert report == original


@pytest.mark.parametrize('field,value', [('status', 'running'), ('error', 'CUDA out of memory'),
    ('global_step', 278), ('policy_weights_changed', False), ('policy_adapter_sha256_after', ''),
    ('data_sha256', 'other'), ('final_adapter', '/unrelated/final'), ('save_reload_max_logit_difference', float('nan')),
    ('train_metrics', {'train_loss': float('nan')})])
def test_no_recovery_of_incomplete_or_different_failure(tmp_path, field, value):
    report, job = fixture(tmp_path)
    report[field] = value
    with pytest.raises(ValueError):
        validate_recovery(report, job)


@pytest.mark.parametrize('field', ['mode', 'data', 'base_model', 'max_steps', 'save_steps', 'seed', 'batch_size', 'gradient_accumulation', 'max_length'])
def test_recovery_cannot_silently_change_training_contract(tmp_path, field):
    report, job = fixture(tmp_path)
    report['arguments'][field] = 'changed'
    with pytest.raises(ValueError):
        validate_recovery(report, job)
