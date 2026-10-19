"""Post-hoc SFT serialization verification, never rewriting historical failure.

Only for the observed terminal AMP/plain precision mismatch. The original
in-memory logits no longer exist: this proves saved weights against the hash
recorded after training, not a newly observed comparison to that old process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from generative.train import require_saved_weight_equality


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_recovery(report, job):
    if job['mode'] != 'sft' or report.get('arguments', {}).get('mode') != 'sft':
        raise ValueError('Recovery is limited to the registered SFT jobs')
    error = report.get('error', '')
    if report.get('status') != 'failed' or not re.fullmatch(
        r'Saved adapter reload differs from trained policy: [0-9.eE+-]+', error
    ):
        raise ValueError('Only the known terminal precision check may be recovered')
    if report.get('global_step') != job['max_steps'] or not report.get('policy_weights_changed'):
        raise ValueError('The full optimizer budget must have actually completed')
    loss = report.get('train_metrics', {}).get('train_loss', float('nan'))
    if not math.isfinite(loss):
        raise ValueError('Training loss must be finite')
    delta = report.get('save_reload_max_logit_difference', float('nan'))
    if not math.isfinite(delta) or delta <= .02 or float(error.rsplit(': ', 1)[1]) != delta:
        raise ValueError('Unexpected original precision-check result')
    if not report.get('policy_adapter_sha256_after'):
        raise ValueError('Original trained weight fingerprint is mandatory')
    args = report['arguments']
    for key in ('data', 'base_model', 'max_steps', 'save_steps', 'seed', 'batch_size', 'gradient_accumulation', 'max_length'):
        if args.get(key) != job.get(key):
            raise ValueError(f'Recovery training configuration changed: {key}')
    if report.get('data_sha256') != job['inputs']['training_data']['sha256']:
        raise ValueError('Recovery data identity changed')
    source = Path(job['reuse_training']['path']).resolve()
    if Path(report['final_adapter']).resolve() != source / 'final':
        raise ValueError('Final adapter must belong to the exact source training')
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job-spec', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    job = json.loads(Path(args.job_spec).read_text())
    source = Path(job['reuse_training']['path'])
    manifest_path = source / 'run_manifest.json'
    original_hash = sha(manifest_path)
    report = json.loads(manifest_path.read_text())
    source = validate_recovery(report, job)
    from robustness.finite_queue import verify_inputs
    verify_inputs({'inputs': job['inputs'], 'code': job['reuse_training']['code'],
                   'code_hashes': job['reuse_training']['code_hashes']})
    checkpoints = sorted(source.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    expected = set(range(job['save_steps'], job['max_steps'] + 1, job['save_steps'])) | {job['max_steps']}
    if {int(p.name.split('-')[-1]) for p in checkpoints} != set(sorted(expected)[-3:]):
        raise ValueError('Retained checkpoint coverage changed')
    state_path = source / f"checkpoint-{job['max_steps']}" / 'trainer_state.json'
    if json.loads(state_path.read_text())['global_step'] != job['max_steps']:
        raise ValueError('Final checkpoint state disagrees with the completed budget')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from accelerate import Accelerator
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from generative.chat import render_prompt

    tokenizer = AutoTokenizer.from_pretrained(job['base_model'], local_files_only=True)
    with Path(job['data']).open() as f:
        row = json.loads(next(line for line in f if line.strip()))
    inputs = tokenizer(render_prompt(tokenizer, row['prompt']), add_special_tokens=True, return_tensors='pt').to('cuda')
    def load():
        base = AutoModelForCausalLM.from_pretrained(job['base_model'], torch_dtype=torch.bfloat16,
                                                   local_files_only=True, attn_implementation='sdpa')
        return PeftModel.from_pretrained(base, report['final_adapter'], local_files_only=True).to('cuda').eval()
    def forward(model):
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            return model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
    model = load()
    trained_hash = report['policy_adapter_sha256_after']
    require_saved_weight_equality(model, trained_hash)
    plain = forward(model)
    accelerator = Accelerator(mixed_precision='bf16')
    model = accelerator.prepare_model(model)
    wrapped = forward(model)
    model = accelerator.unwrap_model(model, keep_fp32_wrapper=False).eval()
    unwrapped = forward(model)
    amp_delta = float((wrapped - plain).abs().max())
    unwrap_delta = float((unwrapped - plain).abs().max())
    if abs(amp_delta - report['save_reload_max_logit_difference']) > 1e-6 or unwrap_delta > .02:
        raise RuntimeError('Original per-task AMP failure was not reproduced; do not recover')
    model.to('cpu')
    torch.cuda.empty_cache()
    other = load()
    require_saved_weight_equality(other, trained_hash)
    fresh = forward(other)
    delta = float((fresh - unwrapped).abs().max())
    if not math.isfinite(delta) or delta > .02:
        raise RuntimeError('Independent plain reconstruction still differs')
    if sha(manifest_path) != original_hash:
        raise RuntimeError('Historical manifest changed during verification')
    recovered = {**report, 'status': 'training_complete_evaluation_pending',
                 'save_reload_max_logit_difference': delta, 'saved_adapter_sha256_verified': trained_hash,
                 'save_reload_scope': 'Post-hoc exact recorded-trained/saved weight identity and fresh plain reconstruction; old in-memory logits were not retained',
                 'verification_recovery': {'source_manifest': str(manifest_path), 'source_manifest_sha256': original_hash,
                    'original_error': report['error'], 'amp_difference_reproduced': amp_delta,
                    'unwrap_difference': unwrap_delta, 'original_manifest_unchanged': True,
                    'source_checkpoint_state': str(state_path), 'source_checkpoint_state_sha256': sha(state_path),
                    'no_additional_optimizer_updates': True}}
    recovered.pop('error', None)
    with (output / 'run_manifest.json').open('x') as f:
        f.write(json.dumps(recovered, indent=2) + '\n')
    print(json.dumps(recovered['verification_recovery']))


if __name__ == '__main__':
    main()
