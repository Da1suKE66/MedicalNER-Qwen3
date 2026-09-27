"""Build immutable SFT-only, matched-data experiments from an audited v2 release.

No semantic adjudication, new gold labels, model calls, or automatic deployment.
The noise arm replaces rows rather than duplicating positive supervision. The
context arms have identical rows, decisions and loss masks, differing in spans.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from generative.chat import render_prompt
from generative.protocol import dumps
from robustness.entity_noise import fingerprint, prompt_payload, protocol_prefix
from .release_v2 import canonical, sha, strict_json
from .supervision_v2 import masked_token_labels, validate_partial_row


def read_rows(path):
    rows = [strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty data or duplicate row IDs')
    for row in rows:
        validate_partial_row(row)
    return rows


def positive_signatures(rows):
    """Count per-source directed labels AND exact cited evidence, not just sets."""
    signatures = Counter()
    for row in rows:
        pairs = {p['pair_id']: p for p in row['pairs']}
        for decision in row['target']['decisions']:
            pair = pairs[decision['pair_id']]
            for edge in decision['relations']:
                signatures[(row['sample_id'], pair['source'], pair['target'],
                            edge['relation'], tuple(sorted(edge['evidence_span_ids'])))] += 1
    return signatures


def noise_replacement(base, auxiliary):
    parents = {r['id']: r for r in base}
    replacements = {}
    for noisy in auxiliary:
        validate_partial_row(noisy)
        meta = noisy['entity_noise']
        parent_id = meta['source_row_id']
        if parent_id not in parents or parent_id in replacements:
            raise ValueError('Missing or duplicate noise parent')
        parent = parents[parent_id]
        if noisy.get('split') != 'train' or parent.get('split') != 'train':
            raise ValueError('Noise must use train-only parents')
        if meta['source_row_sha256'] != fingerprint(parent):
            raise ValueError('Noise parent provenance mismatch')
        if any(noisy.get(k) != parent.get(k) for k in ('group_id', 'sample_id', 'sample_weight', 'spans', 'original_spans')):
            raise ValueError('Noise changed source group, context, or weight')
        if meta.get('construction') not in {'absent_synthetic_name', 'absent_synthetic_name_and_type'}:
            raise ValueError('Only explicitly manufactured absent entities are eligible')
        if positive_signatures([parent]) != positive_signatures([noisy]):
            raise ValueError('Noise changed positive/evidence exposure')
        incident = set(meta['incident_pair_ids'])
        negatives = {p for p, status in noisy['pair_supervision'].items() if status == 'SYNTHETIC_NEGATIVE'}
        if not negatives or negatives != incident:
            raise ValueError('Synthetic negative provenance mismatch')
        # Compare every original directed pair; a silver omission cannot become
        # a newly asserted negative because of the augmentation.
        original_status = {(p['source'], p['target']): parent['pair_supervision'][p['pair_id']] for p in parent['pairs']}
        noisy_status = {(p['source'], p['target']): noisy['pair_supervision'][p['pair_id']] for p in noisy['pairs']}
        if any(noisy_status.get(k) != v for k, v in original_status.items()):
            raise ValueError('Noise changed original pair supervision')
        replacements[parent_id] = deepcopy(noisy)
    result = [replacements.get(row['id'], deepcopy(row)) for row in base]
    if len({r['id'] for r in result}) != len(base) or positive_signatures(result) != positive_signatures(base):
        raise ValueError('Unmatched noise/control exposure')
    return result


def context_pair(base, local):
    full_by_id = {r['id']: r for r in base}
    full_common, local_common = [], []
    for row in local:
        if row['id'] not in full_by_id:
            raise ValueError('Local row has no full-context parent')
        full = full_by_id[row['id']]
        for key in ('sample_id', 'group_id', 'split', 'sample_weight', 'entities', 'pairs'):
            if full.get(key) != row.get(key):
                raise ValueError(f'Unmatched context pair: {key}')
        if positive_signatures([row]) - positive_signatures([full]):
            raise ValueError('Local contains new positive/evidence supervision')
        full_spans = {s['span_id']: s['text'] for s in full['spans']}
        if any(full_spans.get(s['span_id']) != s['text'] for s in row['spans']):
            raise ValueError('Local text is not a literal subset of full spans')
        # Copy the local targets/masks, then restore full context. This yields
        # exactly matched completion token exposure, not only matched labels.
        control = deepcopy(row)
        control['spans'] = deepcopy(full['spans'])
        control['original_spans'] = deepcopy(full['original_spans'])
        control['prompt'] = protocol_prefix() + '\n' + dumps(prompt_payload(control)) + '\n'
        control['experiment_provenance'] = {'contract': 'local_common_targets_full_input_v1',
                                          'full_row_sha256': fingerprint(full),
                                          'local_row_sha256': fingerprint(row)}
        validate_partial_row(control)
        validate_partial_row(row)
        full_common.append(control)
        local_common.append(deepcopy(row))
    return full_common, local_common


def audit_rows(rows, tokenizer=None, max_length=6144):
    counts = Counter(rows=len(rows))
    for row in rows:
        validate_partial_row(row)
        counts.update(row['pair_supervision'].values())
        counts['positive_edges'] += sum(len(d['relations']) for d in row['target']['decisions'])
        if tokenizer is not None:
            ids, labels = masked_token_labels(row, tokenizer)
            prompt_n = len(tokenizer.encode(render_prompt(tokenizer, row['prompt']), add_special_tokens=True))
            total = prompt_n + len(ids) + 1
            if total > max_length:
                raise ValueError(f'Overlength row {row["id"]}: {total}>{max_length}')
            counts['max_sequence_tokens'] = max(counts['max_sequence_tokens'], total)
            counts['prompt_tokens_once'] += prompt_n
            counts['completion_tokens_once'] += len(ids) + 1
            counts['supervised_completion_tokens_once'] += sum(v != -100 for v in labels) + 1
            counts['masked_completion_tokens_once'] += sum(v == -100 for v in labels)
    return dict(counts)


def build(release, output, tokenizer_dir=None, max_length=6144):
    release, output = Path(release), Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing experiment bundle')
    manifest = strict_json((release / 'manifest.json').read_text())
    # Verify every original export, including frozen split and review artifacts.
    for name, digest in manifest['output_sha256'].items():
        if sha((release / name).read_bytes()) != digest:
            raise ValueError(f'Release hash mismatch: {name}')
    variants = {name: {} for name in ('V2_base', 'V2_noise10', 'V2_full_common', 'V2_G5_local_common')}
    for split in ('train', 'tune'):
        base = read_rows(release / 'base' / f'{split}.jsonl')
        local = read_rows(release / 'local' / f'{split}.jsonl')
        if any(r['split'] != split for r in base + local):
            raise ValueError('Frozen split tag mismatch')
        variants['V2_base'][split] = base
        variants['V2_noise10'][split] = (noise_replacement(base, read_rows(release / 'synthetic_negative_auxiliary/train.jsonl'))
                                        if split == 'train' else deepcopy(base))
        common, short = context_pair(base, local)
        variants['V2_full_common'][split] = common
        variants['V2_G5_local_common'][split] = short
    train_groups = {r['group_id'] for v in variants.values() for r in v['train']}
    tune_groups = {r['group_id'] for v in variants.values() for r in v['tune']}
    if train_groups & tune_groups:
        raise ValueError('Train/tune source-group leakage')
    tokenizer = None
    if tokenizer_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if not tokenizer.is_fast or tokenizer.eos_token_id is None:
            raise ValueError('Require a fast tokenizer and explicit EOS')
    audits = {name: {split: audit_rows(rows, tokenizer, max_length) for split, rows in files.items()}
              for name, files in variants.items()}
    output.mkdir(parents=True)
    hashes = {}
    for name, files in variants.items():
        (output / name).mkdir()
        for split, rows in files.items():
            path = output / name / f'{split}.jsonl'
            with path.open('x') as handle:
                handle.write(''.join(canonical(r) + '\n' for r in rows))
            hashes[str(path.relative_to(output))] = sha(path.read_bytes())
    matrix = {'version': 'repair-v2-controlled-20260928',
              'defaults': {'repair': True, 'strict_evidence': True, 'matching': 'nfkc_v2',
                           'pairs_per_batch': 16, 'order': 'hash', 'context': 'full'},
              'variants': {name: ({'context': 'local'} if name == 'V2_G5_local_common' else {}) for name in variants}}
    with (output / 'experiments.json').open('x') as handle:
        handle.write(json.dumps(matrix, indent=2) + '\n')
    hashes['experiments.json'] = sha((output / 'experiments.json').read_bytes())
    report = {'version': matrix['version'], 'source_manifest_sha256': sha((release / 'manifest.json').read_bytes()),
              'source_release': str(release.resolve()), 'output_sha256': hashes, 'audits': audits,
              'code_sha256': sha(Path(__file__).read_bytes()), 'real_tokenizer_checked': tokenizer is not None,
              'tokenizer_config_sha256': sha((Path(tokenizer_dir) / 'tokenizer_config.json').read_bytes()) if tokenizer_dir else None,
              'max_length': max_length, 'train_tune_groups_disjoint': True,
              'noise_positive_evidence_exposure_matched': True,
              'context_rows_targets_masks_matched': True,
              'context_comparison_token_budget_matched': False,
              'eligible_for_existing_rl': False, 'human_adjudicated': False,
              'new_model_results_available': False,
              'caveats': ['Manufactured absent entities are not representative of all real NER errors.',
                          'Full/local control deliberately masks full-only positives to match local supervision.',
                          'Historical complete-graph silver metrics still penalize unannotated predictions; no labels are rewritten.',
                          'Equal steps/positive exposure do not imply equal input-token or compute budget.']}
    with (output / 'manifest.json').open('x') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer-dir')
    parser.add_argument('--max-length', type=int, default=6144)
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.output, args.tokenizer_dir, args.max_length), ensure_ascii=False))


if __name__ == '__main__':
    main()
