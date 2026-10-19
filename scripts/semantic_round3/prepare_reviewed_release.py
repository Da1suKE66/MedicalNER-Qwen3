"""Package the reviewed24 source-grounded preferences; local CPU only.

Every chosen sequence is retained exactly from frozen TRAIN. This packages agent
and teacher evidence, not medical expert gold. No inference or network is used.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'MedicalNER-Qwen3/agent'))
sys.path.insert(0, str(HERE / 'trainer'))
import build_pairs as b
from semantic.data import canonical, sha

RECIPE_SHA = 'b70dae69df3ac276443340c62a1aa6b497ce4994c4ffafc9d16bb70f8bef7d10'
CANDIDATE_SHA = '6413573775641eb7f9f9e28bd8881404ce3cdfaa2a12ae858309df73941b23b8'
PREFERENCES_SHA = '4e983b88c33acaa471150d594321aae09e920748e5c3b5740357153fc4fbdabc'
HOLDOUT_SHA = '1e9455694d5452f223cf3df190ff247f937791c4c6ac309ee9923f1fd3ce6abe'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    out = HERE / 'reviewed24'
    b.require(not out.exists(), 'Never overwrite a released dataset')
    review_dir = HERE / 'root_release_review'
    recipe_path = review_dir / 'selection_recipe.json'
    pref_path = review_dir / 'reviewed_preferences.private.jsonl'
    candidates_path = HERE / 'candidates.private.jsonl'
    b.require(sha(recipe_path.read_bytes()) == RECIPE_SHA, 'Review recipe changed')
    b.require(sha(pref_path.read_bytes()) == PREFERENCES_SHA, 'Reviewed preferences changed')
    b.require(sha(candidates_path.read_bytes()) == CANDIDATE_SHA, 'Original candidates changed')
    recipe = read(recipe_path)
    candidates = {c['id']: c for c in rows(candidates_path)}
    sources, entities, full, locks = b.load_sources(
        ROOT / 'audit_work/full_semantic_20261019/data_v2',
        ROOT / 'audit_work/full_semantic_20261019/relation_complete_overlap_v2')
    held_path = ROOT / 'audit_work/round2_20261019/dpo52_v2/pairs52/mechanical_holdout.jsonl'
    held_blob = b.locked(held_path, HOLDOUT_SHA)
    held = rows(held_path)
    held_groups = {str(p['source_row']['group_id']) for p in held}
    held_donor_groups = {str(p['mutation']['donor']['group_id']) for p in held if 'donor' in p['mutation']}
    prefs = rows(pref_path)
    b.require(len(prefs) == recipe['release_count'] == 24, 'Wrong release count')
    selected = {r['source_row_id']: r for r in recipe['selected']}
    b.require(len(selected) == 24 and len(candidates) == 31, 'Duplicate/missing selected row')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'audit_work/label_correction_20261012/base_input', local_files_only=True)
    result, encoded = [], []
    for pref in prefs:
        entry = selected[pref['source_row_id']]
        c = candidates[entry['candidate_id']]
        row = sources[pref['source_row_id']]
        b.require(row == c['source_row'] and sha(row) == entry['source_row_sha256'], 'Source drift')
        b.require(entry['review']['accepted'] is True and entry['review']['complete_chosen_reviewed'] is True,
                  'Full chosen review not approved')
        b.require(entry['review']['human_adjudicated'] is False, 'Do not fabricate human adjudication')
        b.require(pref['rejected'] == c['rejected'] and sha(pref['rejected']) == entry['rejected_sha256'],
                  'Reviewed negative changed')
        v1_path = HERE / 'api_reviews' / (c['id'] + '.json')
        v1 = read(v1_path)
        v2_path = HERE / 'api_reviews_v2' / (c['id'] + '.json')
        admitted_v1 = v1['accepted_by_two_teacher_calls'] is True
        admitted_v2 = v2_path.exists() and read(v2_path)['admissible_clear_prompt'] is True
        b.require(admitted_v1 or admitted_v2, 'No completed teacher review')
        b.require(str(row['group_id']) not in held_groups | held_donor_groups, 'Holdout overlap')
        b.require(row['metadata']['source_index'] != 188, 'Known questionable source excluded')
        pair = b.make_pair(row, pref['mutation'], rejected=pref['rejected'], review=pref['review'])
        b.verify_pair(pair, sources, entities, full)
        b.require(pair['chosen'] == row['completion'], 'Chosen must remain frozen')
        encoded.append(b.encode_pair(pair, tokenizer, 4096))
        result.append(pair)
        locks[str(v1_path.resolve())] = sha(v1_path.read_bytes())
        if v2_path.exists():
            locks[str(v2_path.resolve())] = sha(v2_path.read_bytes())
    b.require(len({p['id'] for p in result}) == len({str(p['source_row']['group_id']) for p in result}) == 24,
              'Preference or group duplication')
    kinds = dict(Counter(p['mutation']['semantic_change_kind'] for p in result))
    b.require(kinds == recipe['category_counts'], 'Category balance drift')
    for path in (recipe_path, pref_path, candidates_path, held_path):
        locks[str(path.resolve())] = sha(path.read_bytes())
    train_blob = ''.join(canonical(p) + '\n' for p in result).encode()
    manifest = {
        'data_version': b.VERSION, 'release_version': 'reviewed24-bidirectional-semantic-round3-v1',
        'parent_view': 'short_canonical', 'protocol_sha256': b.LOCKS['protocol'],
        'train_pairs': 24, 'train_source_groups': 24, 'mechanical_holdout_pairs': len(held),
        'mechanical_holdout_groups': len(held_groups), 'group_overlap': 0,
        'original_tune_group_overlap': 0, 'holdout_source_and_donor_group_overlap': 0,
        'source188_source_or_donor_occurrences': 0,
        'output_sha256': {'candidate_train.jsonl': sha(train_blob), 'mechanical_holdout.jsonl': sha(held_blob)},
        'mutation_counts': {'reviewed_semantic_contradiction': 24}, 'semantic_change_counts': kinds,
        'no_truncation': True, 'max_length': 4096,
        'max_prompt_plus_completion_tokens': max(len(e['prompt_ids']) + max(len(e['chosen_ids']), len(e['rejected_ids'])) for e in encoded),
        'prompt_tokens_one_pass': sum(len(e['prompt_ids']) for e in encoded),
        'chosen_tokens_one_pass_including_eos': sum(len(e['chosen_ids']) for e in encoded),
        'rejected_tokens_one_pass_including_eos': sum(len(e['rejected_ids']) for e in encoded),
        'completion_length_delta_counts': dict(Counter(len(e['rejected_ids']) - len(e['chosen_ids']) for e in encoded)),
        'full_chosen_records': sum(p['source_row']['target_count'] for p in result),
        'input_file_locks': locks, 'preparer_sha256': sha(Path(__file__).read_bytes()),
        'locked_builder_sha256': sha(Path(b.__file__).read_bytes()),
        'full_chosen_unchanged': True, 'unknown_as_negative': False, 'human_adjudicated': False,
        'medical_gold': False, 'status': 'local_data_release_prepared_not_trained',
        'external_API_calls_this_packaging_step': 0, 'successful_API_calls_in_prior_review': 62,
        'limitations': [
            'Agent and teacher review are not clinical expert adjudication.',
            'Chosen remains existing non-exhaustive silver; this adds preferences, not independent SFT sources.',
            'Seven uncertain full responses remain quarantined; no gold or old training artifact was changed.',
            'Bidirectional labels reduce one-sided preference construction but do not prove model improvement.',
            'Mechanical holdout was drawn from TRAIN and is not a clinical generalization test.',
            'No new DPO training or evaluation has run on this release.'
        ]}
    out.mkdir(mode=0o700)
    for name, blob in [('candidate_train.jsonl', train_blob), ('mechanical_holdout.jsonl', held_blob),
                       ('manifest.json', (canonical(manifest) + '\n').encode())]:
        path = out / name
        path.write_bytes(blob)
        path.chmod(0o600)
    print(canonical({k: manifest[k] for k in ['train_pairs', 'train_source_groups', 'semantic_change_counts',
        'max_prompt_plus_completion_tokens', 'output_sha256', 'full_chosen_records', 'completion_length_delta_counts']}))


if __name__ == '__main__':
    main()
