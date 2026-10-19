"""Build DPO52 locally or remotely using resident TRAIN and source-free recipes.

Never reads evaluation text or predictions. Never transmits any data. The fixed
source-free selection recipe identifies 16 unchanged prior reference pairs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import build_pairs as b
from rebuild_quality_data import reconstruct_preferences
from semantic.data import canonical, sha

PARENT_DATA_SHA='1191b1e6ffa56a352539e86614179e57f89efc760fe580ddb582a1c200bea3b9'
HOLDOUT_SHA='1e9455694d5452f223cf3df190ff247f937791c4c6ac309ee9923f1fd3ce6abe'
OLD16_SHA='101cd3976e6181d2fabb4c8b57c80a643c8d5789b81f889f97b7089b5aacc0b0'
NEW20_SHA='4981d89f5c04be273a767d75fc225832dd3876734053b21b6825d046fa6a6466'
SELECTION_SEED='dpo52-v2-outside-selection-20261019'
LIMITATIONS=[
 '52-pair exploratory DPO: 36 reviewed semantic contrasts and 16 entity-reference contrasts.',
 'Chosen retains complete original silver and uncertainty in unmodified records.',
 'Source188 is excluded as source or donor from preferences; the unchanged SFT parent is not relabelled.',
 'Composition, learning rate and optimizer-update count change together relative to DPO208; no single-factor causal claim.',
 'The retained mechanical holdout comes from original TRAIN and is not a clinical generalization test.',
 'No unlabelled relation is declared false; no new medical facts are generated.',
]


def read(path):return json.loads(Path(path).read_text())
def jsonl_bytes(rows):return ''.join(canonical(row)+'\n' for row in rows).encode()

def selected_outside(old,entities):
    by_id={r['id']:r for r in entities}
    eligible=[p for p in old if p['split']=='train' and p['mutation']['kind']=='outside_entity_table'
        and p['source_row']['metadata']['source_index']!=188
        and by_id[p['mutation']['donor']['row_id']]['metadata']['source_index']!=188]
    eligible.sort(key=lambda p:sha(SELECTION_SEED+':'+p['source_row_sha256']+':'+p['id']))
    b.require(len(eligible)>=16,'Insufficient outside references')
    return eligible[:16]


def build(args):
    out=Path(args.output_dir);b.require(not out.exists(),'Refuse existing output')
    root=Path(args.resident_candidates)
    old=b.read_rows(root/'candidate_train.jsonl');held=b.read_rows(root/'mechanical_holdout.jsonl')
    b.require(sha((root/'candidate_train.jsonl').read_bytes())==PARENT_DATA_SHA and len(old)==208,'Wrong prior208 data')
    b.require(sha((root/'mechanical_holdout.jsonl').read_bytes())==HOLDOUT_SHA and len(held)==32,'Wrong mechanical holdout')
    sources,entities,full,files=b.load_sources(args.base_release,args.overlap_release)
    selected=selected_outside(old,entities)
    selection=read(args.selection_recipe)
    b.require(selection['kind']=='dpo52_source_free_selection_v1' and selection['seed']==SELECTION_SEED,'Wrong selection recipe')
    observed=[{'pair_id':p['id'],'pair_sha256':sha(p),'source_row_id':p['source_row']['id'],
        'source_row_sha256':p['source_row_sha256'],'source_group_id':str(p['source_row']['group_id'])} for p in selected]
    b.require(observed==selection['outside_items'],'Deterministic selection differs from frozen recipe')
    b.require(selection['parent208_file_sha256']==PARENT_DATA_SHA and selection['holdout_file_sha256']==HOLDOUT_SHA,'Selection input hashes changed')
    semantic=[];recipe_hashes=[]
    for recipe_path,count,expected in ((args.semantic16_recipe,16,OLD16_SHA),(args.semantic20_recipe,20,NEW20_SHA)):
        recipe=read(recipe_path)
        b.require(recipe['input_file_sha256']==sha((Path(args.overlap_release)/'short_canonical/train_relations.jsonl').read_bytes()),'Semantic source SHA differs')
        prefs=reconstruct_preferences(list(sources.values()),recipe)
        b.require(len(prefs)==count and sha(jsonl_bytes(prefs))==expected,'Semantic recipe differs from reviewed original preferences')
        for pref in prefs:
            row=sources[pref['source_row_id']]
            b.require(row['metadata']['source_index']!=188,'Source188 excluded')
            pair=b.make_pair(row,pref['mutation'],rejected=pref['rejected'],review=pref['review'])
            semantic.append(pair)
        recipe_hashes.append(sha(Path(recipe_path).read_bytes()))
        files[str(Path(recipe_path).resolve())]=sha(Path(recipe_path).read_bytes())
    # Stable identities and complete original sequences, not shuffled or truncated fragments.
    result=selected+semantic
    b.require(len(result)==52 and len({p['id'] for p in result})==52,'Duplicate or wrong pair count')
    b.require(len({(p['source_row_sha256'],sha(p['rejected'])) for p in result})==52,'Duplicate semantic preference identity')
    original_tune=set(selection['original_frozen_tune_group_ids'])
    hold_groups={str(p['source_row']['group_id']) for p in held}
    held_donor_groups={str(p['mutation']['donor']['group_id']) for p in held if 'donor' in p['mutation']}
    entities_by_id={r['id']:r for r in entities}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    encoded=[]
    for pair in result:
        b.verify_pair(pair,sources,entities,full)
        b.require(pair['chosen']==pair['source_row']['completion'],'Chosen changed')
        b.require(str(pair['source_row']['group_id']) not in original_tune|hold_groups|held_donor_groups,'Source group overlaps original tune or mechanical holdout')
        b.require(pair['source_row']['metadata']['source_index']!=188,'Source188 not allowed')
        if pair['mutation']['kind']=='outside_entity_table':
            donor=entities_by_id[pair['mutation']['donor']['row_id']]
            b.require(donor['metadata']['source_index']!=188 and str(donor['group_id']) not in original_tune|hold_groups|held_donor_groups,'Donor leak or source188 contamination')
        encoded.append(b.encode_pair(pair,tokenizer,args.max_length))
    b.require(set(p['mutation']['kind'] for p in result)=={'outside_entity_table','reviewed_semantic_contradiction'},'Unexpected mutation kinds')
    files.update({str((root/name).resolve()):expected for name,expected in [('candidate_train.jsonl',PARENT_DATA_SHA),('mechanical_holdout.jsonl',HOLDOUT_SHA)]})
    files[str(Path(args.selection_recipe).resolve())]=sha(Path(args.selection_recipe).read_bytes())
    out.mkdir(parents=True,mode=0o700)
    hashes={}
    for name,values in [('candidate_train.jsonl',result),('mechanical_holdout.jsonl',held)]:
        path=out/name;path.write_bytes(jsonl_bytes(values));path.chmod(0o600);hashes[name]=sha(path.read_bytes())
    b.require(hashes['mechanical_holdout.jsonl']==HOLDOUT_SHA,'Holdout bytes changed')
    manifest={'data_version':b.VERSION,'release_version':'dpo52-semantic-focused-v2','parent_view':'short_canonical',
      'protocol_sha256':b.LOCKS['protocol'],'train_pairs':52,'mechanical_holdout_pairs':32,
      'train_source_groups':len({p['source_row']['group_id'] for p in result}),
      'mechanical_holdout_groups':len(hold_groups),'group_overlap':0,
      'original_tune_group_overlap':0,'holdout_source_and_donor_group_overlap':0,
      'source188_source_or_donor_occurrences':0,'external_reviewed_pairs':36,'output_sha256':hashes,
      'input_file_locks':files,'no_truncation':True,'max_length':args.max_length,
      'max_prompt_plus_completion_tokens':max(len(v['prompt_ids'])+max(len(v['chosen_ids']),len(v['rejected_ids'])) for v in encoded),
      'prompt_tokens_one_pass':sum(len(v['prompt_ids']) for v in encoded),
      'chosen_tokens_one_pass_including_eos':sum(len(v['chosen_ids']) for v in encoded),
      'rejected_tokens_one_pass_including_eos':sum(len(v['rejected_ids']) for v in encoded),
      'mutation_counts':dict(Counter(p['mutation']['kind'] for p in result)),
      'semantic_change_counts':dict(Counter(p['mutation'].get('semantic_change_kind') for p in semantic)),
      'old16_and_new20_reconstructed_exactly':True,'old16_jsonl_sha256':OLD16_SHA,'new20_jsonl_sha256':NEW20_SHA,
      'semantic_recipe_file_sha256':recipe_hashes,'selection_recipe_file_sha256':sha(Path(args.selection_recipe).read_bytes()),
      'preparer_sha256':sha(Path(__file__).read_bytes()),'locked_builder_sha256':sha(Path(b.__file__).read_bytes()),
      'limitations':LIMITATIONS,'source_text_uploaded':False,'gpu_or_network_execution_performed':False}
    path=out/'manifest.json';path.write_text(canonical(manifest)+'\n');path.chmod(0o600)
    print(canonical({k:manifest[k] for k in ('train_pairs','train_source_groups','mutation_counts','semantic_change_counts','max_prompt_plus_completion_tokens','output_sha256')}))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('resident-candidates','base-release','overlap-release','tokenizer','selection-recipe','semantic16-recipe','semantic20-recipe','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-length',type=int,default=4096)
    build(p.parse_args())


if __name__=='__main__':main()
