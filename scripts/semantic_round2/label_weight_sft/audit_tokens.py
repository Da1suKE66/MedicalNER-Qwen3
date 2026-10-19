"""CPU-only supervision composition census, no token-loss/gradient claim."""
from collections import Counter,defaultdict
from pathlib import Path
import json,sys,hashlib
HERE=Path(__file__).resolve().parent;W=HERE.parents[2]
sys.path.insert(0,str(W/'MedicalNER-Qwen3/agent'))
from semantic.data import encode_row,sha
from label_weights import weighted_completion
from transformers import AutoTokenizer
SOURCE=W/'audit_work/full_semantic_20261019/relation_complete_overlap_v2/short_canonical/train_relations.jsonl'
expected='4d51e1c99473c5fb392e5dd78b01e7a0ff3024332a68f3b9974fbcbdbd0c1d13'
assert sha(SOURCE.read_bytes())==expected
rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
tok=AutoTokenizer.from_pretrained(W/'medicalner-migration-20261012-v1/tokenizer/qwen3-8b',local_files_only=True)
counts=Counter();by_label=Counter();allweights=0;maxlen=0;totalinput=0;stats=[]
for row in rows:
 encoded=encode_row(row,tok,8192)
 ids,weights,categories=weighted_completion(row['completion'],tok,4.)
 p=len(encoded['input_ids'])-len(ids)
 assert encoded['input_ids'][p:]==ids and encoded['labels']==[-100]*p+ids
 counts.update(categories);allweights+=sum(weights);maxlen=max(maxlen,len(encoded['input_ids']));totalinput+=len(encoded['input_ids'])
 by_label.update(r['relation'] for r in row['target'])
 stats.append({'row_id':row['id'],'prompt_tokens':p,'target_tokens':len(ids),'relation_label_tokens':categories.count('relation_label')})
total=sum(counts.values());labels=counts['relation_label']
report={'status':'PASS','source_sha256':expected,'rows':len(rows),'source_records':len({r['metadata']['source_record_sha256'] for r in rows}),'source_groups':len({r['group_id'] for r in rows}),
 'target_records':sum(len(r['target']) for r in rows),'target_tokens_one_pass_including_eos':total,'max_prompt_plus_completion_tokens':maxlen,'total_input_tokens_one_pass':totalinput,
 'token_categories':dict(counts),'token_fractions':{k:v/total for k,v in counts.items()},'entity_names_plus_evidence_fraction':(counts['source_name']+counts['target_name']+counts['evidence'])/total,
 'label_weight':4.,'weighted_token_mass':allweights,'weighted_label_mass_fraction':4*labels/allweights,
 'relation_record_counts':dict(by_label),'source188_records_preserved':sum(r['metadata']['source_index']==188 for r in rows),
 'prompt_eos_and_token_identity_verified':True,'cross_semantic_field_token_count':0,'actual_label_vs_copy_cross_entropy_measured':False,'actual_gradient_contributions_measured':False,
 'interpretation':'Counts show supervision allocation only. They do not prove entity/evidence copying dominates gradient or remaining errors.',
 'all1000_rows_all_targets_and_eos_preserved':True,'new_gpu_or_network_work':False}
(HERE/'token_census.aggregate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(HERE/'row_token_counts.private.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
