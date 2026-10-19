"""Compile shorter canonical relation SFT only from already-resident train data.

Preserves every original evidence atom. Windows depend solely on source text;
noncontained train evidence receives an explicit auxiliary full-source row.
Inference never reads these training assignments or uses gold fallback.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from semantic.data import canonical, encode_row, sha, source_windows, validate_row
from semantic import protocol as p
from semantic.relation_windows_resident import load_receipt, reconstruct_anchors
from semantic.runtime import verify_title_context

VERSION = 'round2_relation1600_complete_v1'

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(canonical(row)+'\n')

def build(args):
    base,out=Path(args.base_release),Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refuse overwrite')
    receipt=load_receipt(args.equivalence_receipt,args.receipt_sha256)
    raw=(base/'manifest.json').read_bytes(); manifest=json.loads(raw)
    if sha(raw)!=receipt['base_release_manifest_sha256'] or manifest['train_tune_group_overlap']!=0:
        raise ValueError('Frozen base contract mismatch')
    names=('full_source/train_relations.jsonl','title_source_receipts.json','evaluation/train_records.jsonl')
    for name in names:
        if sha((base/name).read_bytes())!=manifest['output_sha256'][name]:
            raise ValueError('Frozen resident file SHA drift')
    if sha((base/names[0]).read_bytes())!=receipt['full_relation_file_sha256']:
        raise ValueError('Frozen full-relation source differs from anchor receipt')
    if sha(Path(p.__file__).read_bytes())!=manifest['protocol_sha256']:
        raise ValueError('Frozen protocol SHA drift')
    full=read_rows(base/names[0]); titles=json.loads((base/names[1]).read_text())
    train={r['original_record_sha256']:r for r in read_rows(base/names[2])}
    anchors,ledger=reconstruct_anchors(full,receipt)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    rows, audits, original, exposures=[],[],Counter(),Counter()
    tokens=Counter(); max_tokens=0; all_windows=0; unknown_windows=0
    for row in full:
        meta,text=row['metadata'],row['task_input']['text']; key=meta['source_record_sha256']
        source=train[key]
        if source['split']!='train' or str(source['source_group_id'])!=row['group_id'] or source['text']!=text:
            raise ValueError('Frozen training source identity mismatch')
        verify_title_context(meta,titles,source_group_id=row['group_id'])
        table=row['task_input']['entities']; byid={e['id']:e for e in table}
        targets=dict(zip(meta['supervised_atom_ids'],row['target']))
        original.update((key,atom) for atom in targets)
        windows=source_windows(text,args.short_chars,args.overlap); all_windows+=len(windows)
        assignment=defaultdict(list)
        for atom,anchor in anchors[key].items():
            containing=[i for i,(a,b) in enumerate(windows) if a<=anchor['start'] and anchor['end']<=b]
            for index in containing or [-1]: assignment[index].append(atom)
        unknown_windows+=sum(i not in assignment for i in range(len(windows)))
        for index,atoms in sorted(assignment.items()):
            def keyfunc(atom):
                t=targets[atom]; a,b=byid[t['source']],byid[t['target']]
                return (a['label'],p.normalize_name(a['name']),b['label'],p.normalize_name(b['name']),t['relation'],anchors[key][atom]['start'])
            atoms.sort(key=keyfunc)
            a,b=(0,len(text)) if index==-1 else windows[index]
            clean=[deepcopy(targets[atom]) for atom in atoms]
            local={**deepcopy(meta),'variant':'short_canonical','order':'canonical',
                'source_window':{'start':a,'end':b,'index':index,'full_source_fallback':index==-1},
                'supervised_atom_ids':atoms,'relation_entity_table_scope':'complete_source_entities',
                'relation_window_supervision':{'version':VERSION,'base_full_row_sha256':sha(row),
                    'anchor_ledger_sha256':sha(ledger),'receipt_sha256':args.receipt_sha256,
                    'scope':'every_containing_source_only_window_of_original_chosen_evidence',
                    'short_chars':args.short_chars,'overlap':args.overlap,
                    'auxiliary_full_source_training_only':index==-1,'runtime_gold_fallback_allowed':False,
                    'unknown_windows_are_not_negative_examples':True,'original_targets_unchanged':True}}
            inputs={'text':text[a:b],'entities':deepcopy(table),'metadata':local}
            new={'data_version':row['data_version'],'id':row['id']+':'+VERSION+':'+str(index),
                 'split':'train','stage':'relations','variant':'short_canonical','group_id':row['group_id'],
                 'metadata':local,'task_input':inputs,'target':clean,'target_count':len(clean),
                 'prompt':p.build_relation_prompt(inputs['text'],table,local,'canonical'),
                 'completion':p.render_relations(clean,table),'input_sha256':sha(inputs),'target_sha256':sha(clean)}
            validate_row(new,require_train=True)
            encoded=encode_row(new,tokenizer,args.max_length)
            tokens.update({x:encoded['_choice_counts'][x] for x in ('prompt_tokens','completion_tokens')})
            max_tokens=max(max_tokens,len(encoded['input_ids']))
            rows.append(new);exposures.update((key,atom) for atom in atoms)
            audits.append({'source_record_sha256':key,'window_index':index,'target_count':len(atoms),
                           'source_window':local['source_window'],'supervised_atom_ids':atoms})
    if set(original)!=set(exposures) or any(n!=1 for n in original.values()):
        raise ValueError('Frozen relation atom coverage changed')
    output=out/'short_canonical/train_relations.jsonl'
    write_rows(output,rows);write_rows(out/'window_audit.private.jsonl',audits)
    fallback=[r for r in rows if r['metadata']['source_window']['index']==-1]
    result={'release_version':VERSION,'status':'compiled_not_trained','completed':True,
        'base_release_manifest_sha256':sha(raw),'protocol_sha256':manifest['protocol_sha256'],
        'compiler_sha256':sha(Path(__file__).read_bytes()),'equivalence_receipt_sha256':args.receipt_sha256,
        'resident_anchor_ledger_sha256':sha(ledger),'external_source_records_read':False,
        'parameters':{'short_chars':args.short_chars,'overlap':args.overlap,'max_length':args.max_length},
        'rows':len(rows),'source_records':len(full),'source_groups':len({r['group_id'] for r in full}),
        'unique_original_relation_atoms':len(original),'unique_atom_coverage_unchanged':True,
        'total_supervised_window_exposures':sum(exposures.values()),'source_only_windows':all_windows,
        'unknown_windows_not_trained':unknown_windows,'auxiliary_full_source_rows':len(fallback),
        'auxiliary_full_source_atoms':sum(r['target_count'] for r in fallback),
        'runtime_gold_fallback_allowed':False,'prompt_and_target_untruncated':True,
        'max_total_tokens':max_tokens,'token_counts':dict(tokens),
        'updates_per_epoch_at_batch1_accum8':math.ceil(len(rows)/8),
        'training_file_sha256':sha(output.read_bytes()),
        'limitations':['Silver label semantics remain unchanged; this does not certify clinical correctness.',
            'Overlapping exposure counts and optimizer update counts change; not an exposure-matched length ablation.',
            'Auxiliary full-source training rows preserve cross-window annotations; inference never follows gold-selected fallback.']}
    write_rows(out/'manifest.json',[result]);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-release','equivalence-receipt','receipt-sha256','tokenizer','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--short-chars',type=int,default=1600)
    parser.add_argument('--overlap',type=int,default=400)
    parser.add_argument('--max-length',type=int,default=8192)
    print(canonical(build(parser.parse_args())))
if __name__=='__main__':main()
