"""Train-only abstention controls; never relabel absent silver annotations.

Both arms retain the immutable v2 rows and receive identical synthetic prompts
and positive targets. The intervention only unmasks source-unsupported pairs in
controlled two-statement excerpts. These are extraction-contract negatives, not
expert-adjudicated clinical negatives. Original/tune/test UNKNOWNs stay UNKNOWN.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from generative.protocol import dumps
from kg_agent.contracts import allowed_relations
from robustness.entity_noise import fingerprint, prompt_payload, protocol_prefix
from .supervision_v2 import serialize_partial, validate_partial_row
from .text_normalization import find_mentions


VERSION = 'abstention-repair-v1'
RELATIONS = {'is_core_symptom_of', 'is_associated_symptom_of'}


def rank(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def fold(value):
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())


def anchors(rows):
    """Only existing supported train positives, never a held-out vocabulary."""
    result = {}
    for row in rows:
        validate_partial_row(row)
        if row.get('split') != 'train' or row.get('eligible_for_rl') is not False:
            raise ValueError('Explicit partial-supervision train rows required')
        if any(k in row for k in ('stress_evaluation', 'heldout')):
            raise ValueError('Held-out artifacts cannot supply synthetic anchors')
        nodes = {e['id']:e for e in row['entities']}
        pairs = {p['pair_id']:p for p in row['pairs']}
        for decision in row['target']['decisions']:
            if row['pair_supervision'][decision['pair_id']] != 'SUPPORTED_SILVER':
                continue
            pair = pairs[decision['pair_id']]
            s,d = nodes[pair['source']],nodes[pair['target']]
            if s['label'] != 'Symptom' or d['label'] != 'Disease':continue
            if any(not 2 <= len(n['name']) <= 140 for n in (s,d)):continue
            if any('\n' in n['name'] or '"' in n['name'] for n in (s,d)):continue
            for edge in decision['relations']:
                if edge['relation'] not in RELATIONS:continue
                # Ambiguous multilabel pairs must not be collapsed to one label.
                if len(decision['relations']) != 1:continue
                key = (str(row['group_id']), fold(s['name']), fold(d['name']), edge['relation'])
                result.setdefault(key, {'source':s['name'],'target':d['name'],'relation':edge['relation'],
                    'group_id':row['group_id'],'sample_id':row['sample_id'], 'parent_id':row['id'],
                    'parent_sha256':fingerprint(row),'evidence_span_ids':edge['evidence_span_ids']})
    return sorted(result.values(),key=lambda a:rank(a))


def distinct_names(*values):
    names = [fold(v) for v in values]
    return len(set(names))==len(names) and not any(a in b or b in a for i,a in enumerate(names) for b in names[i+1:])


def statement(a, index):
    adjective = 'core' if a['relation']=='is_core_symptom_of' else 'associated'
    forms = [
        'For "{d}", "{s}" is listed as {article} {kind} symptom.',
        '"{s}" is {article} {kind} symptom of "{d}".',
        'The description of "{d}" lists "{s}" as {article} {kind} symptom.',
    ]
    return forms[index % len(forms)].format(d=a['target'],s=a['source'],kind=adjective,
                                          article='a' if adjective=='core' else 'an')


def eligible_grounded_negative(a,b):
    return (a['label']=='Symptom' and b['label'] in {'Symptom','Disease'}) or (
        a['label']==b['label']=='Disease' and distinct_names(a['name'],b['name']))


def make_case(first, second, *, missing=None, serial=0):
    """Explicitly constructed relation statements with cautious negative scope.

    Reverse Disease->Symptom interpretations remain UNKNOWN.
    Symptom->symptom, cross-subject symptom->other-disease, and the
    unconnected Disease->Disease pairs qualify for grounded NONE supervision.
"""
    if first['group_id'] != second['group_id']:
        raise ValueError('Keep constructed contexts inside a single train source group')
    common = fold(first['target'])==fold(second['target'])
    if not distinct_names(first['source'],second['source']):raise ValueError('Overlapping symptom names')
    if not common and not distinct_names(first['target'],second['target']):
        raise ValueError('Overlapping disease scopes')
    if not common and missing is not None:raise ValueError('Keep each case within four entities')
    nodes=[]
    def add(name,label):
        for n in nodes:
            if (fold(n['name']),n['label'])==(fold(name),label):return n['id']
        eid=f'N{len(nodes)+1}'
        nodes.append({'id':eid,'entity_id':f'E{len(nodes)+1:03d}','name':name,'label':label})
        return eid
    positives={}
    ordered=[first,second] if int(rank([first,second,serial])[-1],16)%2 else [second,first]
    spans=[]
    for i,a in enumerate(ordered):
        s,d=add(a['source'],'Symptom'),add(a['target'],'Disease')
        span=f'SENT_{i+1:03d}'
        spans.append({'span_id':span,'text':statement(a,serial+i)})
        positives[(s,d)]=(a['relation'],span)
    absent_id=None
    if missing is not None:
        if any(find_mentions(missing['source'],s['text']) for s in spans):raise ValueError('Missing name actually present')
        if not distinct_names(missing['source'],*[n['name'] for n in nodes]):raise ValueError('Overlapping missing name')
        absent_id=add(missing['source'],'Symptom')
    byid={n['id']:n for n in nodes}
    pairs=[{'source':s['id'],'target':d['id'],'allowed':list(allowed_relations(s['label'],d['label']))}
           for s in nodes for d in nodes if s!=d and allowed_relations(s['label'],d['label'])]
    pairs.sort(key=lambda p:rank([(byid[p[k]]['name'],byid[p[k]]['label']) for k in ('source','target')]))
    statuses,proofs,decisions={},{},[]
    for i,p in enumerate(pairs):
        pid=p['pair_id']=f'P{i+1:04d}'
        key=(p['source'],p['target'])
        rel=[]
        if key in positives:
            r,span=positives[key];statuses[pid]='SUPPORTED_SILVER'
            rel=[{'relation':r,'evidence_span_ids':[span]}]
        elif absent_id in key:
            statuses[pid]='SYNTHETIC_NEGATIVE'
            proofs[pid]={'kind':'natural_name_absent_from_constructed_context','absent_entity':absent_id}
        elif eligible_grounded_negative(byid[key[0]],byid[key[1]]):
            statuses[pid]='SYNTHETIC_NEGATIVE'
            proofs[pid]={'kind':'no_explicit_relation_in_two_statement_context',
                         'scope':'synthetic_extraction_contract_not_clinical_absence'}
        else:statuses[pid]='UNKNOWN_SYNTHETIC_UNADJUDICATED'
        decisions.append({'pair_id':pid,'relations':rel})
    identity=rank([first,second,missing,serial])
    row={'id':'abstention:'+identity[:24], 'sample_id':'synthetic:'+identity[:24],
         'group_id':first['group_id'],'split':'train','sample_weight':1.0,
         'data_version':'teacher-silver-repair-v2-abstention-v1','annotation_source':'synthetic_render_of_train_silver',
         'entities':nodes,'pairs':pairs,'spans':spans,'original_spans':{s['span_id']:s['text'] for s in spans},
         'target':{'decisions':decisions},'pair_supervision':statuses,'unknown_reasons':{},
         'loss_mask_contract':'pair-unknown-loss-mask-v1','eligible_for_rl':False,
         'abstention_provenance':{'version':VERSION,'construction':'two_explicit_relation_statements',
             'source_group':first['group_id'],'parents':[first,second],
             'vocabulary_donor':missing,'negative_proofs':proofs,'human_adjudicated':False,
             'synthetic_not_original_medical_text':True}}
    row['prompt']=protocol_prefix()+'\n'+dumps(prompt_payload(row))+'\n'
    refresh(row);validate_case(row)
    return row


def refresh(row):
    row['completion'],row['loss_mask_spans']=serialize_partial(row['target'],row['pair_supervision'])
    row['unknown_reasons']={p:'not_supervised_as_negative' for p,s in row['pair_supervision'].items() if s.startswith('UNKNOWN_')}
    validate_partial_row(row)


def validate_case(row):
    validate_partial_row(row)
    meta=row['abstention_provenance']
    if row['split']!='train' or not meta['synthetic_not_original_medical_text']:
        raise ValueError('Synthetic training provenance required')
    if any(p['group_id']!=row['group_id'] for p in meta['parents']):raise ValueError('Source group drift')
    # Reconstruct the controlled source, not just a self-asserted negative tag.
    actual=[s['text'] for s in row['spans']]
    expected={statement(p,i) for p in meta['parents'] for i in range(3)}
    if len(actual)!=2 or any(s not in expected for s in actual):raise ValueError('Changed constructed source')
    nodes={e['id']:e for e in row['entities']}
    pmap={p['pair_id']:p for p in row['pairs']}
    actual_positives=set()
    for decision in row['target']['decisions']:
        pair=pmap[decision['pair_id']]
        for edge in decision['relations']:
            actual_positives.add((fold(nodes[pair['source']]['name']),edge['relation'],fold(nodes[pair['target']]['name'])))
            if any(sid not in row['original_spans'] for sid in edge['evidence_span_ids']):raise ValueError('Unknown positive evidence')
            for sid in edge['evidence_span_ids']:
                text=row['original_spans'][sid]
                parent=next((a for a in meta['parents'] if fold(a['source'])==fold(nodes[pair['source']]['name'])
                             and fold(a['target'])==fold(nodes[pair['target']]['name']) and a['relation']==edge['relation']),None)
                if parent is None or text not in {statement(parent,i) for i in range(3)}:raise ValueError('Positive evidence drift')
    expected_positives={(fold(p['source']),p['relation'],fold(p['target'])) for p in meta['parents']}
    if actual_positives!=expected_positives:raise ValueError('Constructed positive target changed')
    for pid,status in row['pair_supervision'].items():
        if status!='SYNTHETIC_NEGATIVE':continue
        proof=meta['negative_proofs'].get(pid)
        if not proof:raise ValueError('Negative without construction proof')
        p=pmap[pid]
        if proof['kind']=='natural_name_absent_from_constructed_context':
            eid=proof['absent_entity']
            if eid not in (p['source'],p['target']) or any(find_mentions(nodes[eid]['name'],s) for s in actual):
                raise ValueError('Invalid missing-entity proof')
        elif proof['kind']=='no_explicit_relation_in_two_statement_context':
            a,b=nodes[p['source']],nodes[p['target']]
            if not eligible_grounded_negative(a,b):raise ValueError('Unsupported negative pair')
            if not all(any(find_mentions(n['name'],s) for s in actual) for n in (a,b)):raise ValueError('Grounded negative missing endpoint')
            if any(fold(a['name'])==fold(x['source']) and fold(b['name'])==fold(x['target']) for x in meta['parents']):
                raise ValueError('Constructed positive mislabeled NONE')
        else:raise ValueError('Unknown proof kind')
    return True


def build(base, tune, *, max_cases=512):
    if max_cases<1:raise ValueError('Positive bounded case budget required')
    train_groups={r['group_id'] for r in base}
    if train_groups & {r['group_id'] for r in tune}:raise ValueError('Train/tune group leakage')
    if any(r.get('split')!='tune' for r in tune):raise ValueError('Frozen tune rows required')
    aa=anchors(base);groups=defaultdict(list)
    for a in aa:groups[a['group_id']].append(a)
    combinations=[]
    for group,items in groups.items():
        for i,a in enumerate(items):
            for b in items[i+1:]:
                if not distinct_names(a['source'],b['source']):continue
                common=fold(a['target'])==fold(b['target'])
                if not common and not distinct_names(a['target'],b['target']):continue
                combinations.append((rank([a,b]),a,b,common))
    combinations.sort(key=lambda x:x[0])
    controls,treated=[],[]
    seen=set()
    for _,a,b,common in combinations:
        if len(treated)>=max_cases:break
        missing=None
        if common:
            eligible=[n for n in aa if distinct_names(n['source'],a['source'],b['source'],a['target'])
                      and not any(find_mentions(n['source'],statement(x,j)) for j,x in enumerate((a,b)))]
            missing=min(eligible,key=lambda n:rank([a,b,'vocabulary_donor',n])) if eligible else None
            if missing is None:continue
        row=make_case(a,b,missing=missing,serial=len(treated))
        signature=rank([row['prompt'],row['completion']])
        if signature in seen:continue
        seen.add(signature)
        control=deepcopy(row)
        for pid,proof in row['abstention_provenance']['negative_proofs'].items():
            if proof['kind']=='no_explicit_relation_in_two_statement_context':
                control['pair_supervision'][pid]='UNKNOWN_CONSTRUCTED_ABLATION'
        refresh(control);validate_case(control)
        treated.append(row);controls.append(control)
    if not treated:raise ValueError('No eligible train-only constructed examples')
    result={'V3_missing_only':deepcopy(base)+controls,'V3_grounded_none':deepcopy(base)+treated}
    for rows in result.values():
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate training ID')
        if not any(s=='SYNTHETIC_NEGATIVE' for r in rows for s in r['pair_supervision'].values()):
            raise ValueError('No supervised NONE; do not launch')
    assert all(a['prompt']==b['prompt'] and a['target']==b['target'] for a,b in zip(controls,treated))
    assert all(a==b for a,b in zip(base,result['V3_grounded_none'][:len(base)]))
    return result,{'version':VERSION,'base_rows':len(base),'synthetic_cases':len(treated),
        'source_groups':len({r['group_id'] for r in treated}),
        'identical_input_and_positive_targets':True,'only_intervention':'loss mask on constructed grounded NONE decisions',
        'original_unknowns_unchanged':True,'heldout_used':False,'human_adjudicated':False,
        'eligible_for_existing_rl':False,
        'supervision':{k:dict(Counter(s for r in rows for s in r['pair_supervision'].values())) for k,rows in result.items()}}


def main():
    import argparse
    from .controlled_experiments import read_rows
    p=argparse.ArgumentParser()
    p.add_argument('--base-train',required=True);p.add_argument('--tune',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-cases',type=int,default=512)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise ValueError('Refusing to overwrite an experiment dataset')
    result,report=build(read_rows(args.base_train),read_rows(args.tune),max_cases=args.max_cases)
    out.mkdir(parents=True)
    hashes={}
    for name,rows in result.items():
        path=out/(name+'.jsonl')
        with path.open('x') as f:
            for row in rows:f.write(dumps(row)+'\n')
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report['inputs']={k:{'path':str(Path(v).resolve()),'sha256':hashlib.sha256(Path(v).read_bytes()).hexdigest()}
                      for k,v in [('base_train',args.base_train),('tune',args.tune)]}
    report['output_sha256']=hashes
    with (out/'manifest.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
