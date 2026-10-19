"""Read-only CPU admission of resident full-source data and tokenizer."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from semantic.data import encode_row
from train_label_weighted import read_training_rows,admit_parent,DATA_SHA
from label_weights import attach_weights,weighted_completion


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True);p.add_argument('--label-weight',type=float,default=4.)
    p.add_argument('--parent-adapter')
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('Refusing to overwrite an admission receipt')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha(args.data)==DATA_SHA
    parent=admit_parent(args.parent_adapter,args.tokenizer) if args.parent_adapter else None
    rows=read_training_rows(args.data,'relations');assert len(rows)==1000
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    totals=Counter();lengths=[]
    for row in rows:
        original=encode_row(row,tok,8192)
        result=attach_weights(original,row['completion'],tok,args.label_weight)
        ids,weights,categories=weighted_completion(row['completion'],tok,args.label_weight)
        assert result['input_ids']==original['input_ids'] and result['labels']==original['labels']
        assert ids[-1]==tok.eos_token_id and categories[-1]=='eos' and weights[-1]==1
        totals.update(categories);lengths.append(len(original['input_ids']))
    assert sum(totals.values())==399973 and totals['relation_label']==27752 and totals['eos']==1000
    assert sum(lengths)==1610825 and max(lengths)==5140
    assert sha(args.data)==DATA_SHA
    report={'completed':True,'cpu_only':True,'source_sha256':DATA_SHA,'rows':1000,
            'parent_runtime_contract_checked':bool(parent),
            'target_tokens':sum(totals.values()),'input_tokens':sum(lengths),'max_tokens':max(lengths),
            'categories':dict(totals),'label_weight':args.label_weight,
            'weighted_token_mass':sum(totals.values())+(args.label_weight-1)*totals['relation_label'],
            'full_target_ids_prompt_mask_EOS_unchanged':True,'truncated_rows':0,
            'source_groups':len({row['group_id'] for row in rows}),
            'source_records':len({row['metadata']['source_record_sha256'] for row in rows}),
            'code_sha256':{name:sha(Path(__file__).with_name(name)) for name in
                ['preflight.py','label_weights.py','train_label_weighted.py','verify_torch_math.py']}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
