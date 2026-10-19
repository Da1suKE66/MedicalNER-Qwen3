"""Root-controlled optional experiment; no queue mutation and no data transfer."""
import argparse,json,subprocess,sys,hashlib,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--deadline-unix',type=float)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());here=Path(__file__).resolve().parent
    if args.deadline_unix and args.deadline_unix-time.time()<3000:
        raise ValueError('Less than50min remains for smoke, full training and conservatively budgeted full130 evaluation')
    for name,expected in config['code_sha256'].items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest()==expected,('Code SHA mismatch',name)
    receipts=Path(config['receipt_dir']);smoke=Path(config['smoke_output']);formal=Path(config['formal_output'])
    parent=Path(config['parent_adapter'])
    if len({p.resolve() for p in [receipts,smoke,formal,parent]})!=4:
        raise ValueError('Run outputs must be separate from one another and original parent')
    for output in [receipts,smoke,formal]:
        if output.exists():raise ValueError('Refusing to overwrite any existing run artifact: '+str(output))
    receipts.mkdir(parents=True)
    manifest={'completed':False,'formal_started':False,'smoke_started':False,
              'started_unix':time.time(),'original_parent':str(parent),'config':config,
              'no_queue_mutation':True,'no_training_text_transferred':True}
    def save():
        tmp=receipts/'wrapper_manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n')
        tmp.replace(receipts/'wrapper_manifest.json')
    def run(name,argv,timeout):
        with (receipts/(name+'.log')).open('w') as log:
            subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    save()
    try:
        run('preflight',[sys.executable,str(here/'preflight.py'),'--data',config['data'],
            '--tokenizer',config['base_model'],'--parent-adapter',str(parent),
            '--output',str(receipts/'preflight.json'),'--label-weight','4'],180)
        run('torch_math',[sys.executable,str(here/'verify_torch_math.py'),'--device','cpu'],120)
        math_result=json.loads((receipts/'torch_math.log').read_text().splitlines()[-1])
        assert math_result['passed'] and math_result['trainer_accumulation_regression']['last_partial_group_verified']
        manifest['torch_math']=math_result;save()
        common=[sys.executable,str(here/'train_label_weighted.py'),'--data',config['data'],
                '--data-sha256',config['data_sha256'],'--stage','relations','--base-model',config['base_model'],
                '--parent-adapter',str(parent),'--epochs','1','--learning-rate','2e-5','--batch-size','1',
                '--lora-rank','16','--lora-alpha','32','--dropout','.05','--weight-decay','.01',
                '--max-length','8192','--seed','42','--label-weight','4']
        training_deadline=time.monotonic()+1500
        manifest['smoke_plus_formal_timeout_seconds']=1500
        manifest['conservative_eval_reserve_estimate_seconds']=1800
        save()
        for stage,output,extra,timeout in [
            ('smoke',smoke,['--smoke-only','--max-steps','1','--gradient-accumulation','1','--warmup-ratio','0'],180),
            ('formal',formal,['--gradient-accumulation','8','--warmup-ratio','.05'],1320)]:
            if args.deadline_unix and stage=='formal' and args.deadline_unix-time.time()<2700:
                raise ValueError('Insufficient time remains for full training plus frozen full130 evaluation')
            manifest[stage+'_started']=True;save()
            remaining=training_deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Smoke plus formal1500s training timeout expired')
            run(stage,[*common,'--output-dir',str(output),*extra],min(timeout,remaining))
            receipt=json.loads((output/'training_manifest.json').read_text())
            assert receipt['completed'] and not receipt.get('failed')
            assert receipt['smoke_only'] is (stage=='smoke')
            assert receipt['adapter_initial_sha256']==config['parent_parameters_sha256']
            assert receipt['adapter_final_sha256']!=receipt['adapter_initial_sha256']
            assert receipt['saved_weights_bitwise_verified'] and receipt['saved_adapter_reloaded_exact']
            assert receipt['observed_exposure']['longest_example_seen']
            if stage=='smoke':
                assert receipt['global_step']==1 and receipt['parameters']['warmup_ratio']==0
                assert receipt['observed_exposure']['row_visits']==1
            else:
                seen=receipt['observed_exposure']
                assert seen['unique_rows_seen']==seen['row_visits']==1000
                assert seen['min_visits']==seen['max_visits']==1
                assert seen['supervised_tokens']==399973 and seen['weighted_token_mass']==483229
                assert seen['relation_label_tokens']==27752 and receipt['global_step']==125
                assert receipt['parameters']['warmup_ratio']==.05
            manifest[stage+'_receipt_sha256']=hashlib.sha256((output/'training_manifest.json').read_bytes()).hexdigest()
            manifest[stage+'_completed']=True;save()
        manifest.update(completed=True,finished_unix=time.time(),
                        smoke_weights_never_used_for_formal=True,full130_evaluation_still_required=True)
        save();print(json.dumps(manifest))
    except BaseException as exc:
        manifest.update(failed=True,exception_type=type(exc).__name__,finished_unix=time.time());save();raise


if __name__=='__main__':main()
