"""Run real GPU validation and then a fresh DPO run from the original SFT.

The smoke adapter is never used as the formal parent or accepted for evaluation.
The queue success receipt belongs only to the completed formal training run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

PARENT='/workspace/medicalner/20261019/runs/relation_short_canonical_complete_lr1e4_r16_e2'
PARENT_PARAMETERS='2dc005293f7ef2ffe7cb3de9253117001417c3066115a0624c162f3b7311e9ab'
PLAN_DEADLINE=datetime(2026,10,19,13,20,tzinfo=timezone.utc).timestamp()
LATEST_START=datetime(2026,10,19,12,40,tzinfo=timezone.utc).timestamp()
EXPECTED_TRAIN_SHA='4adfdcea74a3e6f2f5f695eedada04a7a81e1bba65a3b555fcfff641b910122b'
LABEL4_PARAMETERS='a6ffc379c2dec0b60611b1663f1f8820321eaac53ab0c4df9d8191bbba62a4f9'
EVALUATION_RESERVE_SECONDS=1800
TRAINING_TIMEOUT_SECONDS=600


def bounded_training_deadline(now,requested_deadline=None):
    deadline=PLAN_DEADLINE if requested_deadline is None else min(PLAN_DEADLINE,requested_deadline)
    if now>min(LATEST_START,deadline-TRAINING_TIMEOUT_SECONDS-EVALUATION_RESERVE_SECONDS):
        raise TimeoutError('Latest start missed: preserve600s training and1800s evaluation budget')
    end=min(now+TRAINING_TIMEOUT_SECONDS,deadline-EVALUATION_RESERVE_SECONDS)
    if end<=now:raise TimeoutError('Insufficient time: full130 evaluation requires30min reserve')
    return deadline,end


def validate_config(config):
    count=config['expected_pairs']
    if type(count) is not int or count!=24:
        raise ValueError('This isolated continuation requires exactly24 pairs')
    if config['expected_train_sha256']!=EXPECTED_TRAIN_SHA:
        raise ValueError('Reviewed24 frozen train SHA changed')
    common=config['common_argv']
    if not isinstance(common,list) or any(not isinstance(v,str) for v in common):
        raise ValueError('common_argv must be a string list')
    for forbidden in ('--output-dir','--smoke-only','--execute'):
        if forbidden in common:raise ValueError('Stage-only flag present in common argv')
    def option(name):
        if common.count(name)!=1:raise ValueError('Expected exactly one '+name)
        i=common.index(name)
        if i+1>=len(common):raise ValueError('Missing option value '+name)
        return common[i+1]
    if option('--epochs')!='1' or option('--gradient-accumulation')!='4':
        raise ValueError('Formal requires one full epoch and accumulation4')
    for name,value in (('--learning-rate','2e-6'),('--beta','0.1'),('--max-length','4096'),('--seed','42'),('--max-seconds','600')):
        if option(name)!=value:raise ValueError('Frozen continuation parameter changed: '+name)
    if option('--parent-adapter')!=PARENT:
        raise ValueError('Fresh original SFT parent required')
    pair_file=Path(option('--candidates-root'))/'candidate_train.jsonl'
    outputs=[Path(config[k]).resolve() for k in ('smoke_output','formal_output')]
    if len(set(outputs))!=2 or any(v==Path(PARENT).resolve() or v.is_relative_to(Path(PARENT).resolve()) for v in outputs):
        raise ValueError('Smoke/formal output paths must be separate and outside the parent')
    return common,pair_file,count


def check_previous_result(path):
    path=Path(path)
    receipt=json.loads(path.read_text())
    if receipt.get('completed') is not True or receipt.get('failed') or receipt.get('samples')!=130:
        raise ValueError('Label4 full130 must complete before continuation')
    verified=receipt['loaded_adapter_verification']['relations']
    if any(verified.get(k)!=LABEL4_PARAMETERS for k in ('expected_sha256','loaded_sha256','cuda_sha256')):
        raise ValueError('Previous completed result is not the actual label4 adapter')
    metrics=(path.parent/'metrics.json').read_bytes()
    if hashlib.sha256(metrics).hexdigest()!=receipt['metrics_sha256']:
        raise ValueError('Previous label4 metrics digest changed')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--label4-result-delivered',action='store_true',help='Root acknowledges the completed label4 result was delivered before this launch')
    args = parser.parse_args()
    if not args.label4_result_delivered:raise ValueError('Root must deliver label4 full130 result before launching this separate continuation')
    config = json.loads(Path(args.config).read_text())
    for name,digest in config['code_sha256'].items():
        if Path(name).name!=name:raise ValueError('Code lock must name an adjacent file')
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Runtime code lock changed: '+name)
    previous_sha=check_previous_result(config['previous_label4_manifest'])
    trainer = Path(__file__).with_name("semantic_dpo.py")
    common,pairs,count=validate_config(config)
    requested=args.deadline_unix if args.deadline_unix is not None else config.get('deadline_unix')
    deadline,training_end=bounded_training_deadline(time.time(),requested)
    for output in (config['smoke_output'],config['formal_output']):
        if Path(output).exists():raise ValueError('Refusing to overwrite existing output: '+output)
    content = pairs.read_bytes()
    assert hashlib.sha256(content).hexdigest() == config["expected_train_sha256"]
    assert len([line for line in content.splitlines() if line.strip()]) == count
    for stage, output, extra in (("smoke", config["smoke_output"], ["--smoke-only"]),
                                  ("formal", config["formal_output"], [])):
        print(json.dumps({"stage":stage,"status":"starting","parent":"unchanged_original_sft","previous_label4_manifest_sha256":previous_sha,"deadline_scope":"root_arranged_finite_continuation_plan"}),flush=True)
        remaining=training_end-time.time()
        if remaining<=0:raise TimeoutError('Training deadline reached;30min evaluation reserve retained')
        try:
            subprocess.run([sys.executable, str(trainer), "train", *common,
                            "--output-dir", output, *extra, "--execute"], check=True,timeout=remaining)
        except BaseException as exc:
            # Never turn a killed or incomplete trainer into a success receipt.
            if Path(output).exists():
                fail=Path(output)/'wrapper_failure.json'
                if not fail.exists():fail.write_text(json.dumps({'completed':False,'stage':stage,
                    'exception_type':type(exc).__name__,'deadline_unix':deadline,'failed_unix':time.time()})+'\n')
            raise
        receipt = json.loads((Path(output) / "training_manifest.json").read_text())
        assert receipt["completed"] is True and not receipt.get("failed")
        assert receipt["smoke_only"] is (stage == "smoke")
        assert receipt["reference_unchanged"] is True and receipt["policy_weights_changed"] is True
        assert receipt["saved_weights_bitwise_verified"] is True and receipt["saved_reload_exact"] is True
        assert receipt['adapter_initial_sha256']==PARENT_PARAMETERS
        assert receipt['parameters']['parent_adapter']==PARENT
        if stage == "smoke":
            assert receipt["admission"]["smoke_subset_count"] == 4
        else:
            seen = receipt["observed_exposure"]
            assert seen["unique_pairs_seen"] == seen["pairs"] == count
            assert seen["min_visits"] == seen["max_visits"] == 1
            assert seen['successful_backward_microsteps']==count
            assert receipt["global_step"] == count // 4
            assert receipt["current_runtime_contract_check_passed"] is True
        print(json.dumps({"stage": stage, "status": "complete", "updates": receipt["global_step"],
                          "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
