"""Start the separate finite DPO continuation only after verified first delivery."""
from pathlib import Path
import hashlib,json,shlex,subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REMOTE=r'''
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time
p=json.load(sys.stdin)
assert time.time()<=1792413600,'Latest12:40 start passed'
root=Path('/workspace/medicalner/round3_20261019')
old=json.loads((root/'queue/state.json').read_text())
assert old.get('current') is None and len(old['jobs'])==2
assert all(j['status']=='complete' and j['return_code']==0 for j in old['jobs'].values())
manifest=root/'evaluation/label4_cont1epoch_full130_fixed_entity/manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==p['label4_manifest_sha256']
m=json.loads(manifest.read_text())
assert m['completed'] is True and m['samples']==130 and not m['failed']
assert m['metrics_sha256']==p['label4_metrics_sha256']
assert hashlib.sha256(manifest.with_name('metrics.json').read_bytes()).hexdigest()==p['label4_metrics_sha256']
assert p['first_result_delivered_utc'] and p['root_recount_status']=='PASS'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert not gpu,'GPU still occupied; do not preempt'
plan=root/'dpo24/code/launch_plan.prepared.json'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==p['plan_sha256']
runner=Path('/workspace/medicalner/20261019/code_train_v2/scripts/run_semantic_queue.py')
assert hashlib.sha256(runner.read_bytes()).hexdigest()==p['runner_sha256']
state=root/'dpo24/queue'
state.mkdir(mode=0o700)
log=(state/'runner.log').open('xb')
process=subprocess.Popen([sys.executable,str(runner),'--plan',str(plan),'--state-dir',str(state)],
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={**p,'status':'continuation_queue_launched','runner_pid':process.pid,'launched_unix':time.time(),
 'deadline_unix':1792416000,'queue_root':str(state),'previous_queue_finished':True,'GPU_idle_before_launch':True}
(state/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

def main():
    output=HERE/'dpo24_launch_receipt.json'
    assert not output.exists(),'Never repeat a recorded launch'
    state=json.loads((HERE/'state.json').read_text())
    assert state.get('first_result_delivered_utc'),'Report the first complete result before continuing'
    recount_path=HERE/'metrics_review/label4_full130_recount.json'
    recount=json.loads(recount_path.read_text())
    assert recount['status']=='PASS' and recount['samples']==130 and not recount['pilot']
    assert recount['fixed_entity_comparison']['filtered']['exactly_equal'] is True
    runner=ROOT/'MedicalNER-Qwen3/scripts/run_semantic_queue.py'
    plan=HERE/'dpo24_continuation_runtime/launch_plan.prepared.json'
    packet={'first_result_delivered_utc':state['first_result_delivered_utc'],'root_recount_status':recount['status'],
      'root_recount_sha256':hashlib.sha256(recount_path.read_bytes()).hexdigest(),
      'label4_manifest_sha256':recount['inputs_sha256']['manifest.json'],
      'label4_metrics_sha256':recount['inputs_sha256']['metrics.json'],
      'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
      'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest()}
    assert packet['plan_sha256']=='a8ef4e94c4fad57a8844668373bb2890ebcc6e6fde31fe5e4a13da9703ed68c0'
    r=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
      'uhost',shlex.join(['/root/venvs/medicalner/bin/python','-c',REMOTE])],input=json.dumps(packet).encode(),
      stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1400:])
    receipt=json.loads(r.stdout);output.write_text(json.dumps(receipt,indent=2)+'\n');output.chmod(0o600)
    print(json.dumps(receipt))

if __name__=='__main__':main()
