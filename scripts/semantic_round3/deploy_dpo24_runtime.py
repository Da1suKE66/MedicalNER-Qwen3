"""Deploy frozen source/configuration only; do not launch a queue or GPU job."""
from pathlib import Path
import hashlib,json,shlex,subprocess

HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'dpo24_continuation_runtime'
REMOTE=r'''
from pathlib import Path
import hashlib,json,sys
packet=json.load(sys.stdin)
out=Path('/workspace/medicalner/round3_20261019/dpo24/code')
assert not out.exists(),'Refuse existing runtime'
required={'semantic_dpo.py','build_pairs.py','run_smoke_then_train.py','dpo24_config.json','launch_plan.prepared.json'}
assert set(packet['files'])==required
values={n:v['text'].encode() for n,v in packet['files'].items()}
for n,b in values.items():assert hashlib.sha256(b).hexdigest()==packet['files'][n]['sha256']
config=json.loads(values['dpo24_config.json'])
for n,h in config['code_sha256'].items():assert hashlib.sha256(values[n]).hexdigest()==h
assert config['expected_pairs']==24
assert config['expected_train_sha256']=='4adfdcea74a3e6f2f5f695eedada04a7a81e1bba65a3b555fcfff641b910122b'
out.mkdir(parents=True,mode=0o700)
for n,b in values.items():
 p=out/n
 with p.open('xb') as f:f.write(b)
 p.chmod(0o600)
print(json.dumps({'status':'runtime_deployed_not_queued','destination':str(out),'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in values.items()}}))
'''

def main():
    receipt=HERE/'dpo24_runtime_deployment.json'
    assert not receipt.exists(),'Refuse repeated deployment'
    names=('semantic_dpo.py','build_pairs.py','run_smoke_then_train.py','dpo24_config.json','launch_plan.prepared.json')
    files={}
    for n in names:
        b=(RUNTIME/n).read_bytes();files[n]={'text':b.decode(),'sha256':hashlib.sha256(b).hexdigest()}
    assert files['run_smoke_then_train.py']['sha256']=='fc23be805eec1e3f45acbb9e971fe75da8f1c6fb8698d880949c9f021511a979'
    assert files['dpo24_config.json']['sha256']=='11a6b0728f29552f9b3ac2f306abf8fa2d981be6a743c4a444b3bb85f696bbd4'
    r=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'uhost',shlex.join(['/root/venvs/medicalner/bin/python','-c',REMOTE])],input=json.dumps({'files':files}).encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1000:])
    d=json.loads(r.stdout);assert d['sha256']=={n:f['sha256'] for n,f in files.items()}
    receipt.write_text(json.dumps(d,indent=2)+'\n');receipt.chmod(0o600);print(json.dumps(d))

if __name__=='__main__':main()
