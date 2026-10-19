"""Copy the approved preference release to the user's existing training server.

No API calls or GPU work. Destination is new and may not overwrite files.
Credentials and dataset bodies are never logged.
"""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
DESTINATION = '/workspace/medicalner/round3_20261019/dpo24/data'
REMOTE = r'''
import hashlib,json,pathlib,sys
packet=json.load(sys.stdin)
out=pathlib.Path('/workspace/medicalner/round3_20261019/dpo24/data')
assert packet['destination']==str(out)
assert not out.exists(), 'Destination exists; do not overwrite'
assert set(packet['files'])=={'candidate_train.jsonl','mechanical_holdout.jsonl','manifest.json'}
values={name:item['text'].encode() for name,item in packet['files'].items()}
for name,blob in values.items():
 assert hashlib.sha256(blob).hexdigest()==packet['files'][name]['sha256']
m=json.loads(values['manifest.json'])
assert m['train_pairs']==24 and m['human_adjudicated'] is False
assert m['output_sha256']['candidate_train.jsonl']=='4adfdcea74a3e6f2f5f695eedada04a7a81e1bba65a3b555fcfff641b910122b'
for name,digest in m['output_sha256'].items():
 assert hashlib.sha256(values[name]).hexdigest()==digest
out.mkdir(parents=True,mode=0o700)
for name,blob in values.items():
 p=out/name
 with p.open('xb') as f:f.write(blob)
 p.chmod(0o600)
print(json.dumps({'status':'data_deployed_not_trained','destination':str(out),'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in values.items()}}))
'''


def main():
    receipt = HERE / 'dpo24_data_deployment.json'
    if receipt.exists():
        raise FileExistsError('Deployment receipt exists; do not repeat')
    files = {}
    for name in ('candidate_train.jsonl', 'mechanical_holdout.jsonl', 'manifest.json'):
        blob = (HERE / 'reviewed24' / name).read_bytes()
        files[name] = {'text': blob.decode(), 'sha256': hashlib.sha256(blob).hexdigest()}
    packet = {'destination': DESTINATION, 'files': files}
    import shlex
    command = ['/root/venvs/medicalner/bin/python', '-c', REMOTE]
    result = subprocess.run(['ssh', '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', 'uhost', shlex.join(command)],
        input=json.dumps(packet).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        raise RuntimeError('Deployment failed: ' + result.stderr.decode(errors='replace')[-1000:])
    r = json.loads(result.stdout)
    assert r['sha256'] == {n: v['sha256'] for n,v in files.items()}
    receipt.write_text(json.dumps(r,indent=2)+'\n');receipt.chmod(0o600)
    print(json.dumps(r))


if __name__ == '__main__':
    main()
