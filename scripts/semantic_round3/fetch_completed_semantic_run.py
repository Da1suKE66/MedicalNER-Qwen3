"""Download immutable completed results; never submit data or start remote work."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile


REMOTE = r'''
import datetime, hashlib, io, json, pathlib, sys, tarfile
name = sys.argv[1]
expected_samples = int(sys.argv[2])
root = pathlib.Path('/workspace/medicalner/round3_20261019/evaluation') / name
names = ['manifest.json','run_manifest.json','metrics.json','progress.json',
         'generations.jsonl','predictions.raw.jsonl','predictions.filtered.jsonl']
contents = {n:(root/n).read_bytes() for n in names}
m = json.loads(contents['manifest.json'])
assert m == json.loads(contents['run_manifest.json']), 'manifest disagreement'
assert m.get('status') == 'complete' and m.get('complete') is True and m.get('completed') is True and m.get('failed') is False, 'run not complete'
metrics = json.loads(contents['metrics.json'])
assert metrics.get('status') == 'complete' and metrics.get('complete') is True, 'metrics not complete'
assert m['metrics_sha256'] == hashlib.sha256(contents['metrics.json']).hexdigest(), 'metrics hash mismatch'
assert m['samples'] == metrics['samples'] == expected_samples, 'incomplete population'
for n in metrics['artifacts_sha256']:
    assert pathlib.Path(n).name == n and not n.startswith('.'), 'unsafe artifact name'
    assert (root/n).resolve().parent == root.resolve(), 'artifact escapes run directory'
    if n not in contents: contents[n] = (root/n).read_bytes()
    assert metrics['artifacts_sha256'][n] == hashlib.sha256(contents[n]).hexdigest(), n
receipt = {'run':name,'remote_root':str(root),'downloaded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in contents.items()}}
contents['transfer_manifest.json'] = (json.dumps(receipt, sort_keys=True) + '\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
    for n,b in contents.items():
        info=tarfile.TarInfo(n); info.size=len(b); info.mode=0o600
        tar.addfile(info,io.BytesIO(b))
'''


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--expected-samples", type=int, choices=(6,130), default=130)
    parser.add_argument("--expected-modes", choices=("both", "live"), default="live")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.run):
        parser.error("Run name must be an existing simple experiment identifier")
    base = Path(__file__).resolve().parent / "completed_runs"
    base.mkdir(exist_ok=True)
    destination = base / args.run
    if destination.exists():
        raise FileExistsError("Existing result directory is immutable: " + str(destination))
    stage = Path(tempfile.mkdtemp(prefix=".fetch_" + args.run + "_", dir=base))
    stage.chmod(0o700)
    archive = stage / "transfer.tar"
    try:
        command = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "uhost",
                   "/root/venvs/medicalner/bin/python - " + shlex.quote(args.run) + " " + str(args.expected_samples)]
        with archive.open("xb") as stream:
            archive.chmod(0o600)
            result = subprocess.run(command, input=REMOTE.encode(), stdout=stream, stderr=subprocess.PIPE, timeout=120)
        if result.returncode:
            raise RuntimeError("Remote completed-result check failed: " + result.stderr.decode(errors="replace")[-2000:])
        if archive.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("Unexpected result archive size")
        archive_sha = sha(archive.read_bytes())
        required = {"manifest.json", "run_manifest.json", "metrics.json", "progress.json", "generations.jsonl",
                    "predictions.raw.jsonl", "predictions.filtered.jsonl", "transfer_manifest.json"}
        blobs = {}
        with tarfile.open(archive, "r:") as tar:
            for member in tar:
                if not re.fullmatch(r"[a-z][a-z0-9._-]*", member.name) or member.name in blobs or not member.isfile() or member.size > 128 * 1024 * 1024:
                    raise ValueError("Unexpected archive member")
                blobs[member.name] = tar.extractfile(member).read()
        assert required <= set(blobs)
        declared_artifacts = set(json.loads(blobs["metrics.json"])["artifacts_sha256"])
        assert set(blobs) == required | declared_artifacts
        transfer = json.loads(blobs["transfer_manifest.json"])
        assert transfer["run"] == args.run
        assert set(transfer["files"]) == set(blobs) - {"transfer_manifest.json"}
        for name, expected in transfer["files"].items():
            assert len(blobs[name]) == expected["bytes"] and sha(blobs[name]) == expected["sha256"], name
        manifest = json.loads(blobs["manifest.json"])
        metrics = json.loads(blobs["metrics.json"])
        assert manifest == json.loads(blobs["run_manifest.json"])
        assert manifest["complete"] and manifest["completed"] and not manifest["failed"]
        assert manifest["metrics_sha256"] == sha(blobs["metrics.json"])
        modes = {"live", "oracle"} if args.expected_modes == "both" else {"live"}
        assert set(metrics["modes"]) == modes and metrics["samples"] == args.expected_samples
        for mode in modes:
            assert metrics["modes"][mode]["samples"] == args.expected_samples
        for name in ["predictions.raw.jsonl", "predictions.filtered.jsonl"]:
            rows = [json.loads(line) for line in blobs[name].decode().splitlines() if line.strip()]
            assert len(rows) == args.expected_samples * len(modes) and len({row["id"] for row in rows}) == len(rows)
            for mode in modes:
                assert sum(row["mode"] == mode for row in rows) == args.expected_samples
        for name, content in blobs.items():
            with (stage / name).open("xb") as stream:
                os.chmod(stage / name, 0o600)
                stream.write(content)
        local_receipt = {"status":"downloaded_complete_artifacts_sha_verified_independent_metric_recount_pending",
                         "run":args.run,"modes":sorted(modes),"samples":args.expected_samples,
                         "checked_utc":datetime.now(timezone.utc).isoformat(),"archive_sha256":archive_sha,
                         "manifest_sha256":sha(blobs["manifest.json"]),"metrics_sha256":sha(blobs["metrics.json"]),
                         "transfer_manifest_sha256":sha(blobs["transfer_manifest.json"]),
                         "downloader_sha256":sha(Path(__file__).read_bytes()),"remote_work_started":False}
        with (stage / "download_receipt.json").open("x") as stream:
            os.chmod(stage / "download_receipt.json",0o600)
            json.dump(local_receipt,stream,indent=2);stream.write("\n")
        archive.unlink()
        if destination.exists():
            raise FileExistsError("Result destination appeared during download")
        stage.rename(destination)
        print(json.dumps({**local_receipt,"destination":str(destination)}))
    except BaseException as exc:
        with (stage / "download_failed.json").open("x") as stream:
            os.chmod(stage / "download_failed.json",0o600)
            json.dump({"status":"failed_not_a_completed_run","error":f"{type(exc).__name__}: {exc}"},stream)
        raise


if __name__ == "__main__":
    main()
