"""Finite evaluation-only recovery; frozen checkpoints, original sample IDs.

Four disjoint document shards accelerate G1 without changing within-document
generation batches. A merged metric is published only after exact full coverage.
No training, checkpoint modification, data export, or inference API calls.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

BASE = Path('/temp/liluchen/medicalner_agent_generative_20260908')
ROOT = BASE / 'resumed_20260908_1900'
OLD = BASE / 'executions'
RUNS = {'G1_data': OLD / 'G1_data_s42_30162', 'G0': OLD / 'G0_s42_31420'}
MODEL = '/temp/liluchen/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
GOLD = '/temp/liluchen/medicalner_agent_v1_20260825/data/dev_gold_full.json'
CACHE = '/temp/liluchen/medicalner_agent_v1_20260825/reports/dev_predictions_v4_e1r1_full_bigchunk_jsonstop_512.jsonl'
DEPS = '/cache/liluchen/medicalner_eval_resume_20260908/deps'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The shared object-backed mount rejects rename; lock the exact metadata
    # file instead. Readers must retry JSONDecodeError during publication.
    with path.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
        handle.flush()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_shard_hooks(module, indices, total):
    """Retain the evaluator and prompts; only partition document iteration."""
    original_run = module.run_records
    original_build = module.build_batches
    original_metrics = module.evaluate_records
    original_rules = module.rule_ablation

    def subset(records):
        return [records[i] for i in indices] if len(records) == total else records

    # New evaluators bind input identities and batch IDs to explicit original
    # indices. Keep the older remapping path intact for historical snapshots.
    if 'sample_indices' in inspect.signature(original_run).parameters:
        def indexed_kwargs(records, kwargs):
            supplied = kwargs.get('sample_indices')
            expected = list(range(total)) if len(records) == total else list(indices)
            if supplied is not None and list(supplied) != expected:
                raise ValueError('Shard sample_indices alignment changed')
            return {**kwargs, 'sample_indices': list(indices)}

        def indexed_run(records, cached, *args, **kwargs):
            if len(records) != total or (cached is not None and len(cached) != total):
                raise ValueError('Full source/cache alignment changed')
            return original_run(
                subset(records), subset(cached) if cached is not None else None,
                *args, **indexed_kwargs(records, kwargs))

        def indexed_metrics(records, predictions, **kwargs):
            return original_metrics(
                subset(records), predictions, **indexed_kwargs(records, kwargs))

        def indexed_rules(records, predictions, **kwargs):
            return original_rules(
                subset(records), predictions, **indexed_kwargs(records, kwargs))

        module.run_records = indexed_run
        module.evaluate_records = indexed_metrics
        module.rule_ablation = indexed_rules
        return

    def build(*args, sample_id, **kwargs):
        return original_build(*args, sample_id=indices[sample_id], **kwargs)

    def run(records, cached, *args, **kwargs):
        if len(records) != total or (cached is not None and len(cached) != total):
            raise ValueError('Full source/cache alignment changed')
        predictions, raw, audit = original_run(
            subset(records), subset(cached) if cached is not None else None,
            *args, **kwargs)
        for row in raw:
            row['sample_id'] = indices[row['sample_id']]
        return predictions, raw, audit

    module.build_batches = build
    module.run_records = run
    module.evaluate_records = lambda records, predictions: original_metrics(subset(records), predictions)
    module.rule_ablation = lambda records, predictions: original_rules(subset(records), predictions)


def evaluate_shard(index, count, argv):
    import generative.evaluate as module
    gold = argv[argv.index('--gold') + 1]
    output = Path(argv[argv.index('--output-dir') + 1])
    records = json.loads(Path(gold).read_text())
    indices = list(range(index, len(records), count))
    if not indices or not 0 <= index < count:
        raise ValueError('Invalid shard')
    install_shard_hooks(module, indices, len(records))
    sys.argv = [sys.argv[0], *argv]
    module.main()
    dump(output / 'shard.json', {'index': index, 'count': count, 'indices': indices,
         'total': len(records), 'gold_sha256': sha(gold), 'completed_at': stamp()})


def merge(mode):
    from generative.evaluate import evaluate_records, rule_ablation
    destination = ROOT / 'G1_data' / mode
    shards = [ROOT / 'G1_data' / (mode + '_shard_' + str(i)) for i in range(4)]
    with (ROOT / ('merge_' + mode + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / 'metrics.json').exists():
            return
        if not all((p / 'shard.json').exists() for p in shards):
            return
        records = json.loads(Path(GOLD).read_text())
        predictions, raw, seen, counts = [None] * len(records), [], set(), Counter()
        reference = None
        for i, path in enumerate(shards):
            meta = json.loads((path / 'shard.json').read_text())
            report = json.loads((path / 'metrics.json').read_text())
            assert meta['indices'] == list(range(i, len(records), 4))
            assert meta['gold_sha256'] == sha(GOLD) and meta['total'] == len(records)
            args = dict(report['arguments']); args.pop('output_dir')
            if reference is None:
                reference = args
            assert args == reference, 'Shard configuration mismatch'
            rows = [json.loads(s) for s in (path / 'predictions.jsonl').read_text().splitlines()]
            assert len(rows) == len(meta['indices'])
            for index, row in zip(meta['indices'], rows):
                assert index not in seen
                predictions[index] = row; seen.add(index)
            raw.extend(json.loads(s) for s in (path / 'generations.jsonl').read_text().splitlines())
            counts.update(report['coverage_and_generation'])
        assert seen == set(range(len(records))) and all(p is not None for p in predictions)
        raw.sort(key=lambda row: row['sample_id'])
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in [('predictions', predictions), ('generations', raw)]:
            (destination / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        # Publish metrics last, only after full aligned recomputation (not F1 averaging).
        dump(destination / 'metrics.json', {
            'metrics': evaluate_records(records, predictions),
            'coverage_and_generation': dict(counts),
            'rule_ablation': rule_ablation(records, predictions),
            'arguments': reference, 'full_sample_count': len(records),
            'entity_source': 'oracle' if mode == 'oracle' else 'fixed_predicted_cache',
            'not_full_live_pipeline': True, 'merged_shards': 4,
            'exact_once_sample_coverage': True, 'completed_at': stamp()})


def worker(index, port):
    os.umask(0o077)
    ROOT.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / ('worker_' + str(index) + '.json')
    state = {'worker': index, 'port': port, 'pid': os.getpid(), 'host': socket.gethostname(),
             'started_at': stamp(), 'evaluation_only': True, 'status': 'starting'}
    with (ROOT / ('worker_' + str(index) + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        state['gpu_uuid'] = gpu[0].strip()
        if int(gpu[1].strip()) > 1024:
            state.update(status='blocked_gpu_not_idle'); dump(state_path, state); return
        with (ROOT / (state['gpu_uuid'] + '.lock')).open('a') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def run(variant, checkpoint, name, *, oracle=False, tune=False, shard=False):
                code = RUNS[variant] / 'code' / 'agent'
                out = ROOT / variant / name
                state.update(status='running', stage=variant + '/' + name, updated_at=stamp())
                dump(state_path, state)
                args = ['--gold', str(BASE / 'data/tune_records.json') if tune else GOLD,
                        '--base-model', MODEL, '--adapter', str(RUNS[variant] / 'train' / ('checkpoint-' + str(checkpoint))),
                        '--variant', variant, '--output-dir', str(out), '--max-new-tokens', '16000', '--inference-batch-size', '8']
                if not tune:
                    args.extend(['--entity-predictions', CACHE])
                if oracle:
                    args.append('--oracle')
                command = [sys.executable, '-u', str(Path(__file__).resolve()), 'shard', '--index', str(index), '--count', '4', '--', *args] if shard else [sys.executable, '-u', '-m', 'generative.evaluate', *args]
                env = dict(os.environ, PYTHONPATH=DEPS + ':' + str(code), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
                log_path = ROOT / (variant + '_' + name + '.log')
                with log_path.open('w') as log:
                    result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(name + ' exit=' + str(result.returncode))

            try:
                if index < 4:
                    for mode in ['predicted', 'oracle']:
                        run('G1_data', 279, mode + '_shard_' + str(index), oracle=mode == 'oracle', shard=True)
                        merge(mode)
                if index in (0, 1, 4):
                    step = {0: 186, 1: 279, 4: 93}[index]
                    run('G0', step, 'tune_' + str(step), oracle=True, tune=True)
                    if index in (0, 1):
                        state.update(status='waiting_for_all_tuning', updated_at=stamp()); dump(state_path, state)
                        deadline = time.monotonic() + 4 * 3600
                        while not all((ROOT / 'G0' / ('tune_' + str(s)) / 'metrics.json').exists() for s in (93, 186, 279)):
                            if time.monotonic() > deadline:
                                raise TimeoutError('Other G0 tuning did not complete within four hours')
                            time.sleep(10)
                        from generative.run_experiment import select_checkpoint
                        candidates = [{'checkpoint': str(RUNS['G0'] / 'train' / ('checkpoint-' + str(s))), 'step': s,
                            'relation': json.loads((ROOT / 'G0' / ('tune_' + str(s)) / 'metrics.json').read_text())['metrics']['free_text']['relation']} for s in (93, 186, 279)]
                        selected = select_checkpoint(candidates)
                        dump(ROOT / 'G0' / 'selection.json', {'selected': selected, 'all_candidates': candidates, 'test_used_for_selection': False})
                        run('G0', selected['step'], 'predicted' if index == 0 else 'oracle', oracle=index == 1)
                state.update(status='complete', completed_at=stamp()); dump(state_path, state)
            except Exception as exc:
                state.update(status='failed', error=str(exc), updated_at=stamp()); dump(state_path, state)
                raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['launch', 'worker', 'shard'])
    p.add_argument('--index', type=int, required=True); p.add_argument('--port', type=int)
    p.add_argument('--count', type=int, default=4)
    args, remaining = p.parse_known_args()
    if args.mode == 'shard':
        evaluate_shard(args.index, args.count, remaining[1:] if remaining[:1] == ['--'] else remaining)
    elif args.mode == 'worker':
        worker(args.index, args.port)
    else:
        os.umask(0o077); ROOT.mkdir(parents=True, exist_ok=True)
        if (ROOT / ('launch_' + str(args.index) + '.json')).exists():
            raise ValueError('Worker already launched; inspect status before any retry')
        env = dict(os.environ, PYTHONPATH=DEPS + ':' + str(RUNS['G1_data'] / 'code/agent'), PYTHONDONTWRITEBYTECODE='1')
        with (ROOT / ('worker_' + str(args.index) + '.log')).open('w') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--index', str(args.index), '--port', str(args.port)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        result = {'pid': proc.pid, 'port': args.port, 'worker': args.index, 'launched_at': stamp(), 'runner_sha256': sha(__file__)}
        dump(ROOT / ('launch_' + str(args.index) + '.json'), result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
