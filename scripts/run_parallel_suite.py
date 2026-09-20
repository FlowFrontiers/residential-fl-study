#!/usr/bin/env python3
"""Execute the full CPU experiment matrix with dependency-aware worker scheduling."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_pipeline.artifacts import atomic_json, ensure_run_root, file_hash, source_hash
from fl_pipeline.cli import add_arguments, config_from_args
from fl_pipeline.config import CONFIG_NAMES
from fl_pipeline.run_experiment import run_single
from fl_pipeline.run_mia import run_one as loss_probe
from fl_pipeline.run_shadow_mia import run_one as shadow_probe
from scripts.run_suite import commands, PRIMARY

REPO = Path(__file__).resolve().parents[1]


def make_jobs(args):
    reference = json.loads(Path(args.timing_reference).read_text()) if args.timing_reference else None
    jobs = []
    for size in ('small', 'medium'):
        if args.capacity not in ('both', size):
            continue
        variants = [('main', 'stratified', False, CONFIG_NAMES if size == 'small' else PRIMARY),
                    ('temporal', 'temporal', False, PRIMARY)]
        if size == 'small':
            variants.append(('equal', 'stratified', True, PRIMARY))
        for variant, split, equal, configs in variants:
            for name in configs:
                measured = reference['jobs'][f'{size}/{name}'] if reference else None
                if measured:
                    cfg = measured['target_spec']['config']
                    scale = args.rounds * args.local_epochs / (cfg['num_rounds'] * cfg['local_epochs'])
                    cost = measured['target_elapsed_s'] * scale
                    shadow_cost = measured.get('shadow_elapsed_s', 0) * scale * args.n_shadows
                else:
                    cost = (4 if name == 'dp_sgd' else 2 if name == 'feddpa' else 1)
                    cost *= 1.5 if size == 'medium' else 1
                    shadow_cost = cost * args.n_shadows / 2
                for seed in args.seeds:
                    options = dict(vars(args), model_size=size, equal_weight=equal, seeds=[seed])
                    key = f'{size}_{variant}_{name}_{seed}'
                    target = dict(id=key, kind='target', config=name, seed=seed, split=split,
                                  args=options, deps=[], cost=cost)
                    jobs.append(target)
                    if variant == 'main' and name in PRIMARY:
                        for kind, weight in (('loss', cost * 0.01), ('shadow', shadow_cost)):
                            jobs.append(dict(id=f'{key}_{kind}', kind=kind, config=name, seed=seed,
                                             split=split, args=options, deps=[key], cost=weight))
                        target['priority'] = cost + shadow_cost
    for job in jobs:
        job.setdefault('priority', job['cost'])
    return jobs


def execute_job(path):
    job = json.loads(Path(path).read_text())
    args = argparse.Namespace(**job['args'])
    if source_hash() != job['source_sha256']:
        raise ValueError('Pipeline source differs from the scheduled snapshot')
    actual = {key: file_hash(getattr(args, key)) for key in ('home_a', 'home_b')}
    if actual != job['data_sha256']:
        raise ValueError('Datasets differ from the scheduled snapshot')
    config = config_from_args(args)
    if job['kind'] == 'target':
        run_single(args, config, job['config'], job['seed'], job['split'])
    elif job['kind'] == 'loss':
        loss_probe(args, config, job['config'], job['seed'])
    elif job['kind'] == 'shadow':
        shadow_probe(args, config, job['config'], job['seed'])
    else:
        raise ValueError(f'Unknown job kind: {job["kind"]}')


def final_commands(args):
    final = copy.copy(args)
    final.stage = 'all'
    # Reuse completed artifacts to build shared summaries only after every worker exits.
    return [(stage, cmd) for stage, cmd in commands(final) if stage != 'training']


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def run_queue(args, jobs, directory, report):
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    active, finished = {}, set()
    pending = {job['id']: job for job in jobs}
    status = {job['id']: {'state': 'pending'} for job in jobs}
    report.update(status='running', jobs=status, finalization=[])
    report_path = directory / 'progress.json'
    last_write = 0

    def save():
        report.update(elapsed_s=time.monotonic() - started, completed_jobs=len(finished),
                      running_jobs=len(active), pending_jobs=len(pending))
        atomic_json(report_path, report)

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('Invocation wall-clock limit reached; rerun to reuse completed artifacts')

    try:
        while pending or active:
            check_deadline()
            changed = False
            for key, (process, log, t0) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[key]
                status[key].update(state='complete' if code == 0 else 'failed', returncode=code,
                                   elapsed_s=time.monotonic() - t0)
                changed = True
                if code:
                    raise RuntimeError(f'{key} failed with exit {code}; see its worker log')
                finished.add(key)
                print(f'Complete {len(finished)}/{len(jobs)}: {key}', flush=True)
            ready = sorted((j for j in pending.values() if set(j['deps']) <= finished),
                           key=lambda j: (-j['priority'], j['id']))
            for job in ready[:max(0, args.workers - len(active))]:
                key = job['id']
                log_path = directory / f'{key}.log'
                log = log_path.open('w')
                cmd = [sys.executable, str(Path(__file__).resolve()), '--job-file',
                       str(directory / f'{key}.json')]
                try:
                    process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True,
                                               env=dict(os.environ, PYTHONUNBUFFERED='1'))
                except BaseException:
                    log.close()
                    raise
                active[key] = process, log, time.monotonic()
                del pending[key]
                status[key].update(state='running', pid=process.pid, log=str(log_path), argv=cmd,
                                   started_elapsed_s=time.monotonic() - started)
                changed = True
                print(f'Start [{len(active)}/{args.workers}]: {key}', flush=True)
            if pending and not active:
                raise RuntimeError('Unresolved job dependencies')
            if changed or time.monotonic() - last_write >= 30:
                save()
                last_write = time.monotonic()
            if active:
                time.sleep(1)
        report['status'] = 'finalizing'
        save()
        for index, (stage, cmd) in enumerate(final_commands(args)):
            check_deadline()
            key = f'final_{index}_{stage}'
            path = directory / f'{key}.log'
            log = path.open('w')
            process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
            active[key] = process, log, time.monotonic()
            while process.poll() is None:
                check_deadline()
                time.sleep(1)
            _, _, t0 = active.pop(key)
            log.close()
            report['finalization'].append(dict(stage=stage, argv=cmd, log=str(path),
                                                returncode=process.returncode,
                                                elapsed_s=time.monotonic() - t0))
            save()
            if process.returncode:
                raise RuntimeError(f'{key} failed; see {path}')
            print(f'Finalized: {stage}', flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for process, _, _ in active.values():
            stop_process(process)
        for key, (process, log, t0) in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            if key in status:
                status[key].update(state='interrupted', returncode=process.returncode,
                                   elapsed_s=time.monotonic() - t0)
        active.clear()
        save()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--job-file':
        execute_job(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--capacity', choices=['both', 'small', 'medium'], default='both')
    parser.add_argument('--n-shadows', type=int, default=4)
    parser.add_argument('--max-hours', type=float, default=36)
    parser.add_argument('--timing-reference', help='Optional benchmark.json for longest-first scheduling')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.device != 'cpu' or args.num_threads != 1:
        parser.error('This launcher requires --device cpu --num-threads 1')
    if args.workers < 1 or args.n_shadows < 1 or not 0 < args.max_hours < float('inf'):
        parser.error('Require positive workers, shadows, and finite wall-clock limit')
    if args.model_size != 'small':
        parser.error('Use --capacity to select the model capacities')
    config_from_args(args)
    args.shadow_ratio = 0.5
    for key in ('results_dir', 'home_a', 'home_b'):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    jobs = make_jobs(args)
    counts = {kind: sum(j['kind'] == kind for j in jobs) for kind in ('target', 'loss', 'shadow')}
    print(f'Queue: {counts}; shadow trainings: {counts["shadow"] * args.n_shadows}', flush=True)
    if args.dry_run:
        for job in sorted(jobs, key=lambda j: (-j['priority'], j['id'])):
            print(job['id'], 'requires', job['deps'], 'priority', job['priority'])
        return
    root = ensure_run_root(args.results_dir)
    with (root / '.parallel.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another parallel launcher holds this output directory')
        directory = root / f'parallel_{time.time_ns()}'
        directory.mkdir()
        provenance = dict(source_sha256=source_hash(),
                          data_sha256={k: file_hash(getattr(args, k)) for k in ('home_a', 'home_b')})
        for job in jobs:
            atomic_json(directory / f'{job["id"]}.json', dict(job, **provenance))
        report = dict(workers=args.workers, threads_per_worker=1, arguments=vars(args),
                      job_counts=counts, shadow_trainings=counts['shadow'] * args.n_shadows,
                      launcher_sha256=file_hash(__file__), **provenance,
                      timing_context='Concurrent worker schedule; not isolated per-method runtime')
        atomic_json(directory / 'manifest.json', report)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        print(f'Progress and worker logs: {directory}', flush=True)
        run_queue(args, jobs, directory, report)


if __name__ == '__main__':
    main()
