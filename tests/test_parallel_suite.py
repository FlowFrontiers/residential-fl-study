"""Scheduling, failure handling, and artifact equivalence for the CPU queue."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fl_pipeline.artifacts import atomic_json
from fl_pipeline.config import ExperimentConfig
from fl_pipeline.run_experiment import run_single
from scripts.run_parallel_suite import make_jobs, run_queue
from test_pipeline import synthetic_file


class ParallelSuite(unittest.TestCase):
    def test_matrix(self):
        args = argparse.Namespace(timing_reference=None, capacity='both',
                                  rounds=20, local_epochs=5, n_shadows=4,
                                  seeds=[42, 123, 456, 789, 1024])
        jobs = make_jobs(args)
        ids = {j['id'] for j in jobs}
        self.assertEqual(len(ids), len(jobs))
        self.assertEqual({kind: sum(j['kind'] == kind for j in jobs)
                          for kind in ('target', 'loss', 'shadow')},
                         {'target': 85, 'loss': 30, 'shadow': 30})
        self.assertEqual(sum(j['args']['n_shadows'] for j in jobs if j['kind'] == 'shadow'), 120)
        for job in jobs:
            self.assertTrue(set(job['deps']) <= ids)
            if job['kind'] != 'target':
                self.assertEqual(job['split'], 'stratified')
                self.assertFalse(job['args']['equal_weight'])
                self.assertEqual(len(job['deps']), 1)
            if job['args']['model_size'] == 'medium':
                self.assertNotEqual(job['config'], 'feddpa')
                self.assertFalse(job['args']['equal_weight'])

    def test_deadline_prevents_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {}
            args = argparse.Namespace(max_hours=1e-12, workers=1)
            job = {'id': 'pending', 'deps': [], 'priority': 1}
            with self.assertRaises(TimeoutError):
                run_queue(args, [job], Path(tmp), report)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['jobs']['pending']['state'], 'pending')
            self.assertFalse(report['finalization'])

    def test_worker_failure_blocks_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(max_hours=0.05, workers=1)
            job = {'id': 'invalid', 'deps': [], 'priority': 1}
            atomic_json(root / 'invalid.json', {'args': {}, 'source_sha256': 'mismatch'})
            report = {}
            with self.assertRaises(RuntimeError):
                run_queue(args, [job], root, report)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['jobs']['invalid']['state'], 'failed')
            self.assertFalse(report['finalization'])

    def test_parallel_end_to_end_and_resume(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / 'a.parquet', root / 'b.parquet'
            synthetic_file(a, 600, 1)
            synthetic_file(b, 420, 2)
            out = root / 'runs'
            cmd = [sys.executable, 'scripts/run_parallel_suite.py', '--workers', '4',
                   '--device', 'cpu', '--seeds', '42', '--rounds', '1', '--local-epochs', '1',
                   '--n-shadows', '1', '--home-a', str(a), '--home-b', str(b),
                   '--results-dir', str(out), '--max-hours', '0.5']
            dry = subprocess.run(cmd + ['--dry-run'], cwd=repo, capture_output=True, text=True, timeout=120)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse(out.exists())
            run = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1200)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            progress = json.loads(next(out.glob('parallel_*/progress.json')).read_text())
            self.assertEqual(progress['status'], 'complete')
            self.assertEqual(progress['completed_jobs'], 29)
            self.assertEqual(progress['shadow_trainings'], 6)
            for job in progress['jobs'].values():
                self.assertEqual(job['state'], 'complete')
            for manifest in out.glob('parallel_*/*_loss.json'):
                job = json.loads(manifest.read_text())
                target = progress['jobs'][job['deps'][0]]
                attack = progress['jobs'][job['id']]
                self.assertGreaterEqual(attack['started_elapsed_s'] + 0.1,
                                        target['started_elapsed_s'] + target['elapsed_s'])
            for size, suffix in (('small', ''), ('medium', '_baseline_16_medium')):
                numbers = json.loads((out / f'paper_numbers{suffix}.json').read_text())
                self.assertEqual(set(numbers['attacks']), {'mia', 'shadow_mia'})
                self.assertIn('temporal', numbers['evidence'])
                self.assertEqual('equal_weight' in numbers['evidence'], size == 'small')
            args = argparse.Namespace(home_a=str(a), home_b=str(b), results_dir=str(root / 'serial'),
                                      feature_set='baseline_16', model_size='small')
            config = ExperimentConfig(num_rounds=1, local_epochs=1)
            sequential = run_single(args, config, 'baseline_fl', 42)
            target_path = out / 'baseline_16/seed_42/baseline_fl.json'
            target = json.loads(target_path.read_text())
            self.assertEqual(target['spec'], sequential['spec'])
            for home in ('home_a', 'home_b'):
                self.assertEqual(target['rounds'][0][home], sequential['rounds'][0][home])
            again = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1200)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            self.assertEqual(json.loads(target_path.read_text())['checkpoint_sha256'], target['checkpoint_sha256'])
            latest = sorted(out.glob('parallel_*'))[-1]
            for log in latest.glob('*.log'):
                self.assertNotIn('Training ', log.read_text())


if __name__ == '__main__':
    unittest.main()
