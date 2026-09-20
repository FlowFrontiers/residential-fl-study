"""Invariant and end-to-end tests using synthetic flow records."""

import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from opacus.accountants import RDPAccountant

from fl_pipeline.config import BASELINE_16, CORE_CLASSES, CONFIG_NAMES, ExperimentConfig, get_feature_cols
from fl_pipeline.data import prepare_federated_data, transform_features
from fl_pipeline.model import TrafficClassifier
from fl_pipeline.feddpa import compute_fisher_diagonal, clip_and_noise_update, init_local_from_mask
from fl_pipeline.metrics import paired_confidence_interval
from fl_pipeline.mia import compute_mia_metrics, sample_balanced_indices
from fl_pipeline.dp import make_dp_loader, setup_dp_training
from fl_pipeline.train import train_epoch
from fl_pipeline.artifacts import load_target, file_hash, ensure_run_root, atomic_json
from fl_pipeline.run_experiment import prepare_run, run_single
from fl_pipeline.run_mia import run_one as loss_probe
from fl_pipeline.run_shadow_mia import run_one as shadow_probe, make_shadow_data
from fl_pipeline.analyze import load_results, summarize
from scripts.extract_paper_numbers import extract


def synthetic_file(path, n, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.uniform(0, 1000, n) for c in BASELINE_16})
    frame["protocol"] = rng.choice([6, 17], n)
    frame["bidirectional_packets"] = rng.integers(2, 30, n)
    frame["category"] = np.resize(sorted(CORE_CLASSES), n)
    frame["confidence"] = "DPI"
    frame["bidirectional_first_seen_ms"] = np.arange(n) * 1000
    frame.to_parquet(path, index=False)


class Invariants(unittest.TestCase):
    def test_record_isolation_and_subset(self):
        x = np.ones((3, len(BASELINE_16)))
        base = transform_features(x, BASELINE_16)
        x[0, 0] = 1000
        changed = transform_features(x, BASELINE_16)
        np.testing.assert_array_equal(base[1:], changed[1:])
        cols = get_feature_cols("fs_mild", "baseline_16")
        idx = [BASELINE_16.index(c) for c in cols]
        np.testing.assert_array_equal(changed[:, idx], transform_features(x[:, idx], cols))
        with self.assertRaises(ValueError):
            transform_features(np.full_like(x, np.nan), BASELINE_16)

    def test_architecture(self):
        for n in (8, 12, 16):
            model = TrafficClassifier(n, 6)
            self.assertEqual(model.hidden_dims, [16, 16])
        self.assertEqual(sum(p.numel() for p in TrafficClassifier(16, 6).parameters()), 646)
        self.assertEqual(sum(p.numel() for p in TrafficClassifier(16, 6, [128, 64]).parameters()), 10822)

    def test_fisher_per_example(self):
        torch.manual_seed(12)
        model = TrafficClassifier(3, 2, [4, 4])
        x = np.random.default_rng(12).normal(size=(7, 3)).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0])
        expected = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        for a, b in zip(x, y):
            loss = torch.nn.functional.cross_entropy(model(torch.tensor(a)[None]), torch.tensor([b]))
            gradients = torch.autograd.grad(loss, tuple(model.parameters()))
            for (name, _), g in zip(model.named_parameters(), gradients):
                expected[name] += g.square() / len(y)
        actual = compute_fisher_diagonal(model, x, y, 7)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-7, rtol=1e-5)

    def test_full_update_clipping_and_noise(self):
        model = TrafficClassifier(3, 2)
        local = copy.deepcopy(model)
        with torch.no_grad():
            for p in local.parameters():
                p.add_(3)
        update = clip_and_noise_update(local, model, 1, 0)
        self.assertAlmostEqual(torch.cat([d.flatten() for d in update.values()]).norm().item(), 1, places=5)
        torch.manual_seed(5)
        noised = clip_and_noise_update(model, model, 1, 1)
        self.assertTrue(all(torch.all(v != 0) for v in noised.values()))
        mask = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        initialized = init_local_from_mask(model, mask, dict(local.named_parameters()), torch.device("cpu"))
        for a, b in zip(model.parameters(), initialized.parameters()):
            torch.testing.assert_close(a, b)

    def test_empty_poisson_batch(self):
        model = TrafficClassifier(3, 2)
        accountant = RDPAccountant()
        loader = make_dp_loader(np.zeros((8, 3), dtype=np.float32), np.zeros(8, dtype=np.int64), 4)
        model, opt, loader, _ = setup_dp_training(model, torch.optim.Adam(model.parameters()), loader, 1, 1, accountant)
        self.assertEqual(loader.sample_rate, 0.5)
        train_epoch(model, [(torch.empty(0, 3), torch.empty(0, dtype=torch.long))], opt,
                    torch.nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertEqual(sum(n for _, _, n in accountant.history), 1)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_ci_and_output_safety(self):
        self.assertIsNone(paired_confidence_interval([1], [0])["ci_low"])
        with self.assertRaises(ValueError):
            paired_confidence_interval([1, 2], [0])
        with self.assertRaises(ValueError):
            ensure_run_root(Path(__file__).resolve().parents[1] / "results")

    def test_attack_score_conventions(self):
        perfect = compute_mia_metrics(np.zeros(20), np.ones(20))
        self.assertEqual(perfect["auc"], 1)
        self.assertEqual(perfect["tpr_at_1fpr"], 1)
        chance = compute_mia_metrics(np.ones(20), np.ones(20))
        self.assertEqual(chance["auc"], 0.5)
        self.assertEqual(chance["advantage"], 0)
        y = np.repeat(np.arange(6), 30)
        pairs = sample_balanced_indices(y, y, 6, 42, 20)
        self.assertEqual(len(pairs), 6)
        for tr, te in pairs.values():
            self.assertEqual(len(tr), len(te))
            self.assertEqual(len(tr), 20)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        a, b = root / "a.parquet", root / "b.parquet"
        synthetic_file(a, 600, 1)
        synthetic_file(b, 420, 2)
        cls.args = SimpleNamespace(home_a=str(a), home_b=str(b), results_dir=str(root / "runs"),
                                   feature_set="baseline_16", model_size="small", n_shadows=1, shadow_ratio=0.5)
        cls.cfg = ExperimentConfig(num_rounds=2, local_epochs=1, batch_size=128, fisher_n_samples=20)
        cls.records = {}
        for seed in (42, 123):
            for name in CONFIG_NAMES:
                cls.records[seed, name] = run_single(cls.args, cls.cfg, name, seed)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_accounting(self):
        for name in ("dp_sgd", "feddpa"):
            result = self.records[42, name]
            for h in ("home_a", "home_b"):
                self.assertLessEqual(result["rounds"][-1][f"{h}_epsilon"], 8.0001)
                self.assertGreater(result["rounds"][-1][f"{h}_epsilon"], 7.99)
        dp = self.records[42, "dp_sgd"]
        for h in ("home_a", "home_b"):
            self.assertEqual(dp["rounds"][-1][f"{h}_accountant_steps"], dp["sampling"][h]["total_steps"])
            self.assertGreater(dp["rounds"][-1][f"{h}_epsilon"], dp["rounds"][0][f"{h}_epsilon"])

    def test_checkpoint_identity_and_resume(self):
        data, spec, path = prepare_run(self.args, self.cfg, "baseline_fl", 42)
        result, model = load_target(path, spec, "cpu")
        self.assertEqual(file_hash(path.with_suffix(".pt")), result["checkpoint_sha256"])
        with patch("fl_pipeline.run_experiment.train_model", side_effect=AssertionError("must reuse")):
            self.assertEqual(run_single(self.args, self.cfg, "baseline_fl", 42), result)
        with self.assertRaises(ValueError):
            load_target(path, dict(spec, seed=999), "cpu")
        model.eval()
        with torch.no_grad():
            first = model(torch.tensor(data["home_a"]["X_test"][:2]))
            _, other = load_target(path, spec, "cpu")
            torch.testing.assert_close(first, other(torch.tensor(data["home_a"]["X_test"][:2])))

    def test_repeatability(self):
        args = copy.copy(self.args)
        args.results_dir = str(Path(self.tmp.name) / "repeat")
        for name in ("baseline_fl", "fs_mild", "dp_sgd", "feddpa"):
            again = run_single(args, self.cfg, name, 42)
            expected = self.records[42, name]
            for a, b in zip(again["rounds"], expected["rounds"]):
                for home in ("home_a", "home_b"):
                    self.assertEqual(a[home], b[home])

    def test_mismatched_and_incomplete_provenance(self):
        rows = load_results("baseline_16", results_dir=self.args.results_dir, seeds=[42, 123])
        rows["fs_mild"][42]["spec"]["split_hashes"]["home_a"]["train"] = "different"
        with self.assertRaises(ValueError):
            summarize(rows, [42, 123])

    def test_missing_mia_target(self):
        args = copy.copy(self.args)
        args.results_dir = str(Path(self.tmp.name) / "missing")
        for probe in (loss_probe, shadow_probe):
            with self.assertRaises(FileNotFoundError):
                probe(args, self.cfg, "baseline_fl", 42)

    def test_temporal_and_feature_partitions(self):
        a = self.args
        fs = prepare_federated_data(a.home_a, a.home_b, get_feature_cols("fs_mild", "baseline_16"), CORE_CLASSES, 42)
        full = prepare_federated_data(a.home_a, a.home_b, BASELINE_16, CORE_CLASSES, 42)
        self.assertEqual(full["split_hashes"], fs["split_hashes"])
        temporal = prepare_federated_data(a.home_a, a.home_b, BASELINE_16, CORE_CLASSES, 42, split="temporal")
        again = prepare_federated_data(a.home_a, a.home_b, BASELINE_16, CORE_CLASSES, 123, split="temporal")
        self.assertEqual(temporal["split_hashes"], again["split_hashes"])
        self.assertNotEqual(temporal["split_hashes"], full["split_hashes"])
        for h in ("home_a", "home_b"):
            x, y, xo, yo = make_shadow_data(full, h, 0.5, 22)
            self.assertEqual(len(x) + len(xo), full[h]["n_train"])
            self.assertEqual(set(map(tuple, np.concatenate([x, xo]))), set(map(tuple, full[h]["X_train"])))

    def test_attacks_and_extraction(self):
        for seed in (42, 123):
            for name in ("baseline_fl", "fs_mild", "dp_sgd"):
                for probe in (loss_probe, shadow_probe):
                    value = probe(self.args, self.cfg, name, seed)
                    self.assertTrue(0 <= value["equal_avg"]["auc"] <= 1)
        with patch("fl_pipeline.run_shadow_mia.train_model", side_effect=AssertionError("must reuse")):
            shadow_probe(self.args, self.cfg, "baseline_fl", 42)
        result = extract(self.args.results_dir, "baseline_16", "small", [42, 123])
        self.assertEqual(set(result["attacks"]), {"mia", "shadow_mia"})
        self.assertEqual(set(result["main_table"]), set(CONFIG_NAMES))
        atomic_json(Path(self.tmp.name) / "numbers.json", result)
        rows = load_results("baseline_16", results_dir=self.args.results_dir, seeds=[42, 123])
        self.assertEqual(summarize(rows, [42, 123])["std_convention"], "sample (ddof=1)")
        with self.assertRaises(ValueError):
            summarize(rows, [42, 123, 999])

    def test_cli_matrix(self):
        repo = Path(__file__).resolve().parents[1]
        root = Path(self.tmp.name) / "cli"
        cmd = [sys.executable, "scripts/run_suite.py", "--results-dir", str(root),
               "--home-a", self.args.home_a, "--home-b", self.args.home_b,
               "--seeds", "42", "--rounds", "1", "--local-epochs", "1", "--n-shadows", "1"]
        completed = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-8000:])
        for size, suffix in (("small", ""), ("medium", "_baseline_16_medium")):
            result = json.loads((root / f"paper_numbers{suffix}.json").read_text())
            self.assertEqual(set(result["attacks"]), {"mia", "shadow_mia"})
            self.assertIn("temporal", result["evidence"])
            self.assertEqual("equal_weight" in result["evidence"], size == "small")
            folder = "baseline_16" + ("_medium" if size == "medium" else "")
            self.assertTrue((root / "figures" / folder / "convergence.pdf").exists())
        completed = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-8000:])
        self.assertNotIn("Training baseline_fl,", completed.stdout)

    def test_benchmark_cli(self):
        root = Path(self.tmp.name) / "benchmark"
        cmd = [sys.executable, "scripts/benchmark.py", "--results-dir", str(root),
               "--home-a", self.args.home_a, "--home-b", self.args.home_b,
               "--seeds", "42", "--rounds", "1", "--local-epochs", "1"]
        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-8000:])
        report = json.loads((root / "benchmark.json").read_text())
        self.assertTrue(report["projection_complete"])
        self.assertEqual(len(report["jobs"]), 8)
        self.assertGreater(report["projected_training_hours"], 0)


if __name__ == "__main__":
    unittest.main()
