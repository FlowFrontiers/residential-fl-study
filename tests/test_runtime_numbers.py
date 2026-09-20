"""Checks for isolated timing extraction without training or model loading."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.extract_runtime_numbers import CONFIGS, sha256, summarize


class RuntimeNumbersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runtime"
        self.reference = Path(self.temp.name) / "reference"
        self.base = self.root / "baseline_16/seed_42"
        self.base.mkdir(parents=True)
        reference_base = self.reference / "baseline_16/seed_42"
        reference_base.mkdir(parents=True)
        self.provenance = self.root / "provenance"
        self.provenance.mkdir()
        (self.provenance / "run.exit").write_text("0\n")
        (self.provenance / "launch.sh").write_text("measurement launcher")
        manifest = {"seed": 42, "rounds": 20, "local_epochs": 5,
                    "batch_size": 256, "hidden_dims": [16, 16], "device": "cpu",
                    "num_threads": 1, "split": "stratified",
                    "aggregation": "size_proportional", "configs": list(CONFIGS),
                    "source_sha256": "source", "data_sha256": {"home_a": "a", "home_b": "b"},
                    "cpu_model": "test CPU", "timing_context": "sequential",
                    "launcher_sha256": sha256(self.provenance / "launch.sh")}
        (self.provenance / "manifest.json").write_text(json.dumps(manifest))
        for i, name in enumerate(CONFIGS, 1):
            checkpoint = self.base / f"{name}.pt"
            checkpoint.write_bytes(name.encode())
            spec = {"source_sha256": "source", "data_sha256": manifest["data_sha256"],
                    "split": "stratified", "config": {
                        "num_rounds": 20, "equal_weight": False,
                        **{k: manifest[k] for k in ("local_epochs", "batch_size",
                                                   "hidden_dims", "device", "num_threads")}}}
            run = {"seed": 42, "config_name": name, "spec": spec,
                   "checkpoint_sha256": sha256(checkpoint),
                   "rounds": [{"round": r, "time_s": float(i * 10)} for r in range(1, 21)]}
            (self.base / f"{name}.json").write_text(json.dumps(run))
            (reference_base / f"{name}.json").write_text(json.dumps({"spec": spec}))

    def mutate_run(self, change):
        path = self.base / "dp_sgd.json"
        run = json.loads(path.read_text())
        change(run)
        path.write_text(json.dumps(run))

    def test_medians_ratios_and_read_only_extraction(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = summarize(self.root, self.reference)
        self.assertEqual(result["runtime_table"]["dp_sgd"]["median_round_s"], 40)
        self.assertAlmostEqual(result["runtime_table"]["dp_sgd"]["estimated_run_min"], 40 / 3)
        self.assertEqual(result["dp_sgd_over_fs_mild"], 2)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_incomplete_run(self):
        self.mutate_run(lambda r: r["rounds"].pop())
        with self.assertRaisesRegex(ValueError, "Incomplete timing rounds"):
            summarize(self.root, self.reference)

    def test_invalid_duration(self):
        self.mutate_run(lambda r: r["rounds"][0].update(time_s=float("nan")))
        with self.assertRaisesRegex(ValueError, "Invalid round duration"):
            summarize(self.root, self.reference)

    def test_protocol_mismatch(self):
        self.mutate_run(lambda r: r["spec"]["config"].update(num_threads=2))
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            summarize(self.root, self.reference)

    def test_nonzero_exit(self):
        (self.provenance / "run.exit").write_text("124")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            summarize(self.root, self.reference)

    def test_checkpoint_mismatch(self):
        (self.base / "dp_sgd.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            summarize(self.root, self.reference)
