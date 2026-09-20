"""Integrity checks for portable measurement artifacts."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_artifact import sha256, verify


class ArtifactVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "fl_pipeline").mkdir()
        (self.repo / "data").mkdir()
        self.root = self.repo / "artifact"
        self.root.mkdir()
        self.payload = self.root / "results"
        self.payload.mkdir()
        source = hashlib.sha256(b"{}").hexdigest()
        data = {}
        for home, filename in (("home_a", "home_A.parquet"), ("home_b", "home_B.parquet")):
            path = self.repo / "data" / filename
            path.write_bytes(home.encode())
            data[home] = sha256(path)
        protocol = {"source_sha256": source, "data_sha256": data}
        for name, spec in (("target", protocol), ("shadow", {"parent": protocol})):
            checkpoint = self.payload / f"{name}.pt"
            checkpoint.write_bytes(name.encode())
            checkpoint.with_suffix(".json").write_text(json.dumps(
                {"spec": spec, "checkpoint_sha256": sha256(checkpoint)}))
        self.manifest = {"source_sha256": source, "data_sha256": data,
                         "checkpoint_count": 2, "content_roots": ["results"], "files": [
                             {"path": p.relative_to(self.root).as_posix(),
                              "bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(self.payload.iterdir())]}
        (self.root / "repro_manifest.json").write_text(json.dumps(self.manifest))

    def test_targets_and_shadow_parent_identities(self):
        self.assertEqual(verify(self.root, self.repo), (4, 2))

    def test_modified_checkpoint(self):
        (self.payload / "target.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify(self.root, self.repo)

    def test_unlisted_file(self):
        (self.payload / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            verify(self.root, self.repo)

    def test_changed_source(self):
        (self.repo / "fl_pipeline" / "model.py").write_text("# Model\n")
        with self.assertRaisesRegex(ValueError, "source differs"):
            verify(self.root, self.repo)

    def test_invalid_manifest_path(self):
        self.manifest["files"][0]["path"] = "../outside"
        (self.root / "repro_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Invalid artifact path"):
            verify(self.root, self.repo)
