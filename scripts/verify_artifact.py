#!/usr/bin/env python3
"""Verify a measurement snapshot without loading models or training."""

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, repo=REPO):
    root = Path(root).resolve()
    manifest = json.loads((root / "repro_manifest.json").read_text())
    content_roots = manifest["content_roots"]
    require(content_roots and all(name in ("results", "figures") for name in content_roots),
            "Invalid artifact content roots")
    listed = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Invalid artifact path: {relative}")
        require(relative.parts and relative.parts[0] in content_roots,
                f"File outside artifact content roots: {relative}")
        path = root / relative
        require(path.resolve().is_relative_to(root) and path.is_file(),
                f"Missing or external artifact file: {relative}")
        require(entry["path"] not in listed, f"Duplicate manifest entry: {relative}")
        listed.add(entry["path"])
        require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                f"Artifact checksum mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for name in content_roots
              for p in (root / name).rglob("*") if p.is_file()}
    require(actual == listed, "Snapshot contains unlisted or missing files")

    sources = {p.name: sha256(p) for p in sorted((repo / "fl_pipeline").glob("*.py"))}
    source_hash = hashlib.sha256(json.dumps(
        sources, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(source_hash == manifest["source_sha256"], "Pipeline source differs from snapshot")
    for home, name in (("home_a", "home_A.parquet"), ("home_b", "home_B.parquet")):
        path = repo / "data" / name
        require(path.is_file() and sha256(path) == manifest["data_sha256"][home],
                f"Dataset differs or is a Git LFS pointer: {path}; run git lfs pull")

    checkpoints = 0
    for relative in listed:
        if not relative.endswith(".pt"):
            continue
        path = root / relative
        result = json.loads(path.with_suffix(".json").read_text())
        require(sha256(path) == result["checkpoint_sha256"],
                f"Checkpoint/result mismatch: {relative}")
        spec = result["spec"]
        protocol = spec["parent"] if "parent" in spec else spec
        require(protocol["source_sha256"] == manifest["source_sha256"] and
                protocol["data_sha256"] == manifest["data_sha256"],
                f"Model protocol identity mismatch: {relative}")
        checkpoints += 1
    require(checkpoints == manifest["checkpoint_count"], "Checkpoint count mismatch")
    runtime = root / "results/runtime_isolated"
    if runtime.exists() or "runtime_measurement" in manifest:
        try:
            from .extract_runtime_numbers import summarize
        except ImportError:
            from extract_runtime_numbers import summarize
        require("runtime_measurement" in manifest, "Missing isolated timing metadata")
        summary = summarize(runtime, root / "results")
        metadata = manifest["runtime_measurement"]
        require(metadata == {
            "record": "results/runtime_isolated/provenance/manifest.json",
            "exit_record": "results/runtime_isolated/provenance/run.exit",
            "summary": "results/runtime_isolated/runtime_summary.json",
            "checkpoint_count": 5, "seed": summary["seed"],
            "timing_context": summary["timing_context"],
        }, "Isolated timing metadata mismatch")
        require(json.loads((runtime / "runtime_summary.json").read_text()) == summary,
                "Isolated timing summary differs from its inputs")
        require(manifest["paper_inputs"].get("runtime") == metadata["summary"],
                "Paper runtime input mismatch")
    return len(listed), checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=REPO,
                        help="Repository snapshot containing repro_manifest.json")
    args = parser.parse_args()
    files, checkpoints = verify(args.artifact, args.artifact)
    print(f"Verified {files} files and {checkpoints} checkpoints; source and datasets match.")


if __name__ == "__main__":
    main()
