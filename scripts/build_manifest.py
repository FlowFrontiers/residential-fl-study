#!/usr/bin/env python3
"""Build the repository manifest from recorded measurements and file checksums."""

import argparse
import json
from pathlib import Path

from verify_artifact import sha256, verify
from extract_runtime_numbers import summarize


REPO = Path(__file__).resolve().parents[1]


def build(repo):
    execution = json.loads((repo / "results/provenance/execution.json").read_text())
    if execution["status"] != "complete":
        raise ValueError("Cannot package an incomplete execution")
    files = []
    for name in ("results", "figures"):
        for path in sorted((repo / name).rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(repo).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": 2,
        "description": "Residential federated traffic-classification measurement artifact",
        "content_roots": ["results", "figures"],
        "source_sha256": execution["source_sha256"],
        "data_sha256": execution["data_sha256"],
        "checkpoint_count": sum(entry["path"].endswith(".pt") for entry in files),
        "seeds": execution["arguments"]["seeds"],
        "execution": {
            "completed_jobs": execution["completed_jobs"],
            "job_counts": execution["job_counts"],
            "shadow_trainings": execution["shadow_trainings"],
            "elapsed_s": execution["elapsed_s"],
            "workers": execution["workers"],
            "threads_per_worker": execution["threads_per_worker"],
            "timing_context": execution["timing_context"],
            "record": "results/provenance/execution.json",
        },
        "data_files": {
            f"data/{name}": {"bytes": (repo / "data" / name).stat().st_size,
                             "sha256": execution["data_sha256"][home]}
            for home, name in (("home_a", "home_A.parquet"), ("home_b", "home_B.parquet"))
        },
        "paper_inputs": {
            "small": "results/paper_numbers.json",
            "medium": "results/paper_numbers_baseline_16_medium.json",
            "figures": "figures/",
        },
        "files": files,
    }
    runtime = repo / "results/runtime_isolated"
    if runtime.exists():
        summary = summarize(runtime, repo / "results")
        if json.loads((runtime / "runtime_summary.json").read_text()) != summary:
            raise ValueError("Isolated timing summary differs from its inputs")
        manifest["runtime_measurement"] = {
            "record": "results/runtime_isolated/provenance/manifest.json",
            "exit_record": "results/runtime_isolated/provenance/run.exit",
            "summary": "results/runtime_isolated/runtime_summary.json",
            "checkpoint_count": 5,
            "seed": summary["seed"],
            "timing_context": summary["timing_context"],
        }
        manifest["paper_inputs"]["runtime"] = manifest["runtime_measurement"]["summary"]
    (repo / "repro_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return verify(repo, repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    files, checkpoints = build(args.repo.resolve())
    print(f"Manifest built and verified: {files} files, {checkpoints} checkpoints.")


if __name__ == "__main__":
    main()
