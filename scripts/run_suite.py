#!/usr/bin/env python3
"""Run the residential experiment matrix with checkpoint reuse."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_pipeline.cli import add_arguments
from fl_pipeline.artifacts import atomic_json, ensure_run_root, file_hash
from fl_pipeline.config import CONFIG_NAMES

PRIMARY = ["baseline_fl", "fs_mild", "dp_sgd"]


def commands(args):
    common = ["--feature-set", args.feature_set, "--results-dir", args.results_dir,
              "--seeds", *map(str, args.seeds)]
    training = common + ["--device", args.device, "--num-threads", str(args.num_threads),
                         "--rounds", str(args.rounds), "--local-epochs", str(args.local_epochs),
                         "--batch-size", str(args.batch_size), "--home-a", args.home_a, "--home-b", args.home_b]
    jobs = []
    for size in ("small", "medium"):
        if args.capacity != "both" and args.capacity != size:
            continue
        options = training + ["--model-size", size]
        feature = args.feature_set + ("_medium" if size == "medium" else "")
        analysis = ["--feature-set", feature, "--results-dir", args.results_dir,
                    "--seeds", *map(str, args.seeds), "--save-summary"]
        jobs.append(("training", ["-m", "fl_pipeline.run_experiment", *options, "--configs",
                                   *(CONFIG_NAMES if size == "small" else PRIMARY)]))
        if size == "small":
            jobs.append(("training", ["-m", "fl_pipeline.run_experiment", *options,
                                       "--equal-weight", "--configs", *PRIMARY]))
        jobs.append(("temporal", ["scripts/temporal_split_check.py", *options]))
        jobs.append(("mia", ["-m", "fl_pipeline.run_mia", *options, "--configs", *PRIMARY]))
        jobs.append(("shadow", ["-m", "fl_pipeline.run_shadow_mia", *options, "--configs", *PRIMARY,
                                "--n-shadows", str(args.n_shadows)]))
        jobs.append(("analysis", ["-m", "fl_pipeline.analyze", *analysis]))
        if size == "small":
            jobs.append(("analysis", ["-m", "fl_pipeline.analyze", *analysis, "--equal-weight"]))
        jobs.append(("analysis", ["-m", "fl_pipeline.analyze", *analysis, "--split", "temporal"]))
        jobs.append(("analysis", ["scripts/extract_paper_numbers.py", *common, "--model-size", size]))
        for script in ("fig1_class_distribution.py", "fig2_paired_deltas.py", "fig3_convergence.py"):
            jobs.append(("figures", [f"scripts/{script}", *common, "--model-size", size]))
    return [(stage, [sys.executable, *cmd]) for stage, cmd in jobs
            if args.stage == "all" or stage == args.stage]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--capacity", choices=["both", "small", "medium"], default="both")
    parser.add_argument("--stage", choices=["all", "training", "temporal", "mia", "shadow", "analysis", "figures"], default="all")
    parser.add_argument("--n-shadows", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.model_size != "small":
        parser.error("Use --capacity medium or --capacity both for the suite")
    if args.n_shadows < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least one shadow and unique seeds")
    repo = Path(__file__).resolve().parents[1]
    args.results_dir = str(Path(args.results_dir).resolve())
    jobs = commands(args)
    n = len(args.seeds)
    targets = n * ({"small": 11, "medium": 6, "both": 17}[args.capacity])
    shadows = n * 3 * args.n_shadows * (2 if args.capacity == "both" else 1)
    print(f"Complete selected-capacity matrix: {targets} target trainings + {shadows} shadows.")
    print("Both MIA probes reuse main-split target checkpoints. Stage:", args.stage)
    if args.dry_run:
        for stage, cmd in jobs:
            print(f"[{stage}] {shlex.join(cmd)}")
        return
    root = ensure_run_root(args.results_dir)
    # Dataset checksums are printed and included in every target's specification.
    # Custom input files are permitted and remain identifiable by their hashes.
    for path in (args.home_a, args.home_b):
        print(f"Dataset SHA-256: {file_hash(path)}  {path}", flush=True)
    started = time.perf_counter()
    report = {"stage": args.stage, "capacity": args.capacity, "commands": []}
    report_path = root / f"suite_{time.time_ns()}.json"
    for stage, cmd in jobs:
        t0 = time.perf_counter()
        print(f"[{stage}] {shlex.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=repo)
        report["commands"].append({"stage": stage, "argv": cmd,
                                   "elapsed_s": time.perf_counter() - t0, "returncode": result.returncode})
        report["invocation_elapsed_s"] = time.perf_counter() - started
        atomic_json(report_path, report)
        if result.returncode:
            raise SystemExit(result.returncode)
    print(f"Suite elapsed time: {report['invocation_elapsed_s'] / 3600:.2f} h; log: {report_path}")


if __name__ == "__main__":
    main()
