#!/usr/bin/env python3
"""Summarize effect sizes, paired intervals and per-seed directions."""

import argparse
import json
from pathlib import Path
import numpy as np

from .artifacts import atomic_json, ensure_run_root
from .config import CONFIG_NAMES, RESULTS_DIR, SEEDS
from .metrics import paired_confidence_interval


def load_results(feature_set, equal_weight=False, results_dir=RESULTS_DIR, seeds=None, split="stratified"):
    base = Path(results_dir) / feature_set
    if split == "temporal":
        base /= "temporal"
    prefix = "equal_weight_seed_" if equal_weight else "seed_"
    out = {}
    for directory in sorted(base.glob(prefix + "*")):
        if not directory.is_dir():
            continue
        seed = int(directory.name[len(prefix):])
        if seeds is not None and seed not in seeds:
            continue
        for name in CONFIG_NAMES:
            path = directory / f"{name}.json"
            if path.exists():
                out.setdefault(name, {})[seed] = json.loads(path.read_text())
    return out


def final_metrics(result):
    z = result["rounds"][-1]
    return {"home_a": z["home_a"]["macro_f1"], "home_b": z["home_b"]["macro_f1"],
            "combined": (z["home_a"]["macro_f1"] + z["home_b"]["macro_f1"]) / 2,
            "worst": min(z["home_a"]["worst_group_f1"], z["home_b"]["worst_group_f1"])}


def stats(values):
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(results, expected_seeds=None):
    if not results:
        raise ValueError("No result files found")
    expected = set(expected_seeds) if expected_seeds is not None else None
    identities = set()
    provenance = set()
    partitions = {}
    for name, runs in results.items():
        if expected is not None and set(runs) != expected:
            raise ValueError(f"Incomplete seed set for {name}: found {sorted(runs)}, expected {sorted(expected)}")
        for seed, run in runs.items():
            spec = run.get("spec")
            provenance.add(bool(spec))
            if spec:
                if spec["seed"] != seed or spec["config_name"] != name:
                    raise ValueError(f"Run identity disagrees with its path: {name}, seed {seed}")
                paired_partition = (spec["split_hashes"], spec["weights"], spec["classes"])
                if seed in partitions and partitions[seed] != paired_partition:
                    raise ValueError(f"Configurations use different partitions or weights for seed {seed}")
                partitions[seed] = paired_partition
                cfg = json.dumps(spec["config"], sort_keys=True)
                identities.add((spec["source_sha256"], json.dumps(spec["data_sha256"], sort_keys=True),
                                json.dumps(spec["environment"], sort_keys=True), cfg,
                                spec["split"], spec["preprocessing"]["name"]))
                if len(run["rounds"]) != spec["config"]["num_rounds"]:
                    raise ValueError(f"Incomplete rounds: {name}, seed {seed}")
    if len(provenance) > 1:
        raise ValueError("Cannot combine runs with and without provenance")
    if len(identities) > 1:
        raise ValueError("Results mix source, data, execution or protocol settings")
    summary = {"per_config": {}, "paired": {}, "std_convention": "sample (ddof=1)"}
    for name, runs in results.items():
        seeds = sorted(runs)
        rows = [dict(seed=s, **final_metrics(runs[s])) for s in seeds]
        entry = {k: stats([r[k] for r in rows]) for k in ("home_a", "home_b", "combined", "worst")}
        entry.update(seeds=seeds, per_seed=rows)
        summary["per_config"][name] = entry
    if "fs_mild" in results and "dp_sgd" in results:
        if set(results["fs_mild"]) != set(results["dp_sgd"]):
            raise ValueError("Paired configurations must have identical seed sets")
        seeds = sorted(results["fs_mild"])
        fs = [final_metrics(results["fs_mild"][s]) for s in seeds]
        dp = [final_metrics(results["dp_sgd"][s]) for s in seeds]
        summary["paired"]["fs_mild_minus_dp_sgd"] = {
            "seeds": seeds,
            **{k: dict(paired_confidence_interval([r[k] for r in fs], [r[k] for r in dp]),
                       wins=sum(a[k] > b[k] for a, b in zip(fs, dp)))
               for k in ("combined", "worst")}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--equal-weight", action="store_true")
    parser.add_argument("--split", choices=["stratified", "temporal"], default="stratified")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--save-summary", action="store_true")
    args = parser.parse_args()
    result = summarize(load_results(args.feature_set, args.equal_weight, args.results_dir,
                                    args.seeds, args.split), args.seeds)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.save_summary:
        ensure_run_root(args.results_dir)
        base = Path(args.results_dir) / args.feature_set
        if args.split == "temporal":
            base /= "temporal"
        atomic_json(base / ("summary_equal_weight.json" if args.equal_weight else "summary.json"), result)


if __name__ == "__main__":
    main()
