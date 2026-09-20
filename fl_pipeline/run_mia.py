#!/usr/bin/env python3
"""Loss-based membership probe using saved main-split target models."""

import argparse
from pathlib import Path
import time

import numpy as np

from .artifacts import atomic_json, ensure_run_root, load_cached_json, load_target
from .cli import add_arguments, config_from_args
from .mia import sample_balanced_indices, run_mia_attack, N_CAP
from .metrics import paired_confidence_interval
from .run_experiment import prepare_run
from .runtime import configure

HOME_KEYS = ("home_a", "home_b")
PRIMARY = ("baseline_fl", "fs_mild", "dp_sgd")
METRICS = ("auc", "tpr_at_1fpr", "tpr_at_5fpr", "advantage")


def attack_directory(args, kind):
    folder = args.feature_set if args.model_size == "small" else f"{args.feature_set}_{args.model_size}"
    return Path(args.results_dir) / folder / kind


def equal_average(result):
    return {k: float(np.mean([result[h]["macro"][k] for h in HOME_KEYS])) for k in METRICS}


def aggregate_results(all_seed_results, configs):
    summary = {"n_cap": N_CAP, "per_config": {}, "paired_ci": {},
               "seeds": [r[configs[0]]["seed"] for r in all_seed_results]}
    for name in configs:
        out = {}
        for scope in (*HOME_KEYS, "equal_avg"):
            stats = {}
            for key in METRICS:
                vals = [r[name][scope][key] if scope == "equal_avg" else r[name][scope]["macro"][key]
                        for r in all_seed_results]
                stats[f"{key}_mean"] = float(np.mean(vals))
                stats[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            out[scope] = stats
        summary["per_config"][name] = out
    if "fs_mild" in configs:
        for base in ("baseline_fl", "dp_sgd"):
            if base not in configs:
                continue
            for metric, short in (("auc", "auc"), ("tpr_at_1fpr", "tpr1"), ("tpr_at_5fpr", "tpr5")):
                fs = [r["fs_mild"]["equal_avg"][metric] for r in all_seed_results]
                other = [r[base]["equal_avg"][metric] for r in all_seed_results]
                suffix = "baseline" if base == "baseline_fl" else "dp_sgd"
                summary["paired_ci"][f"{short}_fs_mild_minus_{suffix}"] = paired_confidence_interval(fs, other)
    return summary


def run_one(args, config, name, seed):
    started = time.perf_counter()
    data, spec, target = prepare_run(args, config, name, seed)
    training, model = load_target(target, spec, config.device)
    identity = {"target_spec": spec, "checkpoint_sha256": training["checkpoint_sha256"],
                "attack": "loss", "n_cap": N_CAP}
    path = attack_directory(args, "mia") / f"seed_{seed}_{name}.json"
    cached = load_cached_json(path, identity)
    if cached is not None:
        print(f"Reusing {path}")
        return cached
    device = configure(config, seed)
    result = {"config": name, "seed": seed, "identity": identity}
    for home in HOME_KEYS:
        d = data[home]
        indices = sample_balanced_indices(d["y_train"], d["y_test"], data["num_classes"], seed)
        if len(indices) != data["num_classes"]:
            raise ValueError("Loss attack requires at least ten members and nonmembers per retained class")
        result[home] = run_mia_attack(model, d["X_train"], d["y_train"], d["X_test"],
                                    d["y_test"], indices, data["class_names"], device)
    result["equal_avg"] = equal_average(result)
    result["elapsed_s"] = time.perf_counter() - started
    atomic_json(path, result)
    print(f"Saved {path}: AUC={result['equal_avg']['auc']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--configs", nargs="+", choices=PRIMARY, default=list(PRIMARY))
    args = parser.parse_args()
    ensure_run_root(args.results_dir)
    config = config_from_args(args)
    started = time.perf_counter()
    all_results = [{c: run_one(args, config, c, s) for c in args.configs} for s in args.seeds]
    summary = aggregate_results(all_results, args.configs)
    summary.update(attack="loss", invocation_elapsed_s=time.perf_counter() - started,
                   completed_job_elapsed_s=sum(r[c]["elapsed_s"] for r in all_results for c in args.configs))
    atomic_json(attack_directory(args, "mia") / "mia_summary.json", summary)


if __name__ == "__main__":
    main()
