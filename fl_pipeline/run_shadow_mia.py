#!/usr/bin/env python3
"""Shadow membership probe using resamples of the target training pool.

This is a controlled attack diagnostic, not an independent auxiliary-data attack.
Target-test records are never used to fit shadows or attack classifiers.
"""

import argparse
import copy
import time

import numpy as np

from .artifacts import (atomic_json, digest, ensure_run_root, load_cached_json,
                        load_target, save_target)
from .cli import add_arguments, config_from_args
from .data import _make_loader
from .mia import compute_attack_features, train_attack_classifiers, evaluate_shadow_mia, N_CAP
from .run_experiment import prepare_run, train_model
from .run_mia import HOME_KEYS, PRIMARY, aggregate_results, attack_directory, equal_average
from .runtime import configure


def make_shadow_data(data, home_key, shadow_ratio, shadow_seed):
    rng = np.random.RandomState(shadow_seed)
    d = data[home_key]
    n = len(d["y_train"])
    cut = int(n * shadow_ratio)
    if not 0 < cut < n:
        raise ValueError("Both shadow partitions must be nonempty")
    indices = rng.permutation(n)
    train, out = indices[:cut], indices[cut:]
    return d["X_train"][train], d["y_train"][train], d["X_train"][out], d["y_train"][out]


def run_one(args, config, name, seed):
    started = time.perf_counter()
    data, spec, target = prepare_run(args, config, name, seed)
    training, target_model = load_target(target, spec, config.device)
    identity = {"target_spec": spec, "checkpoint_sha256": training["checkpoint_sha256"],
                "attack": "shadow_model", "pool": "target_training_partition",
                "n_cap": N_CAP, "n_shadows": args.n_shadows, "shadow_ratio": args.shadow_ratio}
    directory = attack_directory(args, "shadow_mia")
    path = directory / f"seed_{seed}_{name}.json"
    cached = load_cached_json(path, identity)
    if cached is not None:
        print(f"Reusing {path}")
        return cached
    device = configure(config, seed)
    features = {h: {"x": [], "member": [], "class": []} for h in HOME_KEYS}
    shadow_elapsed = 0.0
    for i in range(args.n_shadows):
        shadow_seed = seed * 1000 + i
        shadow = copy.copy(data)
        splits = {}
        for hk, offset in zip(HOME_KEYS, (0, 7919)):
            xtr, ytr, xout, yout = make_shadow_data(data, hk, args.shadow_ratio, shadow_seed + offset)
            splits[hk] = (xtr, ytr, xout, yout)
            shadow[hk] = {"X_train": xtr, "y_train": ytr, "X_test": xout, "y_test": yout,
                          "n_train": len(ytr), "n_test": len(yout),
                          "train_loader": _make_loader(xtr, ytr, config.batch_size, True, shadow_seed),
                          "test_loader": _make_loader(xout, yout, config.batch_size, False, shadow_seed)}
        na, nb = shadow["home_a"]["n_train"], shadow["home_b"]["n_train"]
        shadow["weights"] = {"home_a": na / (na + nb), "home_b": nb / (na + nb)}
        shadow_spec = {"parent": spec, "shadow_seed": shadow_seed, "shadow_ratio": args.shadow_ratio,
                       "pool": "target_training_partition", "weights": shadow["weights"]}
        job = directory / "models" / f"seed_{seed}_{name}_{digest(shadow_spec)[:12]}_{i}.json"
        if job.exists():
            shadow_results, model = load_target(job, shadow_spec, config.device)
        else:
            t0 = time.perf_counter()
            shadow_results, model = train_model(shadow, config, shadow_seed, name)
            shadow_results["total_time_s"] = time.perf_counter() - t0
            shadow_results = save_target(job, shadow_results, model, shadow_spec)
        shadow_elapsed += shadow_results["total_time_s"]
        for hk in HOME_KEYS:
            xtr, ytr, xout, yout = splits[hk]
            for x, y, member in ((xtr, ytr, 1), (xout, yout, 0)):
                features[hk]["x"].append(compute_attack_features(model, x, y, device))
                features[hk]["member"].append(np.full(len(y), member))
                features[hk]["class"].append(y)
        print(f"Shadow {i + 1}/{args.n_shadows} ready for {name}, seed {seed}", flush=True)
    result = {"attack": "shadow_model", "config": name, "seed": seed,
              "n_shadows": args.n_shadows, "shadow_ratio": args.shadow_ratio, "identity": identity}
    for hk in HOME_KEYS:
        f = features[hk]
        classifiers = train_attack_classifiers(np.concatenate(f["x"]), np.concatenate(f["member"]),
                                               np.concatenate(f["class"]), data["num_classes"])
        d = data[hk]
        result[hk] = evaluate_shadow_mia(
            classifiers, compute_attack_features(target_model, d["X_train"], d["y_train"], device),
            compute_attack_features(target_model, d["X_test"], d["y_test"], device),
            d["y_train"], d["y_test"], data["class_names"], n_cap=N_CAP, seed=seed)
        if result[hk]["macro"]["n_classes_evaluated"] != data["num_classes"]:
            raise ValueError("Shadow attack did not evaluate every retained class")
    result.update(equal_avg=equal_average(result), elapsed_s=time.perf_counter() - started,
                  shadow_training_elapsed_s=shadow_elapsed)
    atomic_json(path, result)
    print(f"Saved {path}: AUC={result['equal_avg']['auc']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--configs", nargs="+", choices=PRIMARY, default=list(PRIMARY))
    parser.add_argument("--n-shadows", type=int, default=4)
    parser.add_argument("--shadow-ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.n_shadows < 1 or not 0 < args.shadow_ratio < 1:
        parser.error("Require n-shadows >= 1 and 0 < shadow-ratio < 1")
    ensure_run_root(args.results_dir)
    config = config_from_args(args)
    started = time.perf_counter()
    all_results = [{c: run_one(args, config, c, s) for c in args.configs} for s in args.seeds]
    summary = aggregate_results(all_results, args.configs)
    summary.update(attack="shadow_model", n_shadows=args.n_shadows, shadow_ratio=args.shadow_ratio,
                   pool="target_training_partition", invocation_elapsed_s=time.perf_counter() - started,
                   completed_job_elapsed_s=sum(r[c]["elapsed_s"] for r in all_results for c in args.configs))
    atomic_json(attack_directory(args, "shadow_mia") / "shadow_mia_summary.json", summary)


if __name__ == "__main__":
    main()
