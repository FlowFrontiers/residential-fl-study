#!/usr/bin/env python3
"""Train or reuse a complete federated target and its checkpoint."""

import argparse
from pathlib import Path

from .config import CONFIG_NAMES, CORE_CLASSES, get_feature_cols
from .cli import add_arguments, config_from_args
from .data import prepare_federated_data
from .artifacts import ensure_run_root, make_spec, load_target, save_target, target_path
from .runtime import configure, timestamp
from .train import run_standard_fl, run_dp_sgd_fl, run_feddpa_fl


def prepare_run(args, config, config_name, seed, split="stratified"):
    device = configure(config, seed)
    data = prepare_federated_data(
        args.home_a, args.home_b, get_feature_cols(config_name, args.feature_set), CORE_CLASSES,
        seed, config.test_ratio, config.batch_size, config.equal_weight,
        split=split, pin_memory=device.type == "cuda")
    spec = make_spec(data, config, seed, config_name, (args.home_a, args.home_b))
    path = target_path(args.results_dir, args.feature_set, args.model_size, seed,
                       config_name, config.equal_weight, split)
    return data, spec, path


def train_model(data, config, seed, config_name):
    if config_name == "dp_sgd":
        return run_dp_sgd_fl(data, config, seed, return_model=True)
    if config_name == "feddpa":
        return run_feddpa_fl(data, config, seed, return_model=True)
    return run_standard_fl(data, config, seed, config_name, return_model=True)


def run_single(args, config, config_name, seed, split="stratified"):
    ensure_run_root(args.results_dir)
    data, spec, path = prepare_run(args, config, config_name, seed, split)
    if path.exists():
        results, _ = load_target(path, spec, config.device)
        print(f"Reusing {path}", flush=True)
        return results
    device = configure(config, seed)
    start = timestamp(device)
    print(f"Training {config_name}, {args.model_size}, seed={seed}, split={split}", flush=True)
    results, model = train_model(data, config, seed, config_name)
    results.update(total_time_s=timestamp(device) - start, feature_cols=data["feature_cols"],
                   weights=data["weights"], audit=data["audit"], test_support=data["test_support"])
    results = save_target(path, results, model, spec)
    print(f"Saved {path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--configs", nargs="+", choices=CONFIG_NAMES, default=CONFIG_NAMES)
    parser.add_argument("--equal-weight", action="store_true")
    parser.add_argument("--split", choices=["stratified", "temporal"], default="stratified")
    args = parser.parse_args()
    ensure_run_root(args.results_dir)
    config = config_from_args(args)
    for seed in args.seeds:
        for name in args.configs:
            run_single(args, config, name, seed, args.split)


if __name__ == "__main__":
    main()
