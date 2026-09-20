#!/usr/bin/env python3
"""Measure full-data and half-training-pool jobs for scheduling estimates."""

import argparse
import copy
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_pipeline.artifacts import atomic_json, ensure_run_root
from fl_pipeline.cli import add_arguments, config_from_args
from fl_pipeline.config import CONFIG_NAMES
from fl_pipeline.data import _make_loader
from fl_pipeline.run_experiment import prepare_run, run_single, train_model
from fl_pipeline.run_shadow_mia import make_shadow_data
from fl_pipeline.runtime import configure, timestamp

PRIMARY = ["baseline_fl", "fs_mild", "dp_sgd"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(rounds=2, local_epochs=1, seeds=[42], results_dir="runs/benchmark")
    args = parser.parse_args()
    if len(args.seeds) != 1:
        parser.error("The benchmark uses exactly one seed")
    root = ensure_run_root(args.results_dir)
    if any(root.glob("suite_*.json")):
        parser.error("Use a dedicated benchmark directory")
    seed = args.seeds[0]
    rows, estimate = {}, 0.0
    start = time.perf_counter()
    for size in ("small", "medium"):
        args.model_size = size
        config = config_from_args(args)
        scale = (20 / config.num_rounds) * (5 / config.local_epochs)
        for name in (CONFIG_NAMES if size == "small" else PRIMARY):
            target = run_single(args, config, name, seed)
            samples = [r["time_s"] for r in target["rounds"]]
            key = f"{size}/{name}"
            row = {"target_round_s": samples, "target_elapsed_s": target["total_time_s"],
                   "target_spec": target["spec"]}
            multiplicity = (3 if size == "small" else 2) if name in PRIMARY else 1
            estimate += 5 * multiplicity * target["total_time_s"] * scale
            if name in PRIMARY:
                data, _, _ = prepare_run(args, config, name, seed)
                shadow = copy.copy(data)
                for hk, offset in (("home_a", 0), ("home_b", 7919)):
                    x, y, xo, yo = make_shadow_data(data, hk, 0.5, seed * 1000 + offset)
                    shadow[hk] = {"X_train": x, "y_train": y, "X_test": xo, "y_test": yo,
                                  "n_train": len(y), "n_test": len(yo),
                                  "train_loader": _make_loader(x, y, args.batch_size, True, seed),
                                  "test_loader": _make_loader(xo, yo, args.batch_size, False, seed)}
                na, nb = shadow["home_a"]["n_train"], shadow["home_b"]["n_train"]
                shadow["weights"] = {"home_a": na / (na + nb), "home_b": nb / (na + nb)}
                device = configure(config, seed)
                t0 = timestamp(device)
                observed, _ = train_model(shadow, config, seed, name)
                row["shadow_elapsed_s"] = timestamp(device) - t0
                row["shadow_round_s"] = [r["time_s"] for r in observed["rounds"]]
                estimate += 5 * 4 * row["shadow_elapsed_s"] * scale
            rows[key] = row
            report = {"jobs": rows, "measured_elapsed_s": time.perf_counter() - start,
                      "projected_training_hours": estimate / 3600,
                      "projection_complete": len(rows) == 8,
                      "projection_matrix": "85 targets plus 120 shadows; 20 rounds, 5 epochs",
                      "caveat": "Linear scheduling estimate only. Excludes attack feature extraction/fitting and I/O. Short-job setup costs and epoch scaling can bias it; allow headroom."}
            atomic_json(root / "benchmark.json", report)
    print(f"Projected training time (not an end-to-end guarantee): {estimate / 3600:.1f} hours")
    print(f"Report: {root / 'benchmark.json'}")


if __name__ == "__main__":
    main()
