#!/usr/bin/env python3
"""Extract table values and paired evidence from one coherent run directory."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl_pipeline.analyze import load_results, summarize, stats
from fl_pipeline.artifacts import atomic_json, ensure_run_root
from fl_pipeline.config import RESULTS_DIR, SEEDS
from fl_pipeline.run_mia import aggregate_results


def validate_related(primary, related):
    for name, entries in related.items():
        for seed, run in entries.items():
            a, b = primary[name][seed]["spec"], run["spec"]
            for field in ("source_sha256", "data_sha256", "environment", "classes", "feature_cols", "preprocessing"):
                if a[field] != b[field]:
                    raise ValueError(f"Related experiment differs on {field}: {name}, seed {seed}")
            ac, bc = dict(a["config"]), dict(b["config"])
            ac.pop("equal_weight")
            bc.pop("equal_weight")
            if ac != bc:
                raise ValueError("Related experiments have different training settings")


def extract(root, feature_set, model_size, seeds):
    subdir = feature_set if model_size == "small" else f"{feature_set}_{model_size}"
    runs = load_results(subdir, results_dir=root, seeds=seeds)
    expected = {"baseline_fl", "fs_mild", "dp_sgd"}
    if model_size == "small":
        expected |= {"fs_aggressive", "feddpa"}
    if set(runs) != expected:
        raise ValueError(f"Expected configurations {sorted(expected)}, found {sorted(runs)}")
    summary = summarize(runs, seeds)
    first = runs["baseline_fl"][seeds[0]]
    classes = sorted(first["audit"]["home_a"]["class_counts_final"])
    result = {"meta": {"feature_set": feature_set, "model_size": model_size, "seeds": seeds,
                       "std_convention": "sample (ddof=1)", "source_sha256": first["spec"]["source_sha256"],
                       "data_sha256": first["spec"]["data_sha256"]},
              "dataset": first["audit"], "main_table": {}, "per_class_table": {},
              "runtime_table": {}, "evidence": {"primary": summary}, "attacks": {}}
    for name, entries in runs.items():
        entry = summary["per_config"][name]
        result["main_table"][name] = {"a_macro": entry["home_a"], "b_macro": entry["home_b"],
                                      "combined": entry["combined"], "worst": entry["worst"],
                                      "per_seed": entry["per_seed"],
                                      "model_params": entries[seeds[0]]["model_params"]}
        result["per_class_table"][name] = {
            h: {c: float(np.mean([entries[s]["rounds"][-1][h]["per_class_f1"][c] for s in seeds]))
                for c in classes} for h in ("home_a", "home_b")}
        times = [r["time_s"] for s in seeds for r in entries[s]["rounds"]]
        rounds = {len(entries[s]["rounds"]) for s in seeds}
        if len(rounds) != 1:
            raise ValueError("Inconsistent round counts")
        median = float(np.median(times))
        result["runtime_table"][name] = {
            "round_time_s": {"median": median, "q25": float(np.percentile(times, 25)),
                             "q75": float(np.percentile(times, 75))},
            "estimated_run_time_min": median * next(iter(rounds)) / 60,
            "observed_run_time_min": stats([entries[s]["total_time_s"] / 60 for s in seeds])}
    eq = load_results(subdir, True, root, seeds)
    if eq:
        if set(eq) != {"baseline_fl", "fs_mild", "dp_sgd"}:
            raise ValueError("Equal-weight outputs must contain all three primary configurations")
        validate_related(runs, eq)
        result["evidence"]["equal_weight"] = summarize(eq, seeds)
    temporal = load_results(subdir, False, root, seeds, "temporal")
    if temporal:
        if set(temporal) != {"baseline_fl", "fs_mild", "dp_sgd"}:
            raise ValueError("Temporal outputs must contain all three primary configurations")
        validate_related(runs, temporal)
        result["evidence"]["temporal"] = summarize(temporal, seeds)
    for kind in ("mia", "shadow_mia"):
        directory = Path(root) / subdir / kind
        if not directory.exists():
            continue
        rows = []
        attack_protocols = set()
        for seed in seeds:
            row = {}
            for name in ("baseline_fl", "fs_mild", "dp_sgd"):
                path = directory / f"seed_{seed}_{name}.json"
                attack = json.loads(path.read_text())
                identity = attack["identity"]
                attack_protocols.add(json.dumps({k: v for k, v in identity.items()
                                                if k not in ("target_spec", "checkpoint_sha256")}, sort_keys=True))
                if (identity["target_spec"] != runs[name][seed]["spec"] or
                        identity["checkpoint_sha256"] != runs[name][seed]["checkpoint_sha256"]):
                    raise ValueError(f"MIA target does not match utility target: {path}")
                row[name] = attack
            rows.append(row)
        if len(attack_protocols) != 1:
            raise ValueError("Attack results have inconsistent protocols")
        result["attacks"][kind] = aggregate_results(rows, ["baseline_fl", "fs_mild", "dp_sgd"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--model-size", choices=["small", "medium"], default="small")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = ensure_run_root(args.results_dir)
    result = extract(root, args.feature_set, args.model_size, args.seeds)
    suffix = "" if args.model_size == "small" and args.feature_set == "baseline_16" else f"_{args.feature_set}_{args.model_size}"
    path = Path(args.output) if args.output else root / f"paper_numbers{suffix}.json"
    ensure_run_root(path.parent)
    atomic_json(path, result)
    print(f"Saved {path}")
    print(json.dumps(result["evidence"]["primary"]["paired"], indent=2))


if __name__ == "__main__":
    main()
