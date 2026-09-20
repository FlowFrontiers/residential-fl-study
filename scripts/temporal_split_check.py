#!/usr/bin/env python3
"""Earlier-80% / later-20% evaluation by flow start time within each home."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl_pipeline.cli import add_arguments, config_from_args
from fl_pipeline.artifacts import atomic_json, ensure_run_root
from fl_pipeline.run_experiment import run_single
from fl_pipeline.analyze import final_metrics, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    root = ensure_run_root(args.results_dir)
    config = config_from_args(args)
    name = "temporal_split_check" + ("_medium" if args.model_size == "medium" else "")
    out = {"split": "temporal_80_20", "model": args.model_size, "seeds": {}}
    all_results = {c: {} for c in ("baseline_fl", "fs_mild", "dp_sgd")}
    for seed in args.seeds:
        rows = {}
        for c in all_results:
            result = run_single(args, config, c, seed, "temporal")
            all_results[c][seed] = result
            metrics = final_metrics(result)
            rows[c] = dict(combined_macro_f1=metrics["combined"], worst_group_f1=metrics["worst"],
                           home_a=result["rounds"][-1]["home_a"], home_b=result["rounds"][-1]["home_b"])
        rows["test_support"] = result["test_support"]
        rows["fs_beats_dp_worst"] = rows["fs_mild"]["worst_group_f1"] > rows["dp_sgd"]["worst_group_f1"]
        out["seeds"][str(seed)] = rows
        atomic_json(root / f"{name}.json", out)
    out["summary"] = summarize(all_results, args.seeds)
    atomic_json(root / f"{name}.json", out)
    print(json_summary(out["summary"]))


def json_summary(summary):
    import json
    return json.dumps(summary["paired"], indent=2)


if __name__ == "__main__":
    main()
