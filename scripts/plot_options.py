"""Common paths and seed selection for paper figures."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_pipeline.config import RESULTS_DIR, SEEDS
from fl_pipeline.artifacts import ensure_run_root
from fl_pipeline.analyze import load_results, summarize


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--feature-set", default="baseline_16")
    parser.add_argument("--model-size", choices=["small", "medium"], default="small")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    root = ensure_run_root(args.results_dir)
    folder = args.feature_set if args.model_size == "small" else f"{args.feature_set}_{args.model_size}"
    runs = load_results(folder, results_dir=root, seeds=args.seeds)
    if not {"baseline_fl", "fs_mild", "dp_sgd"} <= set(runs):
        raise ValueError("Figures require all three primary configurations")
    summarize(runs, args.seeds)
    figures = root / "figures" / folder
    figures.mkdir(parents=True, exist_ok=True)
    return root / folder, figures, args.seeds
