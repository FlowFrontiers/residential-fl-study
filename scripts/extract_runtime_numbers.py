#!/usr/bin/env python3
"""Read isolated timing measurements and print their summary as JSON."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[1]
CONFIGS = ("baseline_fl", "fs_mild", "fs_aggressive", "dp_sgd", "feddpa")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(root, reference):
    root, reference = Path(root), Path(reference)
    provenance = root / "provenance"
    manifest = json.loads((provenance / "manifest.json").read_text())
    if (provenance / "run.exit").read_text().strip() != "0":
        raise ValueError("Isolated measurement did not complete successfully")
    if sha256(provenance / "launch.sh") != manifest["launcher_sha256"]:
        raise ValueError("Timing launcher differs from measurement manifest")
    expected = {"seed": 42, "rounds": 20, "local_epochs": 5,
                "batch_size": 256, "hidden_dims": [16, 16],
                "device": "cpu", "num_threads": 1, "split": "stratified",
                "aggregation": "size_proportional"}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected isolated timing protocol")
    if manifest.get("configs") != list(CONFIGS):
        raise ValueError("Timing manifest must list all five configurations in order")
    base = Path("baseline_16/seed_42")
    if {p.stem for p in (root / base).glob("*.json")} != set(CONFIGS):
        raise ValueError("Timing inputs must contain exactly five configurations")
    table, inputs = {}, {}
    for name in CONFIGS:
        path = root / base / f"{name}.json"
        run = json.loads(path.read_text())
        target = json.loads((reference / base / f"{name}.json").read_text())
        spec = run["spec"]
        if spec != target["spec"]:
            raise ValueError(f"Timing/reference protocol mismatch: {name}")
        if any(spec[k] != manifest[k] for k in ("source_sha256", "data_sha256")):
            raise ValueError(f"Timing/manifest identity mismatch: {name}")
        for key in ("local_epochs", "batch_size", "hidden_dims", "device", "num_threads"):
            if spec["config"][key] != expected[key]:
                raise ValueError(f"Unexpected timing configuration: {name}, {key}")
        if (run["seed"] != 42 or run["config_name"] != name or
                spec["config"]["num_rounds"] != 20 or
                spec["config"]["equal_weight"] or spec["split"] != "stratified"):
            raise ValueError(f"Unexpected timing run identity: {name}")
        if sha256(path.with_suffix(".pt")) != run["checkpoint_sha256"]:
            raise ValueError(f"Timing checkpoint mismatch: {name}")
        rounds = run["rounds"]
        if [r["round"] for r in rounds] != list(range(1, 21)):
            raise ValueError(f"Incomplete timing rounds: {name}")
        times = [r["time_s"] for r in rounds]
        if any(not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError(f"Invalid round duration: {name}")
        median = statistics.median(times)
        table[name] = {"round_count": len(times), "median_round_s": median,
                       "estimated_run_min": median * len(times) / 60,
                       "sum_round_time_min": sum(times) / 60}
        inputs[path.relative_to(root).as_posix()] = sha256(path)
    fs_time = table["fs_mild"]["median_round_s"]
    return {"scope": "Isolated small-model timing only; not utility or attack evidence",
            "seed": 42, "cpu_model": manifest["cpu_model"], "num_threads": 1,
            "timing_context": manifest["timing_context"],
            "source_sha256": manifest["source_sha256"],
            "data_sha256": manifest["data_sha256"],
            "input_sha256": inputs, "runtime_table": table,
            "dp_sgd_over_fs_mild": table["dp_sgd"]["median_round_s"] / fs_time,
            "feddpa_over_fs_mild": table["feddpa"]["median_round_s"] / fs_time}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=REPO / "results/runtime_isolated")
    parser.add_argument("--reference-results", type=Path, default=REPO / "results")
    args = parser.parse_args()
    print(json.dumps(summarize(args.runtime_dir, args.reference_results),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
