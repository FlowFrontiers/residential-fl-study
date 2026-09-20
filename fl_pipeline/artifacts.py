"""Atomic artifacts with data, partition, configuration and code provenance."""

from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .config import PROJECT_ROOT
from .model import TrafficClassifier
from .runtime import environment, resolve_device


@lru_cache(maxsize=16)
def _file_hash(path, size, mtime_ns):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hash(path):
    path = Path(path)
    stat = path.stat()
    return _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hash():
    root = Path(PROJECT_ROOT) / "fl_pipeline"
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".json-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def make_spec(data, config, seed, config_name, data_paths):
    return {"schema": 1, "config_name": config_name, "seed": seed,
            "config": asdict(config), "environment": environment(config),
            "source_sha256": source_hash(),
            "data_sha256": {hk: file_hash(p) for hk, p in zip(("home_a", "home_b"), data_paths)},
            "feature_cols": data["feature_cols"], "classes": data["class_names"],
            "preprocessing": data["preprocessing"], "split": data["split"],
            "split_hashes": data["split_hashes"], "weights": data["weights"],
            "sizes": {hk: {k: data[hk][k] for k in ("n_train", "n_test")}
                      for hk in ("home_a", "home_b")}}


def ensure_run_root(root):
    root = Path(root).resolve()
    published = Path(PROJECT_ROOT) / "results"
    if root == published or published in root.parents:
        raise ValueError("Published results are read-only; select a directory under runs/")
    root.mkdir(parents=True, exist_ok=True)
    return root


def target_path(root, feature_set, model_size, seed, config_name,
                equal_weight=False, split="stratified"):
    folder = feature_set if model_size == "small" else f"{feature_set}_{model_size}"
    base = Path(root) / folder
    if split == "temporal":
        base /= "temporal"
    prefix = "equal_weight_seed_" if equal_weight else "seed_"
    return base / f"{prefix}{seed}" / f"{config_name}.json"


def validate_spec(actual, expected, path):
    if actual != expected:
        differing = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f"Artifact does not match requested run at {path}: {', '.join(differing)}. Use a separate run directory.")


def save_target(path, results, model, spec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"spec": spec, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "input_dim": model.model[0].in_features,
               "num_classes": model.model[-1].out_features, "hidden_dims": list(model.hidden_dims)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-", suffix=".tmp")
    os.close(fd)
    checkpoint = path.with_suffix(".pt")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, checkpoint)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    results = dict(results, spec=spec, checkpoint_sha256=file_hash(checkpoint))
    atomic_json(path, results)
    return results


def load_target(path, spec, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing target {path}; run run_experiment first")
    results = json.loads(path.read_text())
    validate_spec(results.get("spec", {}), spec, path)
    checkpoint = path.with_suffix(".pt")
    if file_hash(checkpoint) != results["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_spec(payload["spec"], spec, checkpoint)
    model = TrafficClassifier(payload["input_dim"], payload["num_classes"], payload["hidden_dims"])
    model.load_state_dict(payload["state_dict"])
    return results, model.to(resolve_device(str(device)))


def load_cached_json(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    validate_spec(result.get("identity", {}), identity, path)
    return result
