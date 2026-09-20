"""Flow filtering, reproducible partitions and fixed per-record transforms."""

import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import PREPROCESSING

TIME_COL = "bidirectional_first_seen_ms"


def load_and_filter(parquet_path: str, feature_cols: List[str],
                    core_classes: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = list(dict.fromkeys(feature_cols + ["category", "confidence", "bidirectional_packets"]))
    df = pd.read_parquet(parquet_path, columns=cols).reset_index(drop=True)
    audit = {"raw_count": len(df), "class_counts_raw": df["category"].value_counts().to_dict()}
    df = df[df["bidirectional_packets"] >= 2]
    audit["after_pkt_filter"] = len(df)
    df = df[df["confidence"] == "DPI"]
    audit["after_dpi_filter"] = len(df)
    df = df[df["category"].isin(core_classes)]
    audit["after_core_filter"] = len(df)
    audit["class_counts_final"] = df["category"].value_counts().to_dict()
    return df, audit


def feature_unit(name: str) -> float:
    if name == "protocol":
        return 255.0
    if name.endswith("duration_ms") or "_piat_" in name:
        return 1000.0  # milliseconds to seconds
    if name.endswith("bytes") or "_ps_" in name:
        return 1024.0  # bytes to KiB
    if name.endswith("packets"):
        return 1.0
    raise ValueError(f"No predefined transform for feature {name}")


def transform_features(values: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """log1p in fixed physical units; protocol uses its fixed numeric range.

    Each output depends only on that input record and public constants. No fitted
    statistics, quantiles, cross-record clipping bounds or test data are used.
    """
    x = np.array(values, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != len(feature_cols):
        raise ValueError("Feature matrix shape does not match feature names")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Flow features must be finite and nonnegative")
    for i, name in enumerate(feature_cols):
        x[:, i] /= feature_unit(name)
        if name == "protocol":
            if (x[:, i] > 1).any():
                raise ValueError("Protocol must be in [0, 255]")
        else:
            x[:, i] = np.log1p(x[:, i])
    return x.astype(np.float32)


def _make_loader(X, y, batch_size, shuffle, seed, pin_memory=False):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      drop_last=False, pin_memory=pin_memory)


def _index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def prepare_federated_data(home_a_path, home_b_path, feature_cols, core_classes,
                           seed, test_ratio=0.2, batch_size=256, equal_weight=False,
                           split="stratified", pin_memory=False):
    if split not in ("stratified", "temporal"):
        raise ValueError("split must be stratified or temporal")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    classes = sorted(core_classes)
    mapping = {name: i for i, name in enumerate(classes)}
    out = {"class_names": classes, "num_classes": len(classes),
           "num_features": len(feature_cols), "feature_cols": list(feature_cols),
           "preprocessing": {"name": PREPROCESSING,
                             "units": {c: feature_unit(c) for c in feature_cols}},
           "split": split, "split_hashes": {}, "audit": {}, "test_support": {}}
    for hk, path in zip(("home_a", "home_b"), (home_a_path, home_b_path)):
        cols = feature_cols + ([TIME_COL] if split == "temporal" else [])
        df, audit = load_and_filter(path, cols, core_classes)
        if split == "temporal":
            if not np.isfinite(df[TIME_COL].to_numpy()).all():
                raise ValueError("Temporal split requires finite flow start times")
            df = df.sort_values(TIME_COL, kind="mergesort")
        y = df["category"].map(mapping).to_numpy(dtype=np.int64)
        if split == "stratified":
            tr, te = train_test_split(np.arange(len(df)), test_size=test_ratio,
                                      random_state=seed, stratify=y)
        else:
            k = int(len(df) * (1 - test_ratio))
            tr, te = np.arange(k), np.arange(k, len(df))
        if not len(tr) or not len(te):
            raise ValueError("Both partitions must be nonempty")
        x = transform_features(df[feature_cols].to_numpy(), feature_cols)
        xtr, xte, ytr, yte = x[tr], x[te], y[tr], y[te]
        out[hk] = {"X_train": xtr, "X_test": xte, "y_train": ytr, "y_test": yte,
                   "n_train": len(tr), "n_test": len(te),
                   "train_loader": _make_loader(xtr, ytr, batch_size, True, seed, pin_memory),
                   "test_loader": _make_loader(xte, yte, batch_size, False, seed, pin_memory)}
        out["split_hashes"][hk] = {"train": _index_hash(df.index.to_numpy()[tr]),
                                   "test": _index_hash(df.index.to_numpy()[te])}
        out["audit"][hk] = audit
        out["test_support"][hk] = {c: int((yte == i).sum()) for i, c in enumerate(classes)}
    na, nb = out["home_a"]["n_train"], out["home_b"]["n_train"]
    out["weights"] = {"home_a": 0.5 if equal_weight else na / (na + nb),
                      "home_b": 0.5 if equal_weight else nb / (na + nb)}
    return out
