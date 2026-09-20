"""Federated training on CPU or CUDA with per-round accounting and timing."""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from opacus.accountants import RDPAccountant

from .model import TrafficClassifier, fedavg
from .metrics import compute_metrics
from .runtime import configure, timestamp
from .dp import compute_delta, calibrate_noise_multiplier, make_dp_loader, setup_dp_training
from .feddpa import (compute_fisher_diagonal, generate_mask, init_local_from_mask,
                     feddpa_local_train, clip_and_noise_update, aggregate_updates,
                     calibrate_feddpa_noise)

HOME_KEYS = ("home_a", "home_b")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(xb)
        # Empty Poisson batches still take a noise-only optimizer step.
        loss = criterion(output, yb) if len(yb) else output.sum()
        loss.backward()
        optimizer.step()


def evaluate(model, loader, class_names, device):
    model.eval()
    predictions, labels = [], []
    loss_sum = torch.zeros((), device=device)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            output = model(xb)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite model output during evaluation")
            loss_sum += nn.functional.cross_entropy(output, yb, reduction="sum")
            predictions.append(output.argmax(1))
            labels.append(yb)
    y_true = torch.cat(labels).cpu().numpy()
    metrics = compute_metrics(y_true, torch.cat(predictions).cpu().numpy(), class_names)
    metrics["loss"] = float(loss_sum.item() / len(y_true))
    return metrics


def _model(data, config, device):
    return TrafficClassifier(data["num_features"], data["num_classes"],
                             config.hidden_dims).to(device)


def _result(model, name, seed):
    return {"config_name": name, "seed": seed, "rounds": [],
            "hidden_dims": list(model.hidden_dims),
            "model_params": sum(p.numel() for p in model.parameters())}


def _finish_round(results, info, models, data, device, started, total_rounds):
    for hk, model in zip(HOME_KEYS, models):
        info[hk] = evaluate(model, data[hk]["test_loader"], data["class_names"], device)
    info["time_s"] = timestamp(device) - started
    results["rounds"].append(info)
    print(f"  Round {info['round']:2d}/{total_rounds} | "
          f"A-F1={info['home_a']['macro_f1']:.3f} | B-F1={info['home_b']['macro_f1']:.3f} | "
          f"A-worst={info['home_a']['worst_group_f1']:.3f} | "
          f"B-worst={info['home_b']['worst_group_f1']:.3f} | ({info['time_s']:.1f}s)", flush=True)


def run_standard_fl(data, config, seed, config_name="baseline_fl", return_model=False):
    device = configure(config, seed)
    model = _model(data, config, device)
    results = _result(model, config_name, seed)
    weights = [data["weights"][h] for h in HOME_KEYS]
    for r in range(1, config.num_rounds + 1):
        t0 = timestamp(device)
        locals_ = []
        for hk in HOME_KEYS:
            local = copy.deepcopy(model)
            opt = torch.optim.Adam(local.parameters(), lr=config.lr)
            for _ in range(config.local_epochs):
                train_epoch(local, data[hk]["train_loader"], opt, nn.CrossEntropyLoss(), device)
            locals_.append(local)
        model = fedavg(locals_, weights).to(device)
        _finish_round(results, {"round": r}, [model, model], data, device, t0, config.num_rounds)
    return (results, model) if return_model else results


def run_dp_sgd_fl(data, config, seed, return_model=False):
    device = configure(config, seed)
    model = _model(data, config, device)
    results = _result(model, "dp_sgd", seed)
    results.update(noise_multipliers={}, deltas={}, sampling={}, privacy={
        "mechanism": "poisson_dp_sgd", "unit": "training_flow",
        "adjacency": "add_remove", "accounted_outputs": "model_updates",
        "rng": "seeded_research_prng", "scope": "fixed_partitions_and_public_protocol_metadata",
        "excluded": ["dataset_publication", "partition_selection", "test_labels", "combined_release_of_multiple_runs"]})
    accountants = {h: RDPAccountant() for h in HOME_KEYS}
    weights = [data["weights"][h] for h in HOME_KEYS]
    for hk in HOME_KEYS:
        n = data[hk]["n_train"]
        steps = math.ceil(n / config.batch_size)
        q = 1.0 / steps  # DPDataLoader.from_data_loader uses 1 / len(loader).
        delta = compute_delta(n)
        total = config.num_rounds * config.local_epochs * steps
        sigma = calibrate_noise_multiplier(config.target_epsilon, delta, q, total)
        results["noise_multipliers"][hk] = sigma
        results["deltas"][hk] = delta
        results["sampling"][hk] = {"sample_rate": q, "total_steps": total}
    for r in range(1, config.num_rounds + 1):
        t0 = timestamp(device)
        locals_, info = [], {"round": r}
        for hk in HOME_KEYS:
            local = copy.deepcopy(model).train()
            opt = torch.optim.Adam(local.parameters(), lr=config.lr)
            loader = make_dp_loader(data[hk]["X_train"], data[hk]["y_train"], config.batch_size)
            local, opt, loader, pe = setup_dp_training(
                local, opt, loader, results["noise_multipliers"][hk], config.max_grad_norm, accountants[hk])
            for _ in range(config.local_epochs):
                train_epoch(local, loader, opt, nn.CrossEntropyLoss(), device)
            info[f"{hk}_epsilon"] = pe.get_epsilon(results["deltas"][hk])
            info[f"{hk}_accountant_steps"] = sum(n for _, _, n in accountants[hk].history)
            locals_.append(local._module)
        model = fedavg(locals_, weights).to(device)
        _finish_round(results, info, [model, model], data, device, t0, config.num_rounds)
    return (results, model) if return_model else results


def run_feddpa_fl(data, config, seed, return_model=False):
    device = configure(config, seed)
    model = _model(data, config, device)
    previous = {h: copy.deepcopy(model) for h in HOME_KEYS}
    results = _result(model, "feddpa", seed)
    results.update(noise_multipliers={}, deltas={}, privacy={
        "mechanism": "full_update_gaussian", "unit": "client_training_partition",
        "adjacency": "replace_one_client_dataset", "sensitivity": 2 * config.max_grad_norm,
        "sample_rate": 1.0, "accounted_outputs": "noised_full_updates",
        "rng": "seeded_research_prng", "scope": "fixed_roster_weights_and_public_protocol_metadata",
        "excluded": ["personalized_models", "personalized_metrics", "dataset_publication", "combined_release_of_multiple_runs"]},
        local_epochs_per_stage=config.local_epochs, local_stages=2,
        evaluation_model="local_after_training")
    accountants = {h: RDPAccountant() for h in HOME_KEYS}
    weights = [data["weights"][h] for h in HOME_KEYS]
    for hk in HOME_KEYS:
        delta = compute_delta(data[hk]["n_train"])
        results["deltas"][hk] = delta
        results["noise_multipliers"][hk] = calibrate_feddpa_noise(config.target_epsilon, delta, config.num_rounds)
    for r in range(1, config.num_rounds + 1):
        t0 = timestamp(device)
        deltas, local_models, info = [], [], {"round": r}
        for hk in HOME_KEYS:
            fisher = compute_fisher_diagonal(previous[hk], data[hk]["X_train"], data[hk]["y_train"],
                                             config.fisher_n_samples, device)
            mask = generate_mask(fisher, config.fisher_threshold)
            old = {n: p.detach() for n, p in previous[hk].named_parameters()}
            local = init_local_from_mask(model, mask, old, device)
            local = feddpa_local_train(local, mask, data[hk]["train_loader"], config.local_epochs,
                                       config.lr, config.lambda_reg, config.max_grad_norm, device)
            deltas.append(clip_and_noise_update(local, model, config.max_grad_norm,
                                                results["noise_multipliers"][hk]))
            previous[hk] = local
            local_models.append(local)
            accountants[hk].step(noise_multiplier=results["noise_multipliers"][hk], sample_rate=1.0)
            info[f"{hk}_epsilon"] = accountants[hk].get_epsilon(results["deltas"][hk])
        model = aggregate_updates(model, deltas, weights)
        _finish_round(results, info, local_models, data, device, t0, config.num_rounds)
    # Only the aggregated model is checkpointed; personalized weights stay local.
    return (results, model) if return_model else results
