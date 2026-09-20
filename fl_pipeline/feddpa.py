"""Dynamic Fisher personalization and adaptive constraints (Yang et al., 2023).

The mask controls local initialization and optimization, not transmitted support.
Full updates are clipped and noised. See docs/methods.md for the release model.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .dp import calibrate_noise_multiplier


def compute_fisher_diagonal(model, X, y, n_samples=5000, device=torch.device("cpu")):
    """Mean squared per-example log-likelihood gradients on a local subsample."""
    model = model.to(device).eval()
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def single_loss(parameters, xb, yb):
        output = functional_call(model, (parameters, buffers), (xb.unsqueeze(0),))
        return F.cross_entropy(output, yb.unsqueeze(0))

    per_sample_grad = vmap(grad(single_loss), in_dims=(None, 0, 0))
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    for start in range(0, len(indices), 256):
        idx = indices[start:start + 256]
        gradients = per_sample_grad(params, torch.as_tensor(X[idx], device=device),
                                    torch.as_tensor(y[idx], device=device))
        for name, values in gradients.items():
            fisher[name] += values.square().sum(0)
    return {n: f / len(indices) for n, f in fisher.items()}


def generate_mask(fisher, threshold=0.4):
    masks = {}
    for name, values in fisher.items():
        span = values.max() - values.min()
        norm = (values - values.min()) / span.clamp_min(1e-10)
        masks[name] = (norm >= threshold).to(values.dtype)
    return masks


def init_local_from_mask(global_model, mask, previous, device):
    model = copy.deepcopy(global_model).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            m = mask[name].to(device)
            param.copy_(m * previous[name].to(device) + (1 - m) * param)
    return model


def feddpa_local_train(model, mask, loader, local_epochs, lr, lambda_reg, clip_norm, device):
    """Personal stage then shared stage, each with local_epochs data passes.

    References are the locally personalized initialization. Parameter gradients
    outside the active mask are zero; each stage has its own Adam optimizer.
    """
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    for personal in (True, False):
        active = {n: m if personal else 1 - m for n, m in mask.items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                diff = torch.cat([((p - reference[n]) * active[n]).flatten()
                                  for n, p in model.named_parameters()])
                norm = torch.linalg.vector_norm(diff)
                penalty = norm if personal else (norm - clip_norm).abs()
                (loss + lambda_reg / 2 * penalty).backward()
                for name, param in model.named_parameters():
                    param.grad.mul_(active[name])
                optimizer.step()
    return model


def clip_and_noise_update(local_model, global_model, clip_norm, noise_multiplier):
    """Gaussian release of the entire clipped update, without revealing a mask.

    Replacement of a client's local data/state can move two clipped updates by
    at most 2*C. noise_multiplier is calibrated relative to that sensitivity.
    """
    base = dict(global_model.named_parameters())
    delta = {n: p.detach() - base[n].detach() for n, p in local_model.named_parameters()}
    norm = torch.linalg.vector_norm(torch.cat([d.flatten() for d in delta.values()]))
    factor = (clip_norm / norm.clamp_min(1e-12)).clamp_max(1.0)
    return {n: d * factor + torch.randn_like(d) * (2 * clip_norm * noise_multiplier)
            for n, d in delta.items()}


def aggregate_updates(global_model, deltas, weights):
    model = copy.deepcopy(global_model)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(sum(w * d[name] for w, d in zip(weights, deltas)))
    return model


def calibrate_feddpa_noise(epsilon, delta, num_rounds):
    return calibrate_noise_multiplier(epsilon, delta, 1.0, num_rounds)
