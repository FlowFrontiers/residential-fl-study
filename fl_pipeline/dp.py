"""DP-SGD utilities: noise calibration, Opacus wrapping, epsilon tracking."""

from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant


def compute_delta(n_train: int) -> float:
    """Delta = min(1e-5, 1/(10*N))."""
    return min(1e-5, 1.0 / (10 * n_train))


@lru_cache(maxsize=128)
def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    num_steps: int,
) -> float:
    """Binary search for noise_multiplier achieving target epsilon.

    Uses Opacus RDP accountant for accurate composition.
    """
    if not (target_epsilon > 0 and 0 < delta < 1 and 0 < sample_rate <= 1 and num_steps > 0):
        raise ValueError("Invalid privacy calibration parameters")

    def epsilon(sigma):
        acct = RDPAccountant()
        acct.history = [(sigma, sample_rate, num_steps)]
        value = acct.get_epsilon(delta)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite RDP accountant result")
        return value

    low, high = 0.0, 1.0
    while epsilon(high) > target_epsilon:
        high *= 2
        if high > 1e6:
            raise ValueError("Requested privacy budget exceeds calibration range")

    for _ in range(100):
        mid = (low + high) / 2
        eps = epsilon(mid)

        if eps > target_epsilon:
            low = mid
        else:
            high = mid  # Can use less noise

        if high - low < 1e-6:
            break

    return high  # Slightly conservative


def make_dp_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Create a DataLoader suitable for Opacus wrapping."""
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def setup_dp_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    noise_multiplier: float,
    max_grad_norm: float,
    accountant: RDPAccountant,
) -> tuple:
    """Wrap model/optimizer/loader with Opacus DP-SGD.

    Uses a persistent RDPAccountant that accumulates across FL rounds.
    Returns (wrapped_model, wrapped_optimizer, wrapped_loader, privacy_engine).
    """
    pe = PrivacyEngine()
    pe.accountant = accountant  # Persistent across rounds

    model, optimizer, data_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        clipping="flat",
        poisson_sampling=True,
    )
    return model, optimizer, data_loader, pe


def get_epsilon(pe: PrivacyEngine, delta: float) -> float:
    """Get current cumulative epsilon from the privacy engine."""
    return pe.get_epsilon(delta)
