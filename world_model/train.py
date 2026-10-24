"""Local training and evaluation for the TCN.

These two functions are what the Flower ClientApp will call in Phase 2
(Week 3 interface: train(...) -> metrics with num_examples, evaluate(...) ->
loss, num_examples, metrics). proximal_mu > 0 turns on the FedProx term;
proximal_mu = 0 is plain FedAvg-style local training.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn

from .metrics import forecast_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model: nn.Module, loader, epochs: int = 1, lr: float = 1e-3,
          proximal_mu: float = 0.0, global_params: list[torch.Tensor] | None = None,
          optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Train in place. Returns {'train_loss', 'num_examples', 'train_time_s'}.

    Pass `optimizer` to keep Adam's state across calls (centralized training);
    leave it None for a fresh optimizer each call (one FL round on a client).
    """
    model.train()
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    if proximal_mu > 0 and global_params is None:
        global_params = [p.detach().clone() for p in model.parameters()]
    t0 = time.perf_counter()
    total, n = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = mse(model(xb), yb)
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return {
        "train_loss": total / max(n, 1),
        "num_examples": len(loader.dataset),
        "train_time_s": time.perf_counter() - t0,
    }


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = [model(torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size]))).numpy()
            for i in range(0, len(x), batch_size)]
    return np.concatenate(outs) if outs else np.zeros((0, *x.shape[1:2], 10), np.float32)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    """Forecast report (overall, per-horizon, per-group MSE) on held-out data."""
    return forecast_report(predict(model, x), y)
