"""PulseNet FL task: model, data loading, local training and evaluation.

This is the only Flower file that touches world_model/ or the data contract.
If the TCN constructor or the .npz layout changes, edit this file only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from world_model.tcn import TCN

N_CHANNELS = 19
N_IN_STEPS = 100
N_OUT_STEPS = 10


# --------------------------------------------------------------------------- model

def build_model(cfg) -> nn.Module:
    """Build the TCN from the run config."""
    return TCN(hidden=int(cfg["hidden-channels"]))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- data

def resolve_data_path(context) -> Path:
    """Find this node's .npz file without hardcoding paths

    Deployment: each Pi's SuperNode is started with
        --node-config "data-path='/path/to/subject_10N.npz'"
    Simulation: Flower gives each virtual node a partition-id; it is mapped to
        a subject in sim-subjects, and the file is read from sim-data-dir.
    """
    node = context.node_config
    if "data-path" in node:
        return Path(str(node["data-path"])).expanduser()

    run = context.run_config
    data_dir = str(run.get("sim-data-dir", ""))
    if not data_dir:
        raise ValueError(
            "No data-path in node config and no sim-data-dir in run config. "
            "Simulation: --run-config \"sim-data-dir='/abs/path/to/data/processed'\""
        )
    subjects = [s.strip() for s in str(run["sim-subjects"]).split(",") if s.strip()]
    pid = int(node["partition-id"])
    n_parts = int(node["num-partitions"])
    if n_parts != len(subjects):
        raise ValueError(
            f"{n_parts} simulated nodes but {len(subjects)} subjects in sim-subjects"
        )
    return Path(data_dir).expanduser() / f"subject_{subjects[pid]}.npz"


def _split_mask(split: np.ndarray, which: str) -> np.ndarray:
    # TODO(check): confirm how prepare_pamap2.py encodes `split`. Strings are
    # assumed here ("train" / "val").
    if split.dtype.kind in ("U", "S"):
        return split.astype(str) == which
    raise ValueError(
        f"Unexpected dtype for 'split': {split.dtype}. Update _split_mask() "
        "to match the data contract."
    )


def load_data(path: Path, batch_size: int, seed: int):
    """Load one subject's windows -> (train_loader, val_loader, n_train, n_val)."""
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    with np.load(path, allow_pickle=False) as d:
        x = d["x"].astype(np.float32)   # [N, 19, 100]
        y = d["y"].astype(np.float32)   # [N, 19, 10]
        split = d["split"]

    if x.shape[1:] != (N_CHANNELS, N_IN_STEPS) or y.shape[1:] != (N_CHANNELS, N_OUT_STEPS):
        raise ValueError(f"{path.name}: x {x.shape}, y {y.shape} don't match the data contract")

    tr, va = _split_mask(split, "train"), _split_mask(split, "val")

    # TODO(check): this assumes x and y are stored already z-scored with the
    # train-split mean/std (the contract also stores mean/std). This guard
    # catches raw values by accident.
    ch_std = x[tr].std(axis=(0, 2))
    if np.any(ch_std > 3.0) or np.any(ch_std < 0.2):
        print(f"[pulsenet] WARNING {path.name}: train per-channel std "
              f"{ch_std.min():.2f}..{ch_std.max():.2f}; data may not be z-scored")

    gen = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr]))
    val_ds = TensorDataset(torch.from_numpy(x[va]), torch.from_numpy(y[va]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, len(train_ds), len(val_ds)


# --------------------------------------------------------------------------- train / eval

def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, mu: float):
    """Local training. MSE loss, Adam (W2 spec), optional FedProx term.

    mu = 0 is plain FedAvg; mu > 0 adds (mu/2) * ||w - w_global||^2.
    Adam is created fresh each round, so its state does not carry across rounds.

    TODO(check): world_model/train.py already supports FedProx. Once its
    signature is confirmed, call it here so FL, B2 and B3 share one training loop.

    Returns (mean MSE over the samples seen, seconds spent).
    """
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    model.train()
    t0 = time.perf_counter()
    loss_sum, n = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = mse(model(xb), yb)
            total = loss
            if global_params is not None:
                prox = sum((p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params))
                total = loss + 0.5 * mu * prox
            total.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)
    return loss_sum / max(n, 1), time.perf_counter() - t0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    """Return (model MSE, persistence MSE) on the validation windows.

    Persistence repeats the last input step for all 10 output steps, so the
    two numbers give the "ratio vs persistence" from the results log.
    """
    model.eval()
    se = pse = 0.0
    n = 0
    for xb, yb in loader:
        pred = model(xb)
        if pred.shape != yb.shape:
            raise ValueError(f"Model output {tuple(pred.shape)} != target {tuple(yb.shape)}")
        persist = xb[:, :, -1:].expand_as(yb)
        se += (pred - yb).pow(2).sum().item()
        pse += (persist - yb).pow(2).sum().item()
        n += yb.numel()
    return se / max(n, 1), pse / max(n, 1)
