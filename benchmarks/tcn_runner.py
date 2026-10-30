"""Train-and-evaluate loop shared by B2 (centralized) and B3 (local-only)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from world_model.data import make_loader
from world_model.metrics import forecast_report, persistence_forecast
from world_model.tcn import TCN, TCNConfig, count_parameters
from world_model.train import predict, set_seed, train


def fit(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, *,
        epochs: int, lr: float, batch_size: int, seed: int, cfg: TCNConfig,
        label: str, log=None) -> tuple[TCN, list[dict]]:
    """Train one TCN; return it and a per-epoch history (train loss, val MSE)."""
    set_seed(seed)
    model = TCN(cfg)
    loader = make_loader(x_tr, y_tr, batch_size, shuffle=True, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # one optimizer for the whole run
    history = []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        m = train(model, loader, epochs=1, optimizer=opt)
        val_mse = forecast_report(predict(model, x_va), y_va)["mse"]
        row = {"label": label, "seed": seed, "epoch": epoch, "train_loss": m["train_loss"],
               "val_mse": val_mse, "epoch_time_s": m["train_time_s"]}
        history.append(row)
        if log:
            log({"train_loss": m["train_loss"], "val_mse": val_mse}, step=epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [{label} seed {seed}] epoch {epoch:>3}/{epochs}  train {m['train_loss']:.4f}  "
                  f"val {val_mse:.4f}  ({time.perf_counter() - t_start:.0f}s)", flush=True)
    return model, history


def report_row(net: TCN, x: np.ndarray, y: np.ndarray, /, **ids) -> dict:
    """Forecast report plus the ratio to persistence on the same windows (<1 beats it)."""
    return add_persistence(forecast_report(predict(net, x), y), x, y, **ids)


def add_persistence(report: dict, x: np.ndarray, y: np.ndarray, /, **ids) -> dict:
    p = forecast_report(persistence_forecast(x, y.shape[2]), y)["mse"]
    return {**ids, **report, "persistence_mse": p, "ratio_vs_persistence": report["mse"] / p}


def save_checkpoint(model: TCN, path: Path, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict(),
                "num_parameters": count_parameters(model), **meta}, path)


def load_checkpoint(path: Path) -> TCN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TCN(TCNConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()