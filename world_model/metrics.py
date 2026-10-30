"""Forecast metrics shared by every benchmark (numpy only).

All MSE values are in z-score units: each channel was standardized with its
subject's own training-split mean and std, so 1.0 means "as wrong as the
channel's typical spread".
"""

from __future__ import annotations

import numpy as np

# Channel order produced by scripts/prepare_pamap2.py
CHANNELS = [
    "hand_acc_x", "hand_acc_y", "hand_acc_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "heart_rate",
]

GROUPS = {
    "acc": [i for i, c in enumerate(CHANNELS) if "_acc_" in c],
    "gyro": [i for i, c in enumerate(CHANNELS) if "_gyro_" in c],
    "hr": [CHANNELS.index("heart_rate")],
}


def forecast_report(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred, target: [N, C, H]. Returns overall, per-horizon and per-group MSE."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    err = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
    report = {
        "num_examples": int(target.shape[0]),
        "mse": float(err.mean()),
    }
    per_h = err.mean(axis=(0, 1))
    for h, v in enumerate(per_h, start=1):
        report[f"mse_h{h}"] = float(v)
    for name, idx in GROUPS.items():
        if max(idx) < err.shape[1]:
            report[f"mse_{name}"] = float(err[:, idx, :].mean())
    return report


def zero_forecast(x: np.ndarray, horizon: int) -> np.ndarray:
    """Predict the channel mean (0 after z-scoring) for every future step."""
    return np.zeros((x.shape[0], x.shape[1], horizon), dtype=x.dtype)


def persistence_forecast(x: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every future step."""
    return np.repeat(x[:, :, -1:], horizon, axis=2)
