"""Write small fake PAMAP2 protocol files (same 54-column .dat format) for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from world_model.pamap2 import RAW_COLUMNS, RAW_FS


def make_subject_file(path: Path, subject: int, bouts: list[tuple[int, float]] | None = None,
                      seed: int = 0, long_dropout: bool = True) -> Path:
    """bouts: (activity_id, seconds). Activity 0 = transient."""
    rng = np.random.default_rng(seed + subject)
    if bouts is None:
        bouts = [(0, 5), (1, 40), (0, 5), (4, 60), (0, 5), (5, 30), (0, 3), (12, 5)]
    rows = []
    t = 0.0
    for act, secs in bouts:
        n = int(secs * RAW_FS)
        tt = t + np.arange(n) / RAW_FS
        block = rng.normal(0, 0.05, size=(n, 54))
        block[:, 0] = tt
        block[:, 1] = act
        freq = 0.3 + 0.4 * (act % 5)
        for name, col in RAW_COLUMNS.items():
            if name in ("timestamp", "activity", "heart_rate"):
                continue
            phase = (col % 7) * 0.5
            block[:, col] += np.sin(2 * np.pi * freq * tt + phase) * (1 + act / 10)
        hr = 70 + 3 * act + np.sin(tt / 10)
        block[:, 2] = np.nan
        block[::11, 2] = hr[::11]                 # ~9 Hz heart rate
        rows.append(block)
        t = tt[-1] + 1 / RAW_FS
    data = np.concatenate(rows)
    hand_x = RAW_COLUMNS["hand_acc_x"]
    data[1000:1003, hand_x] = np.nan               # short dropout: interpolated
    if long_dropout:
        data[6000:6050, hand_x] = np.nan           # long dropout: splits the bout
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.6f")
    return path
