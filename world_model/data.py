"""Load the per-subject files written by scripts/prepare_pamap2.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TRAIN, VAL = 0, 1


def subject_path(processed_dir: Path, subject: int) -> Path:
    return Path(processed_dir) / f"subject_{subject}.npz"


def load_subject(processed_dir: Path, subject: int) -> dict[str, np.ndarray]:
    path = subject_path(processed_dir, subject)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_pamap2.py first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def split_arrays(data: dict, split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, activity) for one split."""
    m = data["split"] == split
    return data["x"][m], data["y"][m], data["activity"][m]


def load_split(processed_dir: Path, subjects: list[int], split: int) -> dict[int, tuple]:
    """{subject: (x, y, activity)} for the requested split."""
    return {s: split_arrays(load_subject(processed_dir, s), split) for s in subjects}


def pooled(parts: dict[int, tuple]) -> tuple[np.ndarray, np.ndarray]:
    xs = [p[0] for p in parts.values()]
    ys = [p[1] for p in parts.values()]
    return np.concatenate(xs), np.concatenate(ys)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)
