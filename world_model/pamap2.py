"""PAMAP2 -> per-subject training files for the PulseNet TCN.

Stand-in for the team's cleaning pipeline, so Phase 1 can run now. It follows
the Week 2 spec and writes one file per subject that matches
docs/data-contract.md (draft below in the module docstring of
scripts/prepare_pamap2.py).

Pipeline per subject (protocol recording):
  1. Read the 19 channels: hand/chest/ankle +-16g accelerometer and gyroscope
     (x, y, z each) plus heart rate.
  2. Heart rate (~9 Hz) is linearly interpolated over time to every sample.
  3. Split into bouts: runs of one non-zero activity with no timestamp gap.
     Activity 0 (transient) rows are dropped.
  4. Inside a bout, IMU gaps of up to 10 samples (0.1 s) are interpolated;
     longer gaps split the bout into separate segments.
  5. Downsample 100 Hz -> 50 Hz by averaging pairs of samples.
  6. Split each segment in time: first ~80% train, then a 110-step gap, then
     validation. Segments too short for both go entirely to train.
  7. Cut samples of 110 steps (100 input + 10 target) every 50 steps.
  8. Standardize every channel with this subject's own training-split mean
     and std (no statistics are shared between subjects).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import CHANNELS

RAW_FS = 100
FS = 50
INPUT_LEN = 100
HORIZON = 10
SAMPLE_LEN = INPUT_LEN + HORIZON          # 110 steps = 2.2 s
STRIDE = 50
GAP = SAMPLE_LEN                          # steps left unused between train and val
VAL_FRAC = 0.2
IMU_GAP_LIMIT = 10                        # raw samples (0.1 s) interpolated inside a bout

# 0-based column index in the 54-column .dat files
_IMU_BASE = {"hand": 3, "chest": 20, "ankle": 37}
_ACC16 = (1, 2, 3)     # offsets inside a 17-column IMU block
_GYRO = (7, 8, 9)

RAW_COLUMNS: dict[str, int] = {"timestamp": 0, "activity": 1}
for _loc, _base in _IMU_BASE.items():
    for _axis, _off in zip("xyz", _ACC16):
        RAW_COLUMNS[f"{_loc}_acc_{_axis}"] = _base + _off
    for _axis, _off in zip("xyz", _GYRO):
        RAW_COLUMNS[f"{_loc}_gyro_{_axis}"] = _base + _off
RAW_COLUMNS["heart_rate"] = 2

ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work",
    11: "car_driving", 12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
    19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping",
}

IMU_CHANNELS = [c for c in CHANNELS if c != "heart_rate"]


def find_raw_file(raw_root: Path, subject: int) -> Path:
    """Locate subject<id>.dat, preferring the Protocol folder."""
    direct = raw_root / f"subject{subject}.dat"
    if direct.exists():
        return direct
    hits = sorted(raw_root.rglob(f"subject{subject}.dat"))
    protocol = [p for p in hits if "protocol" in str(p).lower()]
    if protocol:
        return protocol[0]
    if hits:
        return hits[0]
    raise FileNotFoundError(f"subject{subject}.dat not found under {raw_root}")


def read_subject(path: Path) -> pd.DataFrame:
    cols = sorted(RAW_COLUMNS.values())
    name_by_col = {v: k for k, v in RAW_COLUMNS.items()}
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=cols, dtype=np.float64)
    df = df.rename(columns=name_by_col)
    df = df[["timestamp", "activity", *CHANNELS]]
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _segments(df: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
    """Return (activity, raw-rate array [T, 19]) for every clean segment."""
    df = df.copy()
    df["heart_rate"] = df["heart_rate"].interpolate(method="linear", limit_direction="both")

    act = df["activity"].to_numpy()
    ts = df["timestamp"].to_numpy()
    gap = np.diff(ts, prepend=ts[0]) > 1.5 / RAW_FS
    new_bout = (act != np.roll(act, 1)) | gap
    new_bout[0] = True
    bout_id = np.cumsum(new_bout)

    out: list[tuple[int, np.ndarray]] = []
    for _, bout in df.groupby(bout_id, sort=True):
        a = int(bout["activity"].iloc[0])
        if a == 0:
            continue
        vals = bout[CHANNELS].copy()
        vals[IMU_CHANNELS] = vals[IMU_CHANNELS].interpolate(
            method="linear", limit=IMU_GAP_LIMIT, limit_area="inside"
        )
        arr = vals.to_numpy()
        ok = ~np.isnan(arr).any(axis=1)
        # split on any remaining NaN rows
        edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
        for s, e in zip(edges[::2], edges[1::2]):
            out.append((a, arr[s:e]))
    return out


def _downsample(arr: np.ndarray) -> np.ndarray:
    n = (len(arr) // 2) * 2
    return arr[:n].reshape(-1, 2, arr.shape[1]).mean(axis=1)


def _split_bounds(n: int) -> tuple[int, int | None]:
    """Return (train_end, val_start); val_start is None if the segment is train-only."""
    train_end = min(int((1 - VAL_FRAC) * n), n - GAP - SAMPLE_LEN)
    if train_end < SAMPLE_LEN:
        return n, None
    return train_end, train_end + GAP


def _starts(r0: int, r1: int) -> range:
    return range(r0, r1 - SAMPLE_LEN + 1, STRIDE)


def build_subject(df: pd.DataFrame, subject: int) -> dict:
    segs = []
    for a, raw in _segments(df):
        seg = _downsample(raw)
        if len(seg) >= SAMPLE_LEN:
            segs.append((a, seg))
    if not segs:
        raise ValueError(f"subject {subject}: no usable segments")

    # statistics from training regions only
    train_rows = [seg[: _split_bounds(len(seg))[0]] for _, seg in segs]
    stacked = np.concatenate(train_rows, axis=0)
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), 1e-6)

    xs, acts, splits, seg_ids, starts = [], [], [], [], []
    for sid, (a, seg) in enumerate(segs):
        z = (seg - mean) / std
        train_end, val_start = _split_bounds(len(seg))
        regions = [(0, train_end, 0)]
        if val_start is not None:
            regions.append((val_start, len(seg), 1))
        for r0, r1, split in regions:
            for s in _starts(r0, r1):
                xs.append(z[s : s + SAMPLE_LEN].T)       # [19, 110]
                acts.append(a)
                splits.append(split)
                seg_ids.append(sid)
                starts.append(s)

    w = np.stack(xs).astype(np.float32)                 # [N, 19, 110]
    return {
        "x": w[:, :, :INPUT_LEN],
        "y": w[:, :, INPUT_LEN:],
        "activity": np.asarray(acts, dtype=np.int16),
        "split": np.asarray(splits, dtype=np.int8),     # 0 train, 1 val
        "segment": np.asarray(seg_ids, dtype=np.int32),
        "start": np.asarray(starts, dtype=np.int32),    # step index inside its segment
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "channels": np.asarray(CHANNELS),
        "subject": np.int32(subject),
        "fs": np.int32(FS),
        "input_len": np.int32(INPUT_LEN),
        "horizon": np.int32(HORIZON),
        "stride": np.int32(STRIDE),
        "gap": np.int32(GAP),
    }


def save_subject(data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"subject_{int(data['subject'])}.npz"
    np.savez_compressed(path, **data)
    return path


def summarize(data: dict) -> dict:
    split = data["split"]
    acts = data["activity"]
    return {
        "subject": int(data["subject"]),
        "train_samples": int((split == 0).sum()),
        "val_samples": int((split == 1).sum()),
        "activities_train": sorted({int(a) for a in acts[split == 0]}),
        "activities_val": sorted({int(a) for a in acts[split == 1]}),
    }


def write_summary(summaries: list[dict], out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
