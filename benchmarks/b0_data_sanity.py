"""B0: data sanity checks on data/processed/subject_*.npz.

Checks per subject: sample counts, activities in each split, NaNs, that
validation windows never overlap training windows (and keep the required
gap), and that the training split is standardized.

    python benchmarks/b0_data_sanity.py

*** Benchmark py files composed by Claude and reviewed by Aiden Domingo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_SUBJECTS, markdown_table, new_run_dir, save_results
from world_model.data import TRAIN, VAL, load_subject
from world_model.pamap2 import ACTIVITIES


def check_subject(d: dict) -> tuple[dict, list[str]]:
    problems = []
    split = d["split"]
    sample_len = int(d["input_len"]) + int(d["horizon"])
    if np.isnan(d["x"]).any() or np.isnan(d["y"]).any():
        problems.append("NaN values present")
    min_gap = None
    for seg in np.unique(d["segment"]):
        m = d["segment"] == seg
        tr = d["start"][m & (split == TRAIN)]
        va = d["start"][m & (split == VAL)]
        if len(tr) and len(va):
            g = int(va.min() - (tr.max() + sample_len))
            min_gap = g if min_gap is None else min(min_gap, g)
    if min_gap is not None and min_gap < int(d["gap"]):
        problems.append(f"train/val gap {min_gap} < required {int(d['gap'])}")
    xt = d["x"][split == TRAIN]
    mean_dev = float(np.abs(xt.mean(axis=(0, 2))).max())
    std_dev = float(np.abs(xt.std(axis=(0, 2)) - 1).max())
    if mean_dev > 0.25 or std_dev > 0.25:
        problems.append(f"train split not standardized (mean dev {mean_dev:.2f}, std dev {std_dev:.2f})")
    acts_tr = sorted({int(a) for a in d["activity"][split == TRAIN]})
    acts_va = sorted({int(a) for a in d["activity"][split == VAL]})
    missing = [ACTIVITIES.get(a, str(a)) for a in acts_tr if a not in acts_va]
    row = {
        "subject": int(d["subject"]),
        "train": int((split == TRAIN).sum()),
        "val": int((split == VAL).sum()),
        "val_pct": f"{100 * float((split == VAL).mean()):.1f}",
        "activities": len(acts_tr),
        "acts_without_val": len(missing),
        "segments": int(len(np.unique(d["segment"]))),
        "min_gap_steps": min_gap if min_gap is not None else "",
        "status": "ok" if not problems else "; ".join(problems),
    }
    return row, missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()

    rows, notes = [], []
    for s in args.subjects:
        row, missing = check_subject(load_subject(args.processed, s))
        rows.append(row)
        if missing:
            notes.append(f"- Subject {s}: no validation windows for {', '.join(missing)} (bout too short to split)")

    acts = {}
    for s in args.subjects:
        d = load_subject(args.processed, s)
        for a, n in zip(*np.unique(d["activity"], return_counts=True)):
            acts.setdefault(int(a), {})[s] = int(n)
    act_rows = [{"activity": ACTIVITIES.get(a, a), **{str(s): acts[a].get(s, 0) for s in args.subjects}}
                for a in sorted(acts)]
    extra = "## Windows per activity and subject\n\n" + markdown_table(act_rows, ["activity", *map(str, args.subjects)])
    if notes:
        extra += "\n\n## Notes\n\n" + "\n".join(notes)

    run_dir = new_run_dir(args.results, "b0_data_sanity")
    cols = ["subject", "train", "val", "val_pct", "activities", "acts_without_val", "segments", "min_gap_steps", "status"]
    save_results(run_dir, vars(args), rows, cols, "B0 · Data sanity", extra)
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit("B0 found problems; see the status column")


if __name__ == "__main__":
    main()
