"""B1: naive forecast baselines on each subject's validation split.

  zero         predict the channel mean (0 after z-scoring) -> MSE about 1.0
  persistence  repeat the last observed value for all 10 steps

Every learned model must beat persistence to be worth anything.

    python benchmarks/b1_naive_baselines.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_SUBJECTS, SUMMARY_COLS, new_run_dir, save_results
from world_model.data import VAL, load_split, pooled
from world_model.metrics import forecast_report, persistence_forecast, zero_forecast

BASELINES = {"zero": zero_forecast, "persistence": persistence_forecast}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()

    val = load_split(args.processed, args.subjects, VAL)
    rows = []
    for name, fn in BASELINES.items():
        for s, (x, y, _) in val.items():
            rows.append({"baseline": name, "eval": s, **forecast_report(fn(x, y.shape[2]), y)})
        x, y = pooled(val)
        rows.append({"baseline": name, "eval": "all", **forecast_report(fn(x, y.shape[2]), y)})

    run_dir = new_run_dir(args.results, "b1_naive_baselines")
    save_results(run_dir, vars(args), rows, ["baseline", "eval", "num_examples", *SUMMARY_COLS],
                 "B1 · Naive baselines (validation MSE, z-score units)")


if __name__ == "__main__":
    main()
