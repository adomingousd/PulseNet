"""B3: local-only TCNs - each subject trains its own model on only its own
data (no federation), evaluated on its own validation split. Federated
learning has to beat this to be worth doing.

    python benchmarks/b3_local_only.py --epochs 5      # quick check
    python benchmarks/b3_local_only.py                 # full run (100 epochs)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_SUBJECTS, SUMMARY_COLS, maybe_mlflow, new_run_dir, save_results, write_csv
from tcn_runner import add_persistence, fit, report_row
from world_model.data import TRAIN, VAL, load_split
from world_model.metrics import forecast_report
from world_model.tcn import TCNConfig
from world_model.train import predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--residual-to-last", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    tr = load_split(args.processed, args.subjects, TRAIN)
    va = load_split(args.processed, args.subjects, VAL)
    cfg = TCNConfig(residual_to_last=args.residual_to_last)

    run_dir = new_run_dir(args.results, "b3_local_only")
    rows, history = [], []
    for seed in args.seeds:
        preds, xs, ys = [], [], []
        for s in args.subjects:
            x_tr, y_tr, _ = tr[s]
            x_va, y_va, _ = va[s]
            params = {**vars(args), "seed": seed, "subject": s, **cfg.to_dict()}
            with maybe_mlflow(args.mlflow, "pulsenet-b3-local-only", f"s{s}-seed{seed}", params) as log:
                model, hist = fit(x_tr, y_tr, x_va, y_va, epochs=args.epochs, lr=args.lr,
                                  batch_size=args.batch_size, seed=seed, cfg=cfg, label=f"local {s}", log=log)
                history += hist
                row = report_row(model, x_va, y_va, model="local-only", seed=seed, eval=s)
                rows.append(row)
                log({f"final_{k}": v for k, v in row.items() if isinstance(v, float)})
            preds.append(predict(model, x_va))
            xs.append(x_va)
            ys.append(y_va)
        # pooled over subjects: each window scored by its own subject's model
        x_all, y_all = np.concatenate(xs), np.concatenate(ys)
        rows.append(add_persistence(forecast_report(np.concatenate(preds), y_all), x_all, y_all,
                                    model="local-only", seed=seed, eval="all"))

    write_csv(run_dir / "history.csv", history)
    alls = [r["mse"] for r in rows if r["eval"] == "all"]
    extra = f"Pooled validation MSE across seeds: mean {np.mean(alls):.4f}, std {np.std(alls):.4f} (n={len(alls)})"
    save_results(run_dir, vars(args), rows,
                 ["model", "seed", "eval", "num_examples", *SUMMARY_COLS, "persistence_mse", "ratio_vs_persistence"],
                 "B3 · Local-only TCNs (validation MSE, z-score units)", extra)


if __name__ == "__main__":
    main()
