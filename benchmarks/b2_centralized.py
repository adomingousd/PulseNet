"""B2: centralized TCN - one model trained on the pooled training data of all
subjects (no federation, no DP), evaluated on every subject's validation split.
This is the best case federated training is compared against.

    python benchmarks/b2_centralized.py --epochs 5      # quick check
    python benchmarks/b2_centralized.py                 # full run (100 epochs)
    python benchmarks/b2_centralized.py --seeds 0 1 2 --mlflow
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_SUBJECTS, SUMMARY_COLS, maybe_mlflow, new_run_dir, save_results, write_csv
from tcn_runner import add_persistence, fit, save_checkpoint
from world_model.data import TRAIN, VAL, load_split, pooled
from world_model.tcn import TCNConfig
from world_model.metrics import CHANNELS, forecast_report
from world_model.train import predict

def read_input_channels(path: Path) -> list[str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    channels = config.get("input_channels") if isinstance(config, dict) else None

    if not isinstance(channels, list) or not channels:
        raise ValueError("input_channels must be a nonempty list")
    if any(not isinstance(name, str) or name not in CHANNELS for name in channels):
        raise ValueError("input_channels contains an unknown channel name")
    if len(channels) != len(set(channels)):
        raise ValueError("input_channels contains duplicate names")

    return [name for name in CHANNELS if name in channels]


def evaluation_row(net, x, y, input_indices, **ids):
    forecast = predict(net, x[:, input_indices, :])
    return add_persistence(forecast_report(forecast, y), x, y, **ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--epochs", type=int, default=100, help="100 matches 100 FL rounds x 1 local epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--residual-to-last", action="store_true", help="predict the change from the last value")
    p.add_argument("--hidden", type=int, default=32, help="TCN channels per layer (spec: 32)") # optionally try wider model
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = PyTorch default)")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--mlflow", action="store_true", help="also log to MLflow")
    p.add_argument(
        "--input-config",
        type=Path,
        help="dashboard JSON listing sensor inputs to keep",
    )
    args = p.parse_args()
    input_channels = CHANNELS
    if args.input_config is not None:
        try:
            input_channels = read_input_channels(args.input_config)
        except (OSError, ValueError) as exc:
            p.error(f"--input-config: {exc}")

    if args.residual_to_last and input_channels != CHANNELS:
        p.error("--residual-to-last requires all channels as inputs")

    input_indices = [CHANNELS.index(name) for name in input_channels]
    if args.threads:
        torch.set_num_threads(args.threads)

    tr = load_split(args.processed, args.subjects, TRAIN)
    va = load_split(args.processed, args.subjects, VAL)
    x_tr, y_tr = pooled(tr)
    x_va, y_va = pooled(va)
    cfg = TCNConfig(
        hidden=args.hidden,
        residual_to_last=args.residual_to_last,
        in_channels=len(input_channels),
        out_channels=len(CHANNELS),
    )
    x_tr_inputs = x_tr[:, input_indices, :]
    x_va_inputs = x_va[:, input_indices, :]
    print(f"Using {len(input_channels)} inputs: {', '.join(input_channels)}")
    print(f"Forecasting {len(CHANNELS)} channels")
    print(f"B2 centralized: {len(x_tr)} train / {len(x_va)} val windows from {len(args.subjects)} subjects")

    run_dir = new_run_dir(args.results, "b2_centralized")
    rows, history = [], []
    for seed in args.seeds:
        params = {**vars(args), "seed": seed, **cfg.to_dict()}
        with maybe_mlflow(args.mlflow, "pulsenet-b2-centralized", f"seed{seed}", params) as log:
            model, hist = fit(x_tr_inputs, y_tr, x_va_inputs, y_va, epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch_size, seed=seed, cfg=cfg, label="central", log=log)
            history += hist
            for s, (x, y, _) in va.items():
                rows.append(evaluation_row(model, x, y, input_indices, model="centralized", seed=seed, eval=s))
            final = evaluation_row(model, x_va, y_va, input_indices, model="centralized", seed=seed, eval="all")
            rows.append(final)
            log({f"final_{k}": v for k, v in final.items() if isinstance(v, float)})
        save_checkpoint(model, run_dir / f"model_seed{seed}.pt", trained_on=args.subjects,
                        epochs=args.epochs, seed=seed)

    write_csv(run_dir / "history.csv", history)
    alls = [r["mse"] for r in rows if r["eval"] == "all"]
    extra = f"Pooled validation MSE across seeds: mean {np.mean(alls):.4f}, std {np.std(alls):.4f} (n={len(alls)})"
    save_results(run_dir, {**vars(args), "input_channels": input_channels, "forecast_channels": CHANNELS}, rows,
                 ["model", "seed", "eval", "num_examples", *SUMMARY_COLS, "persistence_mse", "ratio_vs_persistence"],
                 "B2 · Centralized TCN (validation MSE, z-score units)", extra)


if __name__ == "__main__":
    main()
