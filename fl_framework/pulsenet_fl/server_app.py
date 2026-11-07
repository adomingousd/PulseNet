"""PulseNet ServerApp: runs on the Jetson (or the managed local SuperLink in simulation)."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_framework.pulsenet_fl import task

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    torch.manual_seed(int(cfg["seed"]))
    model = task.build_model(cfg)
    print(f"[pulsenet] TCN hidden={cfg['hidden-channels']} params={task.count_params(model):,}")

    min_nodes = int(cfg["min-nodes"])
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        num_rounds=int(cfg["num-server-rounds"]),
        timeout=float(cfg["round-timeout-s"]),
    )

    out = str(cfg.get("save-final-model", ""))
    if out:
        torch.save(result.arrays.to_torch_state_dict(), out)
        print(f"[pulsenet] final global model saved to {out}")
