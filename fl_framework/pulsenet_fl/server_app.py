"""PulseNet ServerApp: runs on the Jetson.
 
Sends the global model to the Pis, averages their updates weighted by sample
count (FedAvg), and repeats for num-server-rounds.
"""
 
from __future__ import annotations
 
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
 
from world_model.tcn import count_parameters
from world_model.train import set_seed
 
from fl_framework.pulsenet_fl.task import build_model
 
app = ServerApp()
 
 
@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
 
    set_seed(int(cfg["seed"]))
    model = build_model(cfg)
    print(f"[pulsenet] TCN hidden={cfg['hidden-channels']} params={count_parameters(model):,}")
 
    n = int(cfg["min-nodes"])
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=n,
        min_evaluate_nodes=n,
        min_available_nodes=n,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        num_rounds=int(cfg["num-server-rounds"]),
        timeout=float(cfg["round-timeout-s"]),
    )
 
    if cfg["save-final-model"]:
        torch.save(result.arrays.to_torch_state_dict(), str(cfg["save-final-model"]))
        print(f"[pulsenet] final global model saved to {cfg['save-final-model']}")