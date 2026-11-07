"""PulseNet ClientApp: runs on each Pi (or as a simulated node on the laptop)."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_framework.pulsenet_fl import task

app = ClientApp()


def _round(msg: Message) -> int:
    # The strategy puts the round number in the config it sends with each message.
    return int(msg.content["config"].get("server-round", 0))


def _load_global_model(msg: Message, context: Context) -> torch.nn.Module:
    model = task.build_model(context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    seed = int(cfg["seed"]) * 1000 + _round(msg)
    torch.manual_seed(seed)

    model = _load_global_model(msg, context)
    train_loader, _, n_train, _ = task.load_data(
        task.resolve_data_path(context), int(cfg["batch-size"]), seed
    )
    train_loss, train_time = task.train(
        model,
        train_loader,
        epochs=int(cfg["local-epochs"]),
        lr=float(cfg["learning-rate"]),
        mu=float(cfg["proximal-mu"]),
    )

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": train_loss,
            "train_time_s": train_time,
            "num-examples": n_train,
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    model = _load_global_model(msg, context)
    _, val_loader, _, n_val = task.load_data(
        task.resolve_data_path(context), int(cfg["batch-size"]), int(cfg["seed"])
    )
    val_mse, persistence_mse = task.evaluate(model, val_loader)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_mse": val_mse,
            "eval_persistence_mse": persistence_mse,
            "num-examples": n_val,
        }),
    })
    return Message(content=content, reply_to=msg)
