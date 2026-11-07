"""PulseNet ClientApp: runs on each Pi (or as a simulated node on the laptop).

Each round the Jetson sends the current global model. This node either trains
it on its own subject's data (train) or scores it on its validation windows
(evaluate). Only weights and metrics are sent back; raw data stays here.
"""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from world_model import data, metrics
from world_model import train as training

from fl_framework.pulsenet_fl.task import build_model, data_location

app = ClientApp()


def _global_model(msg: Message, context: Context):
    model = build_model(context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


def _load(context: Context, split: int):
    """(x, y, activity) for this node's subject and the given split."""
    data_dir, subject = data_location(context)
    return data.split_arrays(data.load_subject(data_dir, subject), split)


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    seed = int(cfg["seed"]) * 1000 + int(msg.content["config"].get("server-round", 0))
    training.set_seed(seed)

    model = _global_model(msg, context)
    x, y, _ = _load(context, data.TRAIN)
    loader = data.make_loader(x, y, int(cfg["batch-size"]), shuffle=True, seed=seed)
    result = training.train(
        model,
        loader,
        epochs=int(cfg["local-epochs"]),
        lr=float(cfg["learning-rate"]),
        proximal_mu=float(cfg["proximal-mu"]),
    )

    reply = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": result["train_loss"],
            "train_time_s": result["train_time_s"],
            "num-examples": result["num_examples"],
        }),
    })
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    model = _global_model(msg, context)
    x, y, _ = _load(context, data.VAL)

    report = training.evaluate(model, x, y)
    persistence = metrics.forecast_report(metrics.persistence_forecast(x, y.shape[2]), y)

    values = {k: v for k, v in report.items() if k != "num_examples"}
    values["persistence_mse"] = persistence["mse"]
    values["num-examples"] = report["num_examples"]
    return Message(content=RecordDict({"metrics": MetricRecord(values)}), reply_to=msg)