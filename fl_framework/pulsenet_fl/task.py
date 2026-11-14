"""Connect Flower's config and world_model/.
 
world_model/ holds the ML code (model, data, training, metrics) and knows
nothing about Flower. This file turns Flower's run/node config into the two
things the apps need: a model and this node's data location.
"""
 
from __future__ import annotations
 
from pathlib import Path
 
from world_model.tcn import TCN, TCNConfig
 
 
def build_model(run_config) -> TCN:
    """Build the TCN from the run config."""
    return TCN(TCNConfig(hidden=int(run_config["hidden-channels"])))
 
 
def data_location(context) -> tuple[Path, int]:
    """Return (data_dir, subject) for this node, with no hardcoded paths.
 
    Deployment: each Pi's SuperNode is started with
        --node-config "data-dir='/path/to/processed' subject-id=101"
    Simulation: Flower numbers the virtual nodes 0..N-1 (partition-id);
        node i gets the i-th subject in sim-subjects, read from sim-data-dir.
    """
    node, run = context.node_config, context.run_config
 
    if "subject-id" in node:
        return Path(str(node["data-dir"])).expanduser(), int(node["subject-id"])
 
    if not run["sim-data-dir"]:
        raise ValueError(
            "Simulation needs --run-config \"sim-data-dir='/abs/path/to/data/processed'\""
        )
    subjects = [int(s) for s in str(run["sim-subjects"]).split(",")]
    if int(node["num-partitions"]) != len(subjects):
        raise ValueError(
            f"{node['num-partitions']} simulated nodes but {len(subjects)} subjects in sim-subjects"
        )
    return Path(str(run["sim-data-dir"])).expanduser(), subjects[int(node["partition-id"])]
