"""Run the configured centralized benchmark and publish completed results."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4


def run_experiment(root, results, channels, subjects, epochs):
    python = os.environ.get("PULSENET_TRAIN_PYTHON", str(root / ".venv/bin/python"))
    script = root / "benchmarks/b2_centralized.py"
    results.mkdir(parents=True, exist_ok=True)
    with (results / ".dashboard-training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A dashboard experiment is already running.") from error

        with tempfile.TemporaryDirectory(dir=results) as temp_dir:
            staging = Path(temp_dir)
            config = {"input_channels": channels}
            config_path = staging / "experiment-inputs.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(
                value for value in (str(root), os.environ.get("PYTHONPATH")) if value
            ))
            command = [
                python, str(script), "--input-config", str(config_path),
                "--subjects", *subjects, "--epochs", str(epochs), "--seeds", "0",
                "--threads", "2", "--results", str(staging),
            ]
            completed = subprocess.run(command, cwd=root, env=env, text=True,
                                       capture_output=True, timeout=1800)
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip()
                                   or f"B2 exited with code {completed.returncode}")
            outputs = list(staging.glob("b2_centralized/*/results.csv"))
            if len(outputs) != 1:
                raise RuntimeError("B2 did not produce a single results.csv.")
            run_dir = outputs[0].parent
            (run_dir / "dashboard-inputs.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
            destination = results / "b2_centralized" / f"{run_dir.name}-{uuid4().hex[:8]}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(destination)
            return destination
