"""Shared helpers for the Phase 1 benchmarks: result folders, tables, MLflow."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_SUBJECTS = list(range(101, 109))
SUMMARY_COLS = ["mse", "mse_h1", "mse_h5", "mse_h10", "mse_acc", "mse_gyro", "mse_hr"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "torch", "flwr"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def new_run_dir(results_root: Path, name: str) -> Path:
    d = Path(results_root) / name / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def save_results(run_dir: Path, config: dict, rows: list[dict], table_cols: list[str],
                 title: str, extra_md: str = "") -> None:
    config = {**config, "git_commit": git_commit(), "versions": versions()}
    write_json(run_dir / "config.json", config)
    write_csv(run_dir / "results.csv", rows)
    md = f"# {title}\n\nRun: {run_dir.name} · commit {config['git_commit']}\n\n"
    md += markdown_table(rows, table_cols) + "\n"
    if extra_md:
        md += "\n" + extra_md + "\n"
    (run_dir / "summary.md").write_text(md)
    print("\n" + markdown_table(rows, table_cols))
    print(f"\nSaved to {run_dir}")


@contextmanager
def maybe_mlflow(enabled: bool, experiment: str, run_name: str, params: dict):
    """Yield a log(metrics, step) function; a no-op unless --mlflow is passed."""
    if not enabled:
        yield lambda metrics, step=None: None
        return
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.set_tags({"git_commit": git_commit(), **{f"v_{k}": v for k, v in versions().items()}})

        def log(metrics: dict, step: int | None = None):
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(clean, step=step)

        yield log
