"""Pre-flight check for a PulseNet FL device (Pi, Jetson or laptop).

Run from the repo copy that contains fl_framework/ and world_model/, with the
FL venv active:

    python scripts/fl_smoke.py                                   # code + libraries
    python scripts/fl_smoke.py --data-dir DIR --subject 101      # also this Pi's data

Prints PASS/FAIL per check and exits 0 only if everything passed. With data,
it also times one local epoch, which is the main part of an FL round on a Pi.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(name, fn) -> bool:
    try:
        detail = fn()
        print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as e:  # report every failure, don't stop at the first
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, help="folder holding subject_<id>.npz")
    p.add_argument("--subject", type=int, help="subject id, e.g. 101")
    p.add_argument("--hidden", type=int, default=32, help="TCN hidden channels")
    a = p.parse_args()

    print(f"host={platform.node()} python={platform.python_version()} arch={platform.machine()}")
    ok = []

    ok.append(check("libraries", lambda: ", ".join(
        f"{pkg} {version(pkg)}" for pkg in ("flwr", "torch", "numpy"))))

    def flower_apps():
        import fl_framework.pulsenet_fl.client_app  # noqa: F401  (checks flwr API names too)
        import fl_framework.pulsenet_fl.server_app  # noqa: F401
    ok.append(check("Flower apps import", flower_apps))

    def model():
        import torch
        from world_model.tcn import TCN, TCNConfig, count_parameters, expected_parameters
        cfg = TCNConfig(hidden=a.hidden)
        m = TCN(cfg)
        n, want = count_parameters(m), expected_parameters(cfg)
        if n != want:
            raise ValueError(f"{n} params, expected {want}")
        out = m(torch.zeros(2, cfg.in_channels, cfg.window))
        if tuple(out.shape) != (2, cfg.in_channels, cfg.horizon):
            raise ValueError(f"output shape {tuple(out.shape)}")
        return f"{n:,} params, output {tuple(out.shape)}"
    ok.append(check("TCN build + forward", model))

    def train_step():
        import numpy as np
        from world_model import data, train
        from world_model.tcn import TCN, TCNConfig
        rng = np.random.default_rng(0)
        x = rng.standard_normal((128, 19, 100), dtype=np.float32)
        y = rng.standard_normal((128, 19, 10), dtype=np.float32)
        r = train.train(TCN(TCNConfig(hidden=a.hidden)), data.make_loader(x, y, 64, shuffle=True))
        return f"loss {r['train_loss']:.3f} on dummy data"
    ok.append(check("one training epoch (dummy data)", train_step))

    if a.data_dir is not None and a.subject is not None:
        def real_data():
            from world_model import data, train
            from world_model.tcn import TCN, TCNConfig
            d = data.load_subject(a.data_dir, a.subject)
            xt, yt, _ = data.split_arrays(d, data.TRAIN)
            xv, _, _ = data.split_arrays(d, data.VAL)
            if len(xt) == 0 or len(xv) == 0:
                raise ValueError(f"train={len(xt)} val={len(xv)} windows")
            t0 = time.perf_counter()
            train.train(TCN(TCNConfig(hidden=a.hidden)), data.make_loader(xt, yt, 64, shuffle=True))
            return f"subject {a.subject}: {len(xt)} train / {len(xv)} val windows, 1 epoch {time.perf_counter() - t0:.1f}s"
        ok.append(check("subject data + timed epoch", real_data))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
