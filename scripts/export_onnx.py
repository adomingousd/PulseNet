"""Export the TCN to ONNX with fixed shapes, for the Hailo Dataflow Compiler.

The Dataflow Compiler runs on x86 Linux, not on the Pi or a Mac, so this only
produces the files to hand to whoever does the compile:
  tcn.onnx            batch 1, input [1, 19, 100] -> output [1, 19, 10]
  calibration.npy     real training windows [N, 19, 100] for INT8 calibration

    python scripts/export_onnx.py                                   # untrained model
    python scripts/export_onnx.py --checkpoint results/b2_centralized/<run>/model_seed0.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from world_model.data import TRAIN, load_split, pooled  # noqa: E402
from world_model.tcn import TCN, TCNConfig, count_parameters  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, help="model .pt from B2/B3 (default: untrained TCN)")
    p.add_argument("--out-dir", type=Path, default=Path("results/onnx"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--calibration-samples", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        from tcn_runner import load_checkpoint
        model = load_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(0)
        model = TCN(TCNConfig()).eval()
    cfg = model.cfg
    dummy = torch.zeros(1, cfg.in_channels, cfg.window)
    path = args.out_dir / "tcn.onnx"
    try:
        torch.onnx.export(model, (dummy,), str(path), input_names=["window"], output_names=["forecast"],
                          opset_version=args.opset)
    except ImportError as e:
        raise SystemExit(f"{e}\nInstall the exporter's dependencies: pip install onnx onnxscript")
    print(f"Wrote {path}  ({count_parameters(model)} parameters, input [1, {cfg.in_channels}, {cfg.window}])")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(path)))
        print("ONNX checker: OK")
    except ImportError:
        print("onnx not installed; skipped the model check (pip install onnx)")

    try:
        import onnxruntime as ort
        x = np.random.default_rng(0).standard_normal((1, cfg.in_channels, cfg.window)).astype(np.float32)
        out = ort.InferenceSession(str(path)).run(None, {"window": x})[0]
        ref = model(torch.from_numpy(x)).detach().numpy()
        print(f"onnxruntime vs PyTorch max abs diff: {np.abs(out - ref).max():.2e}")
    except ImportError:
        pass

    try:
        subjects = list(range(101, 109))
        x, _ = pooled(load_split(args.processed, subjects, TRAIN))
        idx = np.random.default_rng(0).choice(len(x), size=min(args.calibration_samples, len(x)), replace=False)
        np.save(args.out_dir / "calibration.npy", x[idx])
        print(f"Wrote {args.out_dir / 'calibration.npy'}  shape {x[idx].shape}")
    except FileNotFoundError:
        print("No processed data found; skipped calibration.npy")


if __name__ == "__main__":
    main()
