"""Temporal Convolutional Network (TCN) forecaster for PulseNet.

Spec (Week 2 worksheet, section 2C):
  input   [batch, 19 channels, 100 timesteps]   (2 s at 50 Hz)
  4 residual blocks, 32 channels, kernel 3, dilations 1, 2, 4, 8
  receptive field 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61 timesteps
  output  [batch, 19 channels, 10 timesteps]    (next 0.2 s)
  about 31K parameters (24,224 backbone + 6,270 head = 30,494)

The convolutions are written as Conv2d with (1, k) kernels on a [B, C, 1, T]
tensor, with explicit left (causal) zero padding. PyTorch trains this exactly
like a Conv1d, and it is the form the Hailo Dataflow Compiler handles most
reliably, so the ONNX export needs no rewrite later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn


@dataclass
class TCNConfig:
    in_channels: int = 19
    hidden: int = 32
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    horizon: int = 10
    window: int = 100
    # If True, the head predicts the change from the last observed value
    # (output = head + last input step) instead of absolute values.
    residual_to_last: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TCNConfig":
        d = dict(d)
        d["dilations"] = tuple(d.get("dilations", (1, 2, 4, 8)))
        return cls(**d)


class CausalConv(nn.Module):
    """Dilated causal convolution as Conv2d with a (1, k) kernel."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int):
        super().__init__()
        self.pad = nn.ZeroPad2d((dilation * (k - 1), 0, 0, 0))  # left pad only
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, k), dilation=(1, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    """Two causal convs with ReLU, plus a skip connection (1x1 conv if widths differ)."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv(c_in, c_out, k, dilation)
        self.conv2 = CausalConv(c_out, c_out, k, dilation)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.relu(h + self.skip(x))


class TCN(nn.Module):
    def __init__(self, cfg: TCNConfig | None = None):
        super().__init__()
        self.cfg = cfg or TCNConfig()
        c = self.cfg
        blocks = []
        c_in = c.in_channels
        for d in c.dilations:
            blocks.append(ResidualBlock(c_in, c.hidden, c.kernel_size, d))
            c_in = c.hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(c.hidden, c.in_channels * c.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, C, horizon]."""
        b = x.shape[0]
        h = self.backbone(x.unsqueeze(2))       # [B, hidden, 1, T]
        last = h[:, :, 0, -1]                   # features at the last time step
        out = self.head(last).reshape(b, self.cfg.in_channels, self.cfg.horizon)
        if self.cfg.residual_to_last:
            out = out + x[:, :, -1:]
        return out


def receptive_field(cfg: TCNConfig) -> int:
    """Timesteps that can influence one output step (two convs per block)."""
    return 1 + 2 * (cfg.kernel_size - 1) * sum(cfg.dilations)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_parameters(cfg: TCNConfig) -> int:
    """Closed-form parameter count, used by the tests to pin the architecture."""
    k, h, c = cfg.kernel_size, cfg.hidden, cfg.in_channels
    total, c_in = 0, c
    for _ in cfg.dilations:
        total += c_in * h * k + h          # conv1
        total += h * h * k + h             # conv2
        if c_in != h:
            total += c_in * h + h          # 1x1 skip
        c_in = h
    total += h * c * cfg.horizon + c * cfg.horizon  # head
    return total
