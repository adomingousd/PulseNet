# Development Environment

Every machine that runs PulseNet code uses the pinned versions below. Flower must match exactly across the laptop, the Jetson (SuperLink) and the Pis (SuperNodes), and the Python minor version matches the Pis (3.13).

Last updated: 2026-10-23 by Aiden

## Pinned versions

| Package | Version |
|---|---|
| Python | 3.13 |
| Flower (flwr) | 1.42.0 |
| PyTorch | 2.14.1 |

## Machines

| Machine | OS | Python | Status |
|---|---|---|---|
| Aiden's laptop | macOS | 3.13 | Set up: flwr 1.42.0, torch 2.14.1 in `.venv` |
| Raspberry Pi 5 (pi-01 … pi-08) | Raspberry Pi OS Lite (64-bit) | 3.13.5 | Pinned packages to be installed from an offline wheelhouse |
| pulsenet-jetson | Ubuntu 24.04 (JetPack 7.2.1) | — | Waiting on python3-venv and git (offline bundle #2) |
