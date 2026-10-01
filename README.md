# PulseNet

**A privacy-preserving federated learning platform for wearable health sensing.**

COMP 491/492 Senior Capstone, University of San Diego, Shiley-Marcos School of Engineering
Faculty Sponsor: Dr. Nikhil Yadav, Edge & Quantum AI Research Lab

---

## Overview

PulseNet is a federated learning (FL) testbed built on commodity edge hardware. Wearable-class boards replay synthetic physiological signals over BLE to Raspberry Pi 5 nodes, which each train a lightweight temporal world model locally. Only model updates leave each node. Raw data never does. An NVIDIA Jetson AGX Orin aggregates the updates into a global model.

All data is synthetic, derived from public benchmark datasets, so no human subjects are involved and no IRB approval is required.

- **Semester 1 (Build):** End-to-end sensing pipeline, FedAvg baseline, differential privacy layer, and a remote monitoring dashboard.
- **Semester 2 (Research):** Byzantine robustness (Krum, FLTrust, coordinate-wise median) and personalized FL (FedPer).

## Hardware

| Device | Qty | Role |
|---|---|---|
| Raspberry Pi 5 (8GB) | 8 | FL client nodes |
| Hailo-8 AI HAT+ | 8 | NPU accelerator on each Pi 5 |
| Seeed XIAO nRF52840 | 5 | BLE wearable sensing nodes |
| ESP32-S3 | 5 | BLE-to-WiFi bridge tier |
| NVIDIA Jetson AGX Orin 64GB | 1 | Headless aggregation server (SSH only) |
| 2.5GbE managed switch | 1 | Connects the Pi cluster to the campus network |

## Architecture Decisions (Semester 1)

| Area | Choice |
|---|---|
| FL framework | Flower (with adaptations for Hailo-8) |
| World model | Temporal Convolutional Network (TCN) |
| Dataset | PAMAP2, partitioned by subject for non-IID splits |
| Aggregation baseline | FedAvg |
| Privacy layer | Differential privacy |
| Dashboard | Streamlit |

Full reasoning is documented in `docs/`.

## Repository Structure

```
pulsenet/
├── sensing/        # nRF52840 / ESP32-S3 firmware and BLE pipeline
├── fl_framework/   # Flower client/server and aggregation logic
├── world_model/    # TCN model, training, and Hailo-8 compilation
├── dashboard/      # Streamlit monitoring UI
└── docs/           # Architecture decisions, setup guide, experiment notes
```

## Team

| Member | Role |
|---|---|
| Gable Krich | Tech Lead, Sensing Pipeline |
| Aiden Domingo | FL Framework |
| Santiago Guerrero | Dashboard |
| Penelope Yanez | Research & Benchmarking |
| Team | World Model |

## Contributing

- Work happens on the `testing` branch.
- Merges into `main` require a pull request reviewed by at least one other team member.
- Tasks are tracked in Jira (Epic → Story → Task).
- A task is "done" when it is solved cleanly and either reviewed or explicitly agreed to be review-exempt.

## Status

Week 1: Discovery and architecture planning. No implementation yet.
