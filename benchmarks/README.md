# Phase 1 Benchmarks

Run from the repo root with the venv active. Each run writes to `results/<benchmark>/<timestamp>/`: `summary.md` (readable table), `results.csv`, `config.json` (settings, git commit, package versions) and, for B2/B3, `history.csv` (per-epoch loss) and model checkpoints.

| ID | Script | Question it answers |
|---|---|---|
| B0 | `b0_data_sanity.py` | Are the processed files sane: counts, activities, no NaNs, no train/val overlap? |
| B1 | `b1_naive_baselines.py` | How good is "predict the mean" (zero) and "repeat the last value" (persistence)? |
| B2 | `b2_centralized.py` | Best case: one TCN trained on everyone's pooled data |
| B3 | `b3_local_only.py` | Each subject trains alone. FL has to beat this. |

All errors are MSE in z-score units, on each subject's validation split. `ratio_vs_persistence` below 1.0 means the model beats repeating the last value.

Common options: `--epochs`, `--seeds 0 1 2`, `--subjects 101 102 …`, `--residual-to-last` (predict the change from the last value), `--threads N`, `--mlflow`.
