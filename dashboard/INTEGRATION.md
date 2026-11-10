# What to provide for live integration

## 1. One recent BLE packet (later; not a dashboard prerequisite)

On one Pi's terminal, run these read-only commands:

```bash
hostname
sudo cat /etc/pulsenet/node.toml
sudo sh -c 'tail -n 2 /var/lib/pulsenet/PN-*.jsonl'
systemctl status pulsenet-receiver --no-pager
```

Paste the output or save it in a text file. The last command may return a nonzero code when inactive. No files means no saved packets at that location; report that fact. The tail command may print two lines from each daily file; provide one recent record with its filename.

The record's `hex` is received bytes, not automatically sensor values. Provide the firmware source file currently flashed to the board, its branch/commit, and any packet specification. We need channel order, types, units, byte order, sampling interval, timestamps and sequence numbering. Also provide a sample/header of the manually loaded Pi training dataset; that is the CURRENT training source. Raw PAMAP2 .dat, cleaned CSV and processed .npz have different formats.

## 2. FL metrics (ask Aiden)

Provide the exact branch and commit containing the current TCN FL implementation, plus:

- Client train/evaluate functions and server aggregation/evaluation code.
- One completed run's config.json, results.csv and history.csv when available, or a short log showing one round. Do not fabricate files that aren't produced.
- Example per-client reply: node ID, run ID, round number, model version, train/validation example counts, metric names and definitions, training duration and outcome.
- Whether error is training or held-out validation error, standardized or physical units, and how overall error is aggregated.

If this implementation is unpushed, attach the relevant files. No entire PAMAP2 dataset, credentials or model checkpoint is needed merely to agree on telemetry.

## 3. Pi command handling (ask Aiden)

Does any code already receive requests to evaluate, start/stop training or include/exclude a client? If so, provide its file and interface. If none exists, say "not implemented"; that is normal and we can define it together.

Agree on initial supported actions, which component owns them, how a node is identified, and how accepted/rejected/running/completed/failed results are returned. Do not execute arbitrary shell text from the dashboard. FL selection is normally owned by the coordinator, not the receiver. Define whether exclusion affects the next round or an active round; do not assume it stops current training.

## Proposed telemetry agreement (draft, not an implemented API)

Each message needs schema version, node ID, boot ID, increasing report sequence, source timestamp and run/round/model identifiers when applicable. Collector records arrival time. Use collector monotonic elapsed time for heartbeat timeout so node clock jumps do not cause false disconnections. Expose stale/unknown states explicitly. Keep link availability, BLE reception, training activity and FL participation separate. Source timestamps align historical data; also report synchronization state and measured clock offset when available.

Optional fields: CPU percent, memory percent, sustained-high-memory duration, last packet time/count, dataset subject and window counts, train loss, validation MSE, per-channel-group MSE, training time. Missing metrics are unknown, not zero. Do not send raw PAMAP2 records merely for status monitoring.

CSV experiments require compatible channel mapping, sample rate, units, training-only normalization statistics and checkpoint configuration. Evaluate an existing model without changing weights; retrain a separate copy and compare on the same held-out data. Keep these actions separate and explicit.
