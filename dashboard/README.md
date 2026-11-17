# PulseNet dashboard starter

This is a local, demo-only Streamlit interface. It contains eight Pi cards, node filters, example forecast metrics, CSV inspection, configurable heartbeat/memory/clock alerts and acknowledgement history. Controls and model experiments are visibly disabled pending integration. No hardware connection, training, medical-event detector, or real model evaluation is implemented.

## Install in your repository

Copy this entire `dashboard` directory into `PulseNet-transfer` on branch `santi`. Do not replace the repository's root requirements.txt.

From the repository root in VS Code:

```bash
source .venv-dashboard/bin/activate
python -m pip install --no-cache-dir -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open http://localhost:8501. Stop with Ctrl+C. No Jetson, PAMAP2 download or PyTorch is needed.

Add this line to the repository root `.gitignore` if absent:

```text
.venv-dashboard/
```

To upload using GitHub's website, extract the ZIP and upload the **dashboard folder**, not the ZIP, on your `santi` branch. To run locally, also copy that folder into your local repository (a website upload does not update your laptop automatically).

## Expected behavior

Demo nodes are deterministic, not live simulation. Refresh redraws the same values. pi-08 is unresponsive, pi-04 has a BLE disconnect but can still use its local data, pi-06 triggers sustained high memory, and pi-07 reports unsynchronized time. Defaults: heartbeat timeout 30 seconds, memory above 85% for 60 seconds. Changing rules updates cards and alerts. Acknowledgement does not clear a condition. Alert history and settings exist only in the browser session and are not an operational audit log.

CSV checks are preliminary: expected named channels, finite numeric values and at least 110 rows. Timing, units, subject identity, train-only normalization, activity boundaries and gaps still need validation. CSV upload alone never establishes compatibility with a checkpoint. The provided template is five placeholder rows, intentionally too short for training.

## Live architecture

The receiver currently saves BLE packets locally and is restricted to Unix sockets. Keep that boundary. Use a separate status reporter or an agreed server collection mechanism for telemetry. The training client should publish its own activity and metrics. Manually loaded PAMAP2 data does not require BLE connectivity for training. Jetson provides time synchronization; agreement with Jetson alone does not guarantee a correct calendar date.

Future collector on Jetson stores reports; Streamlit reads them. Determine collector port with team before deployment. Never enable laptop internet sharing or bridge the isolated project network. All installs on project devices must follow the offline procedure.

## Files to request from teammates

See INTEGRATION.md. You can start the UI before obtaining them.
