# PulseNet FL deployment runbook (prototype)

Goal: run the Flower app (TCN, FedAvg, no DP) on the closed network: SuperLink on the Jetson, one SuperNode per Pi, launched from a laptop. This is the W4 Session 5 target: all 8 Pis, 20 rounds.

Tested before handoff (Aiden, 2026-11-06): laptop simulation (8 clients, 20 rounds, 0 failures) and offline installs of both bundles in ARM Linux containers (`fl_smoke.py` 5/5 Pi bundle, 4/4 Jetson bundle).

---

## 0. Confirm before going on site

| Item | Assumed here | Confirm with |
|---|---|---|
| Install folder on every device | `~/pulsenet-fl` (user `pulsenet`), separate from the receiver's `/opt/pulsenet` | Gable |
| torch on the Jetson | Yes (the server builds the starting model) | Gable / Aiden |
| Laptop that runs `flwr run` | Needs the repo at the same commit, `flwr==1.42.0` (no torch), and a `pulsenet` connection in its Flower config | Gable / Aiden |
| Ports | Pis → Jetson `9092`, laptop → Jetson `9093`. No firewall yet, so both should be open | Gable |
| Subject mapping | `pi-0N` gets subject `10N` (pi-01 → 101 … pi-08 → 108) | Team (D6) |

---

## 1. Bundle (Aiden, on the Mac, from the repo root)

```
zip -r pulsenet-fl-bundle.zip pyproject.toml fl_framework world_model scripts/fl_smoke.py \
  deploy/requirements-pi.txt deploy/requirements-jetson.txt deploy/wheels-pi deploy/wheels-jetson \
  data/processed/subject_10[1-8].npz -x '*/__pycache__/*'
```

Hand the zip to Gable (USB). On his Windows laptop, unzip it and open PowerShell in the unzipped folder:
```
Expand-Archive pulsenet-fl-bundle.zip -DestinationPath pulsenet-fl-bundle
cd pulsenet-fl-bundle
```
Run every laptop command below from this folder. Don't open the `.py` or `.toml` files in a Windows editor that saves CRLF line endings.

---

## 2. Before anything else: Jetson clock

Follow §4 of Gable's Oct 30 status doc (`timedatectl`, `sudo date -s …`, `sudo hwclock --systohc`). Confirm one Pi shows "System clock synchronized: yes".

---

## 3. Copy files (Gable's laptop, PowerShell, from the unzipped folder)

SSH is password login, so expect a password prompt for each `ssh`/`scp` (3 per Pi).

**Jetson:**
```
ssh pulsenet@10.20.0.10 "mkdir -p ~/pulsenet-fl"
scp -r pyproject.toml fl_framework world_model scripts deploy/requirements-jetson.txt deploy/wheels-jetson pulsenet@10.20.0.10:~/pulsenet-fl/
```

**Pis** (each Pi gets only its own subject's file):
```
foreach ($N in 1..8) {
  ssh "pulsenet@pi-0$N.local" "mkdir -p ~/pulsenet-fl/data"
  scp -r pyproject.toml fl_framework world_model scripts deploy/requirements-pi.txt deploy/wheels-pi "pulsenet@pi-0$N.local:~/pulsenet-fl/"
  scp "data/processed/subject_10$N.npz" "pulsenet@pi-0$N.local:~/pulsenet-fl/data/"
}
```
If `pi-0N.local` doesn't resolve, use `10.20.0.1N` (pi-01 = 10.20.0.11 … pi-08 = 10.20.0.18).

---

## 4. Install and check (over SSH on each device)

**Each Pi** (replace `10N` with that Pi's subject):
```
cd ~/pulsenet-fl
python3 -m venv venv
venv/bin/pip install --no-index --find-links wheels-pi -r requirements-pi.txt
venv/bin/python scripts/fl_smoke.py --data-dir ~/pulsenet-fl/data --subject 10N
```
Expect `5/5 checks passed`. **Write down the "1 epoch Ns" time for each Pi**: it is the first real estimate of round time (W4 target: median round < 110 s).

**Jetson** (`ensurepip` isn't installed, so the venv is made without pip and filled with the system pip):
```
cd ~/pulsenet-fl
python3 -m venv --without-pip venv
python3 -m pip --python venv/bin/python install --no-index --find-links wheels-jetson -r requirements-jetson.txt
venv/bin/python scripts/fl_smoke.py
```
Expect `4/4 checks passed`.

Don't continue until all 9 devices pass.

---

## 5. Start Flower

**Jetson** (one SSH session, leave it open):
```
cd ~/pulsenet-fl && source venv/bin/activate
flower-superlink --insecure --disable-runtime-dependency-installation
```
`--disable-runtime-dependency-installation` is required: by default the SuperLink tries to download the app's dependencies before each run, and with no internet the run hangs at "starting".

**Pis.** For the first test, run pi-01 in the foreground so you can watch it:
```
cd ~/pulsenet-fl && source venv/bin/activate
flower-supernode --insecure --superlink 10.20.0.10:9092 \
  --node-config "data-dir='/home/pulsenet/pulsenet-fl/data' subject-id=101"
```
For all 8, start each in the background with a log file instead. Log in to each Pi first (`ssh pulsenet@pi-0N.local`) and paste the command inside that session; sending it straight from PowerShell mangles the nested quotes. `10N` = that Pi's subject:
```
cd ~/pulsenet-fl && source venv/bin/activate && nohup flower-supernode --insecure --superlink 10.20.0.10:9092 --node-config "data-dir='/home/pulsenet/pulsenet-fl/data' subject-id=10N" > supernode.log 2>&1 &
```
The venv must be activated: the SuperNode starts the ClientApp with the same Python, and that's where torch is installed.

---

## 6. Launch runs (Gable's laptop, PowerShell, from the unzipped bundle folder)

The laptop needs Python with `flwr==1.42.0` (no torch needed). Gable's existing PulseNet venv should already have it (`pip show flwr` to check). The unzipped bundle contains everything `flwr run` packages (`pyproject.toml`, `fl_framework/`, `world_model/`), so run from that folder.

Run `flwr config list` once; it creates the Flower config file if needed and prints its path (on Windows, under your user folder in `.flwr\config.toml`). Add:
```toml
[superlink.pulsenet]
address = "10.20.0.10:9093"
insecure = true
```

Run in stages. Only move on when a stage passes.

| Stage | SuperNodes running | Command | Counts as |
|---|---|---|---|
| A | pi-01 | `flwr run . pulsenet --run-config "min-nodes=1 num-server-rounds=2" --stream` | Connectivity check only |
| B | pi-01, pi-02 | `flwr run . pulsenet --run-config "min-nodes=2 num-server-rounds=2" --stream` | Connectivity check only |
| C | All 8 | See the command below the table | W4 Session 5 run |

Stage C (saves the output to `pi_20r.log` while showing it):
```
flwr run . pulsenet --run-config "num-server-rounds=20" --stream | Tee-Object -FilePath pi_20r.log
```

If the `flwr` CLI misbehaves on Windows, run the same commands from WSL2 (with `flwr==1.42.0` installed there), or bring Aiden's Mac on site.

**Passes when:**
- every round shows `Received N results and 0 failures` (N = SuperNodes running);
- `mse` falls from round to round and ends below `persistence_mse` (0.856);
- for stage C, the curve is close to the laptop reference: `mse` 0.864 (round 1), 0.678 (round 3), 0.491 (round 10), 0.418 (round 20). Pis won't match exactly (different CPU and math libraries); a large gap or a flat curve is a failure.

Save `pi_20r.log` and send it to Aiden with the per-Pi epoch times from step 4.

---

## 7. Stop everything

- Laptop: nothing to stop once the run finishes (`flwr stop <run-id>` to cancel one).
- Pis: `pkill -f flower-supernode` (or Ctrl+C in the foreground session).
- Jetson: Ctrl+C in the SuperLink session.

The BLE receiver (`pulsenet-receiver`) is separate and keeps running throughout.

---

## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| Run stays at "starting" | SuperLink started without `--disable-runtime-dependency-installation`. Restart it with the flag |
| SuperNode can't connect | SuperLink not running, wrong address, or port 9092 blocked. Check `ping 10.20.0.10` from the Pi |
| `flwr run` can't connect | Laptop not on the closed network (Wi-Fi off, static IP set in `ncpa.cpl`), or wrong address in the `pulsenet` connection |
| `FileNotFoundError … subject_10N.npz` | Wrong `data-dir` or `subject-id` in that Pi's `--node-config` |
| `ModuleNotFoundError: torch` in a ClientApp | SuperNode started without the venv activated |
| Fewer results than SuperNodes in a round | Check that Pi's `supernode.log`; the run continues only if `min-nodes` is met |
| Timestamps in 1970 | Jetson clock reset: redo step 2 |
