"""PulseNet interface prototype. Run from repo root: streamlit run dashboard/app.py."""
from datetime import datetime, timezone
import ast
import io
import re
from pathlib import Path
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from subprocess import TimeoutExpired

import pandas as pd
import plotly.express as px
import streamlit as st

from experiment_runner import run_experiment
from pulsenet_live import read_memory

st.set_page_config(page_title="PulseNet | Cluster console", page_icon="📡", layout="wide")
CHANNELS = [f"{site}_{sensor}_{axis}" for site in ("hand", "chest", "ankle")
            for sensor in ("acc", "gyro") for axis in "xyz"] + ["heart_rate"]
BOARDS = ["PN-S01", "PN-E01", "PN-S02", "PN-E02", "PN-S03", "PN-E03", "PN-S04", "PN-E04"]


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def poll_pi(index):
    endpoint = f"10.20.0.{10 + index}:9100"
    try:
        memory = f"{read_memory(endpoint):.1f}%"
        status = "Telemetry reachable"
    except (OSError, ValueError, KeyError):
        memory = "—"
        status = "Telemetry unavailable"
    return {"Pi": f"pi-{index:02}", "Endpoint": endpoint,
            "Exporter status": status, "RAM used": memory}


@st.cache_data(ttl=4)
def pi_readings():
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(poll_pi, range(1, 9)))


@st.fragment(run_every=5)
def live_pi_view():
    st.dataframe(pi_readings(), hide_index=True, width="stretch")
    st.caption(f"Polled at {datetime.now(timezone.utc):%H:%M:%S} UTC")




def format_run_label(benchmark, run_dir):
    benchmark_name = {
        "b2_centralized": "B2 Centralized",
        "b3_local_only": "B3 Local-only",
    }[benchmark]

    try:
        config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        config = {}

    channels = config.get("input_channels") if isinstance(config, dict) else None
    if not isinstance(channels, list) or not channels:
        try:
            dashboard_config = json.loads(
                (run_dir / "dashboard-inputs.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            dashboard_config = {}
        if isinstance(dashboard_config, dict):
            channels = dashboard_config.get("input_channels")

    if not isinstance(channels, list) or not channels:
        sensor_label = "Inputs unrecorded"
    elif set(channels) == set(CHANNELS):
        sensor_label = "All 19 inputs"
    elif set(channels) == {name for name in CHANNELS if "_gyro_" not in name}:
        sensor_label = "Gyroscope inputs excluded (10 inputs)"
    else:
        sensor_label = f"Custom selection ({len(channels)} inputs)"

    return f"{benchmark_name} · {sensor_label} · {run_dir.name}"

def load_experiment_results(results_dir):
    """Read saved benchmarks and the local Flower simulation log."""
    runs = {}
    rounds = []
    for benchmark in ("b2_centralized", "b3_local_only"):
        for path in sorted((results_dir / benchmark).glob("*/results.csv")):
            frame = pd.read_csv(path, dtype={"eval": str})
            if {"seed", "eval", "mse"}.issubset(frame.columns):
                runs[format_run_label(benchmark, path.parent)] = frame
    log = results_dir / "flower-simulation.log"
    if log.is_file():
        round_number = None
        stage = None
        for line in log.read_text(encoding="utf-8").splitlines():
            match = re.search(r"\[ROUND (\d+)/\d+\]", line)
            if match:
                round_number = int(match[1])
                stage = None
            if "aggregate_train:" in line:
                stage = "train"
            if "aggregate_evaluate:" in line:
                stage = "evaluate"
            if round_number is not None and stage == "evaluate" and "Aggregated MetricRecord: " in line:
                metrics = ast.literal_eval(line.split("Aggregated MetricRecord: ", 1)[1])
                rounds.append({**metrics, "round": round_number})
    return runs, pd.DataFrame(rounds)

def demo_nodes():
    states = ["Training", "Training", "Evaluating", "Idle", "Submitted", "Training", "Idle", "Unknown"]
    return pd.DataFrame([dict(node=f"pi-{i:02}", ip=f"10.20.0.{10+i}", board=BOARDS[i-1],
        heartbeat_age_s=120 if i == 8 else i, activity=states[i-1],
        ble="Unknown" if i == 8 else "Disconnected" if i == 4 else "Connected",
        subject=100+i, selected=i in (1, 2, 3, 5, 6), round=12,
        memory_percent=None if i == 8 else 91 if i == 6 else 32+4*i,
        high_memory_duration_s=80 if i == 6 else 0,
        time_sync="Unknown" if i == 8 else "Unsynchronized" if i == 7 else "Synchronized",
        validation_mse=None if i == 8 else round(.12+i*.015, 3),
        data_source="Manually loaded PAMAP2") for i in range(1, 9)])


def history():
    rng = random.Random(42)
    return pd.DataFrame([dict(node=f"pi-{i:02}", round=r,
        validation_mse=.55*math.exp(-r/7)+.06+i*.008+rng.uniform(-.015, .015))
        for i in range(1, 9) for r in range(1, 13)])


def active_alerts(nodes, heartbeat, memory, duration):
    alerts = []
    for row in nodes.to_dict("records"):
        node = row["node"]
        if row["heartbeat_age_s"] > heartbeat:
            alerts.append(dict(node=node, rule="Heartbeat missing", severity="Critical",
                               detail=f"Last report {row['heartbeat_age_s']} s ago; current activity unknown"))
            continue  # Do not interpret stale resource readings as current.
        if row["memory_percent"] is not None and row["memory_percent"] > memory and row["high_memory_duration_s"] >= duration:
            alerts.append(dict(node=node, rule="High memory", severity="Warning",
                               detail=f"{row['memory_percent']}% for {row['high_memory_duration_s']} s"))
        if row["time_sync"] == "Unsynchronized":
            alerts.append(dict(node=node, rule="Clock unsynchronized", severity="Warning",
                               detail="Pi reports no time synchronization"))
    return alerts


st.title("PulseNet")
st.caption("Cluster monitoring · Federated learning · Dataset experiments")
st.warning("Node reports and alerts use demo data. Experiment results come from the local results folder. Live Pi telemetry polls the lab network. Centralized experiments run on this dashboard host.")
with st.sidebar:
    st.header("Workspace")
    st.write("Coordinator: pulsenet-jetson")
    st.code("10.20.0.10")
    st.caption("Deployment target only; this app currently runs locally.")
    st.write("Training source: manually loaded PAMAP2")
    st.caption("Jetson provides network time. Clock agreement does not prove the calendar time is correct.")
    st.button("Refresh view")
    st.caption("Refresh redraws the same reproducible example; it does not poll devices.")
    st.divider()
    st.caption("No raw sensor data is sent to hardware by this prototype.")

nodes = demo_nodes()
st.session_state.setdefault("rules", {"heartbeat": 30, "memory": 85, "duration": 60})
st.session_state.setdefault("acknowledged", [])
st.session_state.setdefault("alert_history", [])
rules = st.session_state.rules
alerts = active_alerts(nodes, **rules)
for alert in alerts:
    key = f"{alert['node']}:{alert['rule']}"
    if not any(a["key"] == key for a in st.session_state.alert_history):
        st.session_state.alert_history.append({"key": key, "observed_at": datetime.now(timezone.utc).isoformat(), **alert})
nodes["connection"] = nodes.heartbeat_age_s.map(lambda a: "Unresponsive" if a > rules["heartbeat"] else "Online")
cols = st.columns(4)
for col, title, value in zip(cols, ["Online", "Selected for FL", "Example round", "Active alerts"],
                            [f"{(nodes.connection == 'Online').sum()}/8", int(nodes.selected.sum()), 12, len(alerts)]):
    col.metric(title, value)
overview, explorer, performance, datasets, alert_tab, controls, telemetry = st.tabs([
    "Overview", "Node explorer", "Prediction performance", "CSV experiments", "Alert rules", "Node controls", "Live Pi telemetry"])

with telemetry:
    live_enabled = st.checkbox("Poll Pi exporters")
    if live_enabled:
        live_pi_view()

with overview:
    st.subheader("Eight-node cluster")
    for start in (0, 4):
        for col, row in zip(st.columns(4), nodes.iloc[start:start+4].to_dict("records")):
            with col.container(border=True):
                st.markdown(f"### {row['node']}")
                st.write(f"{'🟢' if row['connection'] == 'Online' else '🔴'} {row['connection']}")
                st.caption(f"{row['ip']} · {row['board']}")
                st.write("Activity:", row["activity"] if row["connection"] == "Online" else "Unknown (stale report)")
                st.write("BLE:", row["ble"] if row["connection"] == "Online" else "Unknown")
                st.write("FL:", "Selected" if row["selected"] else "Not selected")
                st.caption(f"Last example heartbeat: {row['heartbeat_age_s']} s ago")
    st.info("BLE disconnection does not block training on manually loaded datasets. Unresponsive means reports are missing, not proof a Pi is powered off.")

with explorer:
    left, right = st.columns(2)
    connections = left.multiselect("Connection", list(nodes.connection.unique()), default=list(nodes.connection.unique()))
    activities = right.multiselect("Activity", list(nodes.activity.unique()), default=list(nodes.activity.unique()))
    selected_only = st.checkbox("Only nodes selected for FL")
    filtered = nodes[nodes.connection.isin(connections) & nodes.activity.isin(activities)]
    if selected_only:
        filtered = filtered[filtered.selected]
    st.dataframe(filtered, hide_index=True, width="stretch")
    if not filtered.empty:
        name = st.selectbox("Inspect node", filtered.node.tolist())
        st.json(filtered[filtered.node == name].iloc[0].to_dict())
    else:
        st.info("No nodes match these filters.")

with performance:
    st.subheader("Forecast validation error")
    st.caption("Example mean squared error (MSE) in standardized units; lower is better. These are not experiment results or health-detection accuracy.")
    choices = st.multiselect("Nodes to compare", nodes.node.tolist(), default=["pi-01", "pi-02", "pi-03"])
    values = history()
    values = values[values.node.isin(choices)]
    if not values.empty:
        st.plotly_chart(px.line(values, x="round", y="validation_mse", color="node", markers=True,
                               labels={"validation_mse": "Validation MSE", "round": "FL round"}), width="stretch")
    st.download_button("Download example metrics", history().to_csv(index=False), "demo_metrics.csv", "text/csv")

    st.subheader("Sensor inputs for a new experiment")
    st.caption("Train a centralized TCN on this host. Sensor selection changes the inputs. The model forecasts all 19 channels.")
    groups = st.multiselect(
        "Include sensor groups",
        ["Accelerometer", "Gyroscope", "Heart rate"],
        default=["Accelerometer", "Gyroscope", "Heart rate"],
    )
    channel_groups = {
        "acc": "Accelerometer", "gyro": "Gyroscope", "rate": "Heart rate",
    }
    selected_channels = [
        channel for channel in CHANNELS
        if channel_groups[channel.split("_")[1]] in groups
    ]
    if selected_channels:
        st.write(", ".join(selected_channels))
    else:
        st.warning("Select at least one sensor group.")

    available_subjects = sorted(
        path.stem.removeprefix("subject_")
        for path in (ROOT / "data/processed").glob("subject_*.npz")
        if path.stem.removeprefix("subject_").isdigit()
    )
    subjects = st.multiselect("PAMAP2 subjects", available_subjects,
                              default=available_subjects[:2])
    epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=2)
    if not available_subjects:
        st.info("Place processed PAMAP2 subject files in data/processed/ to enable training.")
    if st.button("Run centralized experiment", disabled=not selected_channels or not subjects):
        with st.spinner("Training and evaluating the TCN…"):
            try:
                destination = run_experiment(ROOT, RESULTS_DIR, selected_channels, subjects, int(epochs))
            except (OSError, RuntimeError, TimeoutExpired) as error:
                st.error(f"Experiment failed: {error}")
            else:
                st.success(f"Completed {destination.name}. Results appear below.")

    st.subheader("Experiment results")
    st.caption(
        "Reads saved runs from the repository’s results/ folder. " 
        "To see how removing gyroscope inputs affects forecast accuracy, compare "
        "runs with the same validation data, forecast targets, and training "
        "settings such as the number of epochs and the random seed."
    )
    st.button("Reload saved results")
    if RESULTS_DIR.is_dir():
        try:
            runs, rounds = load_experiment_results(RESULTS_DIR)
            profiles = []
            summaries = []
            for label, frame in runs.items():
                for row in frame.loc[frame["eval"] == "all"].to_dict("records"):
                    name = f"{label} · seed {row['seed']}"
                    summaries.append({"Run": name, "Validation MSE": row["mse"]})
                    profiles.extend(
                        {"Run": name, "Forecast step": step, "MSE": row[f"mse_h{step}"]}
                        for step in range(1, 11) if f"mse_h{step}" in row
                    )
            if not rounds.empty:
                rounds = rounds.sort_values("round")
                st.plotly_chart(px.line(rounds, x="round", y="mse", markers=True,
                                       labels={"round": "FL round", "mse": "Validation MSE"}),
                                width="stretch")
                final = rounds.iloc[-1]
                name = f"Flower · round {int(final['round'])}"
                summaries.append({"Run": name, "Validation MSE": final["mse"]})
                profiles.extend(
                    {"Run": name, "Forecast step": step, "MSE": final[f"mse_h{step}"]}
                    for step in range(1, 11) if f"mse_h{step}" in final
                )
                with st.expander("Flower evaluation metrics by round"):
                    st.dataframe(rounds, hide_index=True)
            if summaries:
                st.dataframe(pd.DataFrame(summaries), hide_index=True)
            if profiles:
                chart_data = pd.DataFrame(profiles)
                st.plotly_chart(px.line(chart_data, x="Forecast step", y="MSE",
                                       color="Run", markers=True), width="stretch")
            if not summaries:
                st.warning("No aggregate benchmark rows or Flower evaluation rounds were found.")
        except (OSError, KeyError, TypeError, ValueError, SyntaxError) as exc:
            st.error(f"Could not read experiment results: {exc}")

    else:
        st.info("Run a benchmark or Flower simulation to populate the repository’s results/ folder.")

with datasets:
    st.subheader("Inspect a CSV dataset")
    st.write("Upload previews and checks the file locally. It does not change a model or send data to a Pi.")
    st.caption("Expected cleaned CSV: one row per timestep with the 19 named sensor columns below. Raw PAMAP2 .dat files need preprocessing first.")
    with st.expander("Expected sensor columns"):
        st.code(", ".join(CHANNELS))
    sample = pd.DataFrame([{c: 0.0 for c in CHANNELS} for _ in range(5)])
    st.download_button("Download column template (five placeholder rows)", sample.to_csv(index=False), "pamap2_column_template.csv", "text/csv")
    upload = st.file_uploader("CSV file (maximum 10 MiB)", type=["csv"])
    if upload is not None:
        if upload.size > 10*1024*1024:
            st.error("Choose a file smaller than 10 MiB for this initial inspector.")
        else:
            try:
                frame = pd.read_csv(io.BytesIO(upload.getvalue()))
                st.write(f"{len(frame):,} rows · {len(frame.columns)} columns")
                st.dataframe(frame.head(50), hide_index=True)
                missing = [c for c in CHANNELS if c not in frame.columns]
                problems = []
                if missing:
                    problems.append("Missing sensor columns: " + ", ".join(missing))
                if not missing:
                    numeric = frame[CHANNELS].apply(pd.to_numeric, errors="coerce")
                    bad = int((numeric.isna() | numeric.isin([float('inf'), float('-inf')])).sum().sum())
                    if bad:
                        problems.append(f"{bad} missing, nonnumeric or infinite sensor values")
                if len(frame) < 110:
                    problems.append("At least 110 consecutive timesteps are needed for one 100-step input plus 10-step target")
                for problem in problems:
                    st.error(problem)
                if not problems:
                    st.success("Basic column and numeric checks passed. Timing, units, segment continuity and normalization still require verification before model use.")
                st.caption("Preview is limited to 50 rows. File contents are not stored by this app.")
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f"Could not read CSV: {exc}")
    st.button("Evaluate existing model", disabled=True, help="Requires a compatible checkpoint and preprocessing pipeline")
    st.button("Train a separate model copy", disabled=True, help="Requires training integration and a held-out evaluation dataset")

with alert_tab:
    st.subheader("Alert rules · example reports")
    with st.form("rule_settings"):
        heartbeat = st.number_input("Heartbeat timeout (seconds)", 1, 3600, rules["heartbeat"])
        memory = st.slider("Memory warning threshold (%)", 1, 100, rules["memory"])
        duration = st.number_input("Memory must exceed threshold for (seconds)", 1, 3600, rules["duration"])
        if st.form_submit_button("Apply rules"):
            st.session_state.rules = dict(heartbeat=heartbeat, memory=memory, duration=duration)
            st.rerun()
    st.caption("Clock warning triggers on an explicit unsynchronized report. Example sustained-memory duration is supplied by demo reports; live reporters must measure it.")
    for alert in alerts:
        key = f"{alert['node']}:{alert['rule']}"
        with st.container(border=True):
            st.write(f"{alert['severity']} · {alert['node']} · {alert['rule']}")
            st.write(alert["detail"])
            if key in st.session_state.acknowledged:
                st.caption("Acknowledged; condition remains active")
            elif st.button("Acknowledge", key=key):
                st.session_state.acknowledged.append(key)
                st.rerun()
    if not alerts:
        st.success("No example conditions match the current rules.")
    st.subheader("Conditions observed during this session")
    st.dataframe(pd.DataFrame(st.session_state.alert_history), hide_index=True)
    st.caption("Rules, acknowledgements and history last only for this browser session. No emails or notifications are sent. Round-deadline and prediction-degradation rules require live integration.")

with controls:
    st.subheader("Node action requests")
    st.info("Controls are unavailable until Aiden's FL server and a Pi command handler are connected. No commands are sent.")
    st.selectbox("Target node", nodes.node.tolist())
    st.button("Include in future FL rounds", disabled=True)
    st.button("Exclude from future FL rounds", disabled=True)
    st.button("Request evaluation", disabled=True)
    st.caption("Future flow: request → accepted/rejected → running → completed/failed. A button click alone must never be displayed as a completed action.")
