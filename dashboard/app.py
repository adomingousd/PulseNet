"""PulseNet interface prototype. Run from repo root: streamlit run dashboard/app.py."""
from datetime import datetime, timezone
import io
import json
import math
import random

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="PulseNet | Cluster console", page_icon="📡", layout="wide")
CHANNELS = [f"{site}_{sensor}_{axis}" for site in ("hand", "chest", "ankle")
            for sensor in ("acc", "gyro") for axis in "xyz"] + ["heart_rate"]
BOARDS = ["PN-S01", "PN-E01", "PN-S02", "PN-E02", "PN-S03", "PN-E03", "PN-S04", "PN-E04"]


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
st.warning("DEMO MODE — All status reports, metric values and alerts below are fabricated examples. No hardware is connected. No training or evaluation runs from this interface.")
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
overview, explorer, performance, datasets, alert_tab, controls = st.tabs([
    "Overview", "Node explorer", "Prediction performance", "CSV experiments", "Alert rules", "Node controls"])

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
