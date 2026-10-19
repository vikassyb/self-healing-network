from __future__ import annotations

import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from network_engine import DEFAULT_TELEMETRY_THRESHOLDS, LINK_BASE_COSTS, NetworkEngine, ROUTERS

st.set_page_config(page_title="Self-Healing Network AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Self-Healing Network")
st.caption("AI-based failure-risk prediction · automatic route recalculation · interactive network simulation")
st.info("This version is a safe, reproducible simulation. It demonstrates the logic and measures simulated connectivity; it does not change your computer's real network routes.")

engine_state_warning = None
if "engine" not in st.session_state:
    try:
        st.session_state.engine = NetworkEngine(seed=17)
    except sqlite3.Error as exc:
        st.error(f"Could not initialize persistent incident history: {exc}")
        st.stop()
elif not isinstance(st.session_state.engine, NetworkEngine):
    try:
        st.session_state.engine = NetworkEngine(seed=17)
    except sqlite3.Error as exc:
        st.error(f"Could not initialize persistent incident history: {exc}")
        st.stop()
    engine_state_warning = "The stored simulation state was invalid and has been reinitialized."
engine: NetworkEngine = st.session_state.engine
st.session_state.setdefault("telemetry_running", False)
action_error = None

with st.sidebar:
    st.header("Control panel")
    st.write("Advance the simulated network, trigger a failure, or restore the topology.")
    router_choice = st.selectbox("Router to fail", ROUTERS, index=1)
    link_choice = st.selectbox(
        "Link to fail",
        list(LINK_BASE_COSTS),
        format_func=lambda link: f"{link[0]} — {link[1]}",
    )
    risk_threshold = st.slider(
        "High-risk threshold",
        0.30,
        0.90,
        DEFAULT_TELEMETRY_THRESHOLDS["risk_score"],
        0.05,
    )
    auto_heal = st.toggle("Enable automatic rerouting", value=True)
    st.subheader("Live telemetry")
    telemetry_running = st.session_state.telemetry_running
    start_col, pause_col = st.columns(2)
    if start_col.button("▶ Start", disabled=telemetry_running, use_container_width=True):
        st.session_state.telemetry_running = True
        st.rerun()
    if pause_col.button("⏸ Pause", disabled=not telemetry_running, use_container_width=True):
        st.session_state.telemetry_running = False
        st.rerun()
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        min_value=1,
        max_value=10,
        value=2,
        help="The live telemetry fragment updates only while the simulation is running.",
    )
    latency_warning = st.number_input(
        "Latency warning (ms)",
        min_value=0.0,
        value=DEFAULT_TELEMETRY_THRESHOLDS["latency_ms"],
        step=5.0,
    )
    packet_loss_warning = st.number_input(
        "Packet-loss warning (%)",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_TELEMETRY_THRESHOLDS["packet_loss_pct"],
        step=0.5,
    )
    utilization_warning = st.number_input(
        "Utilization warning (%)",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_TELEMETRY_THRESHOLDS["utilization_pct"],
        step=5.0,
    )
    st.divider()
    if st.button(
        "▶ Advance simulation",
        disabled=telemetry_running,
        use_container_width=True,
    ):
        try:
            engine.advance(threshold=risk_threshold)
        except (ValueError, sqlite3.Error) as exc:
            action_error = str(exc)
        else:
            st.rerun()
    if st.button("⚠ Inject router failure", type="primary", use_container_width=True):
        try:
            engine.fail_router(
                router_choice,
                threshold=risk_threshold,
                auto_heal=auto_heal,
            )
            if not auto_heal:
                engine.recover_network(
                    threshold=risk_threshold,
                    auto_heal=False,
                )
        except (ValueError, sqlite3.Error) as exc:
            action_error = str(exc)
        else:
            st.rerun()
    if st.button("⚠ Inject link failure", use_container_width=True):
        try:
            engine.set_link_status(
                *link_choice,
                status="DOWN",
                threshold=risk_threshold,
                auto_heal=auto_heal,
            )
        except (ValueError, sqlite3.Error) as exc:
            action_error = str(exc)
        else:
            st.rerun()
    if st.button("↻ Restore / reset network and telemetry", use_container_width=True):
        try:
            engine.reset()
        except sqlite3.Error as exc:
            action_error = f"Could not persist network reset: {exc}"
        else:
            st.session_state.telemetry_running = False
            st.rerun()
    with st.expander("Advanced simulation scenarios", expanded=False):
        scenario_router_failures = st.multiselect(
            "Scenario router failures",
            ROUTERS,
            key="scenario_router_failures",
        )
        scenario_link_failures = st.multiselect(
            "Scenario link failures",
            list(LINK_BASE_COSTS),
            format_func=lambda link: f"{link[0]} — {link[1]}",
            key="scenario_link_failures",
        )
        scenario_congested_routers = st.multiselect(
            "Congested routers",
            ROUTERS,
            key="scenario_congested_routers",
        )
        scenario_loss_routers = st.multiselect(
            "Routers with packet loss",
            ROUTERS,
            key="scenario_loss_routers",
        )
        scenario_congestion = st.slider(
            "Congestion utilization (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            key="scenario_congestion_pct",
        )
        scenario_packet_loss = st.slider(
            "Injected packet loss (%)",
            min_value=0.0,
            max_value=18.0,
            value=5.0,
            step=0.5,
            key="scenario_packet_loss_pct",
        )
        scenario_seed = st.number_input(
            "Scenario random seed",
            min_value=0,
            max_value=2**32 - 1,
            value=17,
            step=1,
            key="scenario_seed",
        )
        st.caption(
            "Scenario runs begin from a clean simulated topology. Congestion and packet loss "
            "change synthetic router telemetry only."
        )
        if st.button("Run configured scenario", type="primary", use_container_width=True):
            st.session_state.telemetry_running = False
            try:
                engine.run_scenario(
                    router_failures=scenario_router_failures,
                    link_failures=scenario_link_failures,
                    congested_routers=scenario_congested_routers,
                    packet_loss_by_router={
                        router: scenario_packet_loss
                        for router in scenario_loss_routers
                    },
                    congestion_utilization_pct=scenario_congestion,
                    seed=int(scenario_seed),
                    threshold=risk_threshold,
                    auto_heal=auto_heal,
                )
            except (ValueError, sqlite3.Error) as exc:
                action_error = str(exc)
            else:
                st.rerun()
        if st.button("Reset scenario", use_container_width=True):
            try:
                engine.reset()
            except sqlite3.Error as exc:
                action_error = f"Could not persist scenario reset: {exc}"
            else:
                st.session_state.telemetry_running = False
                st.rerun()

if engine_state_warning:
    st.warning(engine_state_warning)
if action_error:
    st.error(f"Simulation action failed: {action_error}")

@st.fragment(run_every=refresh_interval if st.session_state.telemetry_running else None)
def live_telemetry_panel() -> None:
    if st.session_state.telemetry_running:
        try:
            engine.advance(threshold=risk_threshold)
        except (ValueError, sqlite3.Error) as exc:
            st.error(f"Live telemetry update failed: {exc}")
            st.session_state.telemetry_running = False

    st.subheader("Live network telemetry")
    st.caption(
        "Continuously refreshed simulated measurements only; these are not measurements from a physical network."
    )
    thresholds = {
        "latency_ms": latency_warning,
        "packet_loss_pct": packet_loss_warning,
        "utilization_pct": utilization_warning,
        "risk_score": risk_threshold,
    }
    try:
        alerts = engine.telemetry_alerts(thresholds)
    except ValueError as exc:
        st.error(f"Invalid telemetry warning threshold: {exc}")
        return

    state_columns = st.columns(3)
    state_columns[0].metric(
        "Healthy routers",
        sum(alert["state"] == "HEALTHY" for alert in alerts.values()),
    )
    state_columns[1].metric(
        "High-risk routers",
        sum(alert["state"] == "HIGH RISK" for alert in alerts.values()),
    )
    state_columns[2].metric(
        "Failed routers",
        sum(alert["state"] == "FAILED" for alert in alerts.values()),
    )

    state_icons = {"HEALTHY": "🟢", "HIGH RISK": "🟠", "FAILED": "🔴"}
    live_rows = []
    for router in ROUTERS:
        metrics = engine.telemetry[router]
        alert = alerts[router]
        live_rows.append({
            "Router": router,
            "State": f"{state_icons[alert['state']]} {alert['state']}",
            "Latency (ms)": round(metrics["latency_ms"], 1),
            "Packet loss (%)": round(metrics["packet_loss_pct"], 2),
            "Utilization (%)": round(metrics["utilization_pct"], 1),
            "Predicted failure risk (%)": round(engine.risk_scores[router] * 100, 1),
            "Alert details": "; ".join(alert["reasons"]) or "Within configured thresholds",
        })
    st.dataframe(pd.DataFrame(live_rows), hide_index=True, use_container_width=True)

    st.subheader("Historical telemetry")
    st.caption(
        "Charts use persisted simulated observations and retain the most recent 300 samples per router."
    )
    history = pd.DataFrame(engine.incident_store.list_telemetry(limit_per_router=300))
    if not history.empty:
        history["observed_at"] = pd.to_datetime(history["observed_at"], utc=True, errors="coerce")
        history = history.dropna(subset=["observed_at"])
        history["failure_risk_pct"] = history["risk_score"] * 100
        chart_specs = (
            ("latency_ms", "Latency (ms)"),
            ("packet_loss_pct", "Packet loss (%)"),
            ("utilization_pct", "Utilization (%)"),
            ("failure_risk_pct", "Predicted failure risk (%)"),
        )
        chart_tabs = st.tabs([label for _, label in chart_specs])
        for tab, (metric, label) in zip(chart_tabs, chart_specs):
            with tab:
                chart = px.line(
                    history,
                    x="observed_at",
                    y=metric,
                    color="router",
                    markers=True,
                    labels={
                        "observed_at": "Observation time (UTC)",
                        metric: label,
                        "router": "Router",
                    },
                    title=f"Historical {label.lower()} by router",
                )
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No persisted telemetry observations are available yet.")

    st.subheader("Network analytics")
    try:
        incident_metrics = engine.incident_store.analytics()
    except sqlite3.Error as exc:
        st.error(f"Could not calculate network analytics: {exc}")
    else:
        kpi_columns = st.columns(6)
        availability = incident_metrics["network_availability_pct"]
        kpi_columns[0].metric(
            "Simulated availability",
            f"{availability:.2f}%" if availability is not None else "N/A",
            help=(
                "Time-weighted reachability across consecutive persisted observations; "
                "a snapshot's state applies until the next snapshot."
            ),
        )
        mttd = incident_metrics["mttd_seconds"]
        kpi_columns[1].metric(
            "Mean time to detect",
            f"{mttd:.3f} s" if mttd is not None else "N/A",
            help="Mean failure detection timestamp minus failure occurrence timestamp.",
        )
        mttr = incident_metrics["mttr_seconds"]
        kpi_columns[2].metric(
            "Mean time to recover",
            f"{mttr:.3f} s" if mttr is not None else "N/A",
            help="Mean successful recovery completion timestamp minus failure detection timestamp.",
        )
        success_rate = incident_metrics["recovery_success_rate_pct"]
        kpi_columns[3].metric(
            "Recovery success rate",
            f"{success_rate:.2f}%" if success_rate is not None else "N/A",
            help="Successful completed recovery attempts divided by all completed attempts.",
        )
        kpi_columns[4].metric(
            "Observed interval",
            f"{incident_metrics['network_observation_seconds']:.3f} s",
            help="Sum of positive elapsed times between consecutive valid telemetry observations.",
        )
        kpi_columns[5].metric(
            "Observed uptime",
            f"{incident_metrics['network_uptime_seconds']:.3f} s",
            help="Observed intervals whose starting snapshot had an available Host A to Host B route.",
        )
        st.caption(
            "Availability = reachable observed seconds / total observed seconds. "
            "MTTD = mean(detection − failure occurrence). "
            "MTTR = mean(successful recovery completion − detection). "
            "Recovery success rate = successful completed attempts / all completed attempts. "
            "Times are simulated observation timestamps and reported in seconds; no valid interval or pair is shown as N/A."
        )
        st.caption(
            f"Incomplete timestamp pairs: MTTD {incident_metrics['incomplete_detection_timestamps']}; "
            f"MTTR {incident_metrics['incomplete_recovery_timestamps']}. "
            f"In-progress or incomplete recovery attempts: {incident_metrics['incomplete_recovery_attempts']}."
        )
        st.download_button(
            "Download calculated network metrics as CSV",
            data=engine.incident_store.export_metrics_csv(),
            file_name="network_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    try:
        live_route, live_route_note = engine.compute_route(
            threshold=risk_threshold,
            auto_heal=auto_heal,
        )
        if engine.recovery.get("rerouting_enabled") is False:
            prior_route = engine.recovery.get("original_route")
            live_route = prior_route if engine.route_is_available(prior_route) else None
            live_route_note = engine.recovery["result"]
    except ValueError as exc:
        st.error(f"Could not calculate the simulated live route: {exc}")
        live_route = None
        live_route_note = "Route calculation failed."

    if live_route:
        st.success("Current simulated route: " + " → ".join(live_route))
    else:
        st.error("No available simulated route between Host A and Host B.")
    st.write(live_route_note)
    if engine.recovery_state != "IDLE":
        recovery = engine.recovery
        st.write(
            f"**Recovery state:** {engine.recovery_state.replace('_', ' ')}  \n"
            f"**Original route:** {' → '.join(recovery['original_route']) if recovery.get('original_route') else 'Unavailable'}  \n"
            f"**Replacement route:** {' → '.join(recovery['selected_route']) if recovery.get('selected_route') else 'Unavailable'}  \n"
            f"**Verification:** {recovery['result']}"
        )


live_telemetry_panel()

route = None
route_note = "Route calculation has not completed."
try:
    route, route_note = engine.compute_route(threshold=risk_threshold, auto_heal=auto_heal)
    if engine.recovery.get("rerouting_enabled") is False:
        previous_route = engine.recovery.get("original_route")
        route = previous_route if engine.route_is_available(previous_route) else None
        route_note = engine.recovery["result"]
except ValueError as exc:
    st.error(f"Unable to calculate a route: {exc}")

metrics = engine.current_metrics()
active_routers = sum(1 for router in ROUTERS if metrics[router]["status"] == "UP")
high_risk = sum(
    1 for router in ROUTERS
    if metrics[router]["status"] == "UP" and metrics[router]["risk_score"] >= risk_threshold
)
if route is None:
    network_status = "UNREACHABLE"
elif active_routers < len(ROUTERS) or high_risk:
    network_status = "DEGRADED"
else:
    network_status = "HEALTHY"
cols = st.columns(4)
cols[0].metric("Network status", network_status, delta="Route available" if route else "No valid route")
cols[1].metric("Active routers", f"{active_routers}/{len(ROUTERS)}")
cols[2].metric("High-risk routers", str(high_risk))
cols[3].metric("Current route", f"{len(route)-1} hops" if route else "Unavailable")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Network topology")
    graph = engine.graph_for_display()
    pos = {
        "Host A": (0, 1), "R1": (1, 1), "R2": (2, 1.7),
        "R3": (2, 0.3), "R4": (3, 1), "Host B": (4, 1)
    }
    fig, ax = plt.subplots(figsize=(10, 4.8))
    node_colors = []
    for n in graph.nodes:
        if n.startswith("Host"):
            node_colors.append("#7c9ee8")
        elif engine.status.get(n) == "DOWN":
            node_colors.append("#e05d5d")
        elif engine.risk_scores.get(n, 0) >= risk_threshold:
            node_colors.append("#f0b44d")
        else:
            node_colors.append("#54b78a")
    edge_colors = []
    route_edges = set(zip(route, route[1:])) | set((b, a) for a, b in zip(route, route[1:])) if route else set()
    for u, v in graph.edges:
        edge_colors.append("#2774d8" if (u, v) in route_edges else "#b8c0cc")
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=[3 if (u, v) in route_edges else 1.4 for u, v in graph.edges])
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=1800, edgecolors="white", linewidths=1.5)
    labels = {n: f"{n}\n{engine.status.get(n, 'HOST')}" if n in ROUTERS else n for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Green = healthy · Amber = AI high-risk · Red = failed · Blue link = selected route")
    if route:
        st.success("Selected route: " + " → ".join(route))
    else:
        st.error("No available path between Host A and Host B.")
    st.write(route_note)

with right:
    st.subheader("AI failure-risk assessment")
    risk_rows = []
    for r in ROUTERS:
        m = engine.telemetry[r]
        risk_rows.append({
            "Router": r,
            "Actual simulated state": engine.status[r],
            "Latency (ms)": round(m["latency_ms"], 1),
            "Packet loss (%)": round(m["packet_loss_pct"], 2),
            "Utilization (%)": round(m["utilization_pct"], 1),
            "Error rate": round(m["error_rate"], 3),
            "Predicted risk score (%)": round(engine.risk_scores[r] * 100, 1),
            "Risk assessment": "HIGH RISK" if engine.risk_scores[r] >= risk_threshold else "LOWER RISK",
        })
    df = pd.DataFrame(risk_rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Risk assessment is a model prediction; actual simulated state is reported separately. Scores are based on synthetic telemetry and are not validated real-world failure probabilities.")

st.subheader("Recovery verification")
recovery = engine.recovery
st.metric("Recovery state", engine.recovery_state.replace("_", " "))
recovery_cols = st.columns(3)
original_route = recovery.get("original_route")
selected_route = recovery.get("selected_route")
recovery_cols[0].write(
    "**Original route**\n\n" + (" → ".join(original_route) if original_route else "Unavailable")
)
recovery_cols[1].write(
    "**Selected route**\n\n" + (" → ".join(selected_route) if selected_route else "Unavailable")
)
duration_ms = recovery.get("duration_ms")
recovery_cols[2].write(
    "**Simulated recovery duration**\n\n"
    + (f"{duration_ms:.3f} ms" if duration_ms is not None else "Not measured")
)
if recovery.get("verified") is True:
    st.success(recovery["result"])
elif recovery.get("verified") is False:
    st.error(recovery["result"])
else:
    st.info(recovery["result"])

if engine.scenario_result:
    st.subheader("Scenario analysis")
    scenario = engine.scenario_result
    st.caption(f"Scenario seed: {scenario['seed']} · Scenario ID: {scenario['scenario_id']}")
    st.markdown("**Root-cause analysis**")
    st.write(scenario["root_cause"]["explanation"])
    st.write(
        "Affected components: "
        + (", ".join(scenario["root_cause"]["components"]) or "None")
    )
    root_cause = scenario["root_cause"]
    original_scenario_route = root_cause["original_route"]
    post_injection_route = root_cause["route_after_injection"]
    st.write(
        "**Route before injection:** "
        + (" → ".join(original_scenario_route) if original_scenario_route else "Unavailable")
    )
    st.write(
        "**Route after injection, before recovery:** "
        + (" → ".join(post_injection_route) if post_injection_route else "No route available")
    )
    on_original_route = root_cause["failed_components_on_original_route"]
    if on_original_route:
        st.write("**Failed component(s) on original route:** " + ", ".join(on_original_route))
    comparison = pd.DataFrame(scenario["strategy_comparison"])
    comparison = comparison.rename(columns={
        "strategy": "Routing strategy",
        "route": "Route",
        "available": "Available",
        "hop_count": "Hop count",
        "weighted_cost": "Weighted path cost",
    })
    comparison["Route"] = comparison["Route"].map(
        lambda path: " → ".join(path) if path else "No route available"
    )
    st.markdown("**Strategy comparison on the same scenario topology**")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

st.subheader("Predictive model evaluation")
st.caption("Metrics below are computed on a reproducible held-out synthetic test split using the classifier's default decision threshold. They describe this generated dataset only, not real-network performance; the sidebar risk threshold controls live classification and routing.")
evaluation = engine.model_evaluation
metric_columns = st.columns(4)
metric_columns[0].metric("Precision", f"{evaluation['precision']:.3f}")
metric_columns[1].metric("Recall", f"{evaluation['recall']:.3f}")
metric_columns[2].metric("F1-score", f"{evaluation['f1_score']:.3f}")
metric_columns[3].metric(
    "ROC-AUC",
    f"{evaluation['roc_auc']:.3f}" if evaluation["roc_auc"] is not None else "N/A",
)
evaluation_left, evaluation_right = st.columns(2)
with evaluation_left:
    st.markdown("**Confusion matrix**")
    st.caption("Rows = generated actual label (0/1); columns = predicted label (0/1).")
    confusion = pd.DataFrame(
        evaluation["confusion_matrix"],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    st.dataframe(confusion, use_container_width=True)
with evaluation_right:
    st.markdown("**Random Forest feature importance**")
    importance = pd.DataFrame(engine.feature_importance)
    importance["importance"] = importance["importance"].round(4)
    st.dataframe(importance, hide_index=True, use_container_width=True)

st.subheader("Event log")
events = engine.events
if events:
    st.dataframe(pd.DataFrame(list(reversed(events[-30:]))), hide_index=True, use_container_width=True)
else:
    st.write("No events yet. Advance the simulation or inject a router failure to begin.")

try:
    st.subheader("Simulation event replay")
    persisted_scenarios = engine.incident_store.list_simulation_scenarios()
    replay_options = ["All events", *persisted_scenarios]
    replay_scenario = st.selectbox(
        "Scenario to review",
        replay_options,
        key="simulation_replay_scenario",
    )
    replay_events = engine.incident_store.list_simulation_events(
        scenario_id=None if replay_scenario == "All events" else replay_scenario,
        limit=500,
    )
    if replay_events:
        replay_frame = pd.DataFrame(replay_events)
        st.dataframe(
            replay_frame[[
                "timestamp",
                "scenario_id",
                "step",
                "level",
                "event",
                "component",
                "details",
            ]].rename(columns={
                "timestamp": "Timestamp (UTC)",
                "scenario_id": "Scenario ID",
                "step": "Step",
                "level": "Level",
                "event": "Event",
                "component": "Component",
                "details": "Details",
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No persisted simulation events are available to replay.")

    st.subheader("Incident timeline")
    incident_rows = engine.incident_store.list_incidents()
    if incident_rows:
        timeline = pd.DataFrame(incident_rows)
        timestamp_columns = (
            ("timestamp", "Timestamp (UTC)"),
            ("detected_at", "Detected (UTC)"),
            ("recovery_completed_at", "Recovery completed (UTC)"),
        )
        for column, _ in timestamp_columns:
            timeline[column] = pd.to_datetime(timeline[column], utc=True, errors="coerce")
        timeline = timeline.sort_values("timestamp", na_position="last")
        for column, _ in timestamp_columns:
            timeline[column] = timeline[column].map(
                lambda timestamp: (
                    timestamp.isoformat()
                    if pd.notna(timestamp)
                    else "Timestamp unavailable"
                )
            )
        timeline["route_change"] = timeline.apply(
            lambda row: (
                f"{row['original_route'] or 'Unavailable'} → "
                f"{row['replacement_route'] or 'No replacement route'}"
            ),
            axis=1,
        )
        timeline["timeline_event"] = timeline["event_type"].map({
            "router_failure": "Router failure",
            "link_failure": "Link failure",
            "recovery_attempt": "Recovery / route change",
            "network_reset": "Network reset",
        }).fillna(timeline["event_type"])
        timeline = timeline.rename(columns={
            "timestamp": "Timestamp (UTC)",
            "detected_at": "Detected (UTC)",
            "recovery_completed_at": "Recovery completed (UTC)",
            "component": "Component",
            "recovery_outcome": "Recovery result",
            "recovery_duration_ms": "Simulated recovery duration (ms)",
        })
        st.dataframe(
            timeline[[
                "Timestamp (UTC)",
                "Detected (UTC)",
                "Recovery completed (UTC)",
                "timeline_event",
                "Component",
                "route_change",
                "Recovery result",
                "Simulated recovery duration (ms)",
            ]].rename(columns={
                "timeline_event": "Event",
                "route_change": "Route change (original → replacement)",
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No incidents are recorded yet.")

    st.download_button(
        "Download historical telemetry as CSV",
        data=engine.incident_store.export_telemetry_csv(),
        file_name="network_telemetry.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Persistent incident history")
    incident_options = engine.incident_store.filter_options()
    filter_columns = st.columns(3)
    with filter_columns[0]:
        selected_event_types = st.multiselect(
            "Filter by event type",
            incident_options["event_types"],
            key="incident_event_type_filter",
        )
    with filter_columns[1]:
        selected_components = st.multiselect(
            "Filter by component",
            incident_options["components"],
            key="incident_component_filter",
        )
    with filter_columns[2]:
        selected_outcomes = st.multiselect(
            "Filter by recovery outcome",
            incident_options["recovery_outcomes"],
            key="incident_outcome_filter",
        )

    incident_filters = {
        "event_types": selected_event_types or None,
        "components": selected_components or None,
        "recovery_outcomes": selected_outcomes or None,
    }
    filtered_metrics = engine.incident_store.analytics(**incident_filters)
    incident_columns = st.columns(4)
    incident_columns[0].metric("Total incidents", filtered_metrics["total_incidents"])
    incident_columns[1].metric(
        "Successful recoveries",
        filtered_metrics["successful_recoveries"],
    )
    incident_columns[2].metric(
        "Failed recoveries",
        filtered_metrics["failed_recoveries"],
    )
    average_duration = filtered_metrics["average_recovery_duration_ms"]
    incident_columns[3].metric(
        "Average simulated recovery duration",
        f"{average_duration:.3f} ms" if average_duration is not None else "N/A",
        help="Local processing duration of simulated route calculation and verification; not network latency.",
    )
    filtered_incidents = engine.incident_store.list_incidents(**incident_filters)
    if filtered_incidents:
        incident_frame = pd.DataFrame(filtered_incidents)
        st.dataframe(incident_frame, hide_index=True, use_container_width=True)
    else:
        st.write("No incidents match the selected filters.")
    st.download_button(
        "Download incident history as CSV",
        data=engine.incident_store.export_csv(**incident_filters),
        file_name="network_incident_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
except sqlite3.Error as exc:
    st.error(f"Could not load persistent incident history: {exc}")

st.caption(f"Simulation step: {engine.step} · Last refreshed: {datetime.now().strftime('%H:%M:%S')} · Prototype for academic demonstration")
