# Self-Healing Network with AI-Based Predictive Failure Detection

An individual Computer Networks major-project prototype using Python, NetworkX, scikit-learn, and Streamlit.

## What it demonstrates

- Simulated telemetry for routers: latency, packet loss, utilization, error rate, and latency trend.
- A Random Forest classifier trained on reproducible, correlated synthetic telemetry and simulated near-term failure labels.
- Held-out synthetic test metrics (precision, recall, F1-score, confusion matrix, and ROC-AUC when both classes are present), plus model feature importance.
- A network graph with primary and alternate paths.
- Weighted route selection using link cost, router latency, packet loss, utilization, error rate, latency trend, and predicted risk.
- Route calculation excludes failed routers, high-risk routers when automatic rerouting is enabled, and unavailable links.
- Injected router failures automatically trigger recovery only when the automatic-rerouting setting is enabled. When disabled, the prior route is retained if still available; otherwise the dashboard reports that no active route is available instead of showing a newly calculated route.
- Failure injection, verified simulated recovery, reset, event history, and bounded telemetry history.
- A live telemetry dashboard with start/pause controls, configurable warning thresholds, visually distinct healthy/high-risk/failed states, and interactive Plotly time-series charts. Its refresh interval is bounded from 1 to 10 seconds, and it retains at most 300 telemetry snapshots per router.
- Persistent incident records and timestamped per-router telemetry stored in a local SQLite database.
- A network analytics dashboard showing interval-based simulated availability, MTTD, MTTR, recovery success rate, and chronological incident/route-change timeline.
- CSV exports for telemetry history, incident history, and calculated metrics (including units and formulas).
- Reproducible multi-fault scenarios combining router/link outages, synthetic congestion, and packet loss.
- Persistent timestamped simulation-event history with scenario filtering for incident replay.
- Scenario root-cause explanations and a same-topology comparison of the weighted path and shortest-hop routing strategies.

**Academic honesty note:** This project uses synthetic data and a simulated network graph. The held-out metrics describe only a reproducible split of generated data; they are not evidence of real-world prediction accuracy. The model's risk score is an estimate, not a calibrated real-world probability. A risk prediction is distinct from a simulated router being marked DOWN. Recovery verification checks connectivity in the graph simulation only; it does not test real hosts or routers or apply operating-system routes. Recovery duration is the local elapsed time of the simulated route-selection and verification code.

## Recommended setup

Ubuntu 22.04/24.04 is recommended for the later Mininet extension. The dashboard prototype itself also runs on Windows, macOS, and Linux.

### 1. Install Python

Use Python 3.11 or 3.12. Verify:

```bash
python --version
```

On some Linux systems use `python3` instead of `python`.

### 2. Extract the project and open its folder

```bash
cd self_healing_network
```

### 3. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in your browser.

## Dashboard demonstration

1. Click **Start** under Live telemetry to continuously advance the simulated measurements; use **Pause** to stop the bounded fragment refresh.
2. Choose a refresh interval and configure latency, packet-loss, utilization, and predicted-risk warning thresholds in the sidebar.
3. Observe the per-router state table and use the historical Plotly chart tabs to inspect latency, packet loss, utilization, or AI-predicted failure risk.
4. Click **Advance simulation** manually when live telemetry is paused, or let it run while observing the trends.
5. Change the **High-risk threshold** to see how risk-aware route selection changes.
6. Keep **Enable automatic rerouting** on.
7. Choose R2 and click **Inject router failure**. R2 is marked down and the selected route should avoid it when an alternative path is available.
8. Select a link and click **Inject link failure** to simulate an unavailable edge and observe recovery.
9. Review the network analytics definitions, chronological incident timeline, and filtered persistent incident history. Export telemetry, metrics, or incidents as CSV.
10. Click **Restore / reset network and telemetry** to stop live updates and restore the simulation; the reset remains in persistent incident history.
11. Expand **Advanced simulation scenarios** to configure multiple failed routers and links plus synthetic congestion and packet loss, then choose a seed and run the scenario. Each run starts from a clean simulated topology; **Reset scenario** restores that baseline without changing operating-system networking.
12. Review the scenario root-cause explanation and weighted-path versus shortest-hop comparison. Use **Simulation event replay** to filter and review the persisted timestamped event sequence for an individual scenario.

The dashboard reports **HEALTHY** when a route is available and all routers are online and below the selected risk threshold, **DEGRADED** when a route exists but a router is down or high-risk, and **UNREACHABLE** when no active route is available. Automatic recovery verifies simulated graph connectivity before reporting success.

## Architecture

```text
Synthetic network telemetry
          |
          v
Reproducible synthetic telemetry and labels
          |
          v
Stratified train/test split -> Random Forest -> held-out evaluation
          |
          v
Live simulated telemetry -> risk-weighted graph with failed/down links excluded
          |
          v
Weighted shortest-path selection
          |
          v
Recovery state machine -> simulated connectivity verification
          |
          v
Dashboard + event log + recovery measurements
```

Scenario runs additionally apply validated fault combinations to a seeded clean simulation, persist their ordered event sequence in SQLite, explain failed-component/connectivity effects, and compare weighted and shortest-hop routes on the same post-injection graph. Strategy cost and hop counts describe only this simulated topology.

## Project modules

- `app.py`: Streamlit interface, topology visualization, controls, live Plotly telemetry, and event log.
- `network_engine.py`: network graph, synthetic telemetry, ML model, failure injection, and rerouting logic.
- `incident_store.py`: SQLite schema, incident and simulation-event persistence, filtering, analytics, replay retrieval, and CSV export.
- `requirements.txt`: Python dependencies.
- `tests/test_network_engine.py`: engine behavior, multi-fault scenarios, strategy comparison, replay, model reproducibility, input validation, and evaluation tests.
- `tests/test_incident_store.py`: SQLite migration/creation, incident and simulation-event persistence, filters, analytics formulas, CSV export, and validation tests.
- `tests/test_dashboard.py`: Streamlit scenario controls, event replay, analytics, timeline, chart, and CSV control rendering test.

The synthetic classifier uses a fixed seed and stratified train/test split. Only the training partition is used to fit the Random Forest; the dashboard displays metrics calculated on the held-out test partition. Feature importance describes the fitted model's use of these synthetic features, not causal importance in real networks.

On an injected router failure, the recovery workflow progresses through failure detection, rerouting, and simulated connectivity verification before it reports **RECOVERED** or **RECOVERY FAILED**. It records the pre-failure route, selected route, verification result, and elapsed local simulation-processing time. Repeating recovery after completion does not create a second recovery result for the same fault; **Restore / reset network** clears failures, unavailable links, and in-memory recovery history.

Router failures, dashboard or engine-injected link failures, recovery attempts, user-triggered resets, and timestamped telemetry snapshots are persisted in SQLite at `%USERPROFILE%\.self_healing_network\incidents.sqlite3` on Windows (or `~/.self_healing_network/incidents.sqlite3` on other platforms). Set `SELF_HEALING_NETWORK_DB` to use a different database path. Existing incident databases are extended with nullable detection/completion timestamps when opened; old records remain and missing lifecycle timestamps are excluded from duration averages.

Analytics definitions:

- **Simulated network availability (%)** = time in intervals with a reachable Host A to Host B simulated route divided by total observed interval time. Consecutive persisted telemetry observation timestamps define intervals; each interval uses the network reachability recorded at its starting observation. With fewer than two valid observations, availability is **N/A**.
- **MTTD (seconds)** = mean(`detected_at` − failure event timestamp) for router/link failure records with valid, ordered timestamps.
- **MTTR (seconds)** = mean(recovery completion timestamp − detection timestamp) for successfully recovered router/link failures with valid, ordered timestamps.
- **Recovery success rate (%)** = successful completed recovery attempts divided by all completed recovery attempts. In-progress attempts are excluded; with no completed attempts, the value is **N/A**.

Invalid/missing timestamps are excluded rather than replaced with fabricated values and their counts are shown. The reported duration metrics use local simulated event timestamps and are not real network service-level measurements. CSV exports provide telemetry, incident rows, or calculated values together with the metric unit and formula.

## Testing checklist

| Test | Action | Expected result |
|---|---|---|
| T1 Baseline | Start app | All routers show UP and a route is selected |
| T2 Telemetry | Advance 5–10 steps | Metrics, trends, risk scores, and events update |
| T3 Router failure | Fail R2 | R2 shows DOWN; an available alternate route avoids R2 and is verified in the simulation |
| T4 Alternate failure | Restore, then fail R3 | Route avoids R3 if a path remains |
| T5 Automatic rerouting off | Disable toggle and fail a router | No replacement route is silently selected; the old route is shown only if it remains available |
| T6 Risk threshold | Adjust threshold | Set of high-risk routers and selected route may change |
| T7 Reset | Reset network | All routers return UP and metrics are reinitialized |
| T8 No-path handling | Fail multiple routers in code or through repeated injection | App reports no available path rather than crashing |
| T9 Recovery verification | Inject R2 failure | Dashboard shows original/selected routes, recovery state, verification result, and simulated duration |
| T10 No-path recovery | Fail R1 or fail both middle routers in engine tests | Recovery is marked failed and no route is displayed as selected |

Run the engine regression tests from this project directory:

```bash
python -m unittest discover -s tests -v
```

## Evaluation metrics for the report

- **Recovery time:** time between injected failure and the first successful route calculation that avoids the failed component.
- **Packet loss:** in the simulation, report route availability rather than claiming real packet loss. With Mininet, measure packet loss using `ping` or `iperf`.
- **Route availability:** successful route calculations / total route calculations × 100%.
- **False positive rate:** high-risk predictions when no failure is injected / all no-failure observations.
- **Detection lead time:** how many simulation steps before a deliberately introduced failure the model first marks a component high risk. This requires a defined failure-injection experiment.

Do not report fabricated results. Run each test multiple times, record actual output, and compare baseline routing with risk-aware rerouting.

## Optional future Mininet extension

As a later extension, use an Ubuntu VM and Mininet to create a virtual topology with two hosts and redundant paths. Use an OpenFlow controller such as Ryu to install and change forwarding rules. Collect live measurements from host pings and interface counters, pass those measurements into the same feature pipeline, and have the controller reroute around a failed or high-risk link. Test only in your isolated Mininet lab. This is separate from the current synthetic-data phase; the dashboard does not control Mininet or a physical network.

## Individual work plan

1. Day 1: environment setup and run dashboard.
2. Day 2: understand topology and route calculation.
3. Day 3: understand features and synthetic training labels.
4. Day 4: run failure-injection tests and record metrics.
5. Day 5: improve UI, document limitations, and prepare demo/report.
6. Optional advanced phase: integrate Mininet and an SDN controller.
