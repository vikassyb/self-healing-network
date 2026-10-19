from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from math import isfinite
from time import perf_counter
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from incident_store import IncidentStore

ROUTERS = ["R1", "R2", "R3", "R4"]
FEATURES = ["latency_ms", "packet_loss_pct", "utilization_pct", "error_rate", "latency_trend"]
LINK_BASE_COSTS = {
    ("Host A", "R1"): 1.0,
    ("R1", "R2"): 1.0,
    ("R2", "R4"): 1.0,
    ("R1", "R3"): 1.4,
    ("R3", "R4"): 1.0,
    ("R2", "R3"): 1.2,
    ("R4", "Host B"): 1.0,
}
RECOVERY_STATES = (
    "IDLE",
    "FAILURE_DETECTED",
    "REROUTING",
    "RECOVERY_VERIFICATION",
    "RECOVERED",
    "RECOVERY_FAILED",
)
DEFAULT_TELEMETRY_THRESHOLDS = {
    "latency_ms": 100.0,
    "packet_loss_pct": 3.0,
    "utilization_pct": 85.0,
    "risk_score": 0.65,
}
MAX_TELEMETRY_SAMPLES = 300


def generate_synthetic_dataset(seed: int = 17, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate reproducible telemetry and simulated near-term failure labels."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer between 0 and 4294967295.")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 20:
        raise ValueError("n_samples must be an integer of at least 20.")

    rng = np.random.default_rng(seed)
    stress = rng.beta(2.0, 3.0, n_samples)
    latency = np.clip(8 + 115 * stress + rng.normal(0, 10, n_samples), 3, 220)
    utilization = np.clip(12 + 88 * stress + rng.normal(0, 9, n_samples), 5, 100)
    packet_loss = np.clip(0.05 + 12 * stress**2 + rng.normal(0, 0.8, n_samples), 0, 18)
    error_rate = np.clip(0.002 + 0.45 * stress**2 + rng.normal(0, 0.035, n_samples), 0, 0.6)
    latency_trend = np.clip(-3 + 36 * stress + rng.normal(0, 7, n_samples), -10, 40)

    # The label represents an independently generated simulated failure within a
    # hypothetical future window; it is not a real-world observation.
    failure_signal = (
        0.25 * (latency / 120)
        + 0.22 * (packet_loss / 12)
        + 0.22 * (utilization / 100)
        + 0.16 * (error_rate / 0.45)
        + 0.15 * ((latency_trend + 10) / 50)
        + rng.normal(0, 0.08, n_samples)
    )
    labels = (failure_signal >= 0.48).astype(int)
    features = pd.DataFrame({
        "latency_ms": latency,
        "packet_loss_pct": packet_loss,
        "utilization_pct": utilization,
        "error_rate": error_rate,
        "latency_trend": latency_trend,
    }, columns=FEATURES)
    return features, pd.Series(labels, name="simulated_failure")


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """Evaluate predictions against held-out synthetic labels."""
    if not isinstance(model, RandomForestClassifier) or not hasattr(model, "classes_"):
        raise ValueError("model must be a fitted RandomForestClassifier.")
    if not isinstance(X_test, pd.DataFrame) or not set(FEATURES).issubset(X_test.columns):
        raise ValueError(f"X_test must be a DataFrame containing: {', '.join(FEATURES)}.")
    try:
        telemetry = X_test[FEATURES].apply(pd.to_numeric, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError("X_test telemetry features must be numeric.") from exc
    if telemetry.empty or not np.isfinite(telemetry.to_numpy(dtype=float)).all():
        raise ValueError("X_test must contain finite telemetry values and at least one row.")
    if len(telemetry) != len(y_test) or len(y_test) == 0:
        raise ValueError("X_test and y_test must contain the same non-zero number of rows.")
    labels = np.asarray(y_test)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("y_test must contain only binary simulated failure labels (0 or 1).")

    predictions = model.predict(telemetry)
    class_one_index = list(model.classes_).index(1)
    risk_scores = model.predict_proba(telemetry)[:, class_one_index]
    roc_auc = float(roc_auc_score(labels, risk_scores)) if len(np.unique(labels)) == 2 else None
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "roc_auc": roc_auc,
    }


def train_and_evaluate_model(
    seed: int = 17,
    n_samples: int = 5000,
) -> Tuple[RandomForestClassifier, Dict[str, object], List[Dict[str, float]]]:
    """Fit only on a reproducibly split training set and return held-out evaluation."""
    features, labels = generate_synthetic_dataset(seed=seed, n_samples=n_samples)
    class_counts = labels.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Synthetic data must contain at least two examples of each failure label to split and evaluate.")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(X_train[FEATURES], y_train)
    evaluation = evaluate_model(model, X_test, y_test)
    importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(
            zip(FEATURES, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return model, evaluation, importance


def build_model(seed: int = 17) -> RandomForestClassifier:
    """Train and return the synthetic-data demo classifier."""
    model, _, _ = train_and_evaluate_model(seed=seed)
    return model


class NetworkEngine:
    def __init__(self, seed: int = 17, incident_store: Optional[IncidentStore] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.incident_store = incident_store or IncidentStore()
        self.model, self.model_evaluation, self.feature_importance = train_and_evaluate_model(seed)
        self._active_scenario_id: Optional[str] = None
        self.scenario_result: Optional[Dict[str, object]] = None
        self.reset(record_incident=False)

    def reset(
        self,
        record_incident: bool = True,
        scenario_id: Optional[str] = None,
    ) -> None:
        if not isinstance(record_incident, bool):
            raise ValueError("record_incident must be a boolean.")
        if scenario_id is not None and (
            not isinstance(scenario_id, str) or not scenario_id.strip()
        ):
            raise ValueError("Scenario ID must be a non-empty string or None.")
        self._active_scenario_id = scenario_id
        self.scenario_result = None
        if record_incident and hasattr(self, "status"):
            original_route, _ = self.compute_route(auto_heal=False)
            self.incident_store.record_incident(
                event_type="network_reset",
                component="network",
                original_route=original_route,
                recovery_outcome="NETWORK_RESET",
            )
        self.step = 0
        self.status: Dict[str, str] = {r: "UP" for r in ROUTERS}
        self.telemetry: Dict[str, Dict[str, float]] = {}
        for r in ROUTERS:
            self.telemetry[r] = {
                "latency_ms": self.rng.uniform(8, 35),
                "packet_loss_pct": self.rng.uniform(0, 0.5),
                "utilization_pct": self.rng.uniform(20, 65),
                "error_rate": self.rng.uniform(0, 0.03),
                "latency_trend": self.rng.uniform(-1, 2),
            }
        self.risk_scores: Dict[str, float] = {}
        self.link_status: Dict[Tuple[str, str], str] = {
            link: "UP" for link in LINK_BASE_COSTS
        }
        self.recovery_state = "IDLE"
        self.recovery: Dict[str, object] = {
            "original_route": None,
            "selected_route": None,
            "verified": None,
            "result": "No recovery attempt has been made.",
            "duration_ms": None,
        }
        self.recovery_history: List[Dict[str, object]] = []
        self.events: List[dict] = []
        self.history: List[dict] = []
        self._update_risks()
        self._record_history()
        self._event("INFO", "Network initialized", "All routers are online; baseline telemetry collected.")
        if record_incident:
            self._event("INFO", "Network reset", "Simulation state restored to the baseline topology.")

    def _event(
        self,
        level: str,
        event: str,
        details: str,
        component: str = "network",
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        item = {
            "Timestamp UTC": timestamp,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Step": self.step,
            "Level": level,
            "Event": event,
            "Component": component,
            "Details": details,
            "Scenario ID": self._active_scenario_id,
        }
        self.events.append(item)
        self.incident_store.record_simulation_event(
            level=level,
            event=event,
            component=component,
            details=details,
            step=self.step,
            scenario_id=self._active_scenario_id,
            timestamp=timestamp,
        )

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("Risk threshold must be a number between 0 and 1.")
        threshold = float(threshold)
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Risk threshold must be a finite number between 0 and 1.")
        return threshold

    @staticmethod
    def _validate_router(router: str) -> None:
        if router not in ROUTERS:
            raise ValueError(f"Unknown router: {router}")

    def _update_risks(self) -> None:
        for router, metrics in self.telemetry.items():
            row = pd.DataFrame([{k: metrics[k] for k in FEATURES}], columns=FEATURES)
            self.risk_scores[router] = float(self.model.predict_proba(row)[0][1])

    def _record_history(self) -> None:
        observed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        available = self._network_available()
        samples: Dict[str, Dict[str, object]] = {}
        for router, m in self.telemetry.items():
            sample = {
                "step": self.step,
                "timestamp": observed_at,
                "router": router,
                **m,
                "risk_score": self.risk_scores[router],
                "status": self.status[router],
            }
            self.history.append(sample)
            samples[router] = {
                **m,
                "risk_score": self.risk_scores[router],
                "status": self.status[router],
            }
        max_rows = MAX_TELEMETRY_SAMPLES * len(ROUTERS)
        if len(self.history) > max_rows:
            del self.history[:-max_rows]
        self.incident_store.record_telemetry_snapshot(
            step=self.step,
            samples=samples,
            network_available=available,
            observed_at=observed_at,
        )

    def _network_available(self) -> bool:
        graph = self._base_graph()
        graph.remove_nodes_from(
            router for router in ROUTERS if self.status[router] == "DOWN"
        )
        return nx.has_path(graph, "Host A", "Host B")

    @staticmethod
    def _validate_telemetry_thresholds(
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        validated = dict(DEFAULT_TELEMETRY_THRESHOLDS)
        if thresholds is not None and not isinstance(thresholds, dict):
            raise ValueError("Telemetry thresholds must be provided as a mapping.")
        if thresholds:
            unknown = set(thresholds) - set(validated)
            if unknown:
                raise ValueError(f"Unknown telemetry thresholds: {', '.join(sorted(unknown))}.")
            validated.update(thresholds)
        for name, value in validated.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Threshold {name} must be a finite non-negative number.")
        if (
            validated["packet_loss_pct"] > 100
            or validated["utilization_pct"] > 100
            or validated["risk_score"] > 1
        ):
            raise ValueError(
                "Packet-loss and utilization thresholds must be at most 100; risk threshold must be at most 1."
            )
        return validated

    def telemetry_alerts(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, object]]:
        validated = self._validate_telemetry_thresholds(thresholds)
        alerts: Dict[str, Dict[str, object]] = {}
        for router in ROUTERS:
            telemetry = self.telemetry[router]
            reasons: List[str] = []
            if telemetry["latency_ms"] >= validated["latency_ms"]:
                reasons.append(
                    f"Latency {telemetry['latency_ms']:.1f} ms >= {validated['latency_ms']:.1f} ms"
                )
            if telemetry["packet_loss_pct"] >= validated["packet_loss_pct"]:
                reasons.append(
                    f"Packet loss {telemetry['packet_loss_pct']:.2f}% ≥ "
                    f"{validated['packet_loss_pct']:.2f}%"
                )
            if telemetry["utilization_pct"] >= validated["utilization_pct"]:
                reasons.append(
                    f"Utilization {telemetry['utilization_pct']:.1f}% ≥ "
                    f"{validated['utilization_pct']:.1f}%"
                )
            if self.risk_scores[router] >= validated["risk_score"]:
                reasons.append(
                    f"Predicted risk {self.risk_scores[router] * 100:.1f}% ≥ "
                    f"{validated['risk_score'] * 100:.1f}%"
                )
            if self.status[router] == "DOWN":
                reasons.append("Router is marked DOWN in the simulation.")
                state = "FAILED"
            else:
                state = "HIGH RISK" if reasons else "HEALTHY"
            alerts[router] = {"state": state, "reasons": reasons}
        return alerts

    def advance(self, threshold: float = 0.65) -> None:
        threshold = self._validate_threshold(threshold)
        self.step += 1
        for router, m in self.telemetry.items():
            if self.status[router] == "DOWN":
                continue
            old_latency = m["latency_ms"]
            # Correlated drift creates an interpretable rising-risk scenario.
            stress = 1.0 if self.rng.random() < 0.13 else 0.0
            m["latency_ms"] = float(np.clip(old_latency + self.rng.uniform(-5, 7) + stress * self.rng.uniform(15, 35), 3, 220))
            m["latency_trend"] = float(np.clip(m["latency_ms"] - old_latency, -10, 40))
            m["utilization_pct"] = float(np.clip(m["utilization_pct"] + self.rng.uniform(-9, 12) + stress * 12, 5, 100))
            m["packet_loss_pct"] = float(np.clip(m["packet_loss_pct"] + self.rng.uniform(-0.35, 0.7) + stress * 1.2, 0, 18))
            m["error_rate"] = float(np.clip(m["error_rate"] + self.rng.uniform(-0.015, 0.025) + stress * 0.04, 0, 0.6))
        self._update_risks()
        self._record_history()
        self._event("INFO", "Telemetry updated", "Collected new metrics and recalculated AI risk scores.")
        for r in ROUTERS:
            if self.status[r] == "UP" and self.risk_scores[r] >= threshold:
                self._event("WARN", "Elevated failure risk", f"{r} has an elevated demo risk score ({self.risk_scores[r]*100:.1f}%).")

    def fail_router(
        self,
        router: str,
        threshold: float = 0.65,
        auto_heal: bool = True,
    ) -> None:
        self._validate_router(router)
        if self.status[router] == "DOWN":
            self._event("INFO", "Failure injection skipped", f"{router} is already down; existing recovery state is unchanged.")
            return
        original_route, _ = self.compute_route(threshold=threshold, auto_heal=auto_heal)
        failure_incident_id = self._record_failure_incident(router, original_route)
        self.status[router] = "DOWN"
        self._record_history()
        self._begin_recovery(router, original_route, failure_incident_id)
        self._event("CRITICAL", "Router failure detected", f"{router} was deliberately taken offline for the simulation.")
        if auto_heal:
            self.recover_network(threshold=threshold, auto_heal=True)

    def restore_router(self, router: str) -> None:
        self._validate_router(router)
        if self.status[router] == "UP":
            self._event("INFO", "Router restore skipped", f"{router} is already online.")
            return
        self.status[router] = "UP"
        self._record_history()
        self._event("INFO", "Router restored", f"{router} is back online in the simulation.")

    def set_link_status(
        self,
        node_a: str,
        node_b: str,
        status: str,
        threshold: float = 0.65,
        auto_heal: bool = True,
    ) -> None:
        threshold = self._validate_threshold(threshold)
        if not isinstance(auto_heal, bool):
            raise ValueError("Automatic rerouting setting must be a boolean.")
        link = (node_a, node_b)
        reverse_link = (node_b, node_a)
        if link not in LINK_BASE_COSTS and reverse_link not in LINK_BASE_COSTS:
            raise ValueError(f"Unknown network link: {node_a} - {node_b}")
        if status not in ("UP", "DOWN"):
            raise ValueError("Link status must be 'UP' or 'DOWN'.")
        canonical_link = link if link in LINK_BASE_COSTS else reverse_link
        if self.link_status[canonical_link] == status:
            self._event("INFO", "Link status change skipped", f"Simulated link {canonical_link[0]} - {canonical_link[1]} is already {status}.")
            return
        original_route, _ = self.compute_route(threshold=threshold, auto_heal=auto_heal)
        failed_component = f"link {canonical_link[0]} - {canonical_link[1]}"
        if status == "DOWN":
            failure_incident_id = self._record_failure_incident(failed_component, original_route)
        else:
            failure_incident_id = None
        self.link_status[canonical_link] = status
        self._record_history()
        self._event("INFO" if status == "UP" else "WARN", "Link status changed", f"Simulated link {canonical_link[0]} - {canonical_link[1]} is {status}.")
        if status == "DOWN":
            self._begin_recovery(failed_component, original_route, failure_incident_id)
            self.recover_network(threshold=threshold, auto_heal=auto_heal)

    @staticmethod
    def _validate_scenario_names(
        values: Optional[List[str]],
        allowed: List[str],
        label: str,
    ) -> List[str]:
        if values is None:
            return []
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{label} must be a list or tuple.")
        selected = list(values)
        if any(not isinstance(value, str) for value in selected):
            raise ValueError(f"{label} must contain only component names.")
        if len(selected) != len(set(selected)):
            raise ValueError(f"{label} cannot contain duplicates.")
        unknown = [value for value in selected if value not in allowed]
        if unknown:
            raise ValueError(f"Unknown {label.lower()}: {', '.join(map(str, unknown))}.")
        return selected

    def compare_routing_strategies(
        self,
        threshold: float = 0.65,
        auto_heal: bool = True,
    ) -> List[Dict[str, object]]:
        """Compare weighted routing and shortest-hop routing on the same live graph."""
        threshold = self._validate_threshold(threshold)
        if not isinstance(auto_heal, bool):
            raise ValueError("Automatic rerouting setting must be a boolean.")

        graph = self._base_graph()
        excluded_routers = [
            router
            for router in ROUTERS
            if self.status[router] == "DOWN"
            or (auto_heal and self.risk_scores[router] >= threshold)
        ]
        graph.remove_nodes_from(excluded_routers)
        weighted_route, _ = self.compute_route(
            threshold=threshold,
            auto_heal=auto_heal,
        )
        try:
            shortest_hop_route = nx.shortest_path(graph, "Host A", "Host B")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            shortest_hop_route = None

        def describe(route: Optional[List[str]]) -> Dict[str, object]:
            available = self.route_is_available(route)
            weighted_cost = None
            if route and available:
                weighted_cost = sum(
                    graph[u][v]["weight"]
                    for u, v in zip(route, route[1:])
                )
            return {
                "route": route,
                "available": available,
                "hop_count": len(route) - 1 if route else None,
                "weighted_cost": weighted_cost,
            }

        return [
            {"strategy": "Weighted path", **describe(weighted_route)},
            {"strategy": "Shortest hop", **describe(shortest_hop_route)},
        ]

    def run_scenario(
        self,
        router_failures: Optional[List[str]] = None,
        link_failures: Optional[List[Tuple[str, str]]] = None,
        congested_routers: Optional[List[str]] = None,
        packet_loss_by_router: Optional[Dict[str, float]] = None,
        congestion_utilization_pct: float = 90.0,
        seed: Optional[int] = None,
        threshold: float = 0.65,
        auto_heal: bool = True,
    ) -> Dict[str, object]:
        """Run a reproducible multi-fault scenario entirely in the graph simulation."""
        threshold = self._validate_threshold(threshold)
        if not isinstance(auto_heal, bool):
            raise ValueError("Automatic rerouting setting must be a boolean.")
        scenario_seed = self.seed if seed is None else seed
        if (
            isinstance(scenario_seed, bool)
            or not isinstance(scenario_seed, int)
            or not 0 <= scenario_seed <= 2**32 - 1
        ):
            raise ValueError("Scenario seed must be an integer between 0 and 4294967295.")
        if (
            isinstance(congestion_utilization_pct, bool)
            or not isinstance(congestion_utilization_pct, (int, float))
            or not isfinite(congestion_utilization_pct)
            or not 0 <= congestion_utilization_pct <= 100
        ):
            raise ValueError("Congestion utilization must be a finite percentage from 0 to 100.")

        selected_routers = self._validate_scenario_names(
            router_failures, ROUTERS, "Router failures"
        )
        congestion_targets = self._validate_scenario_names(
            congested_routers, ROUTERS, "Congested routers"
        )
        if packet_loss_by_router is not None and not isinstance(packet_loss_by_router, dict):
            raise ValueError("Packet loss must map router names to percentages.")
        loss_targets = packet_loss_by_router or {}
        if any(not isinstance(router, str) for router in loss_targets):
            raise ValueError("Packet-loss router names must be strings.")
        unknown_loss_targets = set(loss_targets) - set(ROUTERS)
        if unknown_loss_targets:
            raise ValueError(
                f"Unknown packet-loss routers: {', '.join(sorted(unknown_loss_targets))}."
            )
        for router, percentage in loss_targets.items():
            if (
                isinstance(percentage, bool)
                or not isinstance(percentage, (int, float))
                or not isfinite(percentage)
                or not 0 <= percentage <= 18
            ):
                raise ValueError(
                    f"Packet loss for {router} must be a finite percentage from 0 to 18."
                )

        if link_failures is not None and not isinstance(link_failures, (list, tuple)):
            raise ValueError("Link failures must be a list or tuple of connected node pairs.")
        canonical_links: List[Tuple[str, str]] = []
        for selected_link in link_failures or []:
            if not isinstance(selected_link, (list, tuple)) or len(selected_link) != 2:
                raise ValueError("Each link failure must identify two connected nodes.")
            node_a, node_b = selected_link
            if not isinstance(node_a, str) or not isinstance(node_b, str):
                raise ValueError("Link endpoints must be node names.")
            link = (node_a, node_b)
            reverse_link = (node_b, node_a)
            if link not in LINK_BASE_COSTS and reverse_link not in LINK_BASE_COSTS:
                raise ValueError(f"Unknown network link: {node_a} - {node_b}")
            canonical = link if link in LINK_BASE_COSTS else reverse_link
            if canonical in canonical_links:
                raise ValueError("Link failures cannot contain duplicate links.")
            canonical_links.append(canonical)

        self.rng = random.Random(scenario_seed)
        scenario_id = f"scenario-{scenario_seed}-{uuid4().hex}"
        self.reset(record_incident=True, scenario_id=scenario_id)
        self._event(
            "INFO",
            "Scenario started",
            f"Started reproducible scenario with seed {scenario_seed}.",
        )
        original_route, _ = self.compute_route(
            threshold=threshold,
            auto_heal=False,
        )

        stress_components: List[str] = []
        for router in congestion_targets:
            metrics = self.telemetry[router]
            metrics["utilization_pct"] = max(
                metrics["utilization_pct"],
                float(congestion_utilization_pct),
            )
            metrics["latency_ms"] = max(
                metrics["latency_ms"],
                min(220.0, 20.0 + metrics["utilization_pct"] * 0.8),
            )
            metrics["latency_trend"] = max(
                metrics["latency_trend"],
                min(40.0, metrics["utilization_pct"] / 5.0),
            )
            stress_components.append(f"congestion at {router}")
            self._event(
                "WARN",
                "Congestion injected",
                f"{router} utilization set to at least {metrics['utilization_pct']:.1f}%.",
                component=router,
            )
        for router, percentage in loss_targets.items():
            self.telemetry[router]["packet_loss_pct"] = float(percentage)
            stress_components.append(f"packet loss at {router}")
            self._event(
                "WARN",
                "Packet loss injected",
                f"{router} packet loss set to {percentage:.2f}%.",
                component=router,
            )
        self._update_risks()

        failures: List[Tuple[str, int]] = []
        for router in selected_routers:
            component = f"router {router}"
            incident_id = self._record_failure_incident(component, original_route)
            self.status[router] = "DOWN"
            failures.append((component, incident_id))
        for node_a, node_b in canonical_links:
            component = f"link {node_a} - {node_b}"
            incident_id = self._record_failure_incident(component, original_route)
            self.link_status[(node_a, node_b)] = "DOWN"
            failures.append((component, incident_id))

        self._record_history()
        for component, incident_id in failures:
            self._begin_recovery(component, original_route, incident_id)
            self._event(
                "CRITICAL",
                "Scenario failure injected",
                f"{component} was made unavailable by the scenario.",
                component=component,
            )

        route_after_faults, _ = self.compute_route(
            threshold=threshold,
            auto_heal=False,
        )
        route_edges = (
            set(zip(original_route, original_route[1:]))
            if original_route
            else set()
        )
        failed_original_route = []
        for component, _ in failures:
            if component.startswith("router ") and component.removeprefix("router ") in (
                original_route or []
            ):
                failed_original_route.append(component)
            elif component.startswith("link "):
                link_nodes = component.removeprefix("link ").split(" - ", 1)
                if len(link_nodes) == 2 and (
                    tuple(link_nodes) in route_edges
                    or tuple(reversed(link_nodes)) in route_edges
                ):
                    failed_original_route.append(component)

        causes = [component for component, _ in failures] + stress_components
        if failures:
            affected = (
                "The original route crossed a failed component."
                if failed_original_route
                else "The original route did not cross the injected failed components."
            )
            connectivity = (
                "At least one route remained after injection."
                if route_after_faults
                else "No Host A to Host B route remained after injection."
            )
            explanation = " ".join((affected, connectivity))
        elif stress_components:
            explanation = (
                "No router or link was disabled; the listed congestion and packet-loss "
                "conditions changed simulated telemetry and route costs."
            )
        else:
            explanation = "No fault or stress condition was selected; baseline connectivity remained."
        root_cause = {
            "components": causes,
            "original_route": original_route,
            "route_after_injection": route_after_faults,
            "failed_components_on_original_route": failed_original_route,
            "route_available_after_injection": route_after_faults is not None,
            "explanation": explanation,
        }
        strategy_comparison = self.compare_routing_strategies(
            threshold=threshold,
            auto_heal=auto_heal,
        )
        self._event(
            "INFO",
            "Root-cause analysis",
            explanation,
            component=", ".join(causes) if causes else "network",
        )
        self._event(
            "INFO",
            "Routing strategy comparison",
            json.dumps(strategy_comparison, sort_keys=True),
        )
        if failures:
            recovery = self.recover_network(
                threshold=threshold,
                auto_heal=auto_heal,
            )
            recovery_state = self.recovery_state
            selected_route = recovery.get("selected_route")
            recovery_verified = recovery.get("verified")
        else:
            recovery_state = "NOT_REQUIRED"
            selected_route = strategy_comparison[0]["route"]
            recovery_verified = self.route_is_available(selected_route)

        self._event(
            "INFO",
            "Scenario completed",
            f"Recovery state: {recovery_state}; simulated connectivity verified: "
            f"{recovery_verified}.",
        )
        self.scenario_result = {
            "scenario_id": scenario_id,
            "seed": scenario_seed,
            "root_cause": root_cause,
            "strategy_comparison": strategy_comparison,
            "recovery_state": recovery_state,
            "selected_route": selected_route,
            "recovery_verified": recovery_verified,
        }
        return dict(self.scenario_result)

    def recover_network(self, threshold: float = 0.65, auto_heal: bool = True) -> Dict[str, object]:
        threshold = self._validate_threshold(threshold)
        if not isinstance(auto_heal, bool):
            raise ValueError("Automatic rerouting setting must be a boolean.")
        if self.recovery_state in ("REROUTING", "RECOVERY_VERIFICATION"):
            return dict(self.recovery)
        if self.recovery_state != "FAILURE_DETECTED":
            return dict(self.recovery)

        components = self.recovery.get("failed_components", [])
        if not components and self.recovery.get("failed_component"):
            components = [self.recovery["failed_component"]]
        incident_id = self.incident_store.record_incident(
            event_type="recovery_attempt",
            component=", ".join(components) if components else "unknown",
            original_route=self.recovery.get("original_route"),
            recovery_outcome="IN_PROGRESS",
        )
        self.recovery["incident_id"] = incident_id
        started_at = perf_counter()
        if not auto_heal:
            self.recovery_state = "RECOVERY_FAILED"
            result = (
                "Automatic rerouting is disabled. The previous route was not changed; "
                "it may be unavailable because of the detected failure."
            )
            self.recovery.update({
                "selected_route": None,
                "verified": False,
                "result": result,
                "duration_ms": (perf_counter() - started_at) * 1000,
                "rerouting_enabled": False,
            })
            self._record_recovery_incident()
            self._complete_failure_incidents()
            self.recovery_history.append({
                **self.recovery,
                "state": self.recovery_state,
            })
            self._event("WARN", "Automatic rerouting disabled", result)
            return dict(self.recovery)

        self.recovery_state = "REROUTING"
        self._event("ACTION", "Rerouting", "Selecting a weighted route that excludes failed routers and unavailable links.")
        route, route_note = self.compute_route(threshold=threshold, auto_heal=auto_heal)
        self.recovery["selected_route"] = route
        self.recovery_state = "RECOVERY_VERIFICATION"
        self._event("ACTION", "Recovery verification", "Checking simulated end-to-end connectivity on the selected route.")

        verified = self._verify_connectivity(route)
        duration_ms = (perf_counter() - started_at) * 1000
        if verified:
            self.recovery_state = "RECOVERED"
            result = f"Simulated connectivity verified. {route_note}"
            self._event("INFO", "Recovery verified", result)
        else:
            self.recovery_state = "RECOVERY_FAILED"
            result = f"Recovery failed: no valid simulated end-to-end connectivity. {route_note}"
            self._event("ERROR", "Recovery failed", result)

        self.recovery.update({
            "verified": verified,
            "result": result,
            "duration_ms": duration_ms,
            "rerouting_enabled": True,
        })
        self._record_recovery_incident()
        self._complete_failure_incidents()
        self.recovery_history.append({
            **self.recovery,
            "state": self.recovery_state,
            "duration_ms": duration_ms,
        })
        return dict(self.recovery)

    def _record_recovery_incident(self) -> None:
        completed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.recovery["completed_at"] = completed_at
        self.incident_store.complete_recovery(
            incident_id=self.recovery["incident_id"],
            replacement_route=self.recovery.get("selected_route"),
            recovery_outcome=self.recovery_state,
            recovery_duration_ms=self.recovery["duration_ms"],
            completed_at=completed_at,
        )

    def _complete_failure_incidents(self) -> None:
        for incident_id in self.recovery.get("failure_incident_ids", []):
            self.incident_store.complete_failure(
                incident_id=incident_id,
                recovery_outcome=self.recovery_state,
                completed_at=self.recovery["completed_at"],
            )

    def _verify_connectivity(self, route: Optional[List[str]]) -> bool:
        if not route or route[0] != "Host A" or route[-1] != "Host B":
            return False
        if any(self.status.get(node) == "DOWN" for node in route):
            return False
        graph = self._base_graph()
        return all(graph.has_edge(node_a, node_b) for node_a, node_b in zip(route, route[1:]))

    def route_is_available(self, route: Optional[List[str]]) -> bool:
        return self._verify_connectivity(route)

    def _begin_recovery(
        self,
        failed_component: str,
        original_route: Optional[List[str]],
        failure_incident_id: Optional[int] = None,
    ) -> None:
        if failure_incident_id is None:
            failure_incident_id = self._record_failure_incident(failed_component, original_route)
        detected_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.incident_store.mark_failure_detected(failure_incident_id, detected_at)
        if self.recovery_state == "FAILURE_DETECTED":
            components = self.recovery.setdefault("failed_components", [])
            if failed_component not in components:
                components.append(failed_component)
            incident_ids = self.recovery.setdefault("failure_incident_ids", [])
            incident_ids.append(failure_incident_id)
            self._event("WARN", "Additional failure detected", f"{failed_component} failed before recovery completed.")
            return
        self.recovery_state = "FAILURE_DETECTED"
        self.recovery = {
            "original_route": original_route,
            "selected_route": None,
            "verified": None,
            "result": f"Failure detected at {failed_component}; recovery has not run yet.",
            "duration_ms": None,
            "failed_component": failed_component,
            "failed_components": [failed_component],
            "failure_incident_ids": [failure_incident_id],
            "rerouting_enabled": None,
        }
        self._event("ACTION", "Failure detected", f"Recovery workflow started for failed {failed_component}.")

    def _record_failure_incident(
        self,
        failed_component: str,
        original_route: Optional[List[str]],
    ) -> int:
        event_type = "link_failure" if failed_component.startswith("link ") else "router_failure"
        return self.incident_store.record_incident(
            event_type=event_type,
            component=failed_component,
            original_route=original_route,
            recovery_outcome="FAILURE_DETECTED",
        )

    def _base_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(["Host A", *ROUTERS, "Host B"])
        for (u, v), base_cost in LINK_BASE_COSTS.items():
            if self.link_status[(u, v)] != "UP":
                continue
            metric_penalty = 0.0
            for node in (u, v):
                if node in ROUTERS:
                    m = self.telemetry[node]
                    metric_penalty += (
                        m["latency_ms"] / 220
                        + m["packet_loss_pct"] / 18
                        + m["utilization_pct"] / 100
                        + m["error_rate"] / 0.6
                        + max(m["latency_trend"], 0) / 40
                        + self.risk_scores[node]
                    )
            graph.add_edge(u, v, weight=base_cost + metric_penalty)
        return graph

    def compute_route(self, threshold: float = 0.65, auto_heal: bool = True) -> Tuple[Optional[List[str]], str]:
        threshold = self._validate_threshold(threshold)
        if not isinstance(auto_heal, bool):
            raise ValueError("Automatic rerouting setting must be a boolean.")
        graph = self._base_graph()
        failed = [r for r in ROUTERS if self.status[r] == "DOWN"]
        high_risk = [r for r in ROUTERS if self.status[r] == "UP" and self.risk_scores[r] >= threshold]
        avoided_high_risk = high_risk if auto_heal else []
        excluded = sorted(set(failed + avoided_high_risk))
        graph.remove_nodes_from(excluded)
        try:
            route = nx.shortest_path(graph, "Host A", "Host B", weight="weight")
            if failed and avoided_high_risk:
                note = f"Route avoids failed routers ({', '.join(failed)}) and high-risk routers ({', '.join(avoided_high_risk)})."
                return route, note
            if failed:
                if high_risk:
                    return route, f"Route avoids failed routers: {', '.join(failed)}. High-risk avoidance is disabled."
                return route, f"Route avoids failed routers: {', '.join(failed)}."
            if avoided_high_risk:
                return route, f"Automatic rerouting avoids high-risk routers: {', '.join(avoided_high_risk)}."
            if high_risk:
                return route, f"No failed routers; high-risk avoidance is disabled for: {', '.join(high_risk)}."
            return route, "No component currently requires avoidance; route selected by weighted path cost."
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, "No route is available under the current failure and risk settings. Restore a router or adjust the risk threshold."

    def graph_for_display(self) -> nx.Graph:
        # Keep failed nodes visible while omitting unavailable links.
        return self._base_graph()

    def current_metrics(self) -> dict:
        return {r: dict(self.telemetry[r], risk_score=self.risk_scores[r], status=self.status[r]) for r in ROUTERS}

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
