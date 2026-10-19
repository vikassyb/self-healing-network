from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

INCIDENT_FIELDS = (
    "id",
    "timestamp",
    "event_type",
    "component",
    "detected_at",
    "recovery_completed_at",
    "original_route",
    "replacement_route",
    "recovery_outcome",
    "recovery_duration_ms",
)
TELEMETRY_FIELDS = (
    "observed_at",
    "step",
    "router",
    "latency_ms",
    "packet_loss_pct",
    "utilization_pct",
    "error_rate",
    "latency_trend",
    "risk_score",
    "status",
    "network_available",
)
SIMULATION_EVENT_FIELDS = (
    "id",
    "timestamp",
    "scenario_id",
    "step",
    "level",
    "event",
    "component",
    "details",
)
METRIC_DEFINITIONS = {
    "network_availability_pct": (
        "%",
        "100 * observed-available interval seconds / total observed interval seconds",
    ),
    "mttd_seconds": (
        "seconds",
        "mean(detected_at - failure timestamp) for failures with valid ordered timestamps",
    ),
    "mttr_seconds": (
        "seconds",
        "mean(recovery completion - detection) for successfully recovered failures",
    ),
    "recovery_success_rate_pct": (
        "%",
        "100 * successful completed recovery attempts / all completed recovery attempts",
    ),
}


class IncidentStore:
    """SQLite-backed persistence for simulated network incidents."""

    def __init__(self, database_path: Optional[str | Path] = None):
        configured_path = database_path or os.environ.get("SELF_HEALING_NETWORK_DB")
        self.database_path = (
            Path(configured_path)
            if configured_path is not None
            else Path.home() / ".self_healing_network" / "incidents.sqlite3"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    detected_at TEXT,
                    recovery_completed_at TEXT,
                    original_route TEXT,
                    replacement_route TEXT,
                    recovery_outcome TEXT,
                    recovery_duration_ms REAL
                )
                """
            )
            incident_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(incidents)").fetchall()
            }
            for column in ("detected_at", "recovery_completed_at"):
                if column not in incident_columns:
                    connection.execute(f"ALTER TABLE incidents ADD COLUMN {column} TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_event_type ON incidents(event_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_component ON incidents(component)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_recovery_outcome ON incidents(recovery_outcome)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    router TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    packet_loss_pct REAL NOT NULL,
                    utilization_pct REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    latency_trend REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    network_available INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_observed_at ON telemetry_samples(observed_at, router)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scenario_id TEXT,
                    step INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    component TEXT NOT NULL,
                    details TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_simulation_events_scenario "
                "ON simulation_events(scenario_id, id)"
            )

    @staticmethod
    def _validated_timestamp(timestamp: Optional[str] = None) -> str:
        if timestamp is None:
            parsed = datetime.now(timezone.utc)
        else:
            if not isinstance(timestamp, str):
                raise ValueError("Timestamps must be ISO-8601 strings with a timezone.")
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("Timestamps must be valid ISO-8601 values.") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("Timestamps must include a timezone.")
        return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")

    @staticmethod
    def _parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
        if not isinstance(timestamp, str) or not timestamp:
            return None
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _serialize_route(route: Optional[Sequence[str]]) -> Optional[str]:
        if route is None:
            return None
        if not isinstance(route, (list, tuple)) or not all(isinstance(node, str) for node in route):
            raise ValueError("Routes must be a sequence of node names or None.")
        return json.dumps(list(route), ensure_ascii=True)

    @staticmethod
    def _display_route(route_json: Optional[str]) -> Optional[str]:
        if route_json is None:
            return None
        route = json.loads(route_json)
        return " -> ".join(route)

    @staticmethod
    def _validate_filter(values: Optional[Sequence[str]]) -> List[str]:
        if values is None:
            return []
        result = [values] if isinstance(values, str) else list(values)
        if any(not isinstance(value, str) or not value.strip() for value in result):
            raise ValueError("Incident filters must contain non-empty strings.")
        return result

    def record_incident(
        self,
        event_type: str,
        component: str,
        original_route: Optional[Sequence[str]] = None,
        replacement_route: Optional[Sequence[str]] = None,
        recovery_outcome: Optional[str] = None,
        recovery_duration_ms: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("Incident event_type must be a non-empty string.")
        if not isinstance(component, str) or not component.strip():
            raise ValueError("Incident component must be a non-empty string.")
        if recovery_outcome is not None and (
            not isinstance(recovery_outcome, str) or not recovery_outcome.strip()
        ):
            raise ValueError("Recovery outcome must be a non-empty string or None.")
        if recovery_duration_ms is not None:
            if (
                isinstance(recovery_duration_ms, bool)
                or not isinstance(recovery_duration_ms, (int, float))
                or not math.isfinite(recovery_duration_ms)
                or recovery_duration_ms < 0
            ):
                raise ValueError("Recovery duration must be a finite non-negative number or None.")

        timestamp = self._validated_timestamp(timestamp)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO incidents (
                    timestamp, event_type, component, original_route,
                    replacement_route, recovery_outcome, recovery_duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    event_type.strip(),
                    component.strip(),
                    self._serialize_route(original_route),
                    self._serialize_route(replacement_route),
                    recovery_outcome,
                    recovery_duration_ms,
                ),
            )
            return int(cursor.lastrowid)

    def complete_recovery(
        self,
        incident_id: int,
        replacement_route: Optional[Sequence[str]],
        recovery_outcome: str,
        recovery_duration_ms: float,
        completed_at: Optional[str] = None,
    ) -> None:
        if not isinstance(recovery_outcome, str) or not recovery_outcome.strip():
            raise ValueError("Recovery outcome must be a non-empty string.")
        if (
            isinstance(recovery_duration_ms, bool)
            or not isinstance(recovery_duration_ms, (int, float))
            or not math.isfinite(recovery_duration_ms)
            or recovery_duration_ms < 0
        ):
            raise ValueError("Recovery duration must be a finite non-negative number.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE incidents
                SET replacement_route = ?, recovery_outcome = ?, recovery_duration_ms = ?,
                    recovery_completed_at = ?
                WHERE id = ? AND event_type = 'recovery_attempt'
                """,
                (
                    self._serialize_route(replacement_route),
                    recovery_outcome.strip(),
                    recovery_duration_ms,
                    self._validated_timestamp(completed_at),
                    incident_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Recovery incident {incident_id} could not be completed.")

    def mark_failure_detected(
        self,
        incident_id: int,
        detected_at: Optional[str] = None,
    ) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE incidents SET detected_at = ?
                WHERE id = ? AND event_type IN ('router_failure', 'link_failure')
                """,
                (self._validated_timestamp(detected_at), incident_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Failure incident {incident_id} could not be marked detected.")

    def complete_failure(
        self,
        incident_id: int,
        recovery_outcome: str,
        completed_at: Optional[str] = None,
    ) -> None:
        if not isinstance(recovery_outcome, str) or not recovery_outcome.strip():
            raise ValueError("Recovery outcome must be a non-empty string.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE incidents
                SET recovery_outcome = ?, recovery_completed_at = ?
                WHERE id = ? AND event_type IN ('router_failure', 'link_failure')
                """,
                (
                    recovery_outcome.strip(),
                    self._validated_timestamp(completed_at),
                    incident_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Failure incident {incident_id} could not be completed.")

    def record_telemetry_snapshot(
        self,
        step: int,
        samples: Dict[str, Dict[str, object]],
        network_available: bool,
        observed_at: Optional[str] = None,
    ) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Telemetry step must be a non-negative integer.")
        if not isinstance(samples, dict) or not samples:
            raise ValueError("Telemetry snapshot must contain router samples.")
        if not isinstance(network_available, bool):
            raise ValueError("Network availability must be a boolean.")
        timestamp = self._validated_timestamp(observed_at)
        rows = []
        expected = (
            "latency_ms",
            "packet_loss_pct",
            "utilization_pct",
            "error_rate",
            "latency_trend",
            "risk_score",
        )
        for router, sample in samples.items():
            if not isinstance(router, str) or not router.strip() or not isinstance(sample, dict):
                raise ValueError("Telemetry samples must map router names to metric mappings.")
            values = []
            for field in expected:
                value = sample.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Telemetry metric {field} must be finite and numeric.")
                values.append(float(value))
            status = sample.get("status")
            if status not in ("UP", "DOWN"):
                raise ValueError("Telemetry router status must be 'UP' or 'DOWN'.")
            rows.append((timestamp, step, router, *values, status, int(network_available)))

        with self._connection() as connection:
            connection.executemany(
                """
                INSERT INTO telemetry_samples (
                    observed_at, step, router, latency_ms, packet_loss_pct,
                    utilization_pct, error_rate, latency_trend, risk_score,
                    status, network_available
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_simulation_event(
        self,
        level: str,
        event: str,
        component: str,
        details: str,
        step: int,
        scenario_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        for name, value in (
            ("level", level),
            ("event", event),
            ("component", component),
            ("details", details),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Simulation event {name} must be a non-empty string.")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Simulation event step must be a non-negative integer.")
        if scenario_id is not None and (
            not isinstance(scenario_id, str) or not scenario_id.strip()
        ):
            raise ValueError("Scenario ID must be a non-empty string or None.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO simulation_events (
                    timestamp, scenario_id, step, level, event, component, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._validated_timestamp(timestamp),
                    scenario_id,
                    step,
                    level.strip(),
                    event.strip(),
                    component.strip(),
                    details.strip(),
                ),
            )
            return int(cursor.lastrowid)

    def list_simulation_events(
        self,
        scenario_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        if scenario_id is not None and (
            not isinstance(scenario_id, str) or not scenario_id.strip()
        ):
            raise ValueError("Scenario ID must be a non-empty string or None.")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError("Simulation event limit must be a positive integer or None.")
        conditions = []
        parameters: List[object] = []
        if scenario_id is not None:
            conditions.append("scenario_id = ?")
            parameters.append(scenario_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM simulation_events"
                + where
                + " ORDER BY id DESC"
                + limit_clause,
                parameters,
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_simulation_scenarios(self) -> List[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT scenario_id
                FROM simulation_events
                WHERE scenario_id IS NOT NULL
                GROUP BY scenario_id
                ORDER BY MAX(id) DESC
                """
            ).fetchall()
        return [row["scenario_id"] for row in rows]

    def list_telemetry(
        self,
        limit_per_router: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        if limit_per_router is not None and (
            isinstance(limit_per_router, bool)
            or not isinstance(limit_per_router, int)
            or limit_per_router <= 0
        ):
            raise ValueError("Telemetry history limit must be a positive integer or None.")
        with self._connection() as connection:
            query = """
                SELECT observed_at, step, router, latency_ms, packet_loss_pct,
                       utilization_pct, error_rate, latency_trend, risk_score,
                       status, network_available
                FROM telemetry_samples
            """
            parameters: Tuple[int, ...] = ()
            if limit_per_router is not None:
                query = f"""
                    SELECT observed_at, step, router, latency_ms, packet_loss_pct,
                           utilization_pct, error_rate, latency_trend, risk_score,
                           status, network_available
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY router ORDER BY observed_at DESC, id DESC
                        ) AS row_rank
                        FROM telemetry_samples
                    )
                    WHERE row_rank <= ?
                """
                parameters = (limit_per_router,)
            query += " ORDER BY observed_at, id"
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def export_telemetry_csv(self) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=TELEMETRY_FIELDS)
        writer.writeheader()
        writer.writerows(self.list_telemetry())
        return output.getvalue()

    @staticmethod
    def _where_clause(
        event_types: Optional[Sequence[str]],
        components: Optional[Sequence[str]],
        recovery_outcomes: Optional[Sequence[str]],
    ) -> Tuple[str, List[str]]:
        conditions = []
        parameters: List[str] = []
        for column, values in (
            ("event_type", event_types),
            ("component", components),
            ("recovery_outcome", recovery_outcomes),
        ):
            selected = IncidentStore._validate_filter(values)
            if selected:
                placeholders = ", ".join("?" for _ in selected)
                conditions.append(f"{column} IN ({placeholders})")
                parameters.extend(selected)
        return (" WHERE " + " AND ".join(conditions) if conditions else "", parameters)

    def list_incidents(
        self,
        event_types: Optional[Sequence[str]] = None,
        components: Optional[Sequence[str]] = None,
        recovery_outcomes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        where, parameters = self._where_clause(event_types, components, recovery_outcomes)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM incidents{where} ORDER BY timestamp DESC, id DESC",
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["original_route"] = self._display_route(item["original_route"])
            item["replacement_route"] = self._display_route(item["replacement_route"])
            result.append(item)
        return result

    def filter_options(self) -> Dict[str, List[str]]:
        options = {}
        with self._connection() as connection:
            for column, key in (
                ("event_type", "event_types"),
                ("component", "components"),
                ("recovery_outcome", "recovery_outcomes"),
            ):
                rows = connection.execute(
                    f"SELECT DISTINCT {column} FROM incidents WHERE {column} IS NOT NULL ORDER BY {column}"
                ).fetchall()
                options[key] = [row[0] for row in rows]
        return options

    def analytics(
        self,
        event_types: Optional[Sequence[str]] = None,
        components: Optional[Sequence[str]] = None,
        recovery_outcomes: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        where, parameters = self._where_clause(event_types, components, recovery_outcomes)
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total_incidents,
                    SUM(CASE WHEN event_type = 'recovery_attempt'
                              AND recovery_outcome = 'RECOVERED' THEN 1 ELSE 0 END)
                        AS successful_recoveries,
                    SUM(CASE WHEN event_type = 'recovery_attempt'
                              AND recovery_outcome = 'RECOVERY_FAILED' THEN 1 ELSE 0 END)
                        AS failed_recoveries,
                    AVG(CASE WHEN event_type = 'recovery_attempt'
                              THEN recovery_duration_ms END)
                        AS average_recovery_duration_ms
                FROM incidents{where}
                """,
                parameters,
            ).fetchone()
            incidents = connection.execute(
                f"""
                SELECT timestamp, detected_at, recovery_completed_at,
                       event_type, recovery_outcome
                FROM incidents{where}
                ORDER BY timestamp, id
                """,
                parameters,
            ).fetchall()
            observations = connection.execute(
                """
                SELECT observed_at, network_available
                FROM telemetry_samples
                GROUP BY observed_at, network_available
                ORDER BY observed_at
                """
            ).fetchall()

        failure_events = [
            incident
            for incident in incidents
            if incident["event_type"] in ("router_failure", "link_failure")
        ]
        detection_durations = []
        incomplete_detection_timestamps = 0
        successful_failure_rows = []
        for incident in failure_events:
            occurred_at = self._parse_timestamp(incident["timestamp"])
            detected_at = self._parse_timestamp(incident["detected_at"])
            if occurred_at is None or detected_at is None or detected_at < occurred_at:
                incomplete_detection_timestamps += 1
            else:
                detection_durations.append((detected_at - occurred_at).total_seconds())
            if incident["recovery_outcome"] == "RECOVERED":
                successful_failure_rows.append(incident)

        repair_durations = []
        incomplete_recovery_timestamps = 0
        for incident in successful_failure_rows:
            detected_at = self._parse_timestamp(incident["detected_at"])
            completed_at = self._parse_timestamp(incident["recovery_completed_at"])
            if detected_at is None or completed_at is None or completed_at < detected_at:
                incomplete_recovery_timestamps += 1
            else:
                repair_durations.append((completed_at - detected_at).total_seconds())

        attempt_rows = [
            incident
            for incident in incidents
            if incident["event_type"] == "recovery_attempt"
        ]
        completed_attempts = [
            incident for incident in attempt_rows
            if incident["recovery_outcome"] in ("RECOVERED", "RECOVERY_FAILED")
        ]
        successful_attempts = sum(
            incident["recovery_outcome"] == "RECOVERED"
            for incident in completed_attempts
        )

        availability_points: Dict[datetime, set[int]] = {}
        for observation in observations:
            observed_at = self._parse_timestamp(observation["observed_at"])
            if observed_at is not None:
                availability_points.setdefault(observed_at, set()).add(
                    int(observation["network_available"])
                )
        ordered_points = [
            (timestamp, next(iter(states)))
            for timestamp, states in sorted(availability_points.items())
            if len(states) == 1
        ]
        total_observation_seconds = 0.0
        available_observation_seconds = 0.0
        for (start, available), (end, _) in zip(ordered_points, ordered_points[1:]):
            duration = (end - start).total_seconds()
            if duration <= 0:
                continue
            total_observation_seconds += duration
            if available:
                available_observation_seconds += duration

        availability_pct = (
            100 * available_observation_seconds / total_observation_seconds
            if total_observation_seconds > 0
            else None
        )
        recovery_success_rate_pct = (
            100 * successful_attempts / len(completed_attempts)
            if completed_attempts
            else None
        )
        metrics = {
            "total_incidents": row["total_incidents"],
            "successful_recoveries": row["successful_recoveries"] or 0,
            "failed_recoveries": row["failed_recoveries"] or 0,
            "average_recovery_duration_ms": row["average_recovery_duration_ms"],
            "network_availability_pct": availability_pct,
            "network_uptime_seconds": available_observation_seconds,
            "network_observation_seconds": total_observation_seconds,
            "mttd_seconds": (
                sum(detection_durations) / len(detection_durations)
                if detection_durations else None
            ),
            "mttd_samples": len(detection_durations),
            "incomplete_detection_timestamps": incomplete_detection_timestamps,
            "mttr_seconds": (
                sum(repair_durations) / len(repair_durations)
                if repair_durations else None
            ),
            "mttr_samples": len(repair_durations),
            "incomplete_recovery_timestamps": incomplete_recovery_timestamps,
            "recovery_success_rate_pct": recovery_success_rate_pct,
            "completed_recovery_attempts": len(completed_attempts),
            "incomplete_recovery_attempts": len(attempt_rows) - len(completed_attempts),
        }
        return metrics

    def export_metrics_csv(self) -> str:
        metrics = self.analytics()
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=("metric", "value", "unit", "formula"))
        writer.writeheader()
        for metric, (unit, formula) in METRIC_DEFINITIONS.items():
            writer.writerow({
                "metric": metric,
                "value": metrics[metric],
                "unit": unit,
                "formula": formula,
            })
        writer.writerow({
            "metric": "network_uptime_seconds",
            "value": metrics["network_uptime_seconds"],
            "unit": "seconds",
            "formula": "sum of observed intervals whose starting observation is reachable",
        })
        writer.writerow({
            "metric": "network_observation_seconds",
            "value": metrics["network_observation_seconds"],
            "unit": "seconds",
            "formula": "sum of positive elapsed times between consecutive valid observations",
        })
        return output.getvalue()

    def export_csv(
        self,
        event_types: Optional[Sequence[str]] = None,
        components: Optional[Sequence[str]] = None,
        recovery_outcomes: Optional[Sequence[str]] = None,
    ) -> str:
        incidents = self.list_incidents(event_types, components, recovery_outcomes)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=INCIDENT_FIELDS)
        writer.writeheader()
        writer.writerows(incidents)
        return output.getvalue()
