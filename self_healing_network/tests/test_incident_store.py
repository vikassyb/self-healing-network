import csv
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from incident_store import INCIDENT_FIELDS, IncidentStore


class IncidentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "nested" / "incidents.sqlite3"
        self.store = IncidentStore(self.database_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_database_is_created_with_incident_table(self) -> None:
        self.assertTrue(self.database_path.is_file())
        incidents = self.store.list_incidents()
        self.assertEqual(incidents, [])
        events = self.store.list_simulation_events()
        self.assertEqual(events, [])

    def test_simulation_events_can_be_recorded_and_replayed_by_scenario(self) -> None:
        scenario_id = "scenario-test"
        first_id = self.store.record_simulation_event(
            level="WARN",
            event="Router failure injected",
            component="R2",
            details="R2 unavailable in the simulation.",
            step=1,
            scenario_id=scenario_id,
            timestamp="2026-10-19T00:00:01+00:00",
        )
        second_id = self.store.record_simulation_event(
            level="ACTION",
            event="Recovery verified",
            component="network",
            details="Alternate route available.",
            step=1,
            scenario_id=scenario_id,
            timestamp="2026-10-19T00:00:02+00:00",
        )
        self.store.record_simulation_event(
            level="INFO",
            event="Unrelated event",
            component="network",
            details="Not part of selected replay.",
            step=0,
        )

        replay = self.store.list_simulation_events(scenario_id=scenario_id)
        limited = self.store.list_simulation_events(scenario_id=scenario_id, limit=1)

        self.assertEqual([row["id"] for row in replay], [first_id, second_id])
        self.assertEqual([row["event"] for row in replay], [
            "Router failure injected",
            "Recovery verified",
        ])
        self.assertEqual(replay[0]["timestamp"], "2026-10-19T00:00:01.000+00:00")
        self.assertEqual(len(limited), 1)
        self.assertEqual(self.store.list_simulation_scenarios(), [scenario_id])
        with self.assertRaises(ValueError):
            self.store.list_simulation_events(limit=0)
        with self.assertRaises(ValueError):
            self.store.record_simulation_event(
                "INFO", "Bad time", "network", "invalid timestamp", 0,
                timestamp="not-a-timestamp",
            )

    def test_existing_incident_database_is_migrated_without_losing_rows(self) -> None:
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        try:
            connection.execute(
                """
                CREATE TABLE incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    original_route TEXT,
                    replacement_route TEXT,
                    recovery_outcome TEXT,
                    recovery_duration_ms REAL
                )
                """
            )
            connection.execute(
                "INSERT INTO incidents (timestamp, event_type, component) "
                "VALUES (?, ?, ?)",
                ("2026-10-19T00:00:00+00:00", "router_failure", "R2"),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = IncidentStore(legacy_path)

        rows = migrated.list_incidents()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["component"], "R2")
        self.assertIsNone(rows[0]["detected_at"])
        self.assertIsNone(rows[0]["recovery_completed_at"])
        self.assertEqual(migrated.list_telemetry(), [])

    def test_incident_insert_preserves_fields_and_routes(self) -> None:
        incident_id = self.store.record_incident(
            event_type="recovery_attempt",
            component="R2",
            original_route=["Host A", "R1", "R2", "R4", "Host B"],
            replacement_route=["Host A", "R1", "R3", "R4", "Host B"],
            recovery_outcome="RECOVERED",
            recovery_duration_ms=1.25,
        )

        record = self.store.list_incidents()[0]
        self.assertEqual(record["id"], incident_id)
        self.assertEqual(record["event_type"], "recovery_attempt")
        self.assertEqual(record["component"], "R2")
        self.assertEqual(record["original_route"], "Host A -> R1 -> R2 -> R4 -> Host B")
        self.assertEqual(record["replacement_route"], "Host A -> R1 -> R3 -> R4 -> Host B")
        self.assertEqual(record["recovery_outcome"], "RECOVERED")
        self.assertEqual(record["recovery_duration_ms"], 1.25)
        self.assertTrue(record["timestamp"])
        reopened_store = IncidentStore(self.database_path)
        self.assertEqual(reopened_store.list_incidents()[0]["id"], incident_id)

    def test_incidents_can_be_filtered_by_all_filter_fields(self) -> None:
        self.store.record_incident("router_failure", "R2", recovery_outcome="FAILURE_DETECTED")
        self.store.record_incident("recovery_attempt", "R2", recovery_outcome="RECOVERED")
        self.store.record_incident("link_failure", "R1 - R3", recovery_outcome="RECOVERY_FAILED")

        selected = self.store.list_incidents(
            event_types=["recovery_attempt"],
            components=["R2"],
            recovery_outcomes=["RECOVERED"],
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["event_type"], "recovery_attempt")
        self.assertEqual(selected[0]["component"], "R2")

    def test_analytics_are_computed_from_stored_records_and_filters(self) -> None:
        self.store.record_incident("router_failure", "R2", recovery_outcome="FAILURE_DETECTED")
        self.store.record_incident(
            "recovery_attempt", "R2", recovery_outcome="RECOVERED", recovery_duration_ms=2.0
        )
        self.store.record_incident(
            "recovery_attempt", "R1", recovery_outcome="RECOVERY_FAILED", recovery_duration_ms=4.0
        )
        self.store.record_incident("network_reset", "network", recovery_outcome="NETWORK_RESET")

        all_metrics = self.store.analytics()
        recovered_only = self.store.analytics(recovery_outcomes=["RECOVERED"])

        self.assertEqual(all_metrics["total_incidents"], 4)
        self.assertEqual(all_metrics["successful_recoveries"], 1)
        self.assertEqual(all_metrics["failed_recoveries"], 1)
        self.assertEqual(all_metrics["average_recovery_duration_ms"], 3.0)
        self.assertEqual(recovered_only["total_incidents"], 1)
        self.assertEqual(recovered_only["average_recovery_duration_ms"], 2.0)

    def test_empty_analytics_have_zero_counts_and_no_average(self) -> None:
        metrics = self.store.analytics()

        self.assertEqual(metrics["total_incidents"], 0)
        self.assertEqual(metrics["successful_recoveries"], 0)
        self.assertEqual(metrics["failed_recoveries"], 0)
        self.assertIsNone(metrics["average_recovery_duration_ms"])

    def test_csv_export_contains_header_and_filtered_incidents(self) -> None:
        self.store.record_incident(
            "recovery_attempt",
            "R2",
            original_route=["Host A", "R1", "R2", "R4", "Host B"],
            replacement_route=["Host A", "R1", "R3", "R4", "Host B"],
            recovery_outcome="RECOVERED",
            recovery_duration_ms=1.25,
        )
        self.store.record_incident("router_failure", "R3", recovery_outcome="FAILURE_DETECTED")

        rows = list(csv.DictReader(io.StringIO(
            self.store.export_csv(event_types=["recovery_attempt"])
        )))

        self.assertEqual(tuple(rows[0].keys()), INCIDENT_FIELDS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["component"], "R2")
        self.assertEqual(rows[0]["original_route"], "Host A -> R1 -> R2 -> R4 -> Host B")
        self.assertEqual(rows[0]["recovery_outcome"], "RECOVERED")

    def test_analytics_calculate_availability_mttd_mttr_and_success_rate(self) -> None:
        start = "2026-10-19T00:00:00+00:00"
        detected = "2026-10-19T00:00:02+00:00"
        completed = "2026-10-19T00:00:08+00:00"
        first_failure = self.store.record_incident(
            "router_failure", "R2", timestamp=start
        )
        self.store.mark_failure_detected(first_failure, detected)
        self.store.complete_failure(first_failure, "RECOVERED", completed)
        first_attempt = self.store.record_incident(
            "recovery_attempt", "R2", timestamp=detected
        )
        self.store.complete_recovery(
            first_attempt, ["Host A", "R1", "R3", "R4", "Host B"],
            "RECOVERED", 6.0, completed,
        )

        second_attempt = self.store.record_incident(
            "recovery_attempt", "R3", timestamp="2026-10-19T00:00:10+00:00"
        )
        self.store.complete_recovery(
            second_attempt, None, "RECOVERY_FAILED", 1.0,
            "2026-10-19T00:00:30+00:00",
        )
        for step, timestamp, available in (
            (0, "2026-10-19T00:00:00+00:00", True),
            (1, "2026-10-19T00:00:10+00:00", False),
            (2, "2026-10-19T00:00:30+00:00", True),
        ):
            self.store.record_telemetry_snapshot(
                step,
                {"R1": {
                    "latency_ms": 10.0,
                    "packet_loss_pct": 0.0,
                    "utilization_pct": 20.0,
                    "error_rate": 0.0,
                    "latency_trend": 0.0,
                    "risk_score": 0.1,
                    "status": "UP" if available else "DOWN",
                }},
                available,
                timestamp,
            )

        metrics = self.store.analytics()

        self.assertAlmostEqual(metrics["network_availability_pct"], 100 / 3)
        self.assertEqual(metrics["network_uptime_seconds"], 10.0)
        self.assertEqual(metrics["network_observation_seconds"], 30.0)
        self.assertEqual(metrics["mttd_seconds"], 2.0)
        self.assertEqual(metrics["mttr_seconds"], 6.0)
        self.assertEqual(metrics["recovery_success_rate_pct"], 50.0)
        self.assertEqual(metrics["completed_recovery_attempts"], 2)

    def test_analytics_handle_empty_observations_and_recovery_history(self) -> None:
        metrics = self.store.analytics()

        self.assertIsNone(metrics["network_availability_pct"])
        self.assertEqual(metrics["network_uptime_seconds"], 0)
        self.assertEqual(metrics["network_observation_seconds"], 0)
        self.assertIsNone(metrics["mttd_seconds"])
        self.assertIsNone(metrics["mttr_seconds"])
        self.assertIsNone(metrics["recovery_success_rate_pct"])
        self.assertEqual(metrics["incomplete_detection_timestamps"], 0)
        self.assertEqual(metrics["incomplete_recovery_timestamps"], 0)

    def test_incomplete_and_invalid_incident_timestamps_are_excluded(self) -> None:
        incomplete_id = self.store.record_incident(
            "router_failure",
            "R2",
            timestamp="2026-10-19T00:00:00+00:00",
        )
        self.store.complete_failure(
            incomplete_id,
            "RECOVERED",
            "2026-10-19T00:00:05+00:00",
        )
        invalid_id = self.store.record_incident(
            "link_failure",
            "R1 - R2",
            timestamp="2026-10-19T00:00:00+00:00",
        )
        with self.store._connection() as connection:
            connection.execute(
                "UPDATE incidents SET detected_at = ?, recovery_outcome = ?, "
                "recovery_completed_at = ? WHERE id = ?",
                ("not-a-timestamp", "RECOVERED", "not-a-timestamp", invalid_id),
            )

        metrics = self.store.analytics()

        self.assertIsNone(metrics["mttd_seconds"])
        self.assertEqual(metrics["incomplete_detection_timestamps"], 2)
        self.assertIsNone(metrics["mttr_seconds"])
        self.assertEqual(metrics["incomplete_recovery_timestamps"], 2)
        with self.assertRaises(ValueError):
            self.store.record_incident("router_failure", "R3", timestamp="invalid")

    def test_telemetry_history_persists_and_exports_to_csv(self) -> None:
        self.store.record_telemetry_snapshot(
            3,
            {"R1": {
                "latency_ms": 12.0,
                "packet_loss_pct": 0.2,
                "utilization_pct": 30.0,
                "error_rate": 0.01,
                "latency_trend": 1.0,
                "risk_score": 0.25,
                "status": "UP",
            }},
            True,
            "2026-10-19T00:00:00+00:00",
        )
        reopened = IncidentStore(self.database_path)
        records = reopened.list_telemetry()
        rows = list(csv.DictReader(io.StringIO(reopened.export_telemetry_csv())))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["router"], "R1")
        self.assertEqual(records[0]["network_available"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], "3")
        self.assertIn("network_available", rows[0])

    def test_metrics_csv_includes_values_units_and_formula_definitions(self) -> None:
        rows = list(csv.DictReader(io.StringIO(self.store.export_metrics_csv())))

        self.assertIn("network_availability_pct", {row["metric"] for row in rows})
        self.assertIn("mttd_seconds", {row["metric"] for row in rows})
        self.assertIn("mttr_seconds", {row["metric"] for row in rows})
        self.assertIn("recovery_success_rate_pct", {row["metric"] for row in rows})
        self.assertTrue(all(row["unit"] and row["formula"] for row in rows))

    def test_invalid_incident_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.record_incident("", "R2")
        with self.assertRaises(ValueError):
            self.store.record_incident("router_failure", "")
        with self.assertRaises(ValueError):
            self.store.record_incident("recovery_attempt", "R2", recovery_duration_ms=-1)
        with self.assertRaises(ValueError):
            self.store.record_incident("recovery_attempt", "R2", original_route=["Host A", 5])


if __name__ == "__main__":
    unittest.main()
