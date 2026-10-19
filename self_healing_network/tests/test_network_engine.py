import unittest
import tempfile

import numpy as np
import pandas as pd
from unittest.mock import patch
from pathlib import Path

from network_engine import (
    FEATURES,
    NetworkEngine,
    ROUTERS,
    evaluate_model,
    generate_synthetic_dataset,
    train_and_evaluate_model,
)
from incident_store import IncidentStore


class NetworkEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.incident_store = IncidentStore(Path(self.temp_dir.name) / "incidents.sqlite3")
        self.engine = NetworkEngine(seed=17, incident_store=self.incident_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_baseline_has_a_route(self) -> None:
        route, _ = self.engine.compute_route(auto_heal=False)

        self.assertIsNotNone(route)
        self.assertEqual(route[0], "Host A")
        self.assertEqual(route[-1], "Host B")

    def test_failed_router_is_avoided_when_an_alternate_path_exists(self) -> None:
        self.engine.fail_router("R2")

        route, _ = self.engine.compute_route(auto_heal=False)

        self.assertIsNotNone(route)
        self.assertNotIn("R2", route)
        self.assertEqual(route[0], "Host A")
        self.assertEqual(route[-1], "Host B")

    def test_weighted_route_prefers_lower_metric_path(self) -> None:
        self.engine.risk_scores.update({"R1": 0.0, "R2": 1.0, "R3": 0.0, "R4": 0.0})
        self.engine.telemetry["R1"].update({
            "latency_ms": 3.0,
            "packet_loss_pct": 0.0,
            "utilization_pct": 5.0,
            "error_rate": 0.0,
            "latency_trend": 0.0,
        })
        self.engine.telemetry["R2"].update({
            "latency_ms": 220.0,
            "packet_loss_pct": 18.0,
            "utilization_pct": 100.0,
            "error_rate": 0.6,
            "latency_trend": 40.0,
        })
        self.engine.telemetry["R3"].update({
            "latency_ms": 3.0,
            "packet_loss_pct": 0.0,
            "utilization_pct": 5.0,
            "error_rate": 0.0,
            "latency_trend": 0.0,
        })
        self.engine.telemetry["R4"].update({
            "latency_ms": 3.0,
            "packet_loss_pct": 0.0,
            "utilization_pct": 5.0,
            "error_rate": 0.0,
            "latency_trend": 0.0,
        })

        route, _ = self.engine.compute_route(auto_heal=False)

        self.assertEqual(route, ["Host A", "R1", "R3", "R4", "Host B"])

    def test_unavailable_link_is_excluded_from_routes(self) -> None:
        self.engine.set_link_status("R1", "R2", "DOWN")

        route, _ = self.engine.compute_route(auto_heal=False)

        self.assertIsNotNone(route)
        self.assertNotIn(("R1", "R2"), list(zip(route, route[1:])))

    def test_link_failure_automatically_reroutes_and_verifies(self) -> None:
        old_route = self.engine.compute_route(auto_heal=False)[0]

        self.engine.set_link_status("R1", "R2", "DOWN", auto_heal=True)

        self.assertEqual(self.engine.recovery_state, "RECOVERED")
        self.assertEqual(self.engine.recovery["original_route"], old_route)
        self.assertTrue(self.engine.recovery["verified"])
        self.assertNotIn(
            ("R1", "R2"),
            list(zip(self.engine.recovery["selected_route"], self.engine.recovery["selected_route"][1:])),
        )

    def test_link_failure_does_not_reroute_when_disabled(self) -> None:
        old_route = self.engine.compute_route(auto_heal=False)[0]

        self.engine.set_link_status("R1", "R2", "DOWN", auto_heal=False)

        self.assertEqual(self.engine.recovery_state, "RECOVERY_FAILED")
        self.assertEqual(self.engine.recovery["original_route"], old_route)
        self.assertIsNone(self.engine.recovery["selected_route"])
        self.assertFalse(self.engine.recovery["verified"])
        self.assertFalse(self.engine.recovery["rerouting_enabled"])

    def test_multi_failure_scenario_applies_router_link_congestion_and_packet_loss(self) -> None:
        result = self.engine.run_scenario(
            router_failures=["R2"],
            link_failures=[("R1", "R2")],
            congested_routers=["R3"],
            packet_loss_by_router={"R3": 2.5},
            congestion_utilization_pct=90.0,
            seed=23,
            threshold=1.0,
        )

        self.assertEqual(self.engine.status["R2"], "DOWN")
        self.assertEqual(self.engine.link_status[("R1", "R2")], "DOWN")
        self.assertGreaterEqual(self.engine.telemetry["R3"]["utilization_pct"], 90.0)
        self.assertEqual(self.engine.telemetry["R3"]["packet_loss_pct"], 2.5)
        self.assertEqual(result["recovery_state"], "RECOVERED")
        self.assertTrue(result["recovery_verified"])
        self.assertTrue(result["root_cause"]["route_available_after_injection"])
        self.assertIn("router R2", result["root_cause"]["failed_components_on_original_route"])
        self.assertEqual(
            [item["strategy"] for item in result["strategy_comparison"]],
            ["Weighted path", "Shortest hop"],
        )
        self.assertTrue(all(item["available"] for item in result["strategy_comparison"]))
        self.assertTrue(all(
            self.engine.route_is_available(item["route"])
            for item in result["strategy_comparison"]
        ))

    def test_multi_router_scenario_reports_unavailable_route_and_failed_recovery(self) -> None:
        result = self.engine.run_scenario(
            router_failures=["R2", "R3"],
            seed=29,
            threshold=1.0,
        )

        self.assertFalse(result["root_cause"]["route_available_after_injection"])
        self.assertEqual(result["recovery_state"], "RECOVERY_FAILED")
        self.assertFalse(result["recovery_verified"])
        self.assertIsNone(result["selected_route"])
        self.assertTrue(all(
            not item["available"] and item["route"] is None
            for item in result["strategy_comparison"]
        ))

    def test_scenario_event_history_is_persistent_and_chronological(self) -> None:
        result = self.engine.run_scenario(
            router_failures=["R2"],
            congested_routers=["R3"],
            packet_loss_by_router={"R3": 4.0},
            seed=31,
            threshold=1.0,
        )
        reopened_store = IncidentStore(self.incident_store.database_path)

        replay = reopened_store.list_simulation_events(
            scenario_id=result["scenario_id"]
        )
        events = [row["event"] for row in replay]

        self.assertGreater(len(replay), 5)
        self.assertEqual([row["id"] for row in replay], sorted(row["id"] for row in replay))
        self.assertTrue(all(row["timestamp"] for row in replay))
        self.assertIn("Scenario started", events)
        self.assertIn("Scenario failure injected", events)
        self.assertIn("Root-cause analysis", events)
        self.assertIn("Routing strategy comparison", events)
        self.assertIn("Recovery verified", events)
        self.assertEqual(
            reopened_store.list_simulation_events(scenario_id="unknown-scenario"),
            [],
        )

    def test_scenario_results_are_reproducible_for_the_same_seed(self) -> None:
        second_store = IncidentStore(Path(self.temp_dir.name) / "repeat.sqlite3")
        second_engine = NetworkEngine(seed=17, incident_store=second_store)
        arguments = {
            "router_failures": ["R2"],
            "congested_routers": ["R3"],
            "packet_loss_by_router": {"R3": 3.0},
            "congestion_utilization_pct": 92.0,
            "seed": 37,
            "threshold": 1.0,
        }

        first_result = self.engine.run_scenario(**arguments)
        second_result = second_engine.run_scenario(**arguments)
        first_result.pop("scenario_id")
        second_result.pop("scenario_id")

        self.assertEqual(first_result, second_result)
        self.assertEqual(self.engine.telemetry, second_engine.telemetry)
        self.assertEqual(self.engine.risk_scores, second_engine.risk_scores)

    def test_scenario_reset_restores_baseline_without_erasing_replay(self) -> None:
        result = self.engine.run_scenario(
            router_failures=["R2"],
            seed=41,
            threshold=1.0,
        )
        persisted_before_reset = self.incident_store.list_simulation_events(
            scenario_id=result["scenario_id"]
        )

        self.engine.reset()

        replay_after_reset = self.incident_store.list_simulation_events(
            scenario_id=result["scenario_id"]
        )
        self.assertTrue(all(state == "UP" for state in self.engine.status.values()))
        self.assertTrue(all(state == "UP" for state in self.engine.link_status.values()))
        self.assertIsNone(self.engine.scenario_result)
        self.assertEqual(replay_after_reset, persisted_before_reset)

    def test_scenario_validation_rejects_bad_inputs_before_reset(self) -> None:
        original_step = self.engine.step
        original_status = dict(self.engine.status)

        with self.assertRaises(ValueError):
            self.engine.run_scenario(
                router_failures=["R2", "R2"],
                seed=17,
            )
        with self.assertRaises(ValueError):
            self.engine.run_scenario(packet_loss_by_router={"R2": 25.0})

        self.assertEqual(self.engine.step, original_step)
        self.assertEqual(self.engine.status, original_status)

    def test_disconnected_source_or_destination_returns_no_route(self) -> None:
        for disconnected_link in (("Host A", "R1"), ("R4", "Host B")):
            with self.subTest(link=disconnected_link):
                store_path = Path(self.temp_dir.name) / f"{disconnected_link[0]}-{disconnected_link[1]}.sqlite3"
                engine = NetworkEngine(seed=17, incident_store=IncidentStore(store_path))
                engine.set_link_status(*disconnected_link, "DOWN", auto_heal=False)

                route, note = engine.compute_route(auto_heal=False)

                self.assertIsNone(route)
                self.assertIn("No route is available", note)
                self.assertEqual(engine.recovery_state, "RECOVERY_FAILED")

    def test_multiple_failed_routers_can_leave_no_route(self) -> None:
        self.engine.fail_router("R2")
        self.engine.fail_router("R3")

        route, note = self.engine.compute_route(auto_heal=False)

        self.assertIsNone(route)
        self.assertIn("No route is available", note)

    def test_failed_host_edge_router_leaves_no_route(self) -> None:
        self.engine.fail_router("R1")

        route, note = self.engine.compute_route(auto_heal=False)

        self.assertIsNone(route)
        self.assertIn("No route is available", note)

    def test_auto_heal_uses_risk_threshold_and_can_be_disabled(self) -> None:
        self.engine.risk_scores["R2"] = 1.0

        rerouted, note = self.engine.compute_route(threshold=0.8, auto_heal=True)
        unchanged_policy, unchanged_note = self.engine.compute_route(threshold=0.8, auto_heal=False)

        self.assertIsNotNone(rerouted)
        self.assertNotIn("R2", rerouted)
        self.assertIn("R2", note)
        self.assertIsNotNone(unchanged_policy)
        self.assertIn("disabled", unchanged_note)

    def test_advance_logs_risks_using_the_selected_threshold(self) -> None:
        threshold = 1.0
        self.engine.advance(threshold=threshold)

        warned_routers = {
            event["Details"].split(" has ", 1)[0]
            for event in self.engine.events
            if event["Event"] == "Elevated failure risk"
        }
        expected_routers = {
            router for router in ROUTERS
            if self.engine.status[router] == "UP" and self.engine.risk_scores[router] >= threshold
        }

        self.assertEqual(warned_routers, expected_routers)

    def test_threshold_validation_rejects_invalid_values(self) -> None:
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    self.engine.compute_route(threshold=threshold)

    def test_telemetry_advance_is_reproducible_and_records_timestamped_history(self) -> None:
        second_store = IncidentStore(Path(self.temp_dir.name) / "second.sqlite3")
        second_engine = NetworkEngine(seed=17, incident_store=second_store)

        self.engine.advance()
        second_engine.advance()

        self.assertEqual(self.engine.telemetry, second_engine.telemetry)
        self.assertEqual(self.engine.risk_scores, second_engine.risk_scores)
        self.assertEqual(self.engine.step, 1)
        self.assertEqual(len(self.engine.history), len(ROUTERS) * 2)
        latest_rows = self.engine.history[-len(ROUTERS):]
        self.assertTrue(all(row["timestamp"].endswith("+00:00") for row in latest_rows))
        self.assertEqual({row["step"] for row in latest_rows}, {1})

    def test_telemetry_alerts_follow_configured_thresholds_and_router_states(self) -> None:
        for router in ROUTERS:
            self.engine.risk_scores[router] = 0.0
            self.engine.telemetry[router].update({
                "latency_ms": 10.0,
                "packet_loss_pct": 0.0,
                "utilization_pct": 10.0,
            })
        thresholds = {
            "latency_ms": 100.0,
            "packet_loss_pct": 3.0,
            "utilization_pct": 85.0,
            "risk_score": 0.65,
        }

        self.assertEqual(self.engine.telemetry_alerts(thresholds)["R2"]["state"], "HEALTHY")
        self.engine.telemetry["R2"]["latency_ms"] = 101.0
        high_risk = self.engine.telemetry_alerts(thresholds)["R2"]
        self.assertEqual(high_risk["state"], "HIGH RISK")
        self.assertTrue(any("Latency" in reason for reason in high_risk["reasons"]))

        self.engine.fail_router("R2", auto_heal=False)
        failed = self.engine.telemetry_alerts(thresholds)["R2"]
        self.assertEqual(failed["state"], "FAILED")
        self.assertTrue(any("marked DOWN" in reason for reason in failed["reasons"]))
        self.engine.restore_router("R2")
        self.engine.telemetry["R2"]["latency_ms"] = 10.0
        self.assertEqual(self.engine.telemetry_alerts(thresholds)["R2"]["state"], "HEALTHY")

    def test_telemetry_alerts_reject_invalid_threshold_mappings(self) -> None:
        for thresholds in (
            [],
            {"unknown": 1.0},
            {"latency_ms": float("nan")},
            {"packet_loss_pct": 101.0},
            {"utilization_pct": 101.0},
            {"risk_score": True},
        ):
            with self.subTest(thresholds=thresholds):
                with self.assertRaises(ValueError):
                    self.engine.telemetry_alerts(thresholds)

    def test_router_validation_and_restore_state(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.fail_router("R5")
        with self.assertRaises(ValueError):
            self.engine.restore_router("R5")

        self.engine.fail_router("R2")
        self.engine.restore_router("R2")

        self.assertEqual(self.engine.status["R2"], "UP")

    def test_reset_restores_network_state(self) -> None:
        self.engine.fail_router("R2")
        self.engine.recover_network()
        self.engine.set_link_status("R1", "R2", "DOWN")
        self.engine.advance()
        self.engine.reset()

        self.assertEqual(self.engine.step, 0)
        self.assertTrue(all(self.engine.status[router] == "UP" for router in ROUTERS))
        self.assertTrue(all(status == "UP" for status in self.engine.link_status.values()))
        self.assertEqual(self.engine.recovery_state, "IDLE")
        self.assertEqual(self.engine.recovery_history, [])
        self.assertIsNotNone(self.engine.compute_route(auto_heal=False)[0])

    def test_recovery_verifies_and_records_success(self) -> None:
        original_route, _ = self.engine.compute_route(auto_heal=False)
        self.engine.fail_router("R2", auto_heal=True)
        recovery = self.engine.recovery

        self.assertEqual(self.engine.recovery_state, "RECOVERED")
        self.assertEqual(recovery["original_route"], original_route)
        self.assertIsNotNone(recovery["selected_route"])
        self.assertNotIn("R2", recovery["selected_route"])
        self.assertTrue(recovery["verified"])
        self.assertGreaterEqual(recovery["duration_ms"], 0.0)
        self.assertEqual(len(self.engine.recovery_history), 1)
        self.assertIn("verified", recovery["result"].lower())

    def test_recovery_fails_when_no_route_exists(self) -> None:
        self.engine.fail_router("R1", auto_heal=True)
        recovery = self.engine.recovery

        self.assertEqual(self.engine.recovery_state, "RECOVERY_FAILED")
        self.assertIsNone(recovery["selected_route"])
        self.assertFalse(recovery["verified"])
        self.assertGreaterEqual(recovery["duration_ms"], 0.0)
        self.assertIn("failed", recovery["result"].lower())

    def test_recovery_fails_if_simulated_connectivity_check_fails(self) -> None:
        with patch.object(self.engine, "_verify_connectivity", return_value=False):
            self.engine.fail_router("R2", auto_heal=True)
        recovery = self.engine.recovery

        self.assertEqual(self.engine.recovery_state, "RECOVERY_FAILED")
        self.assertIsNotNone(recovery["selected_route"])
        self.assertFalse(recovery["verified"])

    def test_recovery_is_idempotent_after_success(self) -> None:
        self.engine.fail_router("R2", auto_heal=True)
        completed = dict(self.engine.recovery)
        event_count = len(self.engine.events)

        repeated = self.engine.recover_network(auto_heal=True)
        self.engine.fail_router("R2", auto_heal=True)

        self.assertEqual(repeated, completed)
        self.assertEqual(self.engine.recovery_state, "RECOVERED")
        self.assertEqual(len(self.engine.recovery_history), 1)
        self.assertEqual(len(self.engine.events), event_count + 1)

    def test_router_failure_does_not_reroute_when_automatic_rerouting_is_disabled(self) -> None:
        old_route = self.engine.compute_route(auto_heal=False)[0]

        self.engine.fail_router("R2", auto_heal=False)
        recovery = self.engine.recover_network(auto_heal=False)

        self.assertEqual(self.engine.recovery_state, "RECOVERY_FAILED")
        self.assertEqual(recovery["original_route"], old_route)
        self.assertIsNone(recovery["selected_route"])
        self.assertFalse(recovery["verified"])
        self.assertFalse(recovery["rerouting_enabled"])

    def test_failures_recoveries_and_reset_are_persisted(self) -> None:
        self.engine.fail_router("R2", auto_heal=True)
        self.engine.set_link_status("R2", "R3", "DOWN", auto_heal=True)
        self.engine.reset()

        records = self.incident_store.list_incidents()
        event_types = [record["event_type"] for record in records]
        self.assertEqual(event_types.count("router_failure"), 1)
        self.assertEqual(event_types.count("link_failure"), 1)
        self.assertEqual(event_types.count("recovery_attempt"), 2)
        self.assertEqual(event_types.count("network_reset"), 1)
        recovery_records = [record for record in records if record["event_type"] == "recovery_attempt"]
        self.assertTrue(all(record["original_route"] for record in recovery_records))
        self.assertTrue(all(record["replacement_route"] for record in recovery_records))
        self.assertTrue(all(record["recovery_duration_ms"] is not None for record in recovery_records))

    def test_failure_lifecycle_and_router_state_observations_are_persisted(self) -> None:
        self.engine.fail_router("R2", auto_heal=True)

        incidents = self.incident_store.list_incidents()
        failure = next(item for item in incidents if item["event_type"] == "router_failure")
        recovery_attempt = next(
            item for item in incidents if item["event_type"] == "recovery_attempt"
        )
        telemetry = self.incident_store.list_telemetry()
        failed_router_observations = [
            row for row in telemetry if row["router"] == "R2" and row["status"] == "DOWN"
        ]

        self.assertIsNotNone(failure["detected_at"])
        self.assertIsNotNone(failure["recovery_completed_at"])
        self.assertEqual(failure["recovery_outcome"], "RECOVERED")
        self.assertIsNotNone(recovery_attempt["recovery_completed_at"])
        self.assertTrue(failed_router_observations)
        self.assertTrue(all(row["network_available"] == 1 for row in failed_router_observations))

    def test_telemetry_history_remains_persistent_across_network_reset(self) -> None:
        self.engine.advance()
        before_reset = len(self.incident_store.list_telemetry())

        self.engine.reset()

        persisted = self.incident_store.list_telemetry()
        self.assertGreater(len(persisted), before_reset)
        self.assertTrue(any(row["step"] == 1 for row in persisted))
        self.assertEqual(self.engine.step, 0)

    def test_prediction_scores_are_finite_and_between_zero_and_one(self) -> None:
        self.assertEqual(set(self.engine.risk_scores), set(ROUTERS))
        for score in self.engine.risk_scores.values():
            self.assertTrue(np.isfinite(score))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_synthetic_data_and_training_are_reproducible(self) -> None:
        features_one, labels_one = generate_synthetic_dataset(seed=23, n_samples=500)
        features_two, labels_two = generate_synthetic_dataset(seed=23, n_samples=500)
        model_one, evaluation_one, importance_one = train_and_evaluate_model(seed=23, n_samples=500)
        model_two, evaluation_two, importance_two = train_and_evaluate_model(seed=23, n_samples=500)

        pd.testing.assert_frame_equal(features_one, features_two)
        pd.testing.assert_series_equal(labels_one, labels_two)
        self.assertEqual(evaluation_one, evaluation_two)
        self.assertEqual(importance_one, importance_two)
        np.testing.assert_array_equal(
            model_one.predict_proba(features_one[FEATURES]),
            model_two.predict_proba(features_two[FEATURES]),
        )

    def test_synthetic_telemetry_and_labels_have_correlated_failure_signals(self) -> None:
        features, labels = generate_synthetic_dataset(seed=17, n_samples=5000)
        correlations = features.assign(simulated_failure=labels).corr()["simulated_failure"]

        self.assertGreater(correlations["latency_ms"], 0.4)
        self.assertGreater(correlations["packet_loss_pct"], 0.4)
        self.assertGreater(correlations["utilization_pct"], 0.4)
        self.assertGreater(correlations["error_rate"], 0.4)
        self.assertGreater(correlations["latency_trend"], 0.3)

    def test_synthetic_generator_rejects_invalid_arguments(self) -> None:
        for seed, n_samples in ((True, 100), (1.5, 100), (-1, 100), (2**32, 100), (1, 19), (1, True)):
            with self.subTest(seed=seed, n_samples=n_samples):
                with self.assertRaises(ValueError):
                    generate_synthetic_dataset(seed=seed, n_samples=n_samples)

    def test_model_evaluation_returns_held_out_metrics_and_importance(self) -> None:
        model, evaluation, importance = train_and_evaluate_model(seed=31, n_samples=500)

        self.assertGreaterEqual(evaluation["precision"], 0.0)
        self.assertLessEqual(evaluation["precision"], 1.0)
        self.assertGreaterEqual(evaluation["recall"], 0.0)
        self.assertLessEqual(evaluation["recall"], 1.0)
        self.assertGreaterEqual(evaluation["f1_score"], 0.0)
        self.assertLessEqual(evaluation["f1_score"], 1.0)
        self.assertEqual(np.asarray(evaluation["confusion_matrix"]).shape, (2, 2))
        self.assertIsNotNone(evaluation["roc_auc"])
        self.assertEqual({entry["feature"] for entry in importance}, set(FEATURES))
        self.assertAlmostEqual(sum(entry["importance"] for entry in importance), 1.0)

        features, _ = generate_synthetic_dataset(seed=32, n_samples=100)
        single_class = evaluate_model(model, features.iloc[:10], pd.Series([0] * 10))
        self.assertIsNone(single_class["roc_auc"])

    def test_model_evaluation_rejects_invalid_inputs(self) -> None:
        model, _, _ = train_and_evaluate_model(seed=37, n_samples=500)
        features, labels = generate_synthetic_dataset(seed=37, n_samples=100)

        with self.assertRaises(ValueError):
            evaluate_model(model, pd.DataFrame({"latency_ms": [10.0]}), pd.Series([0]))
        with self.assertRaises(ValueError):
            evaluate_model(model, features.iloc[:5], labels.iloc[:4])
        with self.assertRaises(ValueError):
            evaluate_model(model, features.iloc[:5], pd.Series([0, 0, 0, 0, 2]))
        invalid_features = features.iloc[:5].copy()
        invalid_features.loc[invalid_features.index[0], "latency_ms"] = np.nan
        with self.assertRaises(ValueError):
            evaluate_model(model, invalid_features, labels.iloc[:5])


if __name__ == "__main__":
    unittest.main()
