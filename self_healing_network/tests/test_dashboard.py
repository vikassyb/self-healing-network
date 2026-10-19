from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class DashboardAnalyticsTests(unittest.TestCase):
    def test_dashboard_renders_analytics_charts_timeline_and_csv_exports(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "dashboard.sqlite3"
            with patch.dict(os.environ, {"SELF_HEALING_NETWORK_DB": str(database_path)}):
                app = AppTest.from_file(str(app_path), default_timeout=30).run()

        self.assertEqual(
            [exception.value for exception in app.exception],
            [],
        )
        rendered_subheaders = {item.value for item in app.subheader}
        self.assertIn("Network analytics", rendered_subheaders)
        self.assertIn("Incident timeline", rendered_subheaders)
        self.assertIn("Simulation event replay", rendered_subheaders)
        self.assertIn("Advanced simulation scenarios", {
            item.label for item in app.get("expander")
        })
        self.assertIn("Scenario router failures", {
            item.label for item in app.multiselect
        })
        self.assertIn("Run configured scenario", {item.label for item in app.button})
        rendered_downloads = {item.label for item in app.get("download_button")}
        self.assertIn("Download calculated network metrics as CSV", rendered_downloads)
        self.assertIn("Download historical telemetry as CSV", rendered_downloads)
        self.assertIn("Download incident history as CSV", rendered_downloads)
        self.assertEqual(len(app.get("plotly_chart")), 4)

    def test_dashboard_runs_configured_scenario(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "scenario-dashboard.sqlite3"
            with patch.dict(os.environ, {"SELF_HEALING_NETWORK_DB": str(database_path)}):
                app = AppTest.from_file(str(app_path), default_timeout=30).run()
                next(
                    item for item in app.get("multiselect")
                    if item.key == "scenario_router_failures"
                ).set_value(["R2"])
                next(
                    item for item in app.button
                    if item.label == "Run configured scenario"
                ).click().run()

        self.assertEqual([exception.value for exception in app.exception], [])
        rendered_subheaders = {item.value for item in app.subheader}
        self.assertIn("Scenario analysis", rendered_subheaders)
        recovery_states = {
            item.value for item in app.metric if item.label == "Recovery state"
        }
        self.assertIn("RECOVERED", recovery_states)


if __name__ == "__main__":
    unittest.main()
