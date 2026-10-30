import json
from pathlib import Path

import pandas as pd
import pytest

from hospital_intelligence.dashboard import create_app
from hospital_intelligence.export import export_data, write_hyper
from hospital_intelligence.services import AnalyticsService


def test_csv_export_reconciles(service: AnalyticsService, tmp_path: Path) -> None:
    manifest = export_data(service, tmp_path)
    encounters = pd.read_csv(tmp_path / "encounters.csv")
    daily = pd.read_csv(tmp_path / "daily_operations.csv")
    assert len(encounters) == len(service.encounters)
    assert daily.admissions.sum() == len(encounters)
    assert encounters.total_cost.sum() == pytest.approx(service.encounters.total_cost.sum())
    assert daily.occupied_bed_hours.sum() == pytest.approx(service.daily.occupied_bed_hours.sum())
    assert manifest["dataset_version"] == service.version
    assert json.loads((tmp_path / "manifest.json").read_text())["synthetic"]


def test_optional_hyper_round_trip(tmp_path: Path) -> None:
    hyper = pytest.importorskip("tableauhyperapi")
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "value": [2.5, float("nan")],
            "flag": pd.array([True, None], dtype="boolean"),
            "date": pd.to_datetime(["2025-01-01", None]),
            "label": ["α", "beta"],
        }
    )
    path = tmp_path / "test.hyper"
    write_hyper(frame, path, "test")
    with hyper.HyperProcess(
        hyper.Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={"log_config": ""}
    ) as process:
        with hyper.Connection(process.endpoint, str(path)) as connection:
            rows = connection.execute_list_query(
                'SELECT "id", "value", "flag", "label" FROM "Extract"."test" ORDER BY "id"'
            )
    assert rows == [[1, 2.5, True, "α"], [2, None, None, "beta"]]


def test_dashboard_routes_and_callbacks(dataset: Path) -> None:
    app = create_app(dataset)
    client = app.server.test_client()
    for route in ("/", "/_dash-layout", "/_dash-dependencies"):
        assert client.get(route).status_code == 200
    layout = client.get("/_dash-layout").get_json()
    assert "Hospital operations" in json.dumps(layout)
    # Exercise the actual callbacks via Dash's HTTP dispatch contract.
    for key, callback in app.callback_map.items():
        if (
            "kpi-row" in key
            or "forecast-chart" in key
            or "simulation-chart" in key
            or "readmission-chart" in key
        ):
            values = {
                "hospital.value": "All hospitals",
                "department.value": "All departments",
                "dates.start_date": "2025-10-03",
                "dates.end_date": "2025-12-31",
                "age.value": "All ages",
                "sex.value": "All sexes",
                "diagnosis.value": "All diagnoses",
                "run-scenario.n_clicks": 0,
                "demand.value": 15,
                "beds.value": 10,
                "los.value": -10,
            }
            inputs = [
                {**entry, "value": values[f"{entry['id']}.{entry['property']}"]}
                for entry in callback["inputs"]
            ]
            states = [
                {**entry, "value": values[f"{entry['id']}.{entry['property']}"]}
                for entry in callback["state"]
            ]
            outputs = [
                {"id": entry.component_id, "property": entry.component_property}
                for entry in callback["output"]
            ]
            response = client.post(
                "/_dash-update-component",
                json={
                    "output": key,
                    "outputs": outputs,
                    "inputs": inputs,
                    "state": states,
                    "changedPropIds": [f"{inputs[0]['id']}.{inputs[0]['property']}"],
                },
            )
            assert response.status_code == 200, response.get_data(as_text=True)
