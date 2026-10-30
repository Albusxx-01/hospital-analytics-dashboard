"""Grain-preserving CSV and optional native Hyper extracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .services import AnalyticsService
from .simulation import Scenario


def write_hyper(frame: pd.DataFrame, destination: Path, table_name: str) -> None:
    try:
        from tableauhyperapi import (
            Connection,
            CreateMode,
            HyperProcess,
            Inserter,
            SqlType,
            TableDefinition,
            TableName,
            Telemetry,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional Tableau dependency: uv sync --extra tableau"
        ) from exc
    definition = TableDefinition(TableName("Extract", table_name))
    for name, dtype in frame.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype):
            sql_type = SqlType.bool()
        elif pd.api.types.is_integer_dtype(dtype):
            sql_type = SqlType.big_int()
        elif pd.api.types.is_numeric_dtype(dtype):
            sql_type = SqlType.double()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = SqlType.timestamp()
        else:
            sql_type = SqlType.text()
        definition.add_column(str(name), sql_type)
    with HyperProcess(
        Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={"log_config": ""}
    ) as hyper:
        with Connection(
            hyper.endpoint, str(destination), CreateMode.CREATE_AND_REPLACE
        ) as connection:
            connection.catalog.create_schema("Extract")
            connection.catalog.create_table(definition)
            with Inserter(connection, definition) as inserter:

                def clean(value: Any) -> Any:
                    if pd.isna(value):
                        return None
                    if isinstance(value, pd.Timestamp):
                        return value.floor("us").to_pydatetime()
                    if isinstance(value, np.generic):
                        return value.item()
                    return value

                inserter.add_rows(
                    tuple(clean(value) for value in row)
                    for row in frame.itertuples(index=False, name=None)
                )
                inserter.execute()


def export_data(service: AnalyticsService, destination: Path, hyper: bool = False) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    forecast = service.forecast("All hospitals", "All departments")
    tables = {
        "encounters": service.encounters,
        "daily_operations": service.daily,
        "ward_stays": service.stays,
        "forecast_network": forecast.forecast,
        "forecast_evaluation": forecast.evaluation,
        "capacity_baseline": service.simulation(
            "All hospitals", "All departments", Scenario()
        ).query("scenario == 'Baseline'"),
    }
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
        if hyper and name in {"encounters", "daily_operations", "ward_stays"}:
            write_hyper(frame, destination / f"{name}.hyper", name)
    manifest = {
        **service.manifest,
        "export_rows": {name: len(frame) for name, frame in tables.items()},
        "hyper": hyper,
        "currency": "USD",
        "synthetic": True,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
