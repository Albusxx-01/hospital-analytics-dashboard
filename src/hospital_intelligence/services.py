"""Dataset-versioned analytics services and precomputed forecasting artifacts."""

import json
import logging
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEPARTMENTS, HOSPITALS, Filters, Settings
from .forecast import ForecastResult, forecast_admissions
from .simulation import Scenario, simulate_capacity

log = logging.getLogger(__name__)


def scope_frame(frame: pd.DataFrame, hospital: str, department: str) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if hospital != "All hospitals":
        mask &= frame.hospital.eq(hospital)
    if department != "All departments":
        mask &= frame.department.eq(department)
    return frame.loc[mask]


def scope_key(hospital: str, department: str) -> str:
    return f"{hospital}__{department}".lower().replace(" ", "_")


def precompute(settings: Settings) -> None:
    manifest = json.loads((settings.processed / "manifest.json").read_text())
    daily = pd.read_parquet(settings.processed / "daily_operations.parquet")
    folder = settings.processed / "forecasts" / manifest["dataset_version"]
    folder.mkdir(parents=True, exist_ok=True)
    for hospital, department in product(
        ("All hospitals", *HOSPITALS), ("All departments", *DEPARTMENTS)
    ):
        scoped = scope_frame(daily, hospital, department)
        series = scoped.groupby("date").admissions.sum()
        result = forecast_admissions(series, seed=manifest["seed"])
        key = scope_key(hospital, department)
        result.forecast.to_parquet(folder / f"{key}.parquet", index=False)
        result.evaluation.to_parquet(folder / f"{key}_evaluation.parquet", index=False)
        (folder / f"{key}.json").write_text(
            json.dumps(
                {
                    "model_name": result.model_name,
                    "training_start": result.training_start,
                    "training_end": result.training_end,
                    "residuals": result.residuals.tolist(),
                }
            )
        )
    log.info("Precomputed 20 hospital/department forecast scopes")


class AnalyticsService:
    """Immutable dataset snapshot; recreating the service invalidates all cached work."""

    def __init__(self, root: Path):
        self.root = root
        folder = root / "processed"
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.version = self.manifest["dataset_version"]
        self.encounters = pd.read_parquet(folder / "encounters.parquet")
        self.daily = pd.read_parquet(folder / "daily_operations.parquet")
        self.stays = pd.read_parquet(folder / "ward_stays.parquet")
        self.cutoff = pd.Timestamp(self.manifest["end"]) + pd.Timedelta(days=1)

    def defaults(self) -> Filters:
        end = pd.Timestamp(self.manifest["end"])
        return Filters(str((end - pd.Timedelta(days=89)).date()), str(end.date()))

    @lru_cache(maxsize=20)
    def forecast(self, hospital: str, department: str) -> ForecastResult:
        folder = self.root / "processed" / "forecasts" / self.version
        key = scope_key(hospital, department)
        metadata = json.loads((folder / f"{key}.json").read_text())
        return ForecastResult(
            pd.read_parquet(folder / f"{key}.parquet"),
            pd.read_parquet(folder / f"{key}_evaluation.parquet"),
            metadata["model_name"],
            metadata["training_start"],
            metadata["training_end"],
            np.array(metadata["residuals"]),
        )

    @lru_cache(maxsize=128)
    def simulation(self, hospital: str, department: str, scenario: Scenario) -> pd.DataFrame:
        encounters = scope_frame(self.encounters, hospital, department)
        daily = scope_frame(self.daily, hospital, department)
        beds = float(daily.loc[daily.date.eq(daily.date.max()), "available_beds"].sum())
        completed = encounters.loc[encounters.discharged_at.notna(), "los_days"].to_numpy()
        current = encounters.loc[encounters.discharged_at.isna(), "elapsed_days"].to_numpy()
        result = simulate_capacity(
            self.forecast(hospital, department).forecast.predicted.to_numpy(),
            completed,
            current,
            beds,
            scenario,
            seed=self.manifest["seed"],
        )
        result["date"] = self.cutoff + pd.to_timedelta(result.day - 1, unit="D")
        return result
