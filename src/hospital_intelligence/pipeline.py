"""Validated transformations with explicit encounter and daily-operational grains."""

import hashlib
import json
import logging

import numpy as np
import pandas as pd

from .config import Settings
from .validation import require, validate_raw

log = logging.getLogger(__name__)


def readmission_flags(encounters: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Attach the next unplanned admission across sites, preserving all index stays.

    A planned admission between discharge and an unplanned admission does not
    hide the latter. Events at exactly 30 days count; censored outcomes are NA.
    """
    frame = encounters.sort_values(["patient_id", "admitted_at"]).copy()
    unplanned = frame.admitted_at.where(frame.admission_type.eq("Emergency"))
    frame["next_unplanned_at"] = unplanned.groupby(frame.patient_id).transform(
        lambda group: group.shift(-1).bfill()
    )
    frame["readmission_eligible"] = (
        frame.discharged_at.notna()
        & frame.disposition.ne("Deceased")
        & (frame.discharged_at + pd.Timedelta(days=30)).le(cutoff)
    )
    delta = frame.next_unplanned_at - frame.discharged_at
    frame["readmitted_30d"] = (delta.ge(pd.Timedelta(0)) & delta.le(pd.Timedelta(days=30))).astype(
        "boolean"
    )
    frame.loc[~frame.readmission_eligible, "readmitted_30d"] = pd.NA
    return frame.drop(columns="next_unplanned_at").sort_values("encounter_id")


def daily_bed_hours(stays: pd.DataFrame) -> pd.DataFrame:
    """Split half-open ward intervals at midnight, retaining fractional bed-hours."""
    start_days = stays.entered_at.dt.normalize()
    last_days = (stays.exited_at - pd.Timedelta(1, unit="ns")).dt.normalize()
    counts = ((last_days - start_days).dt.days + 1).to_numpy()
    repeated = stays.loc[stays.index.repeat(counts)].copy().reset_index(drop=True)
    offsets = (
        np.concatenate([np.arange(count) for count in counts])
        if len(counts)
        else np.array([], dtype=int)
    )
    repeated["date"] = repeated.entered_at.dt.normalize() + pd.to_timedelta(offsets, unit="D")
    start = repeated[["entered_at", "date"]].max(axis=1)
    end = pd.concat([repeated.exited_at, repeated.date + pd.Timedelta(days=1)], axis=1).min(axis=1)
    repeated["occupied_bed_hours"] = (end - start).dt.total_seconds() / 3600
    return repeated.groupby(["date", "hospital", "department"], as_index=False).agg(
        occupied_bed_hours=("occupied_bed_hours", "sum")
    )


def build(settings: Settings) -> dict:
    meta = json.loads((settings.raw / "metadata.json").read_text())
    cutoff = pd.Timestamp(meta["end"]) + pd.Timedelta(days=1)
    names = ["patients", "encounters", "ward_stays", "capacity", "costs"]
    tables = {name: pd.read_parquet(settings.raw / f"{name}.parquet") for name in names}
    validate_raw(tables, cutoff)
    encounter = readmission_flags(tables["encounters"], cutoff).merge(
        tables["patients"], on="patient_id", validate="many_to_one"
    )
    costs = tables["costs"].pivot(index="encounter_id", columns="component", values="amount")
    costs.columns = [f"cost_{name.lower()}" for name in costs.columns]
    costs["total_cost"] = costs.sum(axis=1)
    encounter = encounter.merge(costs, on="encounter_id", validate="one_to_one")
    encounter["los_days"] = (
        encounter.discharged_at - encounter.admitted_at
    ).dt.total_seconds() / 86400
    encounter["elapsed_days"] = (
        encounter.discharged_at.fillna(cutoff) - encounter.admitted_at
    ).dt.total_seconds() / 86400
    daily = tables["capacity"].merge(
        daily_bed_hours(tables["ward_stays"]),
        how="left",
        on=["date", "hospital", "department"],
        validate="one_to_one",
    )
    daily["occupied_bed_hours"] = daily.occupied_bed_hours.fillna(0)
    daily["available_bed_hours"] = daily.available_beds * 24
    arrivals = (
        encounter.assign(date=encounter.admitted_at.dt.normalize())
        .groupby(["date", "hospital", "department"])
        .size()
        .rename("admissions")
    )
    daily = daily.merge(
        arrivals, how="left", on=["date", "hospital", "department"], validate="one_to_one"
    )
    daily["admissions"] = daily.admissions.fillna(0).astype(int)
    daily["occupancy_rate"] = daily.occupied_bed_hours / daily.available_bed_hours
    expected = (
        tables["ward_stays"].exited_at - tables["ward_stays"].entered_at
    ).dt.total_seconds().sum() / 3600
    require(
        bool(np.isclose(daily.occupied_bed_hours.sum(), expected)), "Bed-hour reconciliation failed"
    )
    require(
        bool(np.isclose(encounter.total_cost.sum(), tables["costs"].amount.sum())),
        "Cost reconciliation failed",
    )
    require(int(daily.admissions.sum()) == len(encounter), "Admission reconciliation failed")
    settings.processed.mkdir(parents=True, exist_ok=True)
    output = {
        "encounters": encounter,
        "daily_operations": daily,
        "ward_stays": tables["ward_stays"],
    }
    digest = hashlib.sha256()
    for name, frame in output.items():
        path = settings.processed / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        digest.update(path.read_bytes())
    meta.update(
        {
            "dataset_version": digest.hexdigest()[:16],
            "pipeline_version": 1,
            "tables": {name: len(frame) for name, frame in output.items()},
        }
    )
    (settings.processed / "manifest.json").write_text(json.dumps(meta, indent=2))
    log.info("Built dataset %s: %s", meta["dataset_version"], meta["tables"])
    return meta
