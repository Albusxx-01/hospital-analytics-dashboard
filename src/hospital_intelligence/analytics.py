"""Pure analytics functions: denominators are first-class, never averaged rates."""

import numpy as np
import pandas as pd

from .config import Filters


def select_encounters(
    frame: pd.DataFrame, filters: Filters, *, basis: str = "admitted_at"
) -> pd.DataFrame:
    start, end = pd.Timestamp(filters.start), pd.Timestamp(filters.end) + pd.Timedelta(days=1)
    mask = frame[basis].ge(start) & frame[basis].lt(end)
    for field, value, default in (
        ("hospital", filters.hospital, "All hospitals"),
        ("department", filters.department, "All departments"),
        ("age_band", filters.age, "All ages"),
        ("sex", filters.sex, "All sexes"),
        ("diagnosis", filters.diagnosis, "All diagnoses"),
    ):
        if value != default:
            mask &= frame[field].eq(value)
    return frame.loc[mask].copy()


def select_daily(frame: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    mask = frame.date.ge(pd.Timestamp(filters.start)) & frame.date.le(pd.Timestamp(filters.end))
    if filters.hospital != "All hospitals":
        mask &= frame.hospital.eq(filters.hospital)
    if filters.department != "All departments":
        mask &= frame.department.eq(filters.department)
    return frame.loc[mask].copy()


def occupancy(frame: pd.DataFrame) -> float:
    denominator = frame.available_bed_hours.sum()
    return float(frame.occupied_bed_hours.sum() / denominator) if denominator else float("nan")


def kpis(encounters: pd.DataFrame, daily: pd.DataFrame, filters: Filters) -> dict[str, float]:
    admitted = select_encounters(encounters, filters)
    discharged = select_encounters(encounters, filters, basis="discharged_at")
    eligible = discharged.loc[discharged.readmission_eligible]
    return {
        "admissions": float(len(admitted)),
        "occupancy": occupancy(select_daily(daily, filters)),
        "los": float(discharged.los_days.mean()),
        "readmission": float(eligible.readmitted_30d.mean()) if len(eligible) else float("nan"),
        "cost": float(discharged.total_cost.mean()),
        "eligible": float(len(eligible)),
    }


def wilson_interval(successes: np.ndarray, totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score intervals, with undefined bounds for empty cohorts."""
    n = np.asarray(totals, dtype=float)
    p = np.divide(successes, n, out=np.full_like(n, np.nan), where=n > 0)
    safe = np.where(n > 0, n, np.nan)
    z = 1.959963984540054
    denominator = 1 + z * z / safe
    center = (p + z * z / (2 * safe)) / denominator
    width = z * np.sqrt(p * (1 - p) / safe + z * z / (4 * safe * safe)) / denominator
    return np.maximum(0, center - width), np.minimum(1, center + width)


def readmission_cohorts(encounters: pd.DataFrame, field: str = "department") -> pd.DataFrame:
    eligible = encounters.loc[encounters.readmission_eligible]
    result = (
        eligible.groupby(field, observed=True)
        .agg(eligible=("encounter_id", "size"), readmissions=("readmitted_30d", "sum"))
        .reset_index()
    )
    result["rate"] = result.readmissions.astype(float) / result.eligible
    result["lower"], result["upper"] = wilson_interval(
        result.readmissions.to_numpy(dtype=float), result.eligible.to_numpy()
    )
    return result


def daily_series(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.groupby("date", as_index=False)[
        ["admissions", "occupied_bed_hours", "available_bed_hours"]
    ].sum()
    result["occupancy_rate"] = result.occupied_bed_hours / result.available_bed_hours
    return result
