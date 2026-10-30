"""Seeded synthetic hospital data with coherent patient timelines.

All inpatient stays are single-site, have contiguous ward segments, and have
no overlapping admission for the same patient. Future discharge times are not
written for patients still admitted at the observation cutoff.
"""

import json
import logging

import numpy as np
import pandas as pd

from .config import DEPARTMENTS, DIAGNOSES, HOSPITALS, Settings

log = logging.getLogger(__name__)


def generate(settings: Settings) -> dict[str, pd.DataFrame]:
    if settings.encounters < 100:
        raise ValueError("Use at least 100 encounters for a meaningful demo.")
    rng = np.random.default_rng(settings.seed)
    dates = pd.date_range(settings.start, settings.end)
    if len(dates) < 365:
        raise ValueError("At least one year is required for seasonal analytics.")
    n = settings.encounters
    patient_count = max(100, int(n * 0.66))
    ages = np.clip(rng.normal(58, 20, patient_count).astype(int), 18, 95)
    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:06d}" for i in range(patient_count)],
            "age": ages,
            "sex": rng.choice(["Female", "Male"], patient_count, p=[0.52, 0.48]),
            "comorbidity_count": rng.poisson(np.maximum(ages - 25, 0) / 26),
        }
    )
    patients["age_band"] = pd.cut(
        patients.age, [17, 39, 64, 79, 100], labels=["18–39", "40–64", "65–79", "80+"]
    ).astype(str)
    demand = 1 + 0.20 * np.cos(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    demand *= np.where(dates.dayofweek < 5, 1.10, 0.84)
    demand *= np.linspace(0.94, 1.08, len(dates))
    sampled_days = np.sort(rng.choice(len(dates), n, p=demand / demand.sum()))
    admission_times = dates[sampled_days] + pd.to_timedelta(rng.uniform(0, 24, n), unit="h")
    admission_times = admission_times.sort_values()
    last_discharge = np.full(patient_count, np.datetime64("1900-01-01"), dtype="datetime64[ns]")
    prior_dept = np.zeros(patient_count, dtype=int)
    prior_hospital = np.zeros(patient_count, dtype=int)
    rows: list[dict] = []
    ward_rows: list[dict] = []
    cost_rows: list[dict] = []
    los_means = np.array([4.3, 5.5, 3.6, 5.0])
    for i, admitted in enumerate(admission_times):
        p = int(rng.integers(patient_count))
        while last_discharge[p] > admitted.to_datetime64():
            p = int(rng.integers(patient_count))
        gap = (admitted.to_datetime64() - last_discharge[p]) / np.timedelta64(1, "D")
        repeat = gap < 30
        dept = (
            int(prior_dept[p])
            if repeat and rng.random() < 0.7
            else int(rng.choice(4, p=[0.34, 0.24, 0.22, 0.20]))
        )
        hospital = (
            int(prior_hospital[p])
            if repeat and rng.random() < 0.8
            else int(rng.choice(3, p=[0.43, 0.32, 0.25]))
        )
        emergency = bool(rng.random() < (0.91 if repeat else 0.70))
        severity = int(rng.choice([1, 2, 3], p=[0.48, 0.37, 0.15]))
        los = max(0.35, float(rng.lognormal(np.log(los_means[dept]) - 0.20, 0.52)))
        los *= 0.8 + 0.20 * severity + 0.025 * patients.iloc[p].comorbidity_count
        discharged = admitted + pd.Timedelta(days=los)
        last_discharge[p] = discharged.to_datetime64()
        prior_dept[p], prior_hospital[p] = dept, hospital
        observed_end = min(discharged, settings.cutoff)
        observed_los = (observed_end - admitted).total_seconds() / 86400
        complete = discharged < settings.cutoff
        death = complete and rng.random() < (0.003 + 0.004 * severity)
        encounter_id = f"E{i:07d}"
        department = DEPARTMENTS[dept]
        rows.append(
            {
                "encounter_id": encounter_id,
                "patient_id": patients.iloc[p].patient_id,
                "hospital": HOSPITALS[hospital],
                "department": department,
                "admitted_at": admitted,
                "discharged_at": discharged if complete else pd.NaT,
                "admission_type": "Emergency" if emergency else "Planned",
                "diagnosis": rng.choice(DIAGNOSES[department]),
                "severity": severity,
                "disposition": (
                    "Deceased" if death else ("Post-acute care" if rng.random() < 0.18 else "Home")
                )
                if complete
                else "In hospital",
            }
        )
        stages = ["Emergency", "Ward"] if emergency else ["Ward"]
        if severity == 3:
            stages = (["Emergency"] if emergency else []) + ["ICU", "Ward"]
        weights = np.array(
            [0.06 if s == "Emergency" else 0.30 if s == "ICU" else 1.0 for s in stages], dtype=float
        )
        weights /= weights.sum()
        cursor = admitted
        for j, (stage, weight) in enumerate(zip(stages, weights, strict=True)):
            end = (
                observed_end
                if j == len(stages) - 1
                else cursor + pd.Timedelta(days=observed_los * weight)
            )
            ward_rows.append(
                {
                    "stay_id": f"{encounter_id}-{j}",
                    "encounter_id": encounter_id,
                    "hospital": HOSPITALS[hospital],
                    "department": department,
                    "stage": stage,
                    "entered_at": cursor,
                    "exited_at": end,
                }
            )
            cursor = end
        multiplier = [1.05, 1.0, 0.92][hospital] * (1 + 0.18 * (severity - 1))
        components = {
            "Accommodation": observed_los * [760, 990, 840, 890][dept],
            "Procedures": rng.lognormal(7.3 + 0.3 * dept, 0.65),
            "Medication": observed_los * rng.lognormal(4.9, 0.4),
            "Diagnostics": rng.lognormal(6.5, 0.45),
        }
        for component, amount in components.items():
            cost_rows.append(
                {
                    "encounter_id": encounter_id,
                    "component": component,
                    "amount": round(amount * multiplier, 2),
                }
            )
    encounters = pd.DataFrame(rows)
    # A deceased patient cannot later return. Keep terminal disposition coherent.
    has_later = encounters.duplicated("patient_id", keep="last")
    encounters.loc[has_later & encounters.disposition.eq("Deceased"), "disposition"] = "Home"
    capacity_rows = []
    for h, hospital_name in enumerate(HOSPITALS):
        for d, department in enumerate(DEPARTMENTS):
            mean_arrivals = n / len(dates) * [0.43, 0.32, 0.25][h] * [0.34, 0.24, 0.22, 0.20][d]
            beds = max(4, int(np.ceil(mean_arrivals * los_means[d] * 1.34)))
            for date in dates:
                available = beds - (1 if date.dayofweek == 6 and beds > 10 else 0)
                capacity_rows.append(
                    {
                        "date": date,
                        "hospital": hospital_name,
                        "department": department,
                        "available_beds": available,
                    }
                )
    tables = {
        "patients": patients,
        "encounters": encounters,
        "ward_stays": pd.DataFrame(ward_rows),
        "capacity": pd.DataFrame(capacity_rows),
        "costs": pd.DataFrame(cost_rows),
    }
    settings.raw.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(settings.raw / f"{name}.parquet", index=False)
    (settings.raw / "metadata.json").write_text(
        json.dumps(
            {
                "seed": settings.seed,
                "encounters": n,
                "start": settings.start,
                "end": settings.end,
                "synthetic": True,
                "generator_version": 1,
            },
            indent=2,
        )
    )
    log.info("Generated %s synthetic encounters across %s days", n, len(dates))
    return tables
