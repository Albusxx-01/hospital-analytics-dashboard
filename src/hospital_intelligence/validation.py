"""Fail early on broken business invariants instead of rendering plausible errors."""

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_raw(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> None:
    patients, encounters, stays, costs, capacity = (
        tables[name] for name in ("patients", "encounters", "ward_stays", "costs", "capacity")
    )
    for frame, key in ((patients, "patient_id"), (encounters, "encounter_id"), (stays, "stay_id")):
        require(
            bool(frame[key].notna().all() and frame[key].is_unique), f"Invalid unique key: {key}"
        )
    require(bool(encounters.patient_id.isin(patients.patient_id).all()), "Unknown patient")
    for frame in (stays, costs):
        require(bool(frame.encounter_id.isin(encounters.encounter_id).all()), "Unknown encounter")
    require(bool((encounters.admitted_at < cutoff).all()), "Admission beyond observation cutoff")
    completed = encounters.dropna(subset=["discharged_at"])
    require(
        bool((completed.discharged_at > completed.admitted_at).all()),
        "Nonpositive encounter duration",
    )
    require(bool((completed.discharged_at < cutoff).all()), "Unobserved future discharge")
    require(bool((stays.exited_at > stays.entered_at).all()), "Nonpositive ward stay")
    require(bool((stays.exited_at <= cutoff).all()), "Ward stay beyond observation cutoff")
    ordered = stays.sort_values(["encounter_id", "entered_at"])
    previous = ordered.groupby("encounter_id").exited_at.shift()
    require(
        bool((previous.isna() | ordered.entered_at.eq(previous)).all()), "Noncontiguous ward stays"
    )
    bounds = stays.groupby("encounter_id").agg(
        first=("entered_at", "min"), last=("exited_at", "max")
    )
    aligned = encounters.set_index("encounter_id").join(bounds)
    require(
        bool(aligned["first"].eq(aligned.admitted_at).all()), "Ward admission boundary mismatch"
    )
    require(
        bool(aligned["last"].eq(aligned.discharged_at.fillna(cutoff)).all()),
        "Ward discharge boundary mismatch",
    )
    linked = stays.merge(
        encounters[["encounter_id", "hospital", "department"]],
        on="encounter_id",
        suffixes=("", "_enc"),
        validate="many_to_one",
    )
    require(
        bool(
            (
                linked.hospital.eq(linked.hospital_enc)
                & linked.department.eq(linked.department_enc)
            ).all()
        ),
        "Ward department/hospital mismatch",
    )
    timeline = encounters.sort_values(["patient_id", "admitted_at"])
    prior_end = (
        timeline.assign(end=timeline.discharged_at.fillna(cutoff)).groupby("patient_id").end.shift()
    )
    require(
        bool((prior_end.isna() | timeline.admitted_at.ge(prior_end)).all()),
        "Overlapping patient encounters",
    )
    require(bool(np.isfinite(costs.amount).all() and costs.amount.ge(0).all()), "Invalid costs")
    require(not costs.duplicated(["encounter_id", "component"]).any(), "Duplicate cost component")
    require(set(costs.encounter_id) == set(encounters.encounter_id), "Encounter missing costs")
    require(
        not capacity.duplicated(["date", "hospital", "department"]).any(), "Duplicate capacity key"
    )
    require(bool(capacity.available_beds.gt(0).all()), "Invalid bed capacity")
