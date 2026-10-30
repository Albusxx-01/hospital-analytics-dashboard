from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hospital_intelligence.analytics import kpis, occupancy, readmission_cohorts, select_encounters
from hospital_intelligence.config import Filters, Settings
from hospital_intelligence.generate import generate
from hospital_intelligence.pipeline import daily_bed_hours, readmission_flags
from hospital_intelligence.services import AnalyticsService
from hospital_intelligence.validation import validate_raw


def test_generation_reproducible_and_valid(tmp_path: Path) -> None:
    settings = Settings(root=tmp_path, encounters=100)
    first = generate(settings)
    second = generate(settings)
    for name in first:
        assert_frame_equal(first[name], second[name])
    validate_raw(first, settings.cutoff)
    first["encounters"].loc[0, "patient_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown patient"):
        validate_raw(first, settings.cutoff)


def test_bed_hours_midnight_transfer_and_exclusive_end() -> None:
    stays = pd.DataFrame(
        {
            "entered_at": pd.to_datetime(
                ["2025-01-01 18:00", "2025-01-02 03:00", "2025-01-02 20:00"]
            ),
            "exited_at": pd.to_datetime(
                ["2025-01-02 03:00", "2025-01-02 12:00", "2025-01-03 00:00"]
            ),
            "hospital": ["A", "A", "A"],
            "department": ["X", "Y", "X"],
        }
    )
    daily = daily_bed_hours(stays).set_index(["date", "department"])
    assert daily.loc[(pd.Timestamp("2025-01-01"), "X"), "occupied_bed_hours"] == 6
    assert daily.loc[(pd.Timestamp("2025-01-02"), "X"), "occupied_bed_hours"] == 7
    assert daily.loc[(pd.Timestamp("2025-01-02"), "Y"), "occupied_bed_hours"] == 9
    assert daily.occupied_bed_hours.sum() == 22
    assert pd.Timestamp("2025-01-03") not in daily.index.get_level_values(0)


def test_readmissions_boundaries_censoring_and_intervening_planned_stay() -> None:
    frame = pd.DataFrame(
        [
            ["a", "p1", "2025-01-01", "2025-01-02", "Emergency", "Home"],
            ["b", "p1", "2025-01-05", "2025-01-06", "Planned", "Home"],
            ["c", "p1", "2025-02-01", "2025-02-02", "Emergency", "Home"],
            ["d", "p2", "2025-01-01", "2025-01-02", "Emergency", "Home"],
            ["e", "p2", "2025-02-02", "2025-02-03", "Emergency", "Home"],
            ["f", "p3", "2025-12-20", "2025-12-21", "Emergency", "Home"],
            ["g", "p4", "2025-01-01", "2025-01-02", "Emergency", "Deceased"],
            ["h", "p5", "2025-12-30", None, "Emergency", "In hospital"],
        ],
        columns=[
            "encounter_id",
            "patient_id",
            "admitted_at",
            "discharged_at",
            "admission_type",
            "disposition",
        ],
    )
    frame["admitted_at"] = pd.to_datetime(frame.admitted_at)
    frame["discharged_at"] = pd.to_datetime(frame.discharged_at)
    result = readmission_flags(frame, pd.Timestamp("2026-01-01")).set_index("encounter_id")
    assert result.loc["a", "readmitted_30d"]  # Exactly 30 days, ignoring planned admission.
    assert not result.loc["d", "readmitted_30d"]  # 31 days.
    for key in ("f", "g", "h"):
        assert not result.loc[key, "readmission_eligible"]
        assert pd.isna(result.loc[key, "readmitted_30d"])


def test_occupancy_is_ratio_of_sums_not_average_rate() -> None:
    daily = pd.DataFrame({"occupied_bed_hours": [24, 240], "available_bed_hours": [48, 240]})
    assert occupancy(daily) == pytest.approx(264 / 288)
    assert occupancy(daily) != pytest.approx(0.75)
    assert np.isnan(occupancy(daily.iloc[:0]))


def test_pipeline_reconciles(service: AnalyticsService) -> None:
    assert len(service.encounters) == 600
    assert service.daily.admissions.sum() == 600
    raw_costs = pd.read_parquet(service.root / "raw" / "costs.parquet")
    assert service.encounters.total_cost.sum() == pytest.approx(raw_costs.amount.sum())
    assert service.encounters.encounter_id.is_unique
    assert not service.daily.duplicated(["date", "hospital", "department"]).any()


def test_empty_and_cohort_metrics(service: AnalyticsService) -> None:
    filters = Filters("2030-01-01", "2030-01-31")
    metrics = kpis(service.encounters, service.daily, filters)
    assert metrics["admissions"] == 0
    assert np.isnan(metrics["readmission"])
    assert readmission_cohorts(service.encounters.iloc[:0]).empty
    filters = Filters("2024-01-01", "2025-12-31", age="80+", sex="Female")
    selected = select_encounters(service.encounters, filters)
    assert selected.age_band.eq("80+").all()
    assert selected.sex.eq("Female").all()


def test_discharge_cohort_not_admission_cohort(service: AnalyticsService) -> None:
    filters = service.defaults()
    discharged = select_encounters(service.encounters, filters, basis="discharged_at")
    result = kpis(service.encounters, service.daily, filters)
    assert result["los"] == pytest.approx(discharged.los_days.mean())
    assert result["cost"] == pytest.approx(discharged.total_cost.mean())
    eligible = discharged.loc[discharged.readmission_eligible]
    assert result["eligible"] == len(eligible)
