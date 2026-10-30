import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hospital_intelligence.forecast import forecast_admissions
from hospital_intelligence.simulation import Scenario, simulate_capacity


def daily_counts() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=731)
    rng = np.random.default_rng(42)
    return pd.Series(rng.poisson(40 + 8 * np.sin(np.arange(731) * 2 * np.pi / 365)), index=dates)


def test_forecast_chronology_coverage_and_reproducibility() -> None:
    result = forecast_admissions(daily_counts())
    repeated = forecast_admissions(daily_counts())
    assert_frame_equal(result.forecast, repeated.forecast)
    assert len(result.forecast) == 28
    assert (result.forecast.lower <= result.forecast.upper).all()
    assert (result.forecast.lower >= 0).all()
    assert (result.evaluation.train_end < result.evaluation.test_start).all()
    assert result.evaluation.groupby("model").interval_coverage.count().eq(5).all()
    assert result.model_name == result.evaluation.groupby("model").mae.mean().idxmin()


def test_future_changes_do_not_change_earlier_backtests() -> None:
    series = daily_counts()
    original = forecast_admissions(series)
    changed = series.copy()
    changed.iloc[-28:] += 1000
    altered = forecast_admissions(changed)
    assert_frame_equal(original.evaluation.iloc[:-2], altered.evaluation.iloc[:-2])


def test_forecast_rejects_incomplete_series() -> None:
    with pytest.raises(ValueError, match="280"):
        forecast_admissions(daily_counts().iloc[:100])
    with pytest.raises(ValueError, match="complete"):
        forecast_admissions(daily_counts().drop(daily_counts().index[200]))


def test_simulation_identical_scenario_and_seed() -> None:
    args = (np.full(28, 8.0), np.array([2.0, 3.0, 5.0, 8.0]), np.array([1.0, 2.0, 3.0]), 40.0)
    result = simulate_capacity(*args, simulations=100)
    repeated = simulate_capacity(*args, simulations=100)
    assert_frame_equal(result, repeated)
    baseline = (
        result.loc[result.scenario.eq("Baseline")].reset_index(drop=True).drop(columns="scenario")
    )
    scenario = (
        result.loc[result.scenario.eq("Scenario")].reset_index(drop=True).drop(columns="scenario")
    )
    assert_frame_equal(baseline, scenario)


@pytest.mark.parametrize(
    "scenario", [Scenario(demand_pct=25), Scenario(beds_pct=-20), Scenario(los_pct=20)]
)
def test_pressure_increases_occupancy_and_overflow(scenario: Scenario) -> None:
    result = simulate_capacity(
        np.full(28, 8.0),
        np.array([2.0, 3.0, 5.0, 8.0]),
        np.array([1.0, 2.0, 3.0]),
        40.0,
        scenario,
        simulations=120,
    )
    base = result.loc[result.scenario.eq("Baseline")].reset_index(drop=True)
    changed = result.loc[result.scenario.eq("Scenario")].reset_index(drop=True)
    assert (changed.expected_occupancy >= base.expected_occupancy - 1e-12).all()
    assert (changed.overflow_probability >= base.overflow_probability).all()
    assert changed.overflow_probability.between(0, 1).all()


def test_baseline_is_stable_across_scenario_changes() -> None:
    args = (np.full(28, 8.0), np.array([2.0, 3.0, 5.0, 8.0]), np.array([1.0, 2.0]), 40.0)
    first = simulate_capacity(*args, Scenario(demand_pct=10), simulations=100)
    second = simulate_capacity(*args, Scenario(demand_pct=50), simulations=100)
    assert_frame_equal(
        first.query("scenario == 'Baseline'"), second.query("scenario == 'Baseline'")
    )


def test_simulation_rejects_invalid_scenarios() -> None:
    with pytest.raises(ValueError, match="greater than"):
        simulate_capacity(np.ones(28), np.array([2.0]), np.array([]), 10, Scenario(beds_pct=-100))
