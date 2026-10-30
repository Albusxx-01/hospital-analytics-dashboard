"""Paired Monte Carlo scenarios with identical arrivals/LOS random draws.

Daily Poisson arrivals are transformed through common uniforms; conditional
residual stays for current patients are sampled from completed LOS exceeding
their already-observed stay. Occupancy is daily occupied bed-days, not census.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass(frozen=True)
class Scenario:
    demand_pct: float = 0
    beds_pct: float = 0
    los_pct: float = 0


def simulate_capacity(
    predicted: np.ndarray,
    completed_los: np.ndarray,
    current_elapsed: np.ndarray,
    beds: float,
    scenario: Scenario = Scenario(),
    *,
    simulations: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    predicted = np.asarray(predicted, dtype=float)
    los = np.asarray(completed_los, dtype=float)
    elapsed = np.asarray(current_elapsed, dtype=float)
    if (
        beds <= 0
        or simulations < 1
        or not len(predicted)
        or not np.isfinite(predicted).all()
        or (predicted < 0).any()
    ):
        raise ValueError(
            "Positive capacity, valid arrivals, and at least one simulation are required."
        )
    if not len(los) or not np.isfinite(los).all() or (los <= 0).any():
        raise ValueError("Completed positive length-of-stay observations are required.")
    if not np.isfinite(elapsed).all() or (elapsed < 0).any():
        raise ValueError("Current patient elapsed stays must be nonnegative.")
    if (
        not all(np.isfinite([scenario.demand_pct, scenario.beds_pct, scenario.los_pct]))
        or min(scenario.demand_pct, scenario.beds_pct, scenario.los_pct) <= -100
    ):
        raise ValueError("Scenario percentage changes must be finite and greater than -100%.")
    arrival_seed, los_seed, time_seed, residual_seed = np.random.SeedSequence(seed).spawn(4)
    arrival_rng = np.random.default_rng(arrival_seed)
    los_rng = np.random.default_rng(los_seed)
    time_rng = np.random.default_rng(time_seed)
    residual_rng = np.random.default_rng(residual_seed)
    horizon = len(predicted)
    u_arrivals = arrival_rng.uniform(0.000001, 0.999999, size=(simulations, horizon))
    base_arrivals = poisson.ppf(u_arrivals, predicted).astype(int)
    scenario_arrivals = poisson.ppf(u_arrivals, predicted * (1 + scenario.demand_pct / 100)).astype(
        int
    )
    max_daily = max(int(base_arrivals.max()), int(scenario_arrivals.max()), 1)
    # Patient slot is the leading draw dimension: enlarging the scenario does
    # not change existing baseline patients. Independent streams also keep
    # current-patient residuals stable across scenarios with different demand.
    sampled_los = los_rng.choice(los, size=(max_daily, simulations, horizon)).transpose(1, 2, 0)
    arrival_fraction = time_rng.random((max_daily, simulations, horizon)).transpose(1, 2, 0)
    # Survival-conditioned remaining LOS; tail fallback is an explicit memoryless estimate.
    residual = np.empty((simulations, len(elapsed)))
    for j, age in enumerate(elapsed):
        survivors = los[los > age] - age
        residual[:, j] = (
            residual_rng.choice(survivors, simulations)
            if len(survivors)
            else residual_rng.exponential(max(1, float(los.mean()) / 2), simulations)
        )
    outputs = []
    for label, arrivals, scale, capacity in (
        ("Baseline", base_arrivals, 1.0, beds),
        (
            "Scenario",
            scenario_arrivals,
            1 + scenario.los_pct / 100,
            beds * (1 + scenario.beds_pct / 100),
        ),
    ):
        occupied = np.zeros((simulations, horizon))
        patient_index = np.arange(max_daily)[None, :]
        for day in range(horizon):
            occupied[:, day] += np.clip(residual * scale - day, 0, 1).sum(axis=1)
            for arrival_day in range(day + 1):
                begin = arrival_day + arrival_fraction[:, arrival_day]
                end = begin + sampled_los[:, arrival_day] * scale
                overlap = np.maximum(0, np.minimum(end, day + 1) - np.maximum(begin, day))
                occupied[:, day] += (
                    overlap * (patient_index < arrivals[:, arrival_day, None])
                ).sum(axis=1)
        rates = occupied / capacity
        outputs.append(
            pd.DataFrame(
                {
                    "day": np.arange(1, horizon + 1),
                    "scenario": label,
                    "expected_occupancy": rates.mean(axis=0),
                    "lower": np.quantile(rates, 0.05, axis=0),
                    "upper": np.quantile(rates, 0.95, axis=0),
                    "overflow_probability": (rates > 1).mean(axis=0),
                    "available_beds": capacity,
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)
