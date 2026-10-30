"""Chronological model selection and empirical forecast intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    evaluation: pd.DataFrame
    model_name: str
    training_start: str
    training_end: str
    residuals: np.ndarray


def features(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    t = (dates - origin).days.to_numpy(dtype=float)
    dow = dates.dayofweek.to_numpy()
    return np.column_stack(
        [
            t / 365.25,
            *[(dow == day).astype(float) for day in range(6)],
            np.sin(2 * np.pi * t / 365.25),
            np.cos(2 * np.pi * t / 365.25),
            np.sin(4 * np.pi * t / 365.25),
            np.cos(4 * np.pi * t / 365.25),
        ]
    )


def predict(train: pd.Series, dates: pd.DatetimeIndex, model: str) -> np.ndarray:
    if model == "Weekly seasonal baseline":
        return np.maximum(0, np.resize(train.iloc[-7:].to_numpy(dtype=float), len(dates)))
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=20))
    origin = train.index[0]
    pipeline.fit(features(pd.DatetimeIndex(train.index), origin), train.to_numpy(dtype=float))
    return np.maximum(0, pipeline.predict(features(dates, origin)))


def forecast_admissions(series: pd.Series, horizon: int = 28, seed: int = 42) -> ForecastResult:
    series = series.sort_index().astype(float)
    if len(series) < 280:
        raise ValueError("Forecasting requires at least 280 complete days.")
    expected = pd.date_range(series.index.min(), series.index.max(), freq="D")
    if not series.index.equals(expected) or series.isna().any() or (series < 0).any():
        raise ValueError("Admissions must be a complete, nonnegative daily series.")
    names = ["Regularized seasonal model", "Weekly seasonal baseline"]
    residuals: dict[str, list[np.ndarray]] = {name: [] for name in names}
    rows = []
    # Six expanding-window folds. Each interval is calibrated using ONLY earlier folds.
    for stop in range(len(series) - 6 * horizon, len(series), horizon):
        train, test = series.iloc[:stop], series.iloc[stop : stop + horizon]
        for name in names:
            predicted = predict(train, pd.DatetimeIndex(test.index), name)
            error = test.to_numpy() - predicted
            historical = np.concatenate(residuals[name]) if residuals[name] else np.array([])
            coverage = float("nan")
            if len(historical):
                low, high = np.quantile(historical, [0.05, 0.95])
                coverage = float(
                    np.mean(
                        (test.to_numpy() >= np.maximum(0, predicted + low))
                        & (test.to_numpy() <= np.maximum(0, predicted + high))
                    )
                )
            rows.append(
                {
                    "model": name,
                    "train_end": train.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "mae": float(np.abs(error).mean()),
                    "interval_coverage": coverage,
                }
            )
            residuals[name].append(error)
    evaluation = pd.DataFrame(rows)
    winner = str(evaluation.groupby("model").mae.mean().idxmin())
    future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    point = predict(series, future, winner)
    errors = np.concatenate(residuals[winner])
    rng = np.random.default_rng(seed)
    samples = np.maximum(0, point + rng.choice(errors, size=(2000, horizon)))
    forecast = pd.DataFrame(
        {
            "date": future,
            "predicted": point,
            "lower": np.quantile(samples, 0.05, axis=0),
            "upper": np.quantile(samples, 0.95, axis=0),
        }
    )
    return ForecastResult(
        forecast,
        evaluation,
        winner,
        str(series.index[0].date()),
        str(series.index[-1].date()),
        errors,
    )
