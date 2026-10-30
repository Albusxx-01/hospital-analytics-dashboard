# Hospital Intelligence

A complete Python demonstration of hospital operations analytics, featuring a Plotly Dash application, a reproducible 50,000-encounter synthetic dataset, chronological demand forecasting, paired Monte Carlo capacity scenarios, and Tableau-ready exports.

**Synthetic data throughout. Operational estimates, not clinical recommendations.**

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run hospital demo
uv run hospital serve
```

Open http://127.0.0.1:8050. The demo command generates and validates the data, builds analytics, precomputes forecasts for all hospital/department combinations, and exports CSVs. Startup does not train models or generate data.

In the delivered workspace, the environment and full dataset are already prepared. Start directly with `.venv/bin/hospital serve`; there is no need to regenerate the data. The isolated Python installation leaves the system Python unchanged.

```sh
uv run hospital generate --seed 42 --encounters 50000
uv run hospital build
uv run hospital export --output exports
uv sync --locked --extra tableau
uv run hospital export --output exports --hyper
uv run pytest
uv run mypy src/hospital_intelligence
```

Use `hospital --data-dir /path/to/data <command>` for an alternate data directory. For deployment to a local network, configure `serve --host`; the default binds only to loopback. This demo has no authentication and no clinical-system integration.

## Explore

1. **Overview:** executive metrics, demand trends, capacity utilization, and department performance.
2. **Patient flow:** day/hour arrivals, length-of-stay distributions, and observed care-stage journeys.
3. **Outcomes & costs:** readmission intervals, age cohorts, encounter cost distribution, and cost composition.
4. **Capacity lab:** compare forecasts to a seasonal baseline, inspect backtest metrics, and simulate changes in demand, beds, and length of stay.

See [methodology](docs/methodology.md), [data dictionary](docs/data_dictionary.md), [Tableau instructions](docs/tableau.md), and [demo walkthrough](docs/demo.md).

The delivered package includes [verification results](docs/verification.md), an [overview preview](artifacts/screenshots/overview.png), a [capacity-lab preview](artifacts/screenshots/capacity-lab.png), and an installable Python wheel under `artifacts/dist/`.

## Engineering structure

`src/hospital_intelligence/` separates configuration, generation, validation, transformations, pure analytics, forecasting, simulation, export, cached services, CLI, and presentation. Raw and processed data live under `data/`; exports under `exports/`. These reproducible outputs are ignored by Git. `uv.lock` fixes dependency versions. All filesystem locations are configurable and no external service credentials are needed.

Metrics retain their grain: admission-date encounter counts, discharge-date outcome cohorts, and calendar-date bed-hours. Occupancy does not change with demographic filters because bed capacity has no demographic allocation. Forecasts use the complete historical hospital/department series regardless of dashboard date range.
# hospital-analytics-dashboard
