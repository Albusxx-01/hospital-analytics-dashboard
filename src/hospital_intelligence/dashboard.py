"""Presentation and thin Dash callbacks over the analytics service."""

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from . import charts
from .analytics import kpis, occupancy, select_daily, select_encounters
from .config import DEPARTMENTS, HOSPITALS, Filters
from .services import AnalyticsService, scope_frame
from .simulation import Scenario

GRAPH_CONFIG: Any = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def graph(identifier: str) -> dcc.Graph:
    return dcc.Graph(
        id=identifier,
        config=GRAPH_CONFIG,
        responsive=True,
        style={"height": "330px", "minHeight": "330px", "width": "100%"},
        figure=charts.empty("Preparing analytics…"),
    )


def panel(title: str, subtitle: str, content: Any, tag: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.H3(title), html.P(subtitle)]),
                    html.Span(tag, className="panel-tag") if tag else None,
                ],
                className="panel-heading",
            ),
            content,
        ],
        className="panel",
    )


def dropdown(identifier: str, label: str, options: list[str], value: str) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=identifier),
            dcc.Dropdown(
                options=options, value=value, id=identifier, clearable=False, searchable=True
            ),
        ],
        className="filter-field",
    )


def metric(label: str, value: str, delta: str, caption: str, icon: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [html.Span(label), html.Span(icon, className="metric-icon")],
                className="metric-label",
            ),
            html.Strong(value, className="metric-value"),
            html.Div(
                [html.Span(delta, className="metric-delta"), html.Span(caption)],
                className="metric-foot",
            ),
        ],
        className="metric-card",
    )


def create_app(root: Path = Path("data")) -> Dash:
    service = AnalyticsService(root)
    defaults = service.defaults()
    app = Dash(
        __name__,
        title="Hospital Intelligence | Operations",
        assets_folder=str(Path(__file__).parent / "assets"),
        update_title="",
    )
    navigation_attributes: dict[str, Any] = {"aria-label": "Dashboard views"}
    app.server.config["ANALYTICS_SERVICE"] = service
    app.layout = html.Div(
        [
            dcc.Store(id="active-view", data="overview"),
            dcc.Download(id="download"),
            dcc.Store(id="applied-scenario", data={"demand": 15, "beds": 10, "los": -10}),
            html.A("Skip to dashboard", href="#dashboard-content", className="skip-link"),
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("H", className="brand-mark"),
                            html.Div([html.Strong("HOSPITAL"), html.Span("INTELLIGENCE")]),
                        ],
                        className="brand",
                    ),
                    html.Div("WORKSPACE", className="nav-label"),
                    html.Nav(
                        [
                            html.Button(
                                [html.Span("◫", className="nav-icon"), "Overview"],
                                id="nav-overview",
                                n_clicks=0,
                                className="nav-item active",
                            ),
                            html.Button(
                                [html.Span("⇄", className="nav-icon"), "Patient flow"],
                                id="nav-flow",
                                n_clicks=0,
                                className="nav-item",
                            ),
                            html.Button(
                                [html.Span("◈", className="nav-icon"), "Outcomes & costs"],
                                id="nav-outcomes",
                                n_clicks=0,
                                className="nav-item",
                            ),
                            html.Button(
                                [
                                    html.Span("⌁", className="nav-icon"),
                                    "Capacity lab",
                                    html.Span("SIM", className="nav-chip"),
                                ],
                                id="nav-capacity",
                                n_clicks=0,
                                className="nav-item",
                            ),
                        ],
                        **navigation_attributes,
                    ),
                    html.Div(
                        [
                            html.Div("YOUR NETWORK", className="nav-label"),
                            *[
                                html.Div(
                                    [html.Span("", className="hospital-dot"), name],
                                    className="hospital-name",
                                )
                                for name in HOSPITALS
                            ],
                        ],
                        className="network",
                    ),
                    html.Div(
                        [
                            html.Div("Built for better decisions.", className="sidebar-message"),
                            html.P(
                                "Explore the story behind every admission, every bed, and every day."
                            ),
                            html.Div(
                                [
                                    html.Span("", className="status-dot"),
                                    "Synthetic data · Local demo",
                                ],
                                className="sidebar-status",
                            ),
                        ],
                        className="sidebar-bottom",
                    ),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.Span("Analytics workspace"),
                                    html.Span("/", className="breadcrumb-slash"),
                                    html.Strong("Hospital operations"),
                                ],
                                className="breadcrumb",
                            ),
                            html.Div(
                                [
                                    html.Span("", className="status-dot"),
                                    "Dataset ready",
                                    html.Span("HI", className="avatar"),
                                ],
                                className="header-status",
                            ),
                        ],
                        className="topbar",
                    ),
                    html.Main(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "NETWORK PERFORMANCE",
                                                id="page-eyebrow",
                                                className="eyebrow",
                                            ),
                                            html.H1(
                                                "A healthier view of operations.", id="page-title"
                                            ),
                                            html.P(
                                                "Understand demand. Find pressure points. Plan with confidence.",
                                                id="page-description",
                                                className="page-description",
                                            ),
                                        ]
                                    ),
                                    html.Button(
                                        [html.Span("↓ "), "Export current view"],
                                        id="export-button",
                                        n_clicks=0,
                                        className="button secondary",
                                    ),
                                ],
                                className="page-heading",
                            ),
                            html.Div(
                                [
                                    dropdown(
                                        "hospital",
                                        "Hospital",
                                        ["All hospitals", *HOSPITALS],
                                        defaults.hospital,
                                    ),
                                    dropdown(
                                        "department",
                                        "Department",
                                        ["All departments", *DEPARTMENTS],
                                        defaults.department,
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Reporting period", htmlFor="dates"),
                                            dcc.DatePickerRange(
                                                id="dates",
                                                start_date=defaults.start,
                                                end_date=defaults.end,
                                                min_date_allowed=service.manifest["start"],
                                                max_date_allowed=service.manifest["end"],
                                                display_format="DD MMM YYYY",
                                                minimum_nights=0,
                                            ),
                                        ],
                                        className="filter-field date-field",
                                    ),
                                    html.Button(
                                        "↺  Reset", id="reset", n_clicks=0, className="button reset"
                                    ),
                                ],
                                className="filters",
                            ),
                            html.Div(id="period-caption", className="period-caption"),
                            dcc.Loading(
                                html.Div(id="kpi-row", className="kpi-row"),
                                type="circle",
                                color=charts.TEAL,
                                delay_show=300,
                            ),
                            dcc.Loading(
                                html.Div(
                                    [
                                        html.Section(
                                            [
                                                html.Div(
                                                    [
                                                        panel(
                                                            "Demand & utilization",
                                                            "Daily activity with a 7-day admissions trend",
                                                            graph("trend-chart"),
                                                            "DAILY",
                                                        ),
                                                        panel(
                                                            "Where is capacity tight?",
                                                            "Occupied bed-hours / available bed-hours · click a bar to filter",
                                                            graph("department-chart"),
                                                            "BY DEPARTMENT",
                                                        ),
                                                    ],
                                                    className="chart-grid wide-left",
                                                ),
                                                html.Div(id="insight", className="insight-strip"),
                                                panel(
                                                    "Department performance",
                                                    "Compare volumes, completed stays, and weighted utilization",
                                                    html.Div(
                                                        id="department-table",
                                                        className="table-wrapper",
                                                    ),
                                                ),
                                            ],
                                            id="view-overview",
                                        ),
                                        html.Section(
                                            [
                                                html.Div(
                                                    [
                                                        panel(
                                                            "When patients arrive",
                                                            "Admission counts by weekday and hour",
                                                            graph("heatmap-chart"),
                                                        ),
                                                        panel(
                                                            "The shape of a hospital stay",
                                                            "Completed discharges · median and distribution",
                                                            graph("los-chart"),
                                                        ),
                                                    ],
                                                    className="chart-grid",
                                                ),
                                                panel(
                                                    "The patient journey",
                                                    "Observed transitions for the selected admission cohort · status at dataset cutoff",
                                                    graph("journey-chart"),
                                                    "CARE PATHWAYS",
                                                ),
                                            ],
                                            id="view-flow",
                                            style={"display": "none"},
                                        ),
                                        html.Section(
                                            [
                                                html.Div(
                                                    [
                                                        dropdown(
                                                            "age",
                                                            "Age band",
                                                            [
                                                                "All ages",
                                                                "18–39",
                                                                "40–64",
                                                                "65–79",
                                                                "80+",
                                                            ],
                                                            "All ages",
                                                        ),
                                                        dropdown(
                                                            "sex",
                                                            "Recorded sex",
                                                            ["All sexes", "Female", "Male"],
                                                            "All sexes",
                                                        ),
                                                        dropdown(
                                                            "diagnosis",
                                                            "Diagnosis",
                                                            [
                                                                "All diagnoses",
                                                                *sorted(
                                                                    service.encounters.diagnosis.unique()
                                                                ),
                                                            ],
                                                            "All diagnoses",
                                                        ),
                                                        html.P(
                                                            "Cohort filters affect the charts below and their export. Executive metrics stay at hospital / department level.",
                                                            className="cohort-note",
                                                        ),
                                                    ],
                                                    className="filters cohort-filters",
                                                ),
                                                html.Div(
                                                    [
                                                        panel(
                                                            "30-day readmissions",
                                                            "Unplanned returns · eligible discharge cohort · 95% Wilson intervals",
                                                            graph("readmission-chart"),
                                                        ),
                                                        panel(
                                                            "Outcomes across age groups",
                                                            "Descriptive comparison, without case-mix adjustment",
                                                            graph("age-chart"),
                                                        ),
                                                    ],
                                                    className="chart-grid",
                                                ),
                                                html.Div(
                                                    [
                                                        panel(
                                                            "The cost of a longer stay",
                                                            "Completed encounters · up to 1,500 reproducibly sampled points",
                                                            graph("scatter-chart"),
                                                        ),
                                                        panel(
                                                            "What drives encounter cost?",
                                                            "Mean cost per completed discharge · USD",
                                                            graph("cost-chart"),
                                                        ),
                                                    ],
                                                    className="chart-grid",
                                                ),
                                            ],
                                            id="view-outcomes",
                                            style={"display": "none"},
                                        ),
                                        html.Section(
                                            [
                                                html.Div(
                                                    id="forecast-context", className="insight-strip"
                                                ),
                                                panel(
                                                    "Looking 28 days ahead",
                                                    "Model selected against a weekly seasonal baseline using six chronological backtests",
                                                    graph("forecast-chart"),
                                                    "90% INTERVAL",
                                                ),
                                                html.Div(
                                                    id="model-metrics", className="model-metrics"
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H3(
                                                                    "Make room for what’s next."
                                                                ),
                                                                html.P(
                                                                    "Adjust the operating assumptions and compare the same simulated patient arrivals."
                                                                ),
                                                                html.Span(
                                                                    "400 paired simulations · 28 days",
                                                                    className="panel-tag",
                                                                ),
                                                            ],
                                                            className="scenario-intro",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Demand change"),
                                                                dcc.Slider(
                                                                    -30,
                                                                    50,
                                                                    5,
                                                                    value=15,
                                                                    id="demand",
                                                                    updatemode="drag",
                                                                    marks={
                                                                        -30: "−30%",
                                                                        0: "0",
                                                                        50: "+50%",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                            ],
                                                            className="slider-field",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Available beds"),
                                                                dcc.Slider(
                                                                    -30,
                                                                    50,
                                                                    5,
                                                                    value=10,
                                                                    id="beds",
                                                                    updatemode="drag",
                                                                    marks={
                                                                        -30: "−30%",
                                                                        0: "0",
                                                                        50: "+50%",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                            ],
                                                            className="slider-field",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Length of stay"),
                                                                dcc.Slider(
                                                                    -30,
                                                                    30,
                                                                    5,
                                                                    value=-10,
                                                                    id="los",
                                                                    updatemode="drag",
                                                                    marks={
                                                                        -30: "−30%",
                                                                        0: "0",
                                                                        30: "+30%",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                            ],
                                                            className="slider-field",
                                                        ),
                                                        html.Button(
                                                            "Run scenario →",
                                                            id="run-scenario",
                                                            n_clicks=0,
                                                            className="button primary",
                                                        ),
                                                    ],
                                                    className="scenario-controls",
                                                ),
                                                html.Div(
                                                    id="scenario-summary",
                                                    className="scenario-summary",
                                                ),
                                                html.Div(
                                                    [
                                                        panel(
                                                            "Capacity under pressure",
                                                            "Expected occupancy · dotted line marks available capacity",
                                                            graph("simulation-chart"),
                                                        ),
                                                        panel(
                                                            "How likely is overflow?",
                                                            "Probability that daily bed demand exceeds available capacity",
                                                            graph("risk-chart"),
                                                        ),
                                                    ],
                                                    className="chart-grid",
                                                ),
                                                html.P(
                                                    "Operational estimates based on synthetic data. Capacity is pooled within the selected scope; ward compatibility, queues, staffing, and admission diversion are not modeled. Simulation uncertainty is conditional on the forecast mean.",
                                                    className="method-note",
                                                ),
                                            ],
                                            id="view-capacity",
                                            style={"display": "none"},
                                        ),
                                    ]
                                ),
                                type="circle",
                                color=charts.TEAL,
                                delay_show=400,
                            ),
                            html.Footer(
                                [
                                    html.Span("HOSPITAL INTELLIGENCE"),
                                    html.Span(
                                        f"Synthetic data · Jan 2024–Dec 2025 · USD · Dataset {service.version[:8]}"
                                    ),
                                    html.Span("Python / pandas / NumPy"),
                                ],
                                className="footer",
                            ),
                        ],
                        id="dashboard-content",
                        className="main-content",
                    ),
                ],
                className="main-shell",
            ),
        ],
        className="app-shell",
    )

    view_names = ["overview", "flow", "outcomes", "capacity"]
    headings = {
        "overview": (
            "NETWORK PERFORMANCE",
            "A healthier view of operations.",
            "Understand demand. Find pressure points. Plan with confidence.",
        ),
        "flow": (
            "PATIENT FLOW",
            "Every journey tells a story.",
            "See when patients arrive, how they move, and where time is spent.",
        ),
        "outcomes": (
            "OUTCOMES & COSTS",
            "Better outcomes. Clearer economics.",
            "Explore readmissions, patient cohorts, and the drivers of hospital cost.",
        ),
        "capacity": (
            "CAPACITY LAB",
            "Plan for the pressure ahead.",
            "Turn historical patterns into forecasts and test your next operating decision.",
        ),
    }

    @app.callback(
        Output("active-view", "data"),
        *[Output(f"view-{view}", "style") for view in view_names],
        *[Output(f"nav-{view}", "className") for view in view_names],
        Output("page-eyebrow", "children"),
        Output("page-title", "children"),
        Output("page-description", "children"),
        *[Input(f"nav-{view}", "n_clicks") for view in view_names],
        prevent_initial_call=True,
    )
    def navigate(*_: int) -> tuple:
        selected = str(ctx.triggered_id).removeprefix("nav-")
        return (
            selected,
            *[{"display": "block" if view == selected else "none"} for view in view_names],
            *["nav-item active" if view == selected else "nav-item" for view in view_names],
            *headings[selected],
        )

    @app.callback(
        Output("hospital", "value"),
        Output("department", "value"),
        Output("dates", "start_date"),
        Output("dates", "end_date"),
        Output("age", "value"),
        Output("sex", "value"),
        Output("diagnosis", "value"),
        Input("reset", "n_clicks"),
        Input("department-chart", "clickData"),
        prevent_initial_call=True,
    )
    def reset_or_select(_: int, click_data: dict | None) -> tuple:
        if ctx.triggered_id == "department-chart" and click_data:
            department = click_data["points"][0]["y"]
            return no_update, department, no_update, no_update, no_update, no_update, no_update
        return (
            defaults.hospital,
            defaults.department,
            defaults.start,
            defaults.end,
            "All ages",
            "All sexes",
            "All diagnoses",
        )

    @lru_cache(maxsize=48)
    def operational_figures(filters: Filters) -> tuple:
        admitted = select_encounters(service.encounters, filters)
        discharged = select_encounters(service.encounters, filters, basis="discharged_at")
        daily = select_daily(service.daily, filters)
        return (
            charts.trends(daily),
            charts.department_occupancy(daily),
            charts.arrival_heatmap(admitted),
            charts.los_distribution(discharged),
            charts.journeys(admitted, service.stays),
        )

    @app.callback(
        Output("kpi-row", "children"),
        Output("period-caption", "children"),
        Output("insight", "children"),
        Output("department-table", "children"),
        *[
            Output(identifier, "figure")
            for identifier in (
                "trend-chart",
                "department-chart",
                "heatmap-chart",
                "los-chart",
                "journey-chart",
            )
        ],
        Input("hospital", "value"),
        Input("department", "value"),
        Input("dates", "start_date"),
        Input("dates", "end_date"),
    )
    def update_operations(hospital: str, department: str, start: str, end: str) -> tuple:
        if not start or not end or start > end:
            return (
                [],
                "Select a valid reporting period.",
                "No data",
                "",
                *[charts.empty("Select a valid date range") for _ in range(5)],
            )
        filters = Filters(start, end, hospital, department)
        current = kpis(service.encounters, service.daily, filters)
        duration = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        prior_end = pd.Timestamp(start) - pd.Timedelta(days=1)
        previous = replace(
            filters,
            start=str((prior_end - pd.Timedelta(days=duration - 1)).date()),
            end=str(prior_end.date()),
        )
        prior = kpis(service.encounters, service.daily, previous)
        prior_complete = pd.Timestamp(previous.start) >= pd.Timestamp(service.manifest["start"])

        def formatted(value: float, kind: str) -> str:
            if not np.isfinite(value):
                return "—"
            return {
                "count": f"{value:,.0f}",
                "percent": f"{value:.1%}",
                "days": f"{value:.1f}",
                "money": f"${value:,.0f}",
            }[kind]

        cards = []
        for key, label, kind, icon in (
            ("admissions", "Admissions", "count", "↗"),
            ("occupancy", "Bed occupancy", "percent", "▤"),
            ("los", "Avg. length of stay", "days", "◷"),
            ("readmission", "30-day readmissions", "percent", "↻"),
            ("cost", "Cost / discharge", "money", "$"),
        ):
            delta = "No prior period"
            if prior_complete and np.isfinite(current[key]) and np.isfinite(prior[key]):
                if kind == "percent":
                    delta = f"{(current[key] - prior[key]) * 100:+.1f} pp"
                elif prior[key]:
                    delta = f"{(current[key] / prior[key] - 1):+.1%}"
            cards.append(
                metric(
                    label,
                    formatted(current[key], kind),
                    delta,
                    "vs. previous period" if delta != "No prior period" else "within this dataset",
                    icon,
                )
            )
        daily = select_daily(service.daily, filters)
        grouped = daily.groupby("department")[["occupied_bed_hours", "available_bed_hours"]].sum()
        if len(grouped):
            rates = grouped.occupied_bed_hours / grouped.available_bed_hours
            insight = [
                html.Span("↗", className="insight-icon"),
                html.Div(
                    [
                        html.Strong(
                            f"{rates.idxmax()} has the highest utilization at {rates.max():.1%}. "
                        ),
                        "Open Capacity lab to explore how demand and bed availability could change the picture.",
                    ]
                ),
            ]
        else:
            insight = [html.Span("No operational records match this period.")]
        records = []
        for dept in DEPARTMENTS:
            if department != "All departments" and dept != department:
                continue
            numbers = kpis(service.encounters, service.daily, replace(filters, department=dept))
            records.append(
                html.Tr(
                    [
                        html.Td([html.Span("", className="table-dot"), dept]),
                        html.Td(formatted(numbers["admissions"], "count")),
                        html.Td(formatted(numbers["occupancy"], "percent")),
                        html.Td(formatted(numbers["los"], "days") + " days"),
                        html.Td(formatted(numbers["readmission"], "percent")),
                        html.Td(formatted(numbers["cost"], "money")),
                    ]
                )
            )
        table = html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(label)
                            for label in (
                                "Department",
                                "Admissions",
                                "Occupancy",
                                "Avg. stay",
                                "Readmissions",
                                "Cost / discharge",
                            )
                        ]
                    )
                ),
                html.Tbody(records),
            ]
        )
        caption = f"{pd.Timestamp(start):%d %b %Y} – {pd.Timestamp(end):%d %b %Y} · {duration} days · Readmissions: {int(current['eligible']):,} eligible discharges with complete follow-up · LOS and cost use discharge dates"
        return cards, caption, insight, table, *operational_figures(filters)

    @app.callback(
        *[
            Output(identifier, "figure")
            for identifier in ("readmission-chart", "age-chart", "scatter-chart", "cost-chart")
        ],
        Input("hospital", "value"),
        Input("department", "value"),
        Input("dates", "start_date"),
        Input("dates", "end_date"),
        Input("age", "value"),
        Input("sex", "value"),
        Input("diagnosis", "value"),
    )
    def update_outcomes(
        hospital: str, department: str, start: str, end: str, age: str, sex: str, diagnosis: str
    ) -> tuple:
        if not start or not end or start > end:
            return tuple(charts.empty("Select a valid reporting period") for _ in range(4))
        frame = select_encounters(
            service.encounters,
            Filters(start, end, hospital, department, age, sex, diagnosis),
            basis="discharged_at",
        )
        return (
            charts.cohort_intervals(frame, "department"),
            charts.cohort_intervals(frame, "age_band"),
            charts.cost_scatter(frame),
            charts.cost_breakdown(frame),
        )

    @app.callback(
        Output("forecast-chart", "figure"),
        Output("forecast-context", "children"),
        Output("model-metrics", "children"),
        Input("hospital", "value"),
        Input("department", "value"),
    )
    def update_forecast(hospital: str, department: str) -> tuple:
        result = service.forecast(hospital, department)
        history = scope_frame(service.daily, hospital, department).groupby("date").admissions.sum()
        evaluation = result.evaluation.groupby("model")[["mae", "interval_coverage"]].mean()
        selected = evaluation.loc[[result.model_name], ["mae", "interval_coverage"]].to_numpy(
            dtype=float
        )[0]
        context = [
            html.Span("⌁", className="insight-icon"),
            html.Div(
                [
                    html.Strong(f"{hospital} · {department}. "),
                    f"Training window: {result.training_start} to {result.training_end}. Forecasts use all historical dates; the reporting-period filter does not apply here.",
                ]
            ),
        ]
        baseline_mae = float(
            evaluation.loc[["Weekly seasonal baseline"], ["mae"]].to_numpy(dtype=float)[0, 0]
        )
        metrics = [
            metric(
                "Selected model",
                "Seasonal ridge"
                if result.model_name.startswith("Regularized")
                else "Weekly baseline",
                "6 chronological folds",
                "28 days each",
                "⌁",
            ),
            metric(
                "Backtest error",
                f"{selected[0]:.1f}",
                "Mean absolute error",
                "admissions / day",
                "≈",
            ),
            metric(
                "Interval coverage",
                f"{selected[1]:.1%}",
                "90% nominal interval",
                "calibrated on earlier folds",
                "◎",
            ),
            metric(
                "Baseline error",
                f"{baseline_mae:.1f}",
                "Weekly seasonal baseline",
                "admissions / day",
                "↔",
            ),
        ]
        return charts.forecast_chart(history, result.forecast), context, metrics

    @app.callback(
        Output("simulation-chart", "figure"),
        Output("risk-chart", "figure"),
        Output("scenario-summary", "children"),
        Output("applied-scenario", "data"),
        Input("run-scenario", "n_clicks"),
        Input("hospital", "value"),
        Input("department", "value"),
        State("demand", "value"),
        State("beds", "value"),
        State("los", "value"),
    )
    def update_simulation(
        _: int, hospital: str, department: str, demand: float, beds: float, los: float
    ) -> tuple:
        result = service.simulation(hospital, department, Scenario(demand, beds, los))
        occupancy_figure, risk_figure = charts.simulation_charts(result)
        base = result.loc[result.scenario.eq("Baseline")]
        changed = result.loc[result.scenario.eq("Scenario")]
        summary = [
            html.Strong(
                f"Applied scenario: demand {demand:+g}% · beds {beds:+g}% · stay {los:+g}%"
            ),
            html.Span(
                f"Mean occupancy {base.expected_occupancy.mean():.1%} → {changed.expected_occupancy.mean():.1%}"
            ),
            html.Span(
                f"Mean daily overflow risk {base.overflow_probability.mean():.1%} → {changed.overflow_probability.mean():.1%}"
            ),
        ]
        return occupancy_figure, risk_figure, summary, {"demand": demand, "beds": beds, "los": los}

    @app.callback(
        Output("download", "data"),
        Input("export-button", "n_clicks"),
        State("active-view", "data"),
        State("hospital", "value"),
        State("department", "value"),
        State("dates", "start_date"),
        State("dates", "end_date"),
        State("age", "value"),
        State("sex", "value"),
        State("diagnosis", "value"),
        State("applied-scenario", "data"),
        prevent_initial_call=True,
    )
    def download(
        _: int,
        view: str,
        hospital: str,
        department: str,
        start: str,
        end: str,
        age: str,
        sex: str,
        diagnosis: str,
        applied: dict,
    ) -> Any:
        if not start or not end or start > end:
            return no_update
        filters = Filters(start, end, hospital, department)
        if view == "overview":
            frame = select_daily(service.daily, filters)
        elif view == "flow":
            frame = select_encounters(service.encounters, filters)
        elif view == "outcomes":
            frame = select_encounters(
                service.encounters,
                replace(filters, age=age, sex=sex, diagnosis=diagnosis),
                basis="discharged_at",
            )
        else:
            frame = service.simulation(
                hospital, department, Scenario(applied["demand"], applied["beds"], applied["los"])
            )
        return dcc.send_data_frame(
            frame.to_csv, f"hospital_{view}_{service.version}.csv", index=False
        )

    return app
