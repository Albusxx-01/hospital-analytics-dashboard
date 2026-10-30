"""Consistent Plotly figures built from prefiltered, explicitly grained frames."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analytics import daily_series, readmission_cohorts

TEAL = "#087f8c"
NAVY = "#162c46"
BLUE = "#7399c6"
GOLD = "#c18b37"
PALETTE = [TEAL, BLUE, GOLD, "#8b81b1"]


def style(figure: go.Figure, height: int = 300) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        font={
            "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            "size": 12,
            "color": "#6b7b8e",
        },
        margin={"l": 45, "r": 24, "t": 24, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        hoverlabel={"bgcolor": NAVY, "font_color": "white"},
        legend={"orientation": "h", "y": 1.12, "x": 0, "font_size": 11},
        uirevision="keep-zoom",
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="#edf1f5", zeroline=False, title_font_size=11)
    return figure


def empty(message: str = "No encounters match these filters") -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style(figure)


def trends(daily: pd.DataFrame) -> go.Figure:
    if daily.empty:
        return empty("No operational data in this range")
    data = daily_series(daily)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=data.date,
            y=data.admissions,
            name="Daily admissions",
            marker_color="#deecf1",
            hovertemplate="%{x|%b %d}<br>%{y} admissions<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data.date,
            y=data.admissions.rolling(7, min_periods=1).mean(),
            name="7-day admissions",
            line={"color": TEAL, "width": 3},
            hovertemplate="%{y:.1f} admissions / day<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data.date,
            y=data.occupancy_rate,
            name="Occupancy",
            line={"color": BLUE, "width": 2, "dash": "dot"},
            hovertemplate="%{y:.1%} occupancy<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Admissions", secondary_y=False)
    figure.update_yaxes(title_text="Occupancy", tickformat=".0%", secondary_y=True, showgrid=False)
    return style(figure, 330)


def department_occupancy(daily: pd.DataFrame) -> go.Figure:
    if daily.empty:
        return empty()
    data = daily.groupby("department")[["occupied_bed_hours", "available_bed_hours"]].sum()
    data["rate"] = data.occupied_bed_hours / data.available_bed_hours
    data = data.sort_values("rate")
    figure = go.Figure(
        go.Bar(
            x=data.rate,
            y=data.index,
            orientation="h",
            marker_color=[TEAL if value < 0.9 else GOLD for value in data.rate],
            text=[f"{value:.1%}" for value in data.rate],
            textposition="outside",
            cliponaxis=False,
            customdata=data.index,
            hovertemplate="%{y}<br>%{x:.1%} occupancy<extra>Click to filter</extra>",
        )
    )
    figure.add_vline(x=1, line_dash="dot", line_color="#c5ced8")
    figure.update_xaxes(tickformat=".0%", range=[0, max(1.15, float(data.rate.max()) * 1.22)])
    figure.update_layout(clickmode="event+select", bargap=0.48)
    return style(figure, 330)


def arrival_heatmap(encounters: pd.DataFrame) -> go.Figure:
    if encounters.empty:
        return empty()
    table = pd.crosstab(
        encounters.admitted_at.dt.dayofweek, encounters.admitted_at.dt.hour
    ).reindex(index=range(7), columns=range(24), fill_value=0)
    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=list(range(24)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=[[0, "#f1f7f8"], [0.45, "#7cbfc2"], [1, TEAL]],
            xgap=2,
            ygap=3,
            colorbar={"title": "Arrivals", "thickness": 10},
            hovertemplate="%{y} · %{x}:00<br>%{z} admissions<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(dtick=3)
    return style(figure, 320)


def los_distribution(discharges: pd.DataFrame) -> go.Figure:
    if discharges.empty:
        return empty("No completed stays in this discharge window")
    figure = px.violin(
        discharges,
        x="department",
        y="los_days",
        color="department",
        box=True,
        points=False,
        color_discrete_sequence=PALETTE,
    )
    figure.update_layout(showlegend=False)
    figure.update_yaxes(title_text="Length of stay · days")
    figure.update_xaxes(tickfont_size=10, title_text=None)
    return style(figure, 320)


def journeys(encounters: pd.DataFrame, stays: pd.DataFrame) -> go.Figure:
    if encounters.empty:
        return empty()
    selected = stays.loc[stays.encounter_id.isin(encounters.encounter_id)].sort_values(
        ["encounter_id", "entered_at"]
    )
    pairs: list[tuple[str, str]] = []
    starts = encounters.set_index("encounter_id")
    for encounter_id, group in selected.groupby("encounter_id", sort=False):
        row = starts.loc[encounter_id]
        path = [f"{row.admission_type} admission", *group.stage.tolist(), row.disposition]
        pairs.extend(zip(path[:-1], path[1:]))
    edges = (
        pd.DataFrame(pairs, columns=["source", "target"]).value_counts().reset_index(name="value")
    )
    labels = list(
        dict.fromkeys(
            [
                "Emergency admission",
                "Planned admission",
                "Emergency",
                "ICU",
                "Ward",
                "Home",
                "Post-acute care",
                "Deceased",
                "In hospital",
            ]
        )
    )
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": [NAVY, NAVY, BLUE, GOLD, TEAL, "#9ac4bc", "#8b81b1", "#a9b3c1", "#c0d2e5"],
                "pad": 24,
                "thickness": 16,
                "line_width": 0,
            },
            link={
                "source": [labels.index(x) for x in edges.source],
                "target": [labels.index(x) for x in edges.target],
                "value": edges.value,
                "color": "rgba(8,127,140,0.15)",
                "hovertemplate": "%{source.label} → %{target.label}<br>%{value:,} transitions<extra></extra>",
            },
        )
    )
    return style(figure, 320)


def cohort_intervals(discharges: pd.DataFrame, field: str) -> go.Figure:
    table = readmission_cohorts(discharges, field).sort_values("rate")
    if table.empty:
        return empty("No discharges with complete 30-day follow-up")
    figure = go.Figure(
        go.Scatter(
            x=table.rate,
            y=table[field],
            mode="markers",
            marker={"color": TEAL, "size": 12},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": table.upper - table.rate,
                "arrayminus": table.rate - table.lower,
                "color": BLUE,
                "thickness": 2,
            },
            customdata=np.column_stack([table.eligible, table.readmissions]),
            hovertemplate="%{y}<br>%{x:.1%} readmissions<br>%{customdata[1]} / %{customdata[0]} eligible discharges<extra>95% Wilson interval</extra>",
        )
    )
    figure.update_xaxes(tickformat=".0%", rangemode="tozero")
    return style(figure, 290)


def cost_scatter(discharges: pd.DataFrame) -> go.Figure:
    if discharges.empty:
        return empty()
    sample = discharges.sample(min(1500, len(discharges)), random_state=42)
    figure = px.scatter(
        sample,
        x="los_days",
        y="total_cost",
        color="department",
        color_discrete_sequence=PALETTE,
        opacity=0.55,
        hover_data={
            "encounter_id": True,
            "diagnosis": True,
            "los_days": ":.1f",
            "total_cost": ":$,.0f",
        },
        render_mode="webgl",
    )
    figure.update_yaxes(title_text="Encounter cost · USD", tickprefix="$", tickformat="~s")
    figure.update_xaxes(title_text="Completed length of stay · days")
    figure.update_traces(marker_size=6)
    return style(figure, 340)


def cost_breakdown(discharges: pd.DataFrame) -> go.Figure:
    if discharges.empty:
        return empty()
    components = ["cost_accommodation", "cost_procedures", "cost_medication", "cost_diagnostics"]
    table = discharges.groupby("department")[components].mean()
    figure = go.Figure()
    for name, color in zip(components, PALETTE, strict=True):
        figure.add_bar(
            x=table.index,
            y=table[name],
            name=name.removeprefix("cost_").title(),
            marker_color=color,
        )
    figure.update_layout(barmode="stack", bargap=0.5)
    figure.update_yaxes(title_text="Mean cost per discharge · USD", tickprefix="$", tickformat="~s")
    figure.update_xaxes(tickfont_size=10, title_text=None)
    return style(figure, 340)


def forecast_chart(history: pd.Series, forecast: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    recent = history.iloc[-84:]
    figure.add_scatter(
        x=recent.index,
        y=recent,
        name="Observed admissions",
        mode="lines",
        line={"color": "#a6b7c8", "width": 1.5},
    )
    figure.add_scatter(
        x=forecast.date,
        y=forecast.upper,
        mode="lines",
        line_width=0,
        showlegend=False,
        hoverinfo="skip",
    )
    figure.add_scatter(
        x=forecast.date,
        y=forecast.lower,
        mode="lines",
        line_width=0,
        fill="tonexty",
        fillcolor="rgba(8,127,140,.13)",
        name="90% prediction interval",
        hoverinfo="skip",
    )
    figure.add_scatter(
        x=forecast.date,
        y=forecast.predicted,
        name="28-day forecast",
        mode="lines",
        line={"color": TEAL, "width": 3},
    )
    figure.add_vline(x=str(forecast.date.iloc[0]), line_dash="dot", line_color=TEAL)
    figure.update_yaxes(title_text="Admissions per day", rangemode="tozero")
    return style(figure, 330)


def simulation_charts(simulation: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    occupancy_figure, risk_figure = go.Figure(), go.Figure()
    for name, color in (("Baseline", BLUE), ("Scenario", TEAL)):
        data = simulation.loc[simulation.scenario.eq(name)]
        if name == "Scenario":
            occupancy_figure.add_scatter(
                x=data.date,
                y=data.upper,
                mode="lines",
                line_width=0,
                showlegend=False,
                hoverinfo="skip",
            )
            occupancy_figure.add_scatter(
                x=data.date,
                y=data.lower,
                mode="lines",
                line_width=0,
                fill="tonexty",
                fillcolor="rgba(8,127,140,.10)",
                name="Scenario 90% interval",
                hoverinfo="skip",
            )
        occupancy_figure.add_scatter(
            x=data.date,
            y=data.expected_occupancy,
            name=name,
            mode="lines",
            line={"color": color, "width": 3},
            hovertemplate="%{x|%b %d}<br>%{y:.1%} expected occupancy<extra>" + name + "</extra>",
        )
        risk_figure.add_scatter(
            x=data.date,
            y=data.overflow_probability,
            name=name,
            mode="lines",
            line={"color": color, "width": 3},
            hovertemplate="%{x|%b %d}<br>%{y:.1%} probability<extra>" + name + "</extra>",
        )
    occupancy_figure.add_hline(y=1, line_color=GOLD, line_dash="dot")
    occupancy_figure.update_yaxes(tickformat=".0%", title_text="Occupied / available bed-days")
    risk_figure.update_yaxes(
        tickformat=".0%", range=[0, 1], title_text="Probability of demand > capacity"
    )
    return style(occupancy_figure, 320), style(risk_figure, 320)
