"""
utils.py
--------
Small helpers shared by app.py:
- loading API secrets (local .env or Streamlit Cloud st.secrets)
- building the Plotly visualizations (gauge, radar, membership-function plots)
"""

import os
from typing import Dict

import numpy as np
import plotly.graph_objects as go

from fuzzy_engine import UNIVERSE, MEMBERSHIP_FUNCTIONS
from models import RANGES


def load_secrets_into_env() -> None:
    """Make the LLM API key/config available via os.environ regardless of
    whether the app is running locally (python-dotenv + .env file) or on
    Streamlit Community Cloud (st.secrets). Safe to call every run."""
    from dotenv import load_dotenv

    load_dotenv()  # no-op if there's no .env file (e.g. on Streamlit Cloud)

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
                try:
                    if key in st.secrets and not os.environ.get(key):
                        os.environ[key] = str(st.secrets[key])
                except Exception:  # noqa: BLE001  st.secrets raises if no secrets.toml exists
                    pass
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def score_gauge(score: float, category: str) -> go.Figure:
    """A gauge/speedometer chart for the final 0-100 quality score."""
    color_by_category = {
        "Very Poor": "#d62728",
        "Poor": "#ff7f0e",
        "Average": "#f2c94c",
        "Good": "#8bc34a",
        "Excellent": "#2ca02c",
    }
    bar_color = color_by_category.get(category, "#4c78a8")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            title={"text": f"Environment Quality — {category}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 25], "color": "#fddede"},
                    {"range": [25, 45], "color": "#fde3c6"},
                    {"range": [45, 65], "color": "#fdf3c6"},
                    {"range": [65, 85], "color": "#e2f2c6"},
                    {"range": [85, 100], "color": "#d3f2d3"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def inputs_radar_chart(values: Dict[str, float]) -> go.Figure:
    """Radar chart showing where each input sits within its own 0-100%
    normalized range, so wildly different units (ppm, dB, lux...) can be
    compared visually on one chart."""
    labels = list(values.keys())
    normalized = []
    for k in labels:
        lo, hi = RANGES[k]
        pct = (values[k] - lo) / (hi - lo) * 100
        normalized.append(round(float(np.clip(pct, 0, 100)), 1))

    labels_display = [l.capitalize() for l in labels]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=normalized + normalized[:1],
            theta=labels_display + labels_display[:1],
            fill="toself",
            name="Current readings (% of range)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=350,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


def membership_function_plot(variable: str, crisp_value: float = None) -> go.Figure:
    """Plot the membership functions (curves) for one input/output variable,
    optionally with a vertical line marking the current crisp value — this
    is the key diagram for explaining 'fuzzification' in the viva."""
    universe = UNIVERSE[variable]
    fig = go.Figure()
    for term, curve in MEMBERSHIP_FUNCTIONS[variable].items():
        fig.add_trace(go.Scatter(x=universe, y=curve, mode="lines", name=term))

    if crisp_value is not None:
        fig.add_vline(x=crisp_value, line_dash="dash", line_color="gray",
                       annotation_text=f"input = {crisp_value}")

    fig.update_layout(
        title=f"Membership functions: {variable.capitalize()}",
        xaxis_title=variable,
        yaxis_title="Membership degree",
        yaxis=dict(range=[0, 1.05]),
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def aggregated_output_plot(aggregated: np.ndarray, score: float) -> go.Figure:
    """Plot the aggregated output fuzzy set (Step: Aggregation) together
    with the defuzzified crisp score (Step: Defuzzification)."""
    universe = UNIVERSE["quality"]
    fig = go.Figure()

    for term, curve in MEMBERSHIP_FUNCTIONS["quality"].items():
        fig.add_trace(go.Scatter(x=universe, y=curve, mode="lines", name=term,
                                  line=dict(dash="dot", width=1), opacity=0.4))

    fig.add_trace(go.Scatter(x=universe, y=aggregated, mode="lines", name="Aggregated output",
                              fill="tozeroy", line=dict(width=2, color="#4c78a8")))
    fig.add_vline(x=score, line_color="red", line_width=2,
                  annotation_text=f"Centroid = {score}")

    fig.update_layout(
        title="Aggregated fuzzy output & defuzzified score",
        xaxis_title="Environment Quality (0-100)",
        yaxis_title="Membership degree",
        yaxis=dict(range=[0, 1.05]),
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
