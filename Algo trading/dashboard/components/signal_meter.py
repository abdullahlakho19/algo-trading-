"""
dashboard/components/signal_meter.py
─────────────────────────────────────────────────────────────────────────────
Signal Strength Gauge Component.
Renders a probability score gauge and signal details panel.
─────────────────────────────────────────────────────────────────────────────
"""

import plotly.graph_objects as go
import streamlit as st


def render_probability_gauge(
    probability: float,
    symbol: str = "",
    direction: str = "",
) -> go.Figure:
    """
    Render a gauge chart showing signal probability.
    Green zone: > 75% (trade zone)
    Yellow zone: 50-75% (watch zone)
    Red zone: < 50% (no trade)
    """
    color = "#00E676" if probability >= 0.75 else "#FF9100" if probability >= 0.5 else "#FF5252"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title=dict(
            text=f"{symbol} {direction.upper() if direction else 'SIGNAL'} PROBABILITY",
            font=dict(color="#C9A84C", size=12),
        ),
        number=dict(suffix="%", font=dict(color=color, size=28)),
        delta=dict(reference=75, valueformat=".1f"),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#2A3A4A",
                tickfont=dict(color="#8A94A8"),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0D1B2A",
            borderwidth=0,
            steps=[
                dict(range=[0, 50],  color="#2A0D0D"),
                dict(range=[50, 75], color="#1A1A0D"),
                dict(range=[75, 100], color="#0D2A0D"),
            ],
            threshold=dict(
                line=dict(color="#C9A84C", width=2),
                thickness=0.75,
                value=75,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="#0A0E1A",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#D0D6E0"),
    )

    return fig


def render_signal_details(signal_components: dict) -> None:
    """Render a breakdown of signal components in Streamlit."""
    if not signal_components:
        return

    st.markdown("**Signal Component Breakdown**")
    for name, passed in signal_components.items():
        icon  = "✅" if passed else "❌"
        color = "#00E676" if passed else "#FF5252"
        label = name.replace("_", " ").title()
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; '
            f'padding:4px 8px; background:#0D1B2A; border-radius:4px; margin:2px 0">'
            f'<span style="color:#8A94A8; font-size:12px">{label}</span>'
            f'<span style="color:{color}; font-size:14px">{icon}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_gate_status(gates: list[dict]) -> None:
    """Render all 6 safety gates with pass/fail status."""
    st.markdown("**6-Gate Safety Filter**")
    for gate in gates:
        passed = gate.get("passed", False)
        icon   = "✅" if passed else "❌"
        color  = "#00E676" if passed else "#FF5252"
        name   = gate.get("gate_name", "")
        reason = gate.get("reason", "")
        st.markdown(
            f'<div style="background:#0D1B2A; border-left: 3px solid {color}; '
            f'padding: 6px 10px; margin: 3px 0; border-radius: 3px">'
            f'<span style="color:{color}">{icon} Gate {gate.get("gate_number","")} — {name}</span>'
            f'<br><small style="color:#8A94A8">{reason}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )