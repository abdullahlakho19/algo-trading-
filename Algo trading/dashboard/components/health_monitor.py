"""
dashboard/components/health_monitor.py
─────────────────────────────────────────────────────────────────────────────
Model Health Monitor Component.
Displays A/B/C/D/F grade, performance scores, alerts,
and retraining recommendations.
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import plotly.graph_objects as go


GRADE_CONFIG = {
    "A":   {"color": "#00E676", "label": "EXCELLENT"},
    "B":   {"color": "#00D4FF", "label": "GOOD"},
    "C":   {"color": "#C9A84C", "label": "ACCEPTABLE"},
    "D":   {"color": "#FF9100", "label": "POOR"},
    "F":   {"color": "#FF5252", "label": "FAILING"},
    "N/A": {"color": "#4A5A6A", "label": "NO DATA"},
}


def render_health_grade(report) -> None:
    """Render the overall model health grade card."""
    grade  = getattr(report, "overall_grade", "N/A")
    score  = getattr(report, "overall_score", 0)
    cfg    = GRADE_CONFIG.get(grade, GRADE_CONFIG["N/A"])
    color  = cfg["color"]
    label  = cfg["label"]

    st.markdown(f"""
    <div style="
        background: #0D1B2A;
        border: 2px solid {color};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    ">
        <div style="color: {color}; font-size: 48px; font-weight: bold; line-height: 1">{grade}</div>
        <div style="color: #8A94A8; font-size: 11px; margin-top: 4px">MODEL HEALTH</div>
        <div style="color: {color}; font-size: 13px; font-weight: bold">{label}</div>
        <div style="margin-top: 12px">
            <div style="background: #0A0E1A; border-radius: 6px; height: 10px; overflow: hidden">
                <div style="background: {color}; width: {score}%; height: 100%; border-radius: 6px"></div>
            </div>
            <div style="color: #8A94A8; font-size: 10px; margin-top: 4px">{score:.0f} / 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_health_metrics(report) -> None:
    """Render detailed health metrics."""
    targets = {
        "Win Rate":      (getattr(report, "win_rate", 0),      0.75, "≥ 75%"),
        "Profit Factor": (getattr(report, "profit_factor", 0), 2.0,  "≥ 2.0"),
        "Sharpe Ratio":  (getattr(report, "sharpe_ratio", 0),  2.0,  "≥ 2.0"),
        "Model Accuracy":(getattr(report, "model_accuracy", 0),0.70, "≥ 70%"),
    }

    for metric, (value, target, label) in targets.items():
        pct    = min(1.0, value / target) if target > 0 else 0
        color  = "#00E676" if pct >= 1.0 else "#FF9100" if pct >= 0.7 else "#FF5252"
        bar_w  = int(pct * 100)

        if metric == "Win Rate" or metric == "Model Accuracy":
            display = f"{value:.1%}"
        else:
            display = f"{value:.2f}"

        st.markdown(f"""
        <div style="margin: 6px 0">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8A94A8">
                <span>{metric}</span>
                <span style="color:{color}"><b>{display}</b> (target {label})</span>
            </div>
            <div style="background: #0A0E1A; border-radius: 4px; height: 6px; margin-top: 3px">
                <div style="background:{color}; width:{bar_w}%; height:100%; border-radius:4px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_alerts(report) -> None:
    """Render health alerts and recommendations."""
    alerts  = getattr(report, "alerts", [])
    recs    = getattr(report, "recommendations", [])
    retrain = getattr(report, "needs_retrain", False)

    if retrain:
        st.warning("⚠️ Model retraining recommended")

    if alerts:
        st.markdown("**⚠️ Alerts**")
        for alert in alerts:
            st.markdown(
                f'<div style="background:#2A0D0D; border-left:3px solid #FF5252; '
                f'padding:6px 10px; margin:3px 0; border-radius:3px; '
                f'color:#FF8A80; font-size:12px">{alert}</div>',
                unsafe_allow_html=True,
            )

    if recs:
        st.markdown("**💡 Recommendations**")
        for rec in recs:
            st.markdown(
                f'<div style="background:#0D2A0D; border-left:3px solid #00E676; '
                f'padding:6px 10px; margin:3px 0; border-radius:3px; '
                f'color:#69F0AE; font-size:12px">{rec}</div>',
                unsafe_allow_html=True,
            )


def render_full_health_panel(report) -> None:
    """Render the complete model health panel."""
    col1, col2 = st.columns([1, 2])
    with col1:
        render_health_grade(report)
    with col2:
        render_health_metrics(report)

    st.markdown("")
    render_alerts(report)