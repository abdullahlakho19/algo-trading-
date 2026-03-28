"""
dashboard/components/regime_indicator.py
─────────────────────────────────────────────────────────────────────────────
Market Regime Visual Indicator Component.
Displays current regime classification with color coding
and key metrics as a live status panel.
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st


REGIME_CONFIG = {
    "trending_up":    {"color": "#00E676", "icon": "📈", "label": "TRENDING UP",    "action": "Trade Long"},
    "trending_down":  {"color": "#FF5252", "icon": "📉", "label": "TRENDING DOWN",  "action": "Trade Short"},
    "accumulation":   {"color": "#00D4FF", "icon": "🏦", "label": "ACCUMULATION",   "action": "Long Setups"},
    "distribution":   {"color": "#FF9100", "icon": "🏦", "label": "DISTRIBUTION",   "action": "Short Setups"},
    "balance":        {"color": "#C9A84C", "icon": "⚖️", "label": "BALANCE",        "action": "Range Trade"},
    "volatile":       {"color": "#FF5252", "icon": "⚡", "label": "VOLATILE",       "action": "AVOID"},
    "unknown":        {"color": "#4A5A6A", "icon": "❓", "label": "UNKNOWN",        "action": "Wait"},
    "neutral":        {"color": "#4A5A6A", "icon": "➖", "label": "NEUTRAL",        "action": "Wait"},
}


def render_regime_card(
    regime: str,
    confidence: float,
    trend_strength: float,
    volatility: float,
    bias: str,
) -> None:
    """Render a single regime indicator card in Streamlit."""
    cfg   = REGIME_CONFIG.get(regime, REGIME_CONFIG["unknown"])
    color = cfg["color"]

    st.markdown(f"""
    <div style="
        background: #0D1B2A;
        border: 2px solid {color};
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size: 32px">{cfg['icon']}</div>
        <div style="color: {color}; font-size: 18px; font-weight: bold; margin-top: 6px">
            {cfg['label']}
        </div>
        <div style="color: #8A94A8; font-size: 11px; margin-top: 4px">
            Action: <b style="color:{color}">{cfg['action']}</b>
        </div>
        <hr style="border-color: #1E2A3A; margin: 10px 0">
        <div style="display: flex; justify-content: space-around; font-size: 11px; color: #8A94A8">
            <span>Confidence<br><b style="color:#D0D6E0">{confidence:.0%}</b></span>
            <span>ADX<br><b style="color:#D0D6E0">{trend_strength:.1f}</b></span>
            <span>Vol<br><b style="color:#D0D6E0">{volatility:.3%}</b></span>
            <span>Bias<br><b style="color:#D0D6E0">{bias.upper()}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_multi_regime(regimes: dict) -> None:
    """Render regime cards for multiple symbols."""
    if not regimes:
        st.info("No regime data available.")
        return

    cols = st.columns(min(len(regimes), 4))
    for col, (symbol, result) in zip(cols, regimes.items()):
        with col:
            st.markdown(f"**{symbol}**")
            render_regime_card(
                regime=result.regime.value,
                confidence=result.confidence,
                trend_strength=result.trend_strength,
                volatility=result.volatility,
                bias=result.bias,
            )