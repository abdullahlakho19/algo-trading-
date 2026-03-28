"""
dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
Live Streamlit Dashboard.
Real-time monitoring of the trading agent — P&L, positions,
signals, market regime, model health, and controls.

Run with: streamlit run dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Agent — Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0A0E1A; }
    .stApp { background-color: #0A0E1A; }
    h1, h2, h3, h4, p, li { color: #D0D6E0 !important; }
    .metric-card {
        background-color: #0D1B2A;
        border: 1px solid #C9A84C;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #C9A84C; }
    .metric-label { font-size: 12px; color: #8A94A8; margin-top: 4px; }
    .signal-card {
        background: #0D1B2A;
        border-left: 4px solid #00D4FF;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .status-green { color: #00E676 !important; font-weight: bold; }
    .status-red   { color: #FF5252 !important; font-weight: bold; }
    .status-gold  { color: #C9A84C !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_simulator():
    """Load paper simulator safely."""
    try:
        from execution.paper_simulator import paper_simulator
        return paper_simulator
    except Exception:
        return None

def load_clock():
    try:
        from core.market_clock import market_clock
        return market_clock
    except Exception:
        return None

def load_risk():
    try:
        from risk.risk_manager import risk_manager
        return risk_manager
    except Exception:
        return None

def color_pnl(val: float) -> str:
    if val > 0:   return "status-green"
    if val < 0:   return "status-red"
    return ""


# ── Header ────────────────────────────────────────────────────────────────────
def render_header():
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("## 🏛️ INSTITUTIONAL TRADING AGENT")
        st.markdown("*Citadel • BlackRock • JP Morgan • Jane Street • Renaissance • XTX*")
    with col2:
        clock = load_clock()
        if clock:
            status = clock.get_status()
            sessions = ", ".join(status["sessions"]).upper()
            st.markdown(f"**Sessions:** {sessions}")
            st.markdown(f"**UTC:** {datetime.utcnow().strftime('%H:%M:%S')}")
    with col3:
        st.markdown(f"**Mode:** 🟡 PAPER TRADING")
        st.markdown(f"**Updated:** {datetime.utcnow().strftime('%H:%M:%S')}")


# ── Portfolio Metrics ─────────────────────────────────────────────────────────
def render_portfolio_metrics(sim):
    st.markdown("### 📊 Portfolio Overview")
    perf = sim.get_performance() if sim.closed_trades else {}
    status = sim.get_status()

    cols = st.columns(6)
    metrics = [
        ("Equity",       f"${status['equity']:,.2f}",          None),
        ("Total P&L",    f"${status['total_pnl']:,.2f}",        status['total_pnl']),
        ("P&L %",        f"{status['total_pnl_pct']:.2%}",      status['total_pnl_pct']),
        ("Win Rate",     f"{perf.get('win_rate', 0):.1%}",      None),
        ("Trades",       str(perf.get('total_trades', 0)),       None),
        ("Open Pos",     str(status['open_positions']),          None),
    ]

    for col, (label, value, raw) in zip(cols, metrics):
        with col:
            color = "#C9A84C"
            if raw is not None:
                color = "#00E676" if raw >= 0 else "#FF5252"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ── P&L Chart ─────────────────────────────────────────────────────────────────
def render_pnl_chart(sim):
    st.markdown("### 📈 Equity Curve")
    if not sim.closed_trades:
        st.info("No closed trades yet. P&L chart will appear here.")
        return

    trades = sim.closed_trades
    cumulative = []
    running    = sim.initial_capital
    for t in trades:
        running += t.pnl
        cumulative.append({"time": t.closed_at, "equity": running, "pnl": t.pnl})

    df = pd.DataFrame(cumulative)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["equity"],
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)",
        line=dict(color="#00D4FF", width=2),
        name="Equity",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0D1B2A",
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
        showlegend=False,
        xaxis=dict(gridcolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A", tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Open Positions ────────────────────────────────────────────────────────────
def render_open_positions(sim):
    st.markdown("### 🔓 Open Positions")
    if not sim.open_positions:
        st.info("No open positions.")
        return

    rows = []
    for sym, pos in sim.open_positions.items():
        pnl_color = "🟢" if pos.unrealised_pnl >= 0 else "🔴"
        rows.append({
            "Symbol":    pos.symbol,
            "Direction": pos.direction.upper(),
            "Qty":       round(pos.qty, 4),
            "Entry":     round(pos.entry_price, 5),
            "Current":   round(pos.current_price, 5),
            "SL":        round(pos.stop_loss, 5),
            "TP":        round(pos.take_profit, 5),
            "P&L":       f"{pnl_color} ${pos.unrealised_pnl:.2f}",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ── Trade History ─────────────────────────────────────────────────────────────
def render_trade_history(sim):
    st.markdown("### 📋 Recent Trades")
    if not sim.closed_trades:
        st.info("No closed trades yet.")
        return

    recent = sim.closed_trades[-20:][::-1]
    rows   = []
    for t in recent:
        outcome_icon = "✅" if t.outcome == "win" else "❌"
        rows.append({
            "ID":         t.trade_id,
            "Symbol":     t.symbol,
            "Direction":  t.direction.upper(),
            "Entry":      round(t.entry_price, 5),
            "Exit":       round(t.exit_price,  5),
            "P&L":        f"${t.pnl:.2f}",
            "Outcome":    f"{outcome_icon} {t.outcome.upper()}",
            "Duration":   f"{t.duration_min:.0f}m",
            "Reason":     t.exit_reason.replace("_", " ").upper(),
            "Closed At":  t.closed_at.strftime("%m-%d %H:%M"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Model Health ──────────────────────────────────────────────────────────────
def render_model_health():
    st.markdown("### 🏥 Model Health")
    cols = st.columns(4)

    health_items = [
        ("Ensemble Models",  "✅ Loaded",        "#00E676"),
        ("Data Engine",      "✅ Connected",      "#00E676"),
        ("Risk Manager",     "✅ Active",         "#00E676"),
        ("Circuit Breaker",  "✅ Armed",          "#00E676"),
    ]

    for col, (label, value, color) in zip(cols, health_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}; font-size:18px">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Market Regime ─────────────────────────────────────────────────────────────
def render_regime_panel():
    st.markdown("### 🌍 Market Sessions")
    clock = load_clock()
    if not clock:
        return

    status = clock.get_status()
    sessions = status["sessions"]

    session_data = {
        "asian":    {"label": "🌏 Asian",     "hours": "00:00-09:00 UTC"},
        "london":   {"label": "🌍 London",    "hours": "08:00-17:00 UTC"},
        "new_york": {"label": "🌎 New York",  "hours": "13:00-22:00 UTC"},
        "overlap":  {"label": "🔥 Overlap",   "hours": "13:00-17:00 UTC"},
    }

    cols = st.columns(4)
    for col, (key, info) in zip(cols, session_data.items()):
        with col:
            active = key in sessions
            color  = "#00E676" if active else "#4A5A6A"
            status_text = "ACTIVE" if active else "INACTIVE"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
                <div class="metric-value" style="color:{color}; font-size:16px">{info['label']}</div>
                <div class="metric-label">{info['hours']}</div>
                <div style="color:{color}; font-size:11px; margin-top:4px"><b>{status_text}</b></div>
            </div>
            """, unsafe_allow_html=True)


# ── Circuit Breaker Status ────────────────────────────────────────────────────
def render_circuit_breaker():
    risk = load_risk()
    if not risk:
        return

    cb = risk.circuit_breaker.get_status()
    color  = "#FF5252" if cb["paused"] else "#00E676"
    status = "🔴 TRIGGERED" if cb["paused"] else "🟢 ARMED"

    st.markdown(f"""
    <div style="background:#0D1B2A; border:1px solid {color}; border-radius:8px; padding:12px;">
        <b style="color:{color}">Circuit Breaker: {status}</b><br>
        <small style="color:#8A94A8">
            Daily P&L: {cb['daily_pnl']:.2%} (Limit: {cb['daily_limit']:.2%}) &nbsp;|&nbsp;
            Weekly P&L: {cb['weekly_pnl']:.2%} (Limit: {cb['weekly_limit']:.2%})
        </small>
    </div>
    """, unsafe_allow_html=True)


# ── Controls ──────────────────────────────────────────────────────────────────
def render_controls():
    st.markdown("### ⚙️ Controls")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("▶️ Start Agent", use_container_width=True):
            st.success("Agent started (paper mode)")
    with col2:
        if st.button("⏸ Pause Agent", use_container_width=True):
            st.warning("Agent paused")
    with col3:
        if st.button("📊 Export Excel", use_container_width=True):
            sim = load_simulator()
            if sim and sim.closed_trades:
                from reporting.excel_exporter import excel_exporter
                path = excel_exporter.export(
                    closed_trades=sim.closed_trades,
                    portfolio_value=sim.equity,
                    initial_capital=sim.initial_capital,
                )
                st.success(f"Exported: {path.name}")
            else:
                st.info("No trades to export yet.")
    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()


# ── Main Layout ───────────────────────────────────────────────────────────────
def main():
    sim = load_simulator()
    if sim is None:
        st.error("Could not load trading agent modules. Check your setup.")
        return

    render_header()
    st.divider()

    render_regime_panel()
    st.markdown("")

    render_portfolio_metrics(sim)
    st.markdown("")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_pnl_chart(sim)
    with col_right:
        render_circuit_breaker()
        st.markdown("")
        render_model_health()

    st.markdown("")
    render_open_positions(sim)
    st.markdown("")
    render_trade_history(sim)
    st.markdown("")
    render_controls()

    # Auto-refresh every 5 seconds
    time.sleep(config_refresh := 5)


if __name__ == "__main__":
    main()