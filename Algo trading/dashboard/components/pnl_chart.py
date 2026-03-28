"""
dashboard/components/pnl_chart.py
─────────────────────────────────────────────────────────────────────────────
Live P&L Chart Component.
Renders interactive equity curve, daily P&L bars,
and drawdown chart using Plotly.
─────────────────────────────────────────────────────────────────────────────
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render_equity_curve(trades: list, initial_capital: float = 100_000.0) -> go.Figure:
    """Render full equity curve with drawdown overlay."""
    if not trades:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0E1A",
            plot_bgcolor="#0D1B2A",
            title="Equity Curve — No Trades Yet",
            height=300,
        )
        return fig

    # Build equity series
    equity = [initial_capital]
    dates  = [trades[0].opened_at]
    for t in trades:
        equity.append(equity[-1] + t.pnl)
        dates.append(t.closed_at)

    eq_series = pd.Series(equity)
    drawdown  = (eq_series - eq_series.cummax()) / eq_series.cummax() * 100

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    # Equity curve
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.08)",
        line=dict(color="#00D4FF", width=2),
        name="Equity",
    ), row=1, col=1)

    # Break-even line
    fig.add_hline(
        y=initial_capital, line_dash="dash",
        line_color="#C9A84C", line_width=1,
        annotation_text="Initial Capital",
        annotation_font_color="#C9A84C",
        row=1, col=1,
    )

    # Drawdown
    fig.add_trace(go.Bar(
        x=dates, y=drawdown.tolist(),
        marker_color=["#FF5252" if d < 0 else "#00E676" for d in drawdown],
        name="Drawdown %",
        opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0D1B2A",
        height=380,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis2=dict(gridcolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A", tickprefix="$", tickformat=",.0f"),
        yaxis2=dict(gridcolor="#1E2A3A", ticksuffix="%"),
        title=dict(text="Equity Curve & Drawdown", font=dict(color="#C9A84C", size=13)),
    )

    return fig


def render_daily_pnl(trades: list) -> go.Figure:
    """Render daily P&L bar chart."""
    if not trades:
        return go.Figure()

    df = pd.DataFrame([{"date": t.closed_at.date(), "pnl": t.pnl} for t in trades])
    daily = df.groupby("date")["pnl"].sum().reset_index()

    colors = ["#00E676" if p >= 0 else "#FF5252" for p in daily["pnl"]]

    fig = go.Figure(go.Bar(
        x=daily["date"],
        y=daily["pnl"],
        marker_color=colors,
        name="Daily P&L",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0D1B2A",
        height=220,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A", tickprefix="$"),
        title=dict(text="Daily P&L", font=dict(color="#C9A84C", size=13)),
    )

    return fig


def render_win_loss_pie(trades: list) -> go.Figure:
    """Render win/loss breakdown pie chart."""
    if not trades:
        return go.Figure()

    wins   = sum(1 for t in trades if t.outcome == "win")
    losses = sum(1 for t in trades if t.outcome == "loss")

    fig = go.Figure(go.Pie(
        labels=["Wins", "Losses"],
        values=[wins, losses],
        marker=dict(colors=["#00E676", "#FF5252"]),
        hole=0.5,
        textinfo="label+percent",
        textfont=dict(color="#D0D6E0"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        height=220,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Win/Loss Split", font=dict(color="#C9A84C", size=13)),
    )

    return fig