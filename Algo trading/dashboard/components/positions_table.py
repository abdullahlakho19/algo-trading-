"""
dashboard/components/positions_table.py
─────────────────────────────────────────────────────────────────────────────
Open Positions Table Component.
Renders live positions with P&L, SL/TP progress bars,
and risk exposure per position.
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd


def render_positions_table(positions: dict) -> None:
    """
    Render live open positions with full detail.
    positions: dict of {symbol: SimulatedPosition}
    """
    if not positions:
        st.markdown("""
        <div style="background:#0D1B2A; border:1px solid #1E2A3A; border-radius:8px;
                    padding:20px; text-align:center; color:#4A5A6A">
            No open positions — agent is watching for setups
        </div>
        """, unsafe_allow_html=True)
        return

    for symbol, pos in positions.items():
        pnl_color = "#00E676" if pos.unrealised_pnl >= 0 else "#FF5252"
        dir_color = "#00D4FF" if pos.direction == "long" else "#FF5252"
        dir_icon  = "▲ LONG" if pos.direction == "long" else "▼ SHORT"

        # SL/TP progress bar
        if pos.direction == "long" and pos.stop_loss < pos.take_profit:
            total_range = pos.take_profit - pos.stop_loss
            progress    = (pos.current_price - pos.stop_loss) / total_range if total_range > 0 else 0
        elif pos.direction == "short" and pos.stop_loss > pos.take_profit:
            total_range = pos.stop_loss - pos.take_profit
            progress    = (pos.stop_loss - pos.current_price) / total_range if total_range > 0 else 0
        else:
            progress = 0.5

        progress_pct = max(0, min(100, progress * 100))
        bar_color    = "#00E676" if progress_pct > 50 else "#FF9100" if progress_pct > 25 else "#FF5252"

        st.markdown(f"""
        <div style="background:#0D1B2A; border:1px solid #1E2A3A; border-radius:8px;
                    padding:14px; margin:6px 0">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <span style="color:#D0D6E0; font-size:15px; font-weight:bold">{symbol}</span>
                    <span style="color:{dir_color}; font-size:11px; margin-left:8px;
                                 background:#0A0E1A; padding:2px 6px; border-radius:3px">
                        {dir_icon}
                    </span>
                    <span style="color:#8A94A8; font-size:11px; margin-left:8px">
                        {pos.market.upper()}
                    </span>
                </div>
                <div style="text-align:right">
                    <span style="color:{pnl_color}; font-size:16px; font-weight:bold">
                        ${pos.unrealised_pnl:+.2f}
                    </span>
                </div>
            </div>
            <div style="display:flex; gap:20px; margin-top:10px; font-size:11px; color:#8A94A8">
                <span>Entry: <b style="color:#D0D6E0">{pos.entry_price:.5f}</b></span>
                <span>Current: <b style="color:#D0D6E0">{pos.current_price:.5f}</b></span>
                <span>Qty: <b style="color:#D0D6E0">{pos.qty:.4f}</b></span>
                <span>SL: <b style="color:#FF5252">{pos.stop_loss:.5f}</b></span>
                <span>TP: <b style="color:#00E676">{pos.take_profit:.5f}</b></span>
            </div>
            <div style="margin-top:10px">
                <div style="display:flex; justify-content:space-between; font-size:10px; color:#4A5A6A">
                    <span>SL</span><span>TP Progress: {progress_pct:.0f}%</span><span>TP</span>
                </div>
                <div style="background:#0A0E1A; border-radius:4px; height:6px; margin-top:3px">
                    <div style="background:{bar_color}; width:{progress_pct}%;
                                height:100%; border-radius:4px; transition:width 0.3s"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)