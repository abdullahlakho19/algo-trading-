"""
risk/sl_tp_engine.py
─────────────────────────────────────────────────────────────────────────────
Stop-Loss & Take-Profit Engine.
Mathematically places SL/TP levels using multiple methods.
Never guesses — always uses price structure + ATR.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class SLTPResult:
    """SL/TP calculation output."""
    entry:       float
    stop_loss:   float
    take_profit: float
    risk_reward: float
    sl_distance: float
    tp_distance: float
    method:      str
    notes:       str = ""


class SLTPEngine:
    """
    Calculates stop-loss and take-profit levels using:
    1. ATR-based (default — most reliable)
    2. Structure-based (swing high/low)
    3. Volatility envelope
    4. Fixed R-multiple
    """

    def __init__(self):
        self.min_rr = 2.0    # Minimum R/R ratio required

    # ── ATR Method (Default) ──────────────────────────────────────────────────
    def atr_based(
        self,
        entry:     float,
        direction: str,
        atr:       float,
        sl_mult:   float = 1.5,
        tp_mult:   float = 3.0,
    ) -> SLTPResult:
        """
        ATR-based SL/TP.
        SL: entry ± (ATR × sl_mult)
        TP: entry ± (ATR × tp_mult)
        Default gives 1:2 R/R minimum.
        """
        if direction == "long":
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        else:
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        rr      = tp_dist / max(sl_dist, 1e-10)

        return SLTPResult(
            entry=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            risk_reward=round(rr, 2),
            sl_distance=round(sl_dist, 6),
            tp_distance=round(tp_dist, 6),
            method="atr",
            notes=f"ATR={atr:.5f} SL×{sl_mult} TP×{tp_mult}",
        )

    # ── Structure-Based ───────────────────────────────────────────────────────
    def structure_based(
        self,
        entry:     float,
        direction: str,
        df:        pd.DataFrame,
        lookback:  int = 20,
        buffer:    float = 0.001,    # 0.1% buffer beyond structure
    ) -> SLTPResult:
        """
        Structure-based SL/TP using recent swing highs/lows.
        """
        recent = df.tail(lookback)
        atr    = self._atr(df)

        if direction == "long":
            # SL below most recent swing low
            swing_low = float(recent["low"].min())
            sl        = swing_low * (1 - buffer)
            # TP at recent swing high or 2x SL distance
            swing_high = float(recent["high"].max())
            tp         = max(entry + (entry - sl) * 2, swing_high * (1 + buffer))
        else:
            swing_high = float(recent["high"].max())
            sl         = swing_high * (1 + buffer)
            swing_low  = float(recent["low"].min())
            tp         = min(entry - (sl - entry) * 2, swing_low * (1 - buffer))

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)

        # Sanity check: if SL too wide, fall back to ATR
        if sl_dist > atr * 3:
            return self.atr_based(entry, direction, atr)

        rr = tp_dist / max(sl_dist, 1e-10)
        if rr < self.min_rr:
            # Extend TP to meet minimum R/R
            if direction == "long":
                tp = entry + sl_dist * self.min_rr
            else:
                tp = entry - sl_dist * self.min_rr
            tp_dist = abs(tp - entry)
            rr      = tp_dist / sl_dist

        return SLTPResult(
            entry=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            risk_reward=round(rr, 2),
            sl_distance=round(sl_dist, 6),
            tp_distance=round(tp_dist, 6),
            method="structure",
            notes=f"Swing L: {swing_low:.5f} | Swing H: {swing_high:.5f}",
        )

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    def trailing_stop_level(
        self,
        entry:          float,
        current_price:  float,
        direction:      str,
        initial_sl:     float,
        atr:            float,
        trail_mult:     float = 2.0,
    ) -> float:
        """
        Compute new trailing stop level as price moves in favour.
        Never moves the stop against the trade.
        """
        trail_dist = atr * trail_mult

        if direction == "long":
            new_sl = current_price - trail_dist
            # Only move SL up, never down
            return max(new_sl, initial_sl)
        else:
            new_sl = current_price + trail_dist
            # Only move SL down, never up
            return min(new_sl, initial_sl)

    # ── Partial TP ─────────────────────────────────────────────────────────────
    def partial_tp_levels(
        self,
        entry:     float,
        direction: str,
        atr:       float,
        portions:  list[tuple[float, float]] = None,  # [(rr, % to close)]
    ) -> list[dict]:
        """
        Generate partial take-profit levels.
        Default: close 50% at 1:1 R/R, remaining at 1:3 R/R.
        """
        portions = portions or [(1.0, 0.50), (3.0, 0.50)]
        sl_dist  = atr * 1.5
        levels   = []

        for rr, pct in portions:
            if direction == "long":
                tp_price = entry + sl_dist * rr
            else:
                tp_price = entry - sl_dist * rr
            levels.append({
                "price": round(tp_price, 6),
                "rr":    rr,
                "pct":   pct,
            })
        return levels

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

    def calculate(
        self,
        entry:     float,
        direction: str,
        atr:       float,
        df:        pd.DataFrame = None,
    ) -> SLTPResult:
        """
        Smart SL/TP: uses structure when df provided, ATR as fallback.
        Always ensures minimum R/R is met.
        """
        if df is not None and len(df) >= 20:
            result = self.structure_based(entry, direction, df)
            if result.risk_reward >= self.min_rr:
                return result

        return self.atr_based(entry, direction, atr)


# Singleton
sl_tp_engine = SLTPEngine()