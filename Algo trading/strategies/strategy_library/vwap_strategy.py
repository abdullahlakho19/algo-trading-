"""
strategies/strategy_library/vwap_strategy.py
─────────────────────────────────────────────────────────────────────────────
VWAP Strategy.
Volume-Weighted Average Price is one of the most widely-used
institutional benchmarks. Institutions buy below VWAP and sell above.

Setup:
  - Price rejected at VWAP with volume confirmation → trade in rejection direction
  - Price reclaims VWAP on high volume → trade in reclaim direction
  - VWAP + Standard Deviation bands as dynamic S/R
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class VWAPSignal:
    """VWAP strategy signal output."""
    symbol:    str
    direction: str     # "long" | "short" | "neutral"
    entry:     float
    stop_loss: float
    take_profit: float
    vwap:      float
    vwap_std1_upper: float
    vwap_std1_lower: float
    vwap_std2_upper: float
    vwap_std2_lower: float
    setup_type: str    # "vwap_rejection" | "vwap_reclaim" | "band_mean_revert"
    confidence: float


class VWAPStrategy:
    """
    VWAP-based institutional strategy.
    Uses VWAP as primary reference with standard deviation bands.
    """

    def analyse(self, df: pd.DataFrame, symbol: str) -> VWAPSignal:
        """Full VWAP analysis and signal generation."""
        if len(df) < 20:
            return self._neutral(symbol)

        vwap, std = self._compute_vwap(df)
        current   = float(df["close"].iloc[-1])
        atr       = self._atr(df)

        # VWAP bands
        u1 = vwap + std
        l1 = vwap - std
        u2 = vwap + 2 * std
        l2 = vwap - 2 * std

        # Volume confirmation
        avg_vol    = df["volume"].tail(20).mean()
        last_vol   = float(df["volume"].iloc[-1])
        vol_conf   = last_vol > avg_vol * 1.3

        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])

        setup_type = "none"
        direction  = "neutral"
        confidence = 0.0

        # ── Setup 1: VWAP Rejection ────────────────────────────────────────
        # Price touches VWAP and gets rejected
        if abs(prev_close - vwap) / vwap < 0.002:    # Was at VWAP
            if last_close < vwap and vol_conf:
                direction  = "short"
                setup_type = "vwap_rejection"
                confidence = 0.65
            elif last_close > vwap and vol_conf:
                direction  = "long"
                setup_type = "vwap_rejection"
                confidence = 0.65

        # ── Setup 2: VWAP Reclaim ──────────────────────────────────────────
        # Price was below VWAP, now reclaims it on volume
        elif (prev_close < vwap and last_close > vwap and vol_conf):
            direction  = "long"
            setup_type = "vwap_reclaim"
            confidence = 0.72

        elif (prev_close > vwap and last_close < vwap and vol_conf):
            direction  = "short"
            setup_type = "vwap_reclaim"
            confidence = 0.72

        # ── Setup 3: Band Mean Reversion ───────────────────────────────────
        # Price at 2-std band — mean revert back to VWAP
        elif last_close >= u2:
            direction  = "short"
            setup_type = "band_mean_revert"
            confidence = 0.60
        elif last_close <= l2:
            direction  = "long"
            setup_type = "band_mean_revert"
            confidence = 0.60

        if direction == "neutral":
            return self._neutral(symbol, vwap, u1, l1, u2, l2)

        # SL / TP
        if direction == "long":
            sl = last_close - atr * 1.5
            tp = vwap + (vwap - sl) * 1.5   # Target back to VWAP and beyond
        else:
            sl = last_close + atr * 1.5
            tp = vwap - (sl - vwap) * 1.5

        return VWAPSignal(
            symbol=symbol, direction=direction,
            entry=round(last_close, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            vwap=round(vwap, 6),
            vwap_std1_upper=round(u1, 6),
            vwap_std1_lower=round(l1, 6),
            vwap_std2_upper=round(u2, 6),
            vwap_std2_lower=round(l2, 6),
            setup_type=setup_type,
            confidence=confidence,
        )

    def _compute_vwap(self, df: pd.DataFrame) -> tuple[float, float]:
        """Compute VWAP and standard deviation from intraday data."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol     = df["volume"]
        cum_tpv = (typical * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap    = (cum_tpv / cum_vol).iloc[-1]
        std     = float((typical - float(vwap)).std())
        return float(vwap), std

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

    def _neutral(self, symbol, vwap=0, u1=0, l1=0, u2=0, l2=0) -> VWAPSignal:
        return VWAPSignal(
            symbol=symbol, direction="neutral", entry=0,
            stop_loss=0, take_profit=0, vwap=vwap,
            vwap_std1_upper=u1, vwap_std1_lower=l1,
            vwap_std2_upper=u2, vwap_std2_lower=l2,
            setup_type="none", confidence=0.0,
        )


# Singleton
vwap_strategy = VWAPStrategy()