"""
strategies/strategy_library/momentum_strategy.py
─────────────────────────────────────────────────────────────────────────────
Momentum / Trend Following Strategy.
Enters in the direction of established momentum when multiple
trend indicators converge. Rides the trend until momentum dies.

Setup:
  - ADX > 25 (strong trend confirmed)
  - EMA stack aligned (20 > 50 > 200 for long)
  - RSI in momentum zone (50-70 for long, 30-50 for short)
  - MACD histogram expanding
  - Volume confirming
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class MomentumSignal:
    """Momentum strategy signal."""
    symbol:     str
    direction:  str
    entry:      float
    stop_loss:  float
    take_profit: float
    adx:        float
    rsi:        float
    ema_aligned: bool
    confluence: int
    confidence: float


class MomentumStrategy:
    """
    Pure trend-following momentum strategy.
    Only trades in established trending markets (ADX > 25).
    """

    def analyse(self, df: pd.DataFrame, symbol: str) -> MomentumSignal:
        if len(df) < 60:
            return self._neutral(symbol)

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        # Indicators
        ema20  = close.ewm(span=20, adjust=False).mean()
        ema50  = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        adx    = self._adx(df)
        rsi    = self._rsi(close)
        _, _, macd_hist = self._macd(close)
        atr    = self._atr(df)

        last       = float(close.iloc[-1])
        e20        = float(ema20.iloc[-1])
        e50        = float(ema50.iloc[-1])
        e200       = float(ema200.iloc[-1])
        avg_vol    = df["volume"].tail(20).mean()
        last_vol   = float(df["volume"].iloc[-1])
        vol_ok     = last_vol > avg_vol

        # Pullback detection — price slightly below EMA20 in uptrend (entry)
        pullback_long  = (e20 > e50 > e200 and
                          low.iloc[-1] <= e20 * 1.005 and
                          close.iloc[-1] > e20 * 0.998)
        pullback_short = (e20 < e50 < e200 and
                          high.iloc[-1] >= e20 * 0.995 and
                          close.iloc[-1] < e20 * 1.002)

        confluence = 0
        direction  = "neutral"

        # ── Long Setup ────────────────────────────────────────────────────
        if adx > 25 and e20 > e50 > e200:
            long_signals = sum([
                50 < rsi < 75,
                macd_hist > 0,
                vol_ok,
                pullback_long,
                last > e20,
            ])
            if long_signals >= 3:
                direction  = "long"
                confluence = long_signals

        # ── Short Setup ───────────────────────────────────────────────────
        elif adx > 25 and e20 < e50 < e200:
            short_signals = sum([
                25 < rsi < 50,
                macd_hist < 0,
                vol_ok,
                pullback_short,
                last < e20,
            ])
            if short_signals >= 3:
                direction  = "short"
                confluence = short_signals

        if direction == "neutral":
            return self._neutral(symbol)

        # SL below/above EMA50 for trend trade
        if direction == "long":
            sl = max(float(ema50.iloc[-1]) - atr * 0.5, last - atr * 2.0)
            tp = last + (last - sl) * 3.0
        else:
            sl = min(float(ema50.iloc[-1]) + atr * 0.5, last + atr * 2.0)
            tp = last - (sl - last) * 3.0

        ema_aligned = (e20 > e50 > e200) if direction == "long" else (e20 < e50 < e200)

        return MomentumSignal(
            symbol=symbol,
            direction=direction,
            entry=round(last, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            adx=round(adx, 1),
            rsi=round(rsi, 1),
            ema_aligned=ema_aligned,
            confluence=confluence,
            confidence=round(min(1.0, confluence / 5), 3),
        )

    def _adx(self, df: pd.DataFrame, p: int = 14) -> float:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        atr_s    = tr.ewm(span=p, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(span=p, adjust=False).mean() / (atr_s + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=p, adjust=False).mean() / (atr_s + 1e-10)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        return float(dx.ewm(span=p, adjust=False).mean().iloc[-1])

    def _rsi(self, close: pd.Series, p: int = 14) -> float:
        d = close.diff()
        g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
        return float((100 - 100 / (1 + g / (l + 1e-10))).iloc[-1])

    def _macd(self, close: pd.Series) -> tuple:
        f  = close.ewm(span=12, adjust=False).mean()
        s  = close.ewm(span=26, adjust=False).mean()
        m  = f - s
        sg = m.ewm(span=9,  adjust=False).mean()
        return float(m.iloc[-1]), float(sg.iloc[-1]), float((m - sg).iloc[-1])

    def _atr(self, df: pd.DataFrame, p: int = 14) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=p, adjust=False).mean().iloc[-1])

    def _neutral(self, symbol: str) -> MomentumSignal:
        return MomentumSignal(
            symbol=symbol, direction="neutral", entry=0,
            stop_loss=0, take_profit=0, adx=0,
            rsi=50, ema_aligned=False, confluence=0, confidence=0,
        )


# Singleton
momentum_strategy = MomentumStrategy()