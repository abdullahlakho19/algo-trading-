"""
strategies/strategy_library/mean_reversion_strategy.py
─────────────────────────────────────────────────────────────────────────────
Mean Reversion Strategy.
Trades extreme price deviations from equilibrium back toward the mean.
Only activates in Balance / Range-bound regimes — not in trending markets.

Setup:
  - Market in balance (Choppiness Index > 55, ADX < 20)
  - Price at 2+ standard deviations from mean (Bollinger Band extreme)
  - RSI oversold (<30) or overbought (>70)
  - Volume diminishing (trend losing momentum)
  - Trade back toward mean / POC
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class MeanReversionSignal:
    """Mean reversion strategy signal."""
    symbol:      str
    direction:   str
    entry:       float
    stop_loss:   float
    take_profit: float   # Target: back toward mean
    mean_target: float   # The mean price level
    deviation:   float   # How many std devs from mean
    rsi:         float
    in_range:    bool
    confidence:  float


class MeanReversionStrategy:
    """
    Mean reversion strategy for range-bound markets.
    Fades extreme moves back toward equilibrium.
    """

    def __init__(self):
        self.bb_period    = 20
        self.bb_std       = 2.0
        self.rsi_period   = 14
        self.chop_period  = 14

    def analyse(self, df: pd.DataFrame, symbol: str) -> MeanReversionSignal:
        if len(df) < 60:
            return self._neutral(symbol)

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        # Regime check — only trade mean reversion in ranging markets
        adx       = self._adx(df)
        chop      = self._choppiness(df)
        in_range  = adx < 22 and chop > 55

        if not in_range:
            return self._neutral(symbol)

        # Bollinger Bands
        ma    = close.rolling(self.bb_period).mean()
        std   = close.rolling(self.bb_period).std()
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        last    = float(close.iloc[-1])
        ma_val  = float(ma.iloc[-1])
        std_val = float(std.iloc[-1])
        up_val  = float(upper.iloc[-1])
        lo_val  = float(lower.iloc[-1])

        # Deviation in std devs
        deviation = (last - ma_val) / (std_val + 1e-10)

        rsi     = self._rsi(close)
        atr     = self._atr(df)

        # Volume declining (confirming exhaustion)
        vol_declining = (
            df["volume"].tail(3).mean() < df["volume"].tail(10).mean() * 0.8
        )

        confluence = 0
        direction  = "neutral"

        # ── Long Setup: price at lower band, RSI oversold ──────────────────
        if last <= lo_val and rsi < 35:
            long_signals = sum([
                deviation < -1.8,
                rsi < 30,
                vol_declining,
                last < float(low.tail(20).quantile(0.15)),
            ])
            if long_signals >= 2:
                direction  = "long"
                confluence = long_signals

        # ── Short Setup: price at upper band, RSI overbought ───────────────
        elif last >= up_val and rsi > 65:
            short_signals = sum([
                deviation > 1.8,
                rsi > 70,
                vol_declining,
                last > float(high.tail(20).quantile(0.85)),
            ])
            if short_signals >= 2:
                direction  = "short"
                confluence = short_signals

        if direction == "neutral":
            return self._neutral(symbol)

        # SL: beyond the band extreme
        # TP: back to mean (MA20)
        if direction == "long":
            sl = last - atr * 1.2
            tp = ma_val                # Target the mean
        else:
            sl = last + atr * 1.2
            tp = ma_val

        rr = abs(tp - last) / abs(sl - last)
        if rr < 1.5:
            return self._neutral(symbol)

        return MeanReversionSignal(
            symbol=symbol,
            direction=direction,
            entry=round(last, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            mean_target=round(ma_val, 6),
            deviation=round(deviation, 3),
            rsi=round(rsi, 1),
            in_range=in_range,
            confidence=round(min(1.0, confluence / 4 * 0.7), 3),
        )

    def _adx(self, df: pd.DataFrame, p: int = 14) -> float:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        pdm = h.diff().clip(lower=0)
        mdm = (-l.diff()).clip(lower=0)
        atr = tr.ewm(span=p, adjust=False).mean()
        pdi = 100*pdm.ewm(span=p, adjust=False).mean()/(atr+1e-10)
        mdi = 100*mdm.ewm(span=p, adjust=False).mean()/(atr+1e-10)
        dx  = 100*(pdi-mdi).abs()/(pdi+mdi+1e-10)
        return float(dx.ewm(span=p, adjust=False).mean().iloc[-1])

    def _choppiness(self, df: pd.DataFrame) -> float:
        n  = self.chop_period
        h  = df["high"].tail(n)
        l  = df["low"].tail(n)
        tr = pd.concat([
            df["high"]-df["low"],
            (df["high"]-df["close"].shift()).abs(),
            (df["low"] -df["close"].shift()).abs(),
        ], axis=1).max(axis=1).tail(n)
        hl = float(h.max() - l.min())
        if hl == 0: return 50.0
        return float(100 * np.log10(tr.sum() / hl) / np.log10(n))

    def _rsi(self, close: pd.Series, p: int = 14) -> float:
        d = close.diff()
        g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
        return float((100 - 100/(1 + g/(l+1e-10))).iloc[-1])

    def _atr(self, df: pd.DataFrame, p: int = 14) -> float:
        tr = pd.concat([
            df["high"]-df["low"],
            (df["high"]-df["close"].shift()).abs(),
            (df["low"] -df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=p, adjust=False).mean().iloc[-1])

    def _neutral(self, symbol: str) -> MeanReversionSignal:
        return MeanReversionSignal(
            symbol=symbol, direction="neutral", entry=0,
            stop_loss=0, take_profit=0, mean_target=0,
            deviation=0, rsi=50, in_range=False, confidence=0,
        )


# Singleton
mean_reversion_strategy = MeanReversionStrategy()