"""
intelligence/momentum.py
─────────────────────────────────────────────────────────────────────────────
Momentum & Gradient Convergence Engine.
Measures trend strength and momentum convergence across multiple timeframes.
Confirms that momentum is aligned — not just price direction.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class MomentumResult:
    """Multi-indicator momentum analysis."""
    symbol:       str
    timeframe:    str
    rsi:          float     # 0-100
    macd:         float     # MACD line
    macd_signal:  float     # Signal line
    macd_hist:    float     # Histogram (positive = bullish)
    adx:          float     # Trend strength 0-100
    stoch_k:      float     # Stochastic %K
    stoch_d:      float     # Stochastic %D
    roc:          float     # Rate of change
    convergence:  float     # -1 to +1 (all indicators pointing same way?)
    signal:       str       # "bullish" | "bearish" | "neutral"
    strength:     str       # "strong" | "moderate" | "weak"


class MomentumEngine:
    """
    Computes multiple momentum indicators and their convergence.
    Convergence = multiple indicators pointing in the same direction.
    """

    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> MomentumResult:
        """Full momentum analysis."""
        if len(df) < 50:
            return self._neutral(symbol, timeframe)

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        rsi             = self._rsi(close)
        macd, sig, hist = self._macd(close)
        adx             = self._adx(df)
        stoch_k, stoch_d = self._stochastic(high, low, close)
        roc             = self._roc(close)

        # Convergence score
        bullish_votes = sum([
            rsi > 55,
            hist > 0,
            macd > sig,
            stoch_k > stoch_d and stoch_k > 50,
            roc > 0,
        ])
        bearish_votes = sum([
            rsi < 45,
            hist < 0,
            macd < sig,
            stoch_k < stoch_d and stoch_k < 50,
            roc < 0,
        ])
        total_votes = 5
        convergence = (bullish_votes - bearish_votes) / total_votes

        if convergence > 0.4:
            signal = "bullish"
        elif convergence < -0.4:
            signal = "bearish"
        else:
            signal = "neutral"

        if adx > 30 and abs(convergence) > 0.6:
            strength = "strong"
        elif adx > 20 and abs(convergence) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        return MomentumResult(
            symbol=symbol, timeframe=timeframe,
            rsi=round(rsi, 2), macd=round(macd, 6),
            macd_signal=round(sig, 6), macd_hist=round(hist, 6),
            adx=round(adx, 2), stoch_k=round(stoch_k, 2),
            stoch_d=round(stoch_d, 2), roc=round(roc, 4),
            convergence=round(convergence, 3),
            signal=signal, strength=strength,
        )

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs    = gain / (loss + 1e-10)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    def _macd(
        self, close: pd.Series, fast=12, slow=26, signal=9
    ) -> tuple[float, float, float]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd     = ema_fast - ema_slow
        sig      = macd.ewm(span=signal, adjust=False).mean()
        hist     = macd - sig
        return float(macd.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1])

    def _adx(self, df: pd.DataFrame, period: int = 14) -> float:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        atr_s    = tr.ewm(span=period, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr_s + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr_s + 1e-10)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx      = dx.ewm(span=period, adjust=False).mean()
        return float(adx.iloc[-1])

    def _stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 14, d_period: int = 3
    ) -> tuple[float, float]:
        lowest  = low.rolling(k_period).min()
        highest = high.rolling(k_period).max()
        k = 100 * (close - lowest) / (highest - lowest + 1e-10)
        d = k.rolling(d_period).mean()
        return float(k.iloc[-1]), float(d.iloc[-1])

    def _roc(self, close: pd.Series, period: int = 10) -> float:
        return float(close.pct_change(period).iloc[-1])

    def _neutral(self, symbol: str, timeframe: str) -> MomentumResult:
        return MomentumResult(
            symbol=symbol, timeframe=timeframe,
            rsi=50, macd=0, macd_signal=0, macd_hist=0,
            adx=0, stoch_k=50, stoch_d=50, roc=0,
            convergence=0, signal="neutral", strength="weak",
        )


# Singleton
momentum_engine = MomentumEngine()