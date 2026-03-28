"""
macro/factor_model.py
─────────────────────────────────────────────────────────────────────────────
Factor Model — BlackRock Aladdin philosophy.
Computes factor exposures for each instrument:
  - Momentum factor
  - Volatility factor
  - Liquidity factor
  - Trend factor

Used for risk decomposition and position sizing adjustments.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class FactorExposure:
    """Factor exposures for a single instrument."""
    symbol:     str
    momentum:   float    # -1 to +1 (positive = trending up)
    volatility: float    # 0 to 1 (higher = more volatile)
    liquidity:  float    # 0 to 1 (higher = more liquid)
    trend:      float    # -1 to +1 (positive = strong uptrend)
    composite:  float    # Weighted composite score
    size_adj:   float    # Position size adjustment (0.5 to 1.5)


class FactorModel:
    """
    Computes factor-based risk exposures per instrument.
    Adjusts position sizes based on factor environment.
    """

    WEIGHTS = {
        "momentum":   0.35,
        "volatility": 0.25,
        "liquidity":  0.20,
        "trend":      0.20,
    }

    def compute(self, df: pd.DataFrame, symbol: str) -> FactorExposure:
        """Compute all factor exposures from OHLCV data."""
        if len(df) < 50:
            return self._default(symbol)

        momentum   = self._momentum_factor(df)
        volatility = self._volatility_factor(df)
        liquidity  = self._liquidity_factor(df)
        trend      = self._trend_factor(df)

        # Composite: high momentum + low vol + high liq + strong trend = best conditions
        composite = (
            momentum   * self.WEIGHTS["momentum"]   +
            (1 - volatility) * self.WEIGHTS["volatility"] +
            liquidity  * self.WEIGHTS["liquidity"]  +
            trend      * self.WEIGHTS["trend"]
        )

        # Size adjustment: 0.5 (worst) to 1.5 (best)
        size_adj = 0.5 + composite

        return FactorExposure(
            symbol=symbol,
            momentum=round(momentum, 4),
            volatility=round(volatility, 4),
            liquidity=round(liquidity, 4),
            trend=round(trend, 4),
            composite=round(composite, 4),
            size_adj=round(max(0.25, min(1.5, size_adj)), 3),
        )

    def _momentum_factor(self, df: pd.DataFrame) -> float:
        """12-1 month momentum (normalised to -1, +1)."""
        close = df["close"]
        if len(close) < 20:
            return 0.0
        ret = float(close.pct_change(20).iloc[-1])
        return float(np.clip(ret * 10, -1, 1))

    def _volatility_factor(self, df: pd.DataFrame) -> float:
        """Realised vol normalised to 0-1 (higher = more volatile = worse)."""
        returns = df["close"].pct_change().dropna()
        rv      = float(returns.tail(20).std() * np.sqrt(252))
        return float(np.clip(rv / 0.60, 0, 1))   # 60% annualised vol = max

    def _liquidity_factor(self, df: pd.DataFrame) -> float:
        """Volume consistency as liquidity proxy (0-1, higher = more liquid)."""
        vol = df["volume"].tail(20)
        if vol.mean() == 0:
            return 0.5
        cv = vol.std() / vol.mean()    # Coefficient of variation
        return float(np.clip(1 - cv / 2, 0, 1))

    def _trend_factor(self, df: pd.DataFrame) -> float:
        """Trend strength from EMA alignment (-1 to +1)."""
        close  = df["close"]
        ema20  = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50  = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        last   = float(close.iloc[-1])

        bull = sum([last > ema20, ema20 > ema50, ema50 > ema200])
        bear = sum([last < ema20, ema20 < ema50, ema50 < ema200])
        return (bull - bear) / 3.0

    def _default(self, symbol: str) -> FactorExposure:
        return FactorExposure(
            symbol=symbol, momentum=0.0, volatility=0.5,
            liquidity=0.5, trend=0.0, composite=0.0, size_adj=0.8,
        )


# Singleton
factor_model = FactorModel()