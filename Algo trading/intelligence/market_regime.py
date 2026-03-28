"""
intelligence/market_regime.py
─────────────────────────────────────────────────────────────────────────────
Market Regime Engine.
Classifies the current market condition OBJECTIVELY using quantitative
measures — not visual interpretation.

Regimes:
    ACCUMULATION  — Institutions quietly buying at lower prices
    DISTRIBUTION  — Institutions quietly selling at higher prices
    BALANCE       — Market in equilibrium, no clear directional bias
    TRENDING_UP   — Strong sustained upward move
    TRENDING_DOWN — Strong sustained downward move
    VOLATILE      — High volatility, erratic price action (avoid trading)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from core.logger import get_logger

log = get_logger(__name__)


class Regime(str, Enum):
    ACCUMULATION   = "accumulation"
    DISTRIBUTION   = "distribution"
    BALANCE        = "balance"
    TRENDING_UP    = "trending_up"
    TRENDING_DOWN  = "trending_down"
    VOLATILE       = "volatile"
    UNKNOWN        = "unknown"


@dataclass
class RegimeResult:
    """Output of the market regime classifier."""
    symbol:       str
    timeframe:    str
    regime:       Regime
    confidence:   float      # 0.0 – 1.0
    trend_strength: float    # ADX-equivalent
    volatility:   float      # Normalised ATR
    range_bound:  bool       # True if market is ranging
    bias:         str        # "bullish" | "bearish" | "neutral"
    notes:        str = ""

    @property
    def is_tradeable(self) -> bool:
        """Only trade in trending or accumulation/distribution regimes."""
        return self.regime in (
            Regime.TRENDING_UP, Regime.TRENDING_DOWN,
            Regime.ACCUMULATION, Regime.DISTRIBUTION,
        ) and self.confidence >= 0.6

    @property
    def avoid(self) -> bool:
        """Avoid trading in volatile or unknown regimes."""
        return self.regime in (Regime.VOLATILE, Regime.UNKNOWN)


class MarketRegimeEngine:
    """
    Quantitative market regime classifier.

    Uses multiple indicators to classify regime without visual interpretation:
    - ADX (trend strength)
    - ATR (volatility)
    - Bollinger Band Width (expansion/contraction)
    - Price vs EMA alignment
    - Volume trend (is volume confirming price?)
    - Choppiness Index (trend vs range)
    """

    def __init__(self):
        self.adx_period    = 14
        self.atr_period    = 14
        self.bb_period     = 20
        self.bb_std        = 2.0
        self.chop_period   = 14

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> RegimeResult:
        """
        Classify current market regime from OHLCV data.

        Args:
            df:        OHLCV DataFrame (min 50 candles)
            symbol:    Instrument symbol
            timeframe: Timeframe string

        Returns:
            RegimeResult with regime classification and metrics
        """
        if len(df) < 50:
            log.warning(f"Insufficient data for regime analysis: {symbol}")
            return self._unknown_result(symbol, timeframe)

        adx, plus_di, minus_di = self._compute_adx(df)
        atr         = self._compute_atr(df)
        atr_norm    = self._normalise_atr(atr, df)
        bb_width    = self._compute_bb_width(df)
        choppiness  = self._compute_choppiness(df, atr)
        ema_align   = self._compute_ema_alignment(df)
        vol_trend   = self._compute_volume_trend(df)

        # ── Classification Logic ───────────────────────────────────────────
        regime, confidence, notes = self._classify(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            atr_norm=atr_norm,
            bb_width=bb_width,
            choppiness=choppiness,
            ema_align=ema_align,
            vol_trend=vol_trend,
            df=df,
        )

        bias = self._determine_bias(plus_di, minus_di, ema_align)

        result = RegimeResult(
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            confidence=round(confidence, 3),
            trend_strength=round(float(adx), 2),
            volatility=round(float(atr_norm), 4),
            range_bound=(choppiness > 61.8),
            bias=bias,
            notes=notes,
        )

        log.info(
            f"{symbol} | Regime: {regime.value} | "
            f"Confidence: {confidence:.0%} | ADX: {adx:.1f} | "
            f"Bias: {bias}"
        )

        return result

    # ── Indicator Computations ────────────────────────────────────────────────
    def _compute_adx(
        self, df: pd.DataFrame, period: int = 14
    ) -> tuple[float, float, float]:
        """
        Compute ADX, +DI, -DI.
        ADX > 25 = trending, ADX < 20 = ranging
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr_s      = tr.ewm(span=period, adjust=False).mean()
        plus_di    = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s
        minus_di   = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s
        dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx        = dx.ewm(span=period, adjust=False).mean()

        return float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])

    def _compute_atr(self, df: pd.DataFrame) -> float:
        """Average True Range."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=self.atr_period, adjust=False).mean().iloc[-1])

    def _normalise_atr(self, atr: float, df: pd.DataFrame) -> float:
        """ATR as % of current price — normalised volatility."""
        price = float(df["close"].iloc[-1])
        return atr / price if price > 0 else 0.0

    def _compute_bb_width(self, df: pd.DataFrame) -> float:
        """Bollinger Band Width — measures volatility expansion/contraction."""
        close = df["close"]
        ma    = close.rolling(self.bb_period).mean()
        std   = close.rolling(self.bb_period).std()
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std
        width = (upper - lower) / ma
        return float(width.iloc[-1])

    def _compute_choppiness(self, df: pd.DataFrame, atr: float) -> float:
        """
        Choppiness Index — measures trendiness vs choppiness.
        > 61.8 = choppy/ranging, < 38.2 = trending
        """
        n     = self.chop_period
        high  = df["high"].tail(n)
        low   = df["low"].tail(n)
        close = df["close"].tail(n)

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1).tail(n)

        atr_sum = tr.sum()
        hl_range = high.max() - low.min()

        if hl_range == 0 or atr_sum == 0:
            return 50.0

        chop = 100 * np.log10(atr_sum / hl_range) / np.log10(n)
        return float(chop)

    def _compute_ema_alignment(self, df: pd.DataFrame) -> float:
        """
        EMA alignment score.
        +1.0 = fully bullish (price > EMA20 > EMA50 > EMA200)
        -1.0 = fully bearish
        """
        close   = df["close"]
        ema20   = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50   = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200  = close.ewm(span=200, adjust=False).mean().iloc[-1]
        last    = float(close.iloc[-1])

        score = sum([
            last > ema20,
            ema20 > ema50,
            ema50 > ema200,
        ]) - sum([
            last < ema20,
            ema20 < ema50,
            ema50 < ema200,
        ])

        return score / 3.0

    def _compute_volume_trend(self, df: pd.DataFrame) -> float:
        """
        Volume trend — is volume rising or falling?
        +1 = volume expanding (confirms trend), -1 = volume shrinking
        """
        vol    = df["volume"].tail(20)
        ma_vol = vol.mean()
        recent = vol.tail(5).mean()
        return float(np.clip((recent - ma_vol) / (ma_vol + 1e-10), -1, 1))

    # ── Classification ────────────────────────────────────────────────────────
    def _classify(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        atr_norm: float,
        bb_width: float,
        choppiness: float,
        ema_align: float,
        vol_trend: float,
        df: pd.DataFrame,
    ) -> tuple[Regime, float, str]:
        """
        Combine all indicators into a regime classification with confidence.
        """
        # ── HIGH VOLATILITY — avoid ────────────────────────────────────────
        if atr_norm > 0.05:    # Price moving > 5% per candle on average
            return Regime.VOLATILE, 0.8, f"High ATR: {atr_norm:.2%}"

        # ── TRENDING UP ────────────────────────────────────────────────────
        if adx > 25 and plus_di > minus_di and ema_align > 0.3:
            conf = min(1.0, (adx / 50) * 0.6 + (ema_align + 1) / 2 * 0.4)
            return Regime.TRENDING_UP, conf, f"ADX={adx:.1f} +DI>{minus_di:.1f}"

        # ── TRENDING DOWN ──────────────────────────────────────────────────
        if adx > 25 and minus_di > plus_di and ema_align < -0.3:
            conf = min(1.0, (adx / 50) * 0.6 + (-ema_align + 1) / 2 * 0.4)
            return Regime.TRENDING_DOWN, conf, f"ADX={adx:.1f} -DI>{plus_di:.1f}"

        # ── ACCUMULATION — low ADX, price stabilising at lows ──────────────
        close  = df["close"]
        recent_low   = close.tail(20).min()
        historical_lo = close.quantile(0.15)
        if (adx < 25 and choppiness > 50 and ema_align < 0.2
                and close.iloc[-1] <= historical_lo * 1.05):
            return Regime.ACCUMULATION, 0.65, "Low ADX, price at historical lows"

        # ── DISTRIBUTION — low ADX, price stabilising at highs ─────────────
        historical_hi = close.quantile(0.85)
        if (adx < 25 and choppiness > 50 and ema_align > -0.2
                and close.iloc[-1] >= historical_hi * 0.95):
            return Regime.DISTRIBUTION, 0.65, "Low ADX, price at historical highs"

        # ── BALANCE / RANGING ──────────────────────────────────────────────
        if choppiness > 55 or adx < 20:
            return Regime.BALANCE, 0.70, f"Chop={choppiness:.1f} ADX={adx:.1f}"

        return Regime.UNKNOWN, 0.40, "Mixed signals"

    def _determine_bias(
        self, plus_di: float, minus_di: float, ema_align: float
    ) -> str:
        score = (plus_di - minus_di) / (plus_di + minus_di + 1e-10) + ema_align
        if score > 0.2:
            return "bullish"
        elif score < -0.2:
            return "bearish"
        return "neutral"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _unknown_result(self, symbol: str, timeframe: str) -> RegimeResult:
        return RegimeResult(
            symbol=symbol, timeframe=timeframe,
            regime=Regime.UNKNOWN, confidence=0.0,
            trend_strength=0.0, volatility=0.0,
            range_bound=False, bias="neutral",
        )


# Singleton
market_regime_engine = MarketRegimeEngine()