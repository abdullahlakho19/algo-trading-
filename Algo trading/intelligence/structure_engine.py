"""
intelligence/structure_engine.py
─────────────────────────────────────────────────────────────────────────────
Market Structure Engine.
Detects Break of Structure (BoS) and Change of Character (CHoCH)
in real-time — core Smart Money Concepts (SMC) analysis.

BoS  = Break of Structure   → trend continuation signal
CHoCH = Change of Character → potential trend reversal signal
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from core.logger import get_logger

log = get_logger(__name__)


class StructureSignal(str, Enum):
    BOS_BULLISH   = "bos_bullish"    # Higher high broken — continue up
    BOS_BEARISH   = "bos_bearish"    # Lower low broken — continue down
    CHOCH_BULLISH = "choch_bullish"  # Bearish structure broken — reversal up
    CHOCH_BEARISH = "choch_bearish"  # Bullish structure broken — reversal down
    NONE          = "none"


@dataclass
class SwingPoint:
    """A detected swing high or swing low."""
    index:     int
    timestamp: pd.Timestamp
    price:     float
    kind:      str            # "high" | "low"
    broken:    bool = False   # Has this level been broken?


@dataclass
class StructureResult:
    """Output of the structure engine for one timeframe."""
    symbol:       str
    timeframe:    str
    signal:       StructureSignal
    swing_highs:  list[SwingPoint] = field(default_factory=list)
    swing_lows:   list[SwingPoint] = field(default_factory=list)
    last_hh:      float = 0.0      # Last Higher High
    last_hl:      float = 0.0      # Last Higher Low
    last_lh:      float = 0.0      # Last Lower High
    last_ll:      float = 0.0      # Last Lower Low
    trend:        str = "neutral"  # "bullish" | "bearish" | "neutral"
    bos_level:    float = 0.0      # Level that was broken
    choch_level:  float = 0.0      # Level that triggered CHoCH
    confidence:   float = 0.0


class StructureEngine:
    """
    Real-time market structure analysis.
    Identifies swing points and monitors for BoS / CHoCH events.
    """

    def __init__(self, swing_lookback: int = 5):
        """
        Args:
            swing_lookback: Candles on each side to confirm a swing point
        """
        self.swing_lookback = swing_lookback

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> StructureResult:
        """
        Analyse market structure and detect BoS / CHoCH.

        Args:
            df:        OHLCV DataFrame (min 50 candles)
            symbol:    Instrument symbol
            timeframe: Timeframe string

        Returns:
            StructureResult with signal, swing points, and structure context
        """
        if len(df) < 20:
            log.warning(f"Insufficient data for structure analysis: {symbol}")
            return StructureResult(
                symbol=symbol, timeframe=timeframe, signal=StructureSignal.NONE
            )

        swing_highs = self._find_swing_highs(df)
        swing_lows  = self._find_swing_lows(df)

        result = StructureResult(
            symbol=symbol,
            timeframe=timeframe,
            signal=StructureSignal.NONE,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        )

        # Determine structure (HH/HL or LH/LL)
        self._classify_structure(result, swing_highs, swing_lows)

        # Check for BoS / CHoCH on latest candle
        self._detect_signals(result, df, swing_highs, swing_lows)

        log.debug(
            f"{symbol} | {timeframe} | Structure: {result.trend} | "
            f"Signal: {result.signal.value}"
        )

        return result

    # ── Swing Point Detection ─────────────────────────────────────────────────
    def _find_swing_highs(self, df: pd.DataFrame) -> list[SwingPoint]:
        """
        Find confirmed swing highs.
        A swing high: candle whose high is the highest of N candles on each side.
        """
        highs  = df["high"].values
        n      = self.swing_lookback
        result = []

        for i in range(n, len(highs) - n):
            window = highs[i - n: i + n + 1]
            if highs[i] == window.max():
                result.append(SwingPoint(
                    index=i,
                    timestamp=df.index[i],
                    price=float(highs[i]),
                    kind="high",
                ))

        return result[-20:]   # Keep last 20 swing highs

    def _find_swing_lows(self, df: pd.DataFrame) -> list[SwingPoint]:
        """Find confirmed swing lows."""
        lows   = df["low"].values
        n      = self.swing_lookback
        result = []

        for i in range(n, len(lows) - n):
            window = lows[i - n: i + n + 1]
            if lows[i] == window.min():
                result.append(SwingPoint(
                    index=i,
                    timestamp=df.index[i],
                    price=float(lows[i]),
                    kind="low",
                ))

        return result[-20:]

    # ── Structure Classification ──────────────────────────────────────────────
    def _classify_structure(
        self,
        result: StructureResult,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
    ) -> None:
        """
        Classify trend structure as bullish (HH+HL) or bearish (LH+LL).
        """
        if len(highs) >= 2:
            if highs[-1].price > highs[-2].price:
                result.last_hh = highs[-1].price   # Higher High
            else:
                result.last_lh = highs[-1].price   # Lower High

        if len(lows) >= 2:
            if lows[-1].price > lows[-2].price:
                result.last_hl = lows[-1].price    # Higher Low
            else:
                result.last_ll = lows[-1].price    # Lower Low

        # Classify trend
        has_hh = result.last_hh > 0
        has_hl = result.last_hl > 0
        has_lh = result.last_lh > 0
        has_ll = result.last_ll > 0

        if has_hh and has_hl:
            result.trend = "bullish"
        elif has_lh and has_ll:
            result.trend = "bearish"
        else:
            result.trend = "neutral"

    # ── BoS / CHoCH Detection ─────────────────────────────────────────────────
    def _detect_signals(
        self,
        result: StructureResult,
        df: pd.DataFrame,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
    ) -> None:
        """
        Detect Break of Structure or Change of Character on the latest candle.

        BoS   = continuation: bullish trend breaks prior HH → BoS bullish
        CHoCH = reversal: bullish trend breaks prior HL → CHoCH bearish
        """
        last_close = float(df["close"].iloc[-1])
        last_low   = float(df["low"].iloc[-1])
        last_high  = float(df["high"].iloc[-1])

        if not highs or not lows:
            return

        last_swing_high = highs[-1].price
        last_swing_low  = lows[-1].price

        # ── BULLISH BREAK OF STRUCTURE ─────────────────────────────────────
        # Price closes above the last swing high → trend continuation up
        if result.trend == "bullish" and last_close > last_swing_high:
            result.signal    = StructureSignal.BOS_BULLISH
            result.bos_level = last_swing_high
            result.confidence = self._score_confidence(df, "bullish")
            return

        # ── BEARISH BREAK OF STRUCTURE ─────────────────────────────────────
        # Price closes below the last swing low → trend continuation down
        if result.trend == "bearish" and last_close < last_swing_low:
            result.signal    = StructureSignal.BOS_BEARISH
            result.bos_level = last_swing_low
            result.confidence = self._score_confidence(df, "bearish")
            return

        # ── BULLISH CHANGE OF CHARACTER ────────────────────────────────────
        # In a bearish trend, price breaks above last Lower High → potential reversal
        if result.trend == "bearish" and last_close > last_swing_high:
            result.signal      = StructureSignal.CHOCH_BULLISH
            result.choch_level = last_swing_high
            result.confidence  = self._score_confidence(df, "bullish") * 0.85
            return

        # ── BEARISH CHANGE OF CHARACTER ────────────────────────────────────
        # In a bullish trend, price breaks below last Higher Low → potential reversal
        if result.trend == "bullish" and last_close < last_swing_low:
            result.signal      = StructureSignal.CHOCH_BEARISH
            result.choch_level = last_swing_low
            result.confidence  = self._score_confidence(df, "bearish") * 0.85
            return

    # ── Confidence Scoring ────────────────────────────────────────────────────
    def _score_confidence(self, df: pd.DataFrame, direction: str) -> float:
        """Score the confidence of a structure signal (0.0-1.0)."""
        close = df["close"]
        vol   = df["volume"]

        # Is volume above average on the break candle?
        vol_ratio = vol.iloc[-1] / vol.tail(20).mean()

        # Is momentum in the right direction?
        momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        if direction == "bullish":
            momentum_ok = momentum > 0
        else:
            momentum_ok = momentum < 0

        # Score components
        vol_score      = min(1.0, vol_ratio / 2.0)     # Max out at 2x avg vol
        momentum_score = 0.3 if momentum_ok else 0.0
        base_score     = 0.5

        return round(min(1.0, base_score + vol_score * 0.3 + momentum_score), 3)

    # ── Key Levels ────────────────────────────────────────────────────────────
    def get_key_levels(self, result: StructureResult) -> list[dict]:
        """Return all swing levels as structured key level objects."""
        levels = []
        for sh in result.swing_highs[-5:]:
            levels.append({
                "level": sh.price, "type": "SWING_HIGH",
                "strength": "strong", "timestamp": sh.timestamp,
            })
        for sl in result.swing_lows[-5:]:
            levels.append({
                "level": sl.price, "type": "SWING_LOW",
                "strength": "strong", "timestamp": sl.timestamp,
            })
        return sorted(levels, key=lambda x: x["level"])


# Singleton
structure_engine = StructureEngine()