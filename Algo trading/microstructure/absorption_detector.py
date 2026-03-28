"""
microstructure/absorption_detector.py
─────────────────────────────────────────────────────────────────────────────
Volume Absorption Detector.
Detects when institutions are absorbing supply or demand at key levels —
a core signal that smart money is entering or defending a position.

Absorption = high volume + small price move = someone is taking the other side.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class AbsorptionResult:
    """Output of absorption detection."""
    symbol:          str
    timeframe:       str
    detected:        bool
    absorption_type: str      # "buying_absorption" | "selling_absorption" | "none"
    strength:        float    # 0.0 – 1.0
    price_level:     float    # Price where absorption occurred
    volume_ratio:    float    # Volume relative to average
    signal:          str      # "bullish" | "bearish" | "neutral"


class AbsorptionDetector:
    """
    Detects volume absorption patterns.

    Buying Absorption:
        - Large volume candle at support
        - Small body (close near open)
        - Price doesn't fall despite selling pressure
        → Institutions absorbing sell orders (bullish)

    Selling Absorption:
        - Large volume candle at resistance
        - Small body (close near open)
        - Price doesn't rise despite buying pressure
        → Institutions absorbing buy orders (bearish)
    """

    def __init__(self):
        self.vol_threshold  = 2.0    # Volume must be 2x average
        self.body_threshold = 0.3    # Body < 30% of range = absorption candle
        self.lookback       = 20

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> AbsorptionResult:
        """Detect absorption on the latest candles."""
        if len(df) < self.lookback + 5:
            return self._no_result(symbol, timeframe)

        last    = df.iloc[-1]
        recent  = df.tail(self.lookback)

        # Volume ratio vs average
        avg_vol     = recent["volume"].mean()
        vol_ratio   = float(last["volume"]) / (avg_vol + 1e-10)

        # Body ratio (small body = possible absorption)
        hl_range    = float(last["high"] - last["low"])
        body        = abs(float(last["close"] - last["open"]))
        body_ratio  = body / (hl_range + 1e-10)

        # Close position in range (0 = at low, 1 = at high)
        close_loc   = (float(last["close"]) - float(last["low"])) / (hl_range + 1e-10)

        # Is price at a key area?
        at_support    = float(last["low"]) <= float(recent["low"].quantile(0.20))
        at_resistance = float(last["high"]) >= float(recent["high"].quantile(0.80))

        # Absorption conditions
        high_vol     = vol_ratio >= self.vol_threshold
        small_body   = body_ratio <= self.body_threshold

        buying_absorption = (
            high_vol and small_body
            and at_support
            and close_loc > 0.4    # Closed mid-to-high despite big down wicks
        )

        selling_absorption = (
            high_vol and small_body
            and at_resistance
            and close_loc < 0.6    # Closed mid-to-low despite big up wicks
        )

        if buying_absorption:
            strength = min(1.0, (vol_ratio / self.vol_threshold) * (1 - body_ratio))
            result   = AbsorptionResult(
                symbol=symbol, timeframe=timeframe,
                detected=True,
                absorption_type="buying_absorption",
                strength=round(strength, 3),
                price_level=float(last["low"]),
                volume_ratio=round(vol_ratio, 2),
                signal="bullish",
            )
        elif selling_absorption:
            strength = min(1.0, (vol_ratio / self.vol_threshold) * (1 - body_ratio))
            result   = AbsorptionResult(
                symbol=symbol, timeframe=timeframe,
                detected=True,
                absorption_type="selling_absorption",
                strength=round(strength, 3),
                price_level=float(last["high"]),
                volume_ratio=round(vol_ratio, 2),
                signal="bearish",
            )
        else:
            result = self._no_result(symbol, timeframe)
            result.volume_ratio = round(vol_ratio, 2)

        if result.detected:
            log.info(
                f"{symbol} | Absorption detected: {result.absorption_type} | "
                f"Vol ratio: {vol_ratio:.1f}x | Strength: {result.strength:.2f}"
            )

        return result

    # ── Multi-candle Absorption ───────────────────────────────────────────────
    def detect_composite(
        self, df: pd.DataFrame, symbol: str, lookback: int = 5
    ) -> dict:
        """
        Detect absorption over multiple candles (composite pattern).
        Used for higher timeframe analysis.
        """
        recent = df.tail(lookback)
        avg_vol   = df["volume"].tail(20).mean()
        total_vol = recent["volume"].sum()
        vol_ratio = total_vol / (avg_vol * lookback + 1e-10)

        price_range = float(recent["high"].max() - recent["low"].min())
        close_move  = abs(float(recent["close"].iloc[-1] - recent["close"].iloc[0]))
        efficiency  = close_move / (price_range + 1e-10)

        # Low efficiency + high volume = absorption (lots of volume, little movement)
        is_absorbing = vol_ratio > 1.5 and efficiency < 0.3

        return {
            "symbol":       symbol,
            "absorbing":    is_absorbing,
            "vol_ratio":    round(vol_ratio, 2),
            "efficiency":   round(efficiency, 3),
            "type":         "composite_absorption" if is_absorbing else "none",
        }

    def _no_result(self, symbol: str, timeframe: str) -> AbsorptionResult:
        return AbsorptionResult(
            symbol=symbol, timeframe=timeframe,
            detected=False, absorption_type="none",
            strength=0.0, price_level=0.0,
            volume_ratio=0.0, signal="neutral",
        )


# Singleton
absorption_detector = AbsorptionDetector()