"""
microstructure/darkpool_tracker.py
─────────────────────────────────────────────────────────────────────────────
Dark Pool & Block Trade Tracker — JP Morgan philosophy.
Detects large off-exchange institutional prints that signal
significant directional commitment from smart money.

Dark pool prints appear as sudden large volume spikes with
minimal price impact — institutions executing large orders quietly.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class DarkPoolPrint:
    """A detected dark pool / block trade event."""
    symbol:       str
    timestamp:    pd.Timestamp
    price:        float
    volume:       float
    vol_ratio:    float     # vs average
    direction:    str       # "bullish" | "bearish" | "unknown"
    confidence:   float


@dataclass
class DarkPoolResult:
    """Output of dark pool analysis."""
    symbol:        str
    timeframe:     str
    prints:        list[DarkPoolPrint] = field(default_factory=list)
    net_direction: str   = "neutral"   # Net direction of recent prints
    signal:        str   = "neutral"
    confidence:    float = 0.0
    total_dark_vol: float = 0.0


class DarkPoolTracker:
    """
    Identifies potential dark pool activity from unusual volume prints.

    Methodology:
    - Candles with volume > N * average = potential block trade
    - If price moves less than expected for that volume = dark pool
    - Track net direction of block prints for bias
    """

    def __init__(self):
        self.vol_spike_threshold = 3.0   # Volume must be 3x average
        self.price_efficiency_max = 0.4  # Max price efficiency (low = dark pool)
        self.lookback            = 50

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> DarkPoolResult:
        """
        Detect dark pool prints in recent candles.
        """
        if len(df) < self.lookback:
            return DarkPoolResult(symbol=symbol, timeframe=timeframe)

        prints     = []
        avg_vol    = df["volume"].tail(self.lookback).mean()
        avg_range  = (df["high"] - df["low"]).tail(self.lookback).mean()

        for i in range(max(0, len(df) - self.lookback), len(df)):
            row       = df.iloc[i]
            vol_ratio = float(row["volume"]) / (avg_vol + 1e-10)

            if vol_ratio < self.vol_spike_threshold:
                continue

            # Price efficiency = actual move / expected move for this volume
            actual_range = float(row["high"] - row["low"])
            expected     = avg_range * (vol_ratio ** 0.5)   # Expected proportional range
            efficiency   = actual_range / (expected + 1e-10)

            if efficiency > self.price_efficiency_max:
                continue   # Too much price movement = not a dark pool print

            # Determine direction from close location
            close_loc = (float(row["close"]) - float(row["low"])) / (actual_range + 1e-10)
            if close_loc > 0.6:
                direction = "bullish"
            elif close_loc < 0.4:
                direction = "bearish"
            else:
                direction = "unknown"

            confidence = min(1.0, vol_ratio / self.vol_spike_threshold * 0.7 +
                            (1 - efficiency) * 0.3)

            prints.append(DarkPoolPrint(
                symbol=symbol,
                timestamp=df.index[i],
                price=float(row["close"]),
                volume=float(row["volume"]),
                vol_ratio=round(vol_ratio, 2),
                direction=direction,
                confidence=round(confidence, 3),
            ))

        # Net direction from recent prints
        if prints:
            bull_prints = sum(1 for p in prints if p.direction == "bullish")
            bear_prints = sum(1 for p in prints if p.direction == "bearish")
            total_dark_vol = sum(p.volume for p in prints)

            if bull_prints > bear_prints:
                net_dir = "bullish"
                signal  = "bullish"
                conf    = bull_prints / len(prints)
            elif bear_prints > bull_prints:
                net_dir = "bearish"
                signal  = "bearish"
                conf    = bear_prints / len(prints)
            else:
                net_dir = "neutral"
                signal  = "neutral"
                conf    = 0.3

            if prints:
                log.info(
                    f"{symbol} | Dark Pool: {len(prints)} prints | "
                    f"Net: {net_dir} | Vol: {total_dark_vol:,.0f}"
                )

            return DarkPoolResult(
                symbol=symbol, timeframe=timeframe,
                prints=prints, net_direction=net_dir,
                signal=signal, confidence=round(conf, 3),
                total_dark_vol=round(total_dark_vol, 2),
            )

        return DarkPoolResult(symbol=symbol, timeframe=timeframe)

    # ── Recent Print Summary ──────────────────────────────────────────────────
    def get_recent_prints(
        self, result: DarkPoolResult, n: int = 5
    ) -> list[dict]:
        """Return the N most recent prints as dicts."""
        recent = sorted(result.prints, key=lambda p: p.timestamp, reverse=True)[:n]
        return [
            {
                "timestamp":  p.timestamp.isoformat(),
                "price":      p.price,
                "volume":     p.volume,
                "vol_ratio":  p.vol_ratio,
                "direction":  p.direction,
                "confidence": p.confidence,
            }
            for p in recent
        ]


# Singleton
darkpool_tracker = DarkPoolTracker()