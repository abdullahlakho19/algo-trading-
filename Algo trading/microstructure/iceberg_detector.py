"""
microstructure/iceberg_detector.py
─────────────────────────────────────────────────────────────────────────────
Iceberg Order Detector.
Identifies hidden large institutional orders that are being systematically
filled in small pieces to hide their true size.

Signs of iceberg orders:
  - Repeated prints at the same price level
  - High volume that doesn't move price
  - Consistent absorption at bid or ask
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import Counter
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class IcebergResult:
    """Output of iceberg detection."""
    symbol:       str
    detected:     bool
    side:         str      # "bid" (buy iceberg) | "ask" (sell iceberg) | "none"
    price_level:  float    # Suspected iceberg price
    hit_count:    int      # How many times this level has been tested
    confidence:   float    # 0.0 – 1.0
    signal:       str      # "bullish" | "bearish" | "neutral"


class IcebergDetector:
    """
    Detects potential iceberg orders from OHLCV and tick patterns.

    Key signals:
    1. Price repeatedly touches same level but doesn't break through
    2. High volume at a fixed price with minimal movement
    3. Consistent wicks getting "cut" at the same level
    """

    def __init__(self):
        self.level_tolerance = 0.001    # 0.1% price tolerance for "same level"
        self.min_hits        = 3         # Min touches to suspect iceberg
        self.lookback        = 50

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str
    ) -> IcebergResult:
        """
        Detect iceberg orders from OHLCV price action.
        """
        if len(df) < self.lookback:
            return self._no_result(symbol)

        recent = df.tail(self.lookback)

        # Look for repeated highs and lows at same level
        bid_iceberg = self._detect_at_level(recent, side="bid")
        ask_iceberg = self._detect_at_level(recent, side="ask")

        if bid_iceberg["detected"] and bid_iceberg["confidence"] >= 0.6:
            return IcebergResult(
                symbol=symbol,
                detected=True,
                side="bid",
                price_level=bid_iceberg["level"],
                hit_count=bid_iceberg["hits"],
                confidence=bid_iceberg["confidence"],
                signal="bullish",
            )
        elif ask_iceberg["detected"] and ask_iceberg["confidence"] >= 0.6:
            return IcebergResult(
                symbol=symbol,
                detected=True,
                side="ask",
                price_level=ask_iceberg["level"],
                hit_count=ask_iceberg["hits"],
                confidence=ask_iceberg["confidence"],
                signal="bearish",
            )

        return self._no_result(symbol)

    # ── Level Analysis ────────────────────────────────────────────────────────
    def _detect_at_level(self, df: pd.DataFrame, side: str) -> dict:
        """
        Check for repeated touches at a single price level.
        side="bid" → look at lows (hidden buyer defending a floor)
        side="ask" → look at highs (hidden seller capping a ceiling)
        """
        prices = df["low"].values if side == "bid" else df["high"].values
        avg_price = prices.mean()
        tolerance = avg_price * self.level_tolerance

        # Round prices to tolerance grid
        rounded = np.round(prices / tolerance) * tolerance
        counter = Counter(rounded)

        if not counter:
            return {"detected": False}

        most_common_level, hit_count = counter.most_common(1)[0]

        if hit_count < self.min_hits:
            return {"detected": False}

        # Validate: did price respect this level? (bounced off each time)
        touches_df = df[np.isclose(
            (df["low"] if side == "bid" else df["high"]).values,
            most_common_level, atol=tolerance
        )]

        if len(touches_df) < self.min_hits:
            return {"detected": False}

        # Volume check — is there unusual volume at this level?
        avg_vol      = df["volume"].mean()
        touch_vol    = touches_df["volume"].mean()
        vol_ratio    = touch_vol / (avg_vol + 1e-10)

        confidence = min(1.0, (hit_count / 8) * 0.5 + min(vol_ratio / 2, 0.5))

        return {
            "detected":   True,
            "level":      round(float(most_common_level), 6),
            "hits":       hit_count,
            "vol_ratio":  round(vol_ratio, 2),
            "confidence": round(confidence, 3),
        }

    # ── Live Orderbook Detection ──────────────────────────────────────────────
    def analyse_orderbook(self, orderbook: dict, symbol: str) -> dict:
        """
        Detect icebergs from live orderbook (crypto / direct access).
        Looks for suspiciously large resting orders at a single level.
        """
        if not orderbook:
            return {"detected": False}

        bids = orderbook.get("bids", pd.DataFrame())
        asks = orderbook.get("asks", pd.DataFrame())

        result = {"detected": False, "symbol": symbol}

        if not bids.empty:
            max_bid_size = bids["size"].max()
            avg_bid_size = bids["size"].mean()
            if max_bid_size > avg_bid_size * 5:
                result.update({
                    "detected": True, "side": "bid",
                    "level": float(bids.loc[bids["size"].idxmax(), "price"]),
                    "size": float(max_bid_size),
                    "signal": "bullish",
                })

        if not asks.empty:
            max_ask_size = asks["size"].max()
            avg_ask_size = asks["size"].mean()
            if max_ask_size > avg_ask_size * 5:
                result.update({
                    "detected": True, "side": "ask",
                    "level": float(asks.loc[asks["size"].idxmax(), "price"]),
                    "size": float(max_ask_size),
                    "signal": "bearish",
                })

        if result["detected"]:
            log.info(f"{symbol} | Iceberg detected on {result.get('side')} @ {result.get('level')}")

        return result

    def _no_result(self, symbol: str) -> IcebergResult:
        return IcebergResult(
            symbol=symbol, detected=False, side="none",
            price_level=0.0, hit_count=0, confidence=0.0, signal="neutral",
        )


# Singleton
iceberg_detector = IcebergDetector()