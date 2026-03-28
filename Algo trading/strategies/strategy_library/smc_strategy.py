"""
strategies/strategy_library/smc_strategy.py
─────────────────────────────────────────────────────────────────────────────
Smart Money Concepts (SMC) Strategy.
Combines Order Blocks, Fair Value Gaps, Break of Structure,
and Liquidity sweeps to trade with institutional flow.

Core SMC logic:
  1. Identify Break of Structure (BoS) → confirms trend direction
  2. Find Order Block — last bearish candle before bullish BoS (or vice versa)
  3. Wait for price to retrace into Order Block
  4. Confirm with Fair Value Gap and volume
  5. Enter with tight SL below Order Block
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrderBlock:
    """An institutional order block level."""
    index:     int
    timestamp: pd.Timestamp
    high:      float
    low:       float
    mid:       float
    direction: str       # "bullish" | "bearish"
    valid:     bool = True
    tested:    bool = False


@dataclass
class FairValueGap:
    """A Fair Value Gap (imbalance) in price."""
    index:     int
    high:      float
    low:       float
    mid:       float
    direction: str   # "bullish" | "bearish"
    filled:    bool = False


@dataclass
class SMCSignal:
    """Output of SMC analysis."""
    symbol:       str
    direction:    str       # "long" | "short" | "neutral"
    entry:        float
    stop_loss:    float
    take_profit:  float
    order_block:  OrderBlock = None
    fvg:          FairValueGap = None
    confluence:   int = 0
    confidence:   float = 0.0
    notes:        str = ""


class SMCStrategy:
    """
    Smart Money Concepts full strategy implementation.
    Detects Order Blocks, Fair Value Gaps, and liquidity sweeps.
    """

    def __init__(self):
        self.ob_lookback  = 50    # Candles to look back for order blocks
        self.fvg_lookback = 30

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str
    ) -> SMCSignal:
        """
        Full SMC analysis. Returns a trade signal if setup is valid.
        """
        if len(df) < 60:
            return SMCSignal(symbol=symbol, direction="neutral",
                             entry=0, stop_loss=0, take_profit=0)

        # Step 1 — Find structure
        from intelligence.structure_engine import structure_engine
        structure = structure_engine.analyse(df, symbol, "primary")

        # Step 2 — Find order blocks
        order_blocks = self._find_order_blocks(df)

        # Step 3 — Find Fair Value Gaps
        fvgs = self._find_fair_value_gaps(df)

        # Step 4 — Find liquidity sweeps
        sweep = self._detect_liquidity_sweep(df)

        current_price = float(df["close"].iloc[-1])
        atr           = self._atr(df)

        # Step 5 — Score setups
        return self._build_signal(
            symbol, df, structure, order_blocks, fvgs,
            sweep, current_price, atr
        )

    # ── Order Block Detection ─────────────────────────────────────────────────
    def _find_order_blocks(self, df: pd.DataFrame) -> list[OrderBlock]:
        """
        Find institutional order blocks.
        Bullish OB: last bearish candle before a strong bullish move
        Bearish OB: last bullish candle before a strong bearish move
        """
        blocks = []
        lookback = min(self.ob_lookback, len(df) - 3)

        for i in range(2, lookback):
            idx   = len(df) - 1 - i
            candle = df.iloc[idx]
            next3  = df.iloc[idx + 1: idx + 4]

            # Bullish OB: bearish candle followed by 3 strong bullish candles
            if (candle["close"] < candle["open"] and
                    all(next3["close"] > next3["open"]) and
                    next3["close"].iloc[-1] > candle["high"] * 1.001):

                blocks.append(OrderBlock(
                    index=idx,
                    timestamp=df.index[idx],
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    mid=float((candle["high"] + candle["low"]) / 2),
                    direction="bullish",
                ))

            # Bearish OB: bullish candle followed by 3 strong bearish candles
            elif (candle["close"] > candle["open"] and
                      all(next3["close"] < next3["open"]) and
                      next3["close"].iloc[-1] < candle["low"] * 0.999):

                blocks.append(OrderBlock(
                    index=idx,
                    timestamp=df.index[idx],
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    mid=float((candle["high"] + candle["low"]) / 2),
                    direction="bearish",
                ))

        return blocks[-10:]   # Keep most recent 10

    # ── Fair Value Gap Detection ───────────────────────────────────────────────
    def _find_fair_value_gaps(self, df: pd.DataFrame) -> list[FairValueGap]:
        """
        Detect Fair Value Gaps (3-candle imbalance).
        FVG exists when candle 1 high < candle 3 low (bullish gap)
        or candle 1 low > candle 3 high (bearish gap).
        """
        fvgs = []
        for i in range(1, min(self.fvg_lookback, len(df) - 1)):
            c1 = df.iloc[-i - 1]
            c3 = df.iloc[-i + 1] if i > 0 else df.iloc[-1]

            # Bullish FVG
            if float(c1["high"]) < float(c3["low"]):
                fvgs.append(FairValueGap(
                    index=-i,
                    high=float(c3["low"]),
                    low=float(c1["high"]),
                    mid=float((c3["low"] + c1["high"]) / 2),
                    direction="bullish",
                ))
            # Bearish FVG
            elif float(c1["low"]) > float(c3["high"]):
                fvgs.append(FairValueGap(
                    index=-i,
                    high=float(c1["low"]),
                    low=float(c3["high"]),
                    mid=float((c1["low"] + c3["high"]) / 2),
                    direction="bearish",
                ))

        return fvgs[:5]

    # ── Liquidity Sweep Detection ──────────────────────────────────────────────
    def _detect_liquidity_sweep(self, df: pd.DataFrame) -> dict:
        """
        Detect a liquidity sweep — price briefly breaks a key level
        then reverses (stop-hunt by institutions).
        """
        if len(df) < 20:
            return {"detected": False}

        recent = df.tail(20)
        last   = df.iloc[-1]

        # Check if last candle swept a recent high/low then reversed
        recent_high = float(recent["high"].iloc[:-1].max())
        recent_low  = float(recent["low"].iloc[:-1].min())

        swept_high = (float(last["high"]) > recent_high and
                      float(last["close"]) < recent_high)

        swept_low  = (float(last["low"]) < recent_low and
                      float(last["close"]) > recent_low)

        if swept_high:
            return {"detected": True, "type": "high_sweep", "level": recent_high,
                    "signal": "bearish"}
        if swept_low:
            return {"detected": True, "type": "low_sweep", "level": recent_low,
                    "signal": "bullish"}
        return {"detected": False}

    # ── Signal Builder ────────────────────────────────────────────────────────
    def _build_signal(
        self, symbol, df, structure, order_blocks,
        fvgs, sweep, price, atr
    ) -> SMCSignal:
        """Combine all SMC components into a trade signal."""
        confluence = 0
        direction  = "neutral"
        entry      = price
        best_ob    = None
        best_fvg   = None
        notes      = []

        from intelligence.structure_engine import StructureSignal

        # ── Bullish Setup ──────────────────────────────────────────────────
        if structure.signal in (StructureSignal.BOS_BULLISH, StructureSignal.CHOCH_BULLISH):
            direction   = "long"
            confluence += 1
            notes.append("bullish_structure")

            # Find most recent valid bullish OB that price is near
            for ob in reversed(order_blocks):
                if ob.direction == "bullish" and ob.low <= price <= ob.high * 1.01:
                    best_ob     = ob
                    confluence += 1
                    notes.append(f"ob_at_{ob.mid:.4f}")
                    break

            # Find bullish FVG near price
            for fvg in fvgs:
                if fvg.direction == "bullish" and fvg.low <= price <= fvg.high:
                    best_fvg    = fvg
                    confluence += 1
                    notes.append("fvg_support")
                    break

            # Liquidity sweep adds confluence
            if sweep.get("detected") and sweep.get("signal") == "bullish":
                confluence += 1
                notes.append("liquidity_sweep")

        # ── Bearish Setup ──────────────────────────────────────────────────
        elif structure.signal in (StructureSignal.BOS_BEARISH, StructureSignal.CHOCH_BEARISH):
            direction   = "short"
            confluence += 1
            notes.append("bearish_structure")

            for ob in reversed(order_blocks):
                if ob.direction == "bearish" and ob.low * 0.99 <= price <= ob.high:
                    best_ob     = ob
                    confluence += 1
                    notes.append(f"ob_at_{ob.mid:.4f}")
                    break

            for fvg in fvgs:
                if fvg.direction == "bearish" and fvg.low <= price <= fvg.high:
                    best_fvg    = fvg
                    confluence += 1
                    notes.append("fvg_resistance")
                    break

            if sweep.get("detected") and sweep.get("signal") == "bearish":
                confluence += 1
                notes.append("liquidity_sweep")

        if direction == "neutral" or confluence < 2:
            return SMCSignal(symbol=symbol, direction="neutral",
                             entry=0, stop_loss=0, take_profit=0)

        # Calculate SL/TP
        if direction == "long":
            sl = (best_ob.low - atr * 0.5) if best_ob else (price - atr * 1.5)
            tp = price + (price - sl) * 2.5
        else:
            sl = (best_ob.high + atr * 0.5) if best_ob else (price + atr * 1.5)
            tp = price - (sl - price) * 2.5

        confidence = min(1.0, confluence / 5)

        return SMCSignal(
            symbol=symbol,
            direction=direction,
            entry=round(price, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            order_block=best_ob,
            fvg=best_fvg,
            confluence=confluence,
            confidence=round(confidence, 3),
            notes=" | ".join(notes),
        )

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


# Singleton
smc_strategy = SMCStrategy()