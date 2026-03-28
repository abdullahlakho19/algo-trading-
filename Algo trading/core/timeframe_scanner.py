"""
core/timeframe_scanner.py
─────────────────────────────────────────────────────────────────────────────
Multi-timeframe scanning orchestrator.
Scans from Daily down to 1m and builds a unified market context object
for each symbol. Higher timeframes set the directional bias; lower
timeframes find precise entries.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe:   str
    trend:       str          # "bullish" | "bearish" | "neutral"
    bias:        float        # -1.0 to +1.0
    key_levels:  list[float]  = field(default_factory=list)
    notes:       str          = ""


@dataclass
class MultiTimeframeContext:
    """
    Unified multi-timeframe picture for a single symbol.
    This object flows through the entire intelligence pipeline.
    """
    symbol:         str
    timestamp:      pd.Timestamp
    macro_bias:     str        = "neutral"   # From Daily/4H
    entry_bias:     str        = "neutral"   # From 1H/15m
    micro_bias:     str        = "neutral"   # From 5m/1m
    alignment:      float      = 0.0         # -1 to +1 (all TFs aligned?)
    timeframes:     dict       = field(default_factory=dict)  # TF → TimeframeAnalysis
    tradeable:      bool       = False


class TimeframeScanner:
    """
    Orchestrates analysis across all timeframes for a given symbol.

    Hierarchy:
        Daily  → macro directional bias
        4H     → intermediate structure
        1H     → entry context + confirmation zone
        15m    → trigger timeframe
        5m     → entry timing
        1m     → order flow / microstructure
    """

    # Timeframe weights for alignment score (higher TF = more weight)
    TF_WEIGHTS = {
        "1d":  0.35,
        "4h":  0.25,
        "1h":  0.20,
        "15m": 0.10,
        "5m":  0.07,
        "1m":  0.03,
    }

    def __init__(self):
        self.timeframes = config.timeframes.ALL

    def scan(self, symbol: str, data: dict[str, pd.DataFrame]) -> MultiTimeframeContext:
        """
        Main entry point. Takes a dict of {timeframe: OHLCV DataFrame}
        and returns a MultiTimeframeContext.

        Args:
            symbol: instrument ticker / pair
            data:   dict mapping timeframe strings to OHLCV DataFrames
        """
        ctx = MultiTimeframeContext(
            symbol=symbol,
            timestamp=pd.Timestamp.utcnow(),
        )

        tf_analyses: dict[str, TimeframeAnalysis] = {}
        alignment_score = 0.0

        for tf in self.timeframes:
            df = data.get(tf)
            if df is None or len(df) < config.timeframes.MIN_CANDLES:
                log.debug(f"{symbol} | {tf} — insufficient data, skipping.")
                continue

            analysis = self._analyse_timeframe(symbol, tf, df)
            tf_analyses[tf] = analysis

            # Accumulate weighted alignment score
            weight = self.TF_WEIGHTS.get(tf, 0.05)
            alignment_score += analysis.bias * weight

        ctx.timeframes = tf_analyses
        ctx.alignment  = round(alignment_score, 4)

        # Set directional biases from TF groupings
        ctx.macro_bias  = self._group_bias(tf_analyses, ["1d", "4h"])
        ctx.entry_bias  = self._group_bias(tf_analyses, ["1h", "15m"])
        ctx.micro_bias  = self._group_bias(tf_analyses, ["5m", "1m"])

        # Tradeable only if macro and entry biases agree
        ctx.tradeable = (
            ctx.macro_bias != "neutral"
            and ctx.entry_bias == ctx.macro_bias
            and abs(ctx.alignment) >= 0.3
        )

        log.info(
            f"{symbol} | MTF Scan | Macro: {ctx.macro_bias} | "
            f"Entry: {ctx.entry_bias} | Alignment: {ctx.alignment:.2f} | "
            f"Tradeable: {ctx.tradeable}"
        )

        return ctx

    def _analyse_timeframe(
        self, symbol: str, tf: str, df: pd.DataFrame
    ) -> TimeframeAnalysis:
        """
        Analyse a single timeframe OHLCV dataframe.
        Returns trend direction and a normalised bias score.
        """
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        # ── Simple trend detection using EMAs ─────────────────────────────────
        ema_20  = close.ewm(span=20,  adjust=False).mean()
        ema_50  = close.ewm(span=50,  adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        last_close = close.iloc[-1]
        last_20    = ema_20.iloc[-1]
        last_50    = ema_50.iloc[-1]
        last_200   = ema_200.iloc[-1]

        # Bullish structure: price > EMA20 > EMA50 > EMA200
        bullish_signals = sum([
            last_close > last_20,
            last_20 > last_50,
            last_50 > last_200,
            close.iloc[-1] > close.iloc[-5],   # Recent momentum up
        ])

        bearish_signals = sum([
            last_close < last_20,
            last_20 < last_50,
            last_50 < last_200,
            close.iloc[-1] < close.iloc[-5],
        ])

        # Bias: +1.0 = fully bullish, -1.0 = fully bearish
        bias = (bullish_signals - bearish_signals) / 4.0

        if bias > 0.25:
            trend = "bullish"
        elif bias < -0.25:
            trend = "bearish"
        else:
            trend = "neutral"

        # Key levels: recent swing highs and lows
        recent = df.tail(50)
        swing_highs = recent["high"].nlargest(3).tolist()
        swing_lows  = recent["low"].nsmallest(3).tolist()
        key_levels  = sorted(set(swing_highs + swing_lows))

        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            bias=round(bias, 4),
            key_levels=key_levels,
            notes=f"EMA20={last_20:.4f} EMA50={last_50:.4f} EMA200={last_200:.4f}",
        )

    def _group_bias(
        self, analyses: dict[str, TimeframeAnalysis], tfs: list[str]
    ) -> str:
        """Average bias across a group of timeframes."""
        biases = [analyses[tf].bias for tf in tfs if tf in analyses]
        if not biases:
            return "neutral"
        avg = sum(biases) / len(biases)
        if avg > 0.2:
            return "bullish"
        elif avg < -0.2:
            return "bearish"
        return "neutral"


# Singleton
timeframe_scanner = TimeframeScanner()