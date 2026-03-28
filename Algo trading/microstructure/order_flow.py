"""
microstructure/order_flow.py
─────────────────────────────────────────────────────────────────────────────
Order Flow & Market Microstructure Engine — Citadel Securities philosophy.
Analyses bid/ask delta, volume imbalance, and aggressive order detection
to reveal who is in control BEFORE price moves.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrderFlowResult:
    """Complete order flow analysis output."""
    symbol:           str
    timeframe:        str
    delta:            float    # Cumulative buy vol - sell vol (positive = buyers winning)
    delta_pct:        float    # Delta as % of total volume
    buy_vol:          float    # Estimated buy volume
    sell_vol:         float    # Estimated sell volume
    total_vol:        float
    imbalance:        float    # -1.0 to +1.0 (positive = buyer dominance)
    aggressive_side:  str      # "buyers" | "sellers" | "neutral"
    exhaustion:       bool     # True if delta diverging from price (potential reversal)
    momentum:         float    # Cumulative delta trend (-1 to +1)
    signal:           str      # "bullish" | "bearish" | "neutral"
    confidence:       float    # 0.0 – 1.0


class OrderFlowEngine:
    """
    Analyses order flow from OHLCV data using the Bid/Ask estimation method.

    Without raw tick data (Level 2), we estimate buy/sell volume using:
    - Candle body direction
    - Wick analysis
    - Volume spread analysis (VSA)
    - Close position within range (closer to high = buying pressure)
    """

    def __init__(self):
        self.lookback = 20    # Candles for context

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> OrderFlowResult:
        """
        Estimate order flow from OHLCV data.

        Args:
            df:        OHLCV DataFrame
            symbol:    Instrument symbol
            timeframe: Timeframe string

        Returns:
            OrderFlowResult with delta, imbalance, and signal
        """
        if len(df) < 20:
            return self._neutral_result(symbol, timeframe)

        # Estimate buy/sell volume per candle
        buy_vol_series, sell_vol_series = self._estimate_volume_split(df)

        # Compute delta (last N candles)
        window = min(self.lookback, len(df))
        recent_buy  = buy_vol_series.tail(window).sum()
        recent_sell = sell_vol_series.tail(window).sum()
        total_vol   = recent_buy + recent_sell + 1e-10

        delta     = recent_buy - recent_sell
        delta_pct = delta / total_vol
        imbalance = delta / total_vol   # -1 to +1

        # Aggressive side
        if imbalance > 0.15:
            aggressive_side = "buyers"
        elif imbalance < -0.15:
            aggressive_side = "sellers"
        else:
            aggressive_side = "neutral"

        # Delta momentum — is delta trending up or down?
        delta_series = buy_vol_series - sell_vol_series
        delta_ma = delta_series.rolling(10).mean()
        if len(delta_ma.dropna()) >= 2:
            momentum = float(np.sign(delta_ma.iloc[-1] - delta_ma.iloc[-5]))
        else:
            momentum = 0.0

        # Exhaustion check — price making highs but delta declining (bearish div)
        exhaustion = self._check_exhaustion(df, delta_series)

        # Final signal
        if aggressive_side == "buyers" and not exhaustion and momentum > 0:
            signal = "bullish"
            confidence = min(1.0, abs(imbalance) * 2)
        elif aggressive_side == "sellers" and not exhaustion and momentum < 0:
            signal = "bearish"
            confidence = min(1.0, abs(imbalance) * 2)
        else:
            signal = "neutral"
            confidence = 0.3

        result = OrderFlowResult(
            symbol=symbol,
            timeframe=timeframe,
            delta=round(delta, 2),
            delta_pct=round(delta_pct, 4),
            buy_vol=round(recent_buy, 2),
            sell_vol=round(recent_sell, 2),
            total_vol=round(total_vol, 2),
            imbalance=round(imbalance, 4),
            aggressive_side=aggressive_side,
            exhaustion=exhaustion,
            momentum=momentum,
            signal=signal,
            confidence=round(confidence, 3),
        )

        log.debug(
            f"{symbol} | OrderFlow | Delta: {delta:.0f} | "
            f"Imbalance: {imbalance:.2%} | {aggressive_side} | {signal}"
        )

        return result

    # ── Volume Split Estimation ───────────────────────────────────────────────
    def _estimate_volume_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """
        Estimate buy vs sell volume per candle using multiple methods.

        Method: Close Location Volume (CLV)
        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        CLV > 0 = close near high = buying pressure
        CLV < 0 = close near low  = selling pressure
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        vol   = df["volume"]

        hl_range = (high - low).replace(0, np.nan)
        clv      = ((close - low) - (high - close)) / hl_range
        clv      = clv.fillna(0).clip(-1, 1)

        buy_vol  = vol * ((clv + 1) / 2)    # CLV=1 → 100% buy, CLV=-1 → 0% buy
        sell_vol = vol * ((1 - clv) / 2)

        return buy_vol, sell_vol

    # ── Exhaustion Detection ──────────────────────────────────────────────────
    def _check_exhaustion(
        self, df: pd.DataFrame, delta_series: pd.Series
    ) -> bool:
        """
        Detect delta exhaustion — price and delta diverging.
        Bullish exhaustion: price making higher highs, delta making lower highs
        Bearish exhaustion: price making lower lows, delta making higher lows
        """
        if len(df) < 10:
            return False

        price_recent  = df["close"].tail(10)
        delta_recent  = delta_series.tail(10)

        price_slope = float(np.polyfit(range(10), price_recent.values, 1)[0])
        delta_slope = float(np.polyfit(range(10), delta_recent.values, 1)[0])

        # Divergence: price and delta moving in opposite directions
        return (price_slope > 0 and delta_slope < -price_slope * 0.5) or \
               (price_slope < 0 and delta_slope > -price_slope * 0.5)

    # ── Orderbook Imbalance (when live data available) ────────────────────────
    def analyse_orderbook(self, orderbook: dict, symbol: str) -> dict:
        """
        Analyse real orderbook depth for live imbalance.
        Called when live orderbook data is available (crypto).
        """
        if not orderbook:
            return {}

        bid_vol = float(orderbook.get("bid_volume", 0))
        ask_vol = float(orderbook.get("ask_volume", 0))
        total   = bid_vol + ask_vol + 1e-10
        imbalance = (bid_vol - ask_vol) / total

        top_bids = orderbook.get("bids", pd.DataFrame())
        top_asks = orderbook.get("asks", pd.DataFrame())

        # Walls — large resting orders (potential support/resistance)
        bid_wall = float(top_bids["size"].max()) if not top_bids.empty else 0
        ask_wall = float(top_asks["size"].max()) if not top_asks.empty else 0

        return {
            "symbol":     symbol,
            "imbalance":  round(imbalance, 4),
            "bid_vol":    round(bid_vol, 2),
            "ask_vol":    round(ask_vol, 2),
            "bid_wall":   round(bid_wall, 2),
            "ask_wall":   round(ask_wall, 2),
            "spread":     orderbook.get("spread"),
            "signal":     "bullish" if imbalance > 0.2 else "bearish" if imbalance < -0.2 else "neutral",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _neutral_result(self, symbol: str, timeframe: str) -> OrderFlowResult:
        return OrderFlowResult(
            symbol=symbol, timeframe=timeframe,
            delta=0, delta_pct=0, buy_vol=0, sell_vol=0, total_vol=0,
            imbalance=0, aggressive_side="neutral",
            exhaustion=False, momentum=0, signal="neutral", confidence=0,
        )


# Singleton
order_flow_engine = OrderFlowEngine()