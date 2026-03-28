"""
data_feeds/options_flow.py
─────────────────────────────────────────────────────────────────────────────
Options Flow Monitor.
Detects unusual options activity that signals large institutional
directional bets BEFORE they show up in the stock price.

Sources:
  - Alpaca (options data — requires Alpaca Options subscription)
  - Yahoo Finance (estimated IV from option chains — free)
  - Estimated from price action when no options data available
─────────────────────────────────────────────────────────────────────────────
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class OptionsActivity:
    """A single unusual options activity event."""
    symbol:         str
    expiry:         str
    strike:         float
    option_type:    str      # "call" | "put"
    volume:         int
    open_interest:  int
    vol_oi_ratio:   float    # Volume / OI (> 3 = unusual)
    implied_vol:    float    # IV percentage
    premium:        float    # Total premium paid
    side:           str      # "bullish" | "bearish"
    unusual:        bool     # True if volume >> OI


@dataclass
class OptionsFlowResult:
    """Aggregated options flow signal."""
    symbol:           str
    signal:           str    # "bullish" | "bearish" | "neutral"
    confidence:       float
    call_put_ratio:   float  # > 1 = more calls = bullish
    iv_percentile:    float  # 0-100 (current IV vs 52-week range)
    unusual_activity: list[OptionsActivity] = field(default_factory=list)
    net_premium:      float = 0.0   # Positive = net call buying (bullish)
    timestamp:        datetime = field(default_factory=datetime.utcnow)

    @property
    def is_strong_signal(self) -> bool:
        return (abs(self.call_put_ratio - 1) > 0.5 and
                self.confidence > 0.6 and
                len(self.unusual_activity) >= 1)


class OptionsFlowMonitor:
    """
    Monitors options market for unusual institutional activity.
    Uses Yahoo Finance option chains as the free data source.
    """

    def __init__(self):
        self.alpaca_key    = os.getenv("ALPACA_API_KEY", "")
        self._cache:       dict[str, OptionsFlowResult] = {}
        self._cache_ttl    = 60    # Minutes

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(self, symbol: str) -> OptionsFlowResult:
        """
        Analyse options flow for a stock symbol.

        Args:
            symbol: Stock ticker (e.g. "AAPL")

        Returns:
            OptionsFlowResult with signal and unusual activity
        """
        cached = self._get_cache(symbol)
        if cached:
            return cached

        # Try yFinance option chain (free)
        result = self._analyse_yfinance(symbol)

        self._set_cache(symbol, result)
        return result

    # ── yFinance Options Chain ────────────────────────────────────────────────
    def _analyse_yfinance(self, symbol: str) -> OptionsFlowResult:
        """Fetch and analyse option chain from yFinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Get next 2 expiry dates
            expiries = ticker.options[:2]
            if not expiries:
                return self._empty_result(symbol)

            all_calls   = []
            all_puts    = []
            unusual     = []

            for expiry in expiries:
                chain = ticker.option_chain(expiry)
                calls = chain.calls
                puts  = chain.puts

                if calls is None or puts is None:
                    continue

                # Find unusual volume (volume > 3x open interest)
                for _, row in calls.iterrows():
                    vol = int(row.get("volume", 0) or 0)
                    oi  = int(row.get("openInterest", 1) or 1)
                    ratio = vol / max(oi, 1)
                    iv    = float(row.get("impliedVolatility", 0) or 0)
                    prem  = float(row.get("lastPrice", 0) or 0) * vol * 100

                    all_calls.append({"vol": vol, "oi": oi, "iv": iv, "prem": prem})

                    if ratio >= 3 and vol >= 100:
                        unusual.append(OptionsActivity(
                            symbol=symbol, expiry=expiry,
                            strike=float(row.get("strike", 0)),
                            option_type="call", volume=vol,
                            open_interest=oi, vol_oi_ratio=round(ratio, 2),
                            implied_vol=round(iv, 4),
                            premium=round(prem, 2),
                            side="bullish", unusual=True,
                        ))

                for _, row in puts.iterrows():
                    vol = int(row.get("volume", 0) or 0)
                    oi  = int(row.get("openInterest", 1) or 1)
                    ratio = vol / max(oi, 1)
                    iv    = float(row.get("impliedVolatility", 0) or 0)
                    prem  = float(row.get("lastPrice", 0) or 0) * vol * 100

                    all_puts.append({"vol": vol, "oi": oi, "iv": iv, "prem": prem})

                    if ratio >= 3 and vol >= 100:
                        unusual.append(OptionsActivity(
                            symbol=symbol, expiry=expiry,
                            strike=float(row.get("strike", 0)),
                            option_type="put", volume=vol,
                            open_interest=oi, vol_oi_ratio=round(ratio, 2),
                            implied_vol=round(iv, 4),
                            premium=round(prem, 2),
                            side="bearish", unusual=True,
                        ))

            # Compute aggregate metrics
            total_call_vol = sum(c["vol"] for c in all_calls)
            total_put_vol  = sum(p["vol"] for p in all_puts)
            cp_ratio = total_call_vol / max(total_put_vol, 1)

            net_prem = sum(u.premium for u in unusual if u.side == "bullish") - \
                       sum(u.premium for u in unusual if u.side == "bearish")

            # Average IV
            all_ivs = [c["iv"] for c in all_calls + all_puts if c["iv"] > 0]
            avg_iv  = float(np.mean(all_ivs)) if all_ivs else 0.0

            # Signal from CP ratio and net premium
            if cp_ratio > 1.5 and net_prem > 0:
                signal     = "bullish"
                confidence = min(1.0, (cp_ratio - 1) * 0.4 + 0.3)
            elif cp_ratio < 0.67 and net_prem < 0:
                signal     = "bearish"
                confidence = min(1.0, (1 / cp_ratio - 1) * 0.4 + 0.3)
            else:
                signal     = "neutral"
                confidence = 0.3

            result = OptionsFlowResult(
                symbol=symbol,
                signal=signal,
                confidence=round(confidence, 3),
                call_put_ratio=round(cp_ratio, 3),
                iv_percentile=round(avg_iv * 100, 1),
                unusual_activity=unusual[:10],    # Top 10 unusual
                net_premium=round(net_prem, 2),
            )

            log.info(
                f"Options Flow | {symbol} | {signal} | "
                f"C/P: {cp_ratio:.2f} | Unusual: {len(unusual)} | "
                f"Net Prem: ${net_prem:,.0f}"
            )
            return result

        except Exception as e:
            log.warning(f"Options flow analysis failed for {symbol}: {e}")
            return self._empty_result(symbol)

    # ── IV Percentile ─────────────────────────────────────────────────────────
    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Compute IV Rank — where is current IV vs 52-week range?
        0 = historically low, 100 = historically high.
        """
        try:
            import yfinance as yf
            ticker  = yf.Ticker(symbol)
            hist    = ticker.history(period="1y")
            if hist.empty:
                return 50.0

            # Estimate realised vol as IV proxy
            returns = hist["Close"].pct_change().dropna()
            rv_series = returns.rolling(20).std() * np.sqrt(252)
            current   = float(rv_series.iloc[-1])
            lo        = float(rv_series.min())
            hi        = float(rv_series.max())
            rank      = (current - lo) / (hi - lo + 1e-10) * 100
            return round(rank, 1)
        except Exception:
            return 50.0

    # ── Cache ─────────────────────────────────────────────────────────────────
    def _get_cache(self, symbol: str) -> OptionsFlowResult | None:
        c = self._cache.get(symbol)
        if not c:
            return None
        age = (datetime.utcnow() - c.timestamp).total_seconds() / 60
        return c if age < self._cache_ttl else None

    def _set_cache(self, symbol: str, result: OptionsFlowResult) -> None:
        self._cache[symbol] = result

    def _empty_result(self, symbol: str) -> OptionsFlowResult:
        return OptionsFlowResult(
            symbol=symbol, signal="neutral",
            confidence=0.0, call_put_ratio=1.0,
            iv_percentile=50.0,
        )


# Singleton
options_flow_monitor = OptionsFlowMonitor()