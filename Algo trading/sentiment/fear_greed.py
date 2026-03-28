"""
sentiment/fear_greed.py
─────────────────────────────────────────────────────────────────────────────
Fear & Greed Index Engine.
Fetches the CNN Fear & Greed Index (stocks/macro) and the
Crypto Fear & Greed Index — both free, no API key needed.

Interpretation (contrarian):
  0-25   = Extreme Fear  → potential BUY zone (institutions accumulating)
  26-45  = Fear          → cautious, look for longs
  46-55  = Neutral       → no directional edge from sentiment
  56-75  = Greed         → cautious, look for shorts
  76-100 = Extreme Greed → potential SELL zone (institutions distributing)
─────────────────────────────────────────────────────────────────────────────
"""

import requests
from datetime import datetime
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class FearGreedResult:
    """Fear & Greed Index reading."""
    index_type:     str      # "crypto" | "stock"
    value:          int      # 0-100
    label:          str      # "Extreme Fear" → "Extreme Greed"
    signal:         str      # "bullish" | "bearish" | "neutral" (contrarian)
    contrarian_str: str      # Human-readable contrarian interpretation
    timestamp:      datetime = None

    @property
    def is_extreme(self) -> bool:
        return self.value <= 25 or self.value >= 75

    @property
    def zone(self) -> str:
        if self.value <= 25:   return "extreme_fear"
        if self.value <= 45:   return "fear"
        if self.value <= 55:   return "neutral"
        if self.value <= 75:   return "greed"
        return "extreme_greed"


class FearGreedEngine:
    """
    Fetches and interprets Fear & Greed indices.
    CONTRARIAN: buy fear, sell greed.
    """

    CRYPTO_FG_URL = "https://api.alternative.me/fng/?limit=1"

    def get_crypto_fear_greed(self) -> FearGreedResult:
        """
        Fetch Crypto Fear & Greed Index from alternative.me.
        Completely free, no API key needed.
        Updates daily.
        """
        try:
            r    = requests.get(self.CRYPTO_FG_URL, timeout=10)
            data = r.json()["data"][0]
            val  = int(data["value"])
            lbl  = data["value_classification"]
            ts   = datetime.utcfromtimestamp(int(data["timestamp"]))

            signal, cs = self._interpret(val)

            result = FearGreedResult(
                index_type="crypto",
                value=val,
                label=lbl,
                signal=signal,
                contrarian_str=cs,
                timestamp=ts,
            )

            log.info(f"Crypto F&G: {val} ({lbl}) → {signal} | {cs}")
            return result

        except Exception as e:
            log.warning(f"Crypto Fear & Greed fetch failed: {e}")
            return self._neutral_result("crypto")

    def get_stock_fear_greed(self) -> FearGreedResult:
        """
        Estimate stock market sentiment from VIX proxy (yFinance).
        CNN Fear & Greed doesn't have a free API — we approximate it.
        """
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")

            if hist.empty:
                return self._neutral_result("stock")

            vix_val = float(hist["Close"].iloc[-1])

            # Convert VIX to F&G scale (inverse relationship)
            # VIX 10 = Extreme Greed (~90), VIX 40+ = Extreme Fear (~10)
            fg_val = max(0, min(100, int(100 - (vix_val - 10) * 2.5)))

            signal, cs = self._interpret(fg_val)
            label = self._value_to_label(fg_val)

            result = FearGreedResult(
                index_type="stock",
                value=fg_val,
                label=label,
                signal=signal,
                contrarian_str=cs,
                timestamp=datetime.utcnow(),
            )

            log.info(f"Stock F&G (VIX proxy): {fg_val} ({label}) → {signal}")
            return result

        except Exception as e:
            log.warning(f"Stock Fear & Greed estimate failed: {e}")
            return self._neutral_result("stock")

    def get_combined(self) -> dict:
        """Get both indices and a combined interpretation."""
        crypto = self.get_crypto_fear_greed()
        stock  = self.get_stock_fear_greed()

        combined_val = (crypto.value + stock.value) // 2
        signal, cs   = self._interpret(combined_val)

        return {
            "crypto":         crypto,
            "stock":          stock,
            "combined_value": combined_val,
            "combined_signal": signal,
            "interpretation": cs,
            "market_mood":    self._value_to_label(combined_val),
        }

    # ── Interpretation ────────────────────────────────────────────────────────
    def _interpret(self, value: int) -> tuple[str, str]:
        """
        Contrarian interpretation of Fear & Greed.
        Fear = potential long opportunity
        Greed = potential short/exit opportunity
        """
        if value <= 20:
            return "bullish", "Extreme Fear — Institutions likely accumulating. Strong contrarian BUY zone."
        elif value <= 35:
            return "bullish", "Fear present — Market oversold. Look for long setups."
        elif value <= 55:
            return "neutral", "Balanced sentiment — No strong directional bias from F&G."
        elif value <= 75:
            return "bearish", "Greed building — Market overbought. Tighten stops on longs."
        else:
            return "bearish", "Extreme Greed — Institutions likely distributing. Strong contrarian SELL zone."

    def _value_to_label(self, value: int) -> str:
        if value <= 20:  return "Extreme Fear"
        if value <= 35:  return "Fear"
        if value <= 55:  return "Neutral"
        if value <= 75:  return "Greed"
        return "Extreme Greed"

    def _neutral_result(self, index_type: str) -> FearGreedResult:
        return FearGreedResult(
            index_type=index_type, value=50, label="Neutral",
            signal="neutral", contrarian_str="No data available.",
            timestamp=datetime.utcnow(),
        )


# Singleton
fear_greed_engine = FearGreedEngine()