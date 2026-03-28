"""
macro/regime_overlay.py
─────────────────────────────────────────────────────────────────────────────
Macro Regime Overlay — BlackRock Aladdin philosophy.
Classifies the broader macro environment as Risk-On or Risk-Off
and adjusts trading behaviour accordingly.

Risk-On:  Markets are calm, appetite for risk is high → normal trading
Risk-Off: Fear elevated, correlations spike → reduce size, be cautious
Crisis:   Systemic stress → stop trading, preserve capital
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from core.logger import get_logger

log = get_logger(__name__)


class MacroRegime(str, Enum):
    RISK_ON    = "risk_on"
    RISK_OFF   = "risk_off"
    CRISIS     = "crisis"
    NEUTRAL    = "neutral"


@dataclass
class MacroRegimeResult:
    """Output of macro regime classification."""
    regime:          MacroRegime
    confidence:      float
    size_multiplier: float    # Adjust position sizes by this factor
    notes:           str = ""

    @property
    def should_trade(self) -> bool:
        return self.regime != MacroRegime.CRISIS

    @property
    def is_risk_on(self) -> bool:
        return self.regime == MacroRegime.RISK_ON


class MacroRegimeOverlay:
    """
    Classifies macro environment using cross-asset signals.

    Uses proxy indicators since live VIX/credit spreads require
    premium data — estimates from available price data.
    """

    def __init__(self):
        # Risk-off thresholds
        self.high_vol_threshold  = 0.025    # 2.5% normalised ATR = elevated
        self.crisis_threshold    = 0.05     # 5% = crisis level

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def classify(
        self,
        spy_df:  pd.DataFrame = None,    # S&P 500 proxy
        vix_df:  pd.DataFrame = None,    # VIX proxy (if available)
        btc_df:  pd.DataFrame = None,    # BTC (risk asset)
        gold_df: pd.DataFrame = None,    # Gold (safe haven)
    ) -> MacroRegimeResult:
        """
        Classify macro regime from cross-asset data.
        All inputs are optional — uses what's available.
        """
        scores = []

        if spy_df is not None and not spy_df.empty:
            scores.append(self._score_equity(spy_df))

        if btc_df is not None and not btc_df.empty:
            scores.append(self._score_crypto_risk(btc_df))

        if gold_df is not None and not gold_df.empty:
            scores.append(self._score_safe_haven(gold_df))

        if not scores:
            return MacroRegimeResult(
                regime=MacroRegime.NEUTRAL,
                confidence=0.3,
                size_multiplier=0.8,
                notes="No macro data available",
            )

        avg_score = np.mean(scores)

        # Map score to regime
        # avg_score: 1.0 = risk-on, 0.0 = risk-off, -1.0 = crisis
        if avg_score > 0.3:
            regime      = MacroRegime.RISK_ON
            size_mult   = 1.0
            confidence  = min(1.0, avg_score)
        elif avg_score < -0.3:
            regime      = MacroRegime.CRISIS
            size_mult   = 0.0
            confidence  = min(1.0, abs(avg_score))
        elif avg_score < 0:
            regime      = MacroRegime.RISK_OFF
            size_mult   = 0.5
            confidence  = 0.6
        else:
            regime      = MacroRegime.NEUTRAL
            size_mult   = 0.8
            confidence  = 0.4

        result = MacroRegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            size_multiplier=size_mult,
            notes=f"Score: {avg_score:.2f} from {len(scores)} signals",
        )

        log.info(f"Macro Regime: {regime.value} | Confidence: {confidence:.1%} | Size: {size_mult}x")
        return result

    # ── Component Scorers ─────────────────────────────────────────────────────
    def _score_equity(self, df: pd.DataFrame) -> float:
        """Score from equity index. Trending up = risk-on."""
        if len(df) < 50:
            return 0.0
        close   = df["close"]
        ema20   = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50   = close.ewm(span=50, adjust=False).mean().iloc[-1]
        last    = float(close.iloc[-1])

        atr     = self._atr_norm(df)

        # Bullish alignment = risk-on
        if last > ema20 > ema50 and atr < self.high_vol_threshold:
            return 0.8
        elif last < ema20 < ema50 and atr > self.high_vol_threshold:
            return -0.6
        elif atr > self.crisis_threshold:
            return -1.0
        return 0.0

    def _score_crypto_risk(self, df: pd.DataFrame) -> float:
        """BTC trending up = risk appetite on."""
        if len(df) < 20:
            return 0.0
        ret_20 = float(df["close"].pct_change(20).iloc[-1])
        if ret_20 > 0.10:
            return 0.6
        elif ret_20 < -0.20:
            return -0.7
        return ret_20 * 2

    def _score_safe_haven(self, df: pd.DataFrame) -> float:
        """Gold surging = risk-off signal."""
        if len(df) < 10:
            return 0.0
        ret_10 = float(df["close"].pct_change(10).iloc[-1])
        if ret_10 > 0.03:
            return -0.5   # Gold up = risk-off
        elif ret_10 < -0.02:
            return 0.3    # Gold down = risk-on
        return 0.0

    def _atr_norm(self, df: pd.DataFrame) -> float:
        """Normalised ATR."""
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        return atr / float(close.iloc[-1]) if close.iloc[-1] > 0 else 0.0


# Singleton
macro_regime_overlay = MacroRegimeOverlay()