"""
sentiment/sentiment_engine.py
─────────────────────────────────────────────────────────────────────────────
Master Sentiment Engine.
Aggregates all sentiment sources into a single sentiment score
and integrates it into the probability scoring pipeline.

Sources & Weights:
  News (FinBERT)      40%  — Most reliable, specific to symbol
  Fear & Greed        30%  — Macro context, contrarian signal
  Social (Reddit)     20%  — Buzz and retail momentum
  COT Positioning     10%  — Institutional long/short bias (Forex)

Sentiment is a FILTER, not a primary signal:
  - Strong confirming sentiment → increase probability score by up to +0.10
  - Strong opposing sentiment  → reduce probability score by up to -0.15
  - Extreme sentiment          → contrarian override flag
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class SentimentSnapshot:
    """Complete sentiment picture for one symbol at one moment."""
    symbol:            str
    composite_score:   float     # -1.0 to +1.0 weighted aggregate
    composite_label:   str       # "bullish" | "bearish" | "neutral"
    confidence:        float     # 0-1 overall confidence
    news_score:        float
    fear_greed_score:  float
    social_score:      float
    cot_score:         float
    is_contrarian:     bool      # Extreme retail sentiment = contrarian flag
    contrarian_dir:    str       # Direction the contrarian signal favours
    signal_boost:      float     # Amount to add to probability score (-0.15 to +0.10)
    sources_active:    list[str] = field(default_factory=list)
    timestamp:         datetime  = field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        boost_str = f"+{self.signal_boost:.2f}" if self.signal_boost >= 0 else f"{self.signal_boost:.2f}"
        return (
            f"Sentiment | {self.symbol} | {self.composite_label} "
            f"({self.composite_score:.2f}) | Boost: {boost_str} | "
            f"Sources: {len(self.sources_active)}"
        )


class SentimentEngine:
    """
    Master sentiment aggregator.
    Combines all sentiment sources and integrates with the signal pipeline.
    """

    # Source weights
    WEIGHTS = {
        "news":        0.40,
        "fear_greed":  0.30,
        "social":      0.20,
        "cot":         0.10,
    }

    # Signal boost limits
    MAX_BOOST      =  0.10   # Max positive adjustment to probability
    MAX_REDUCTION  = -0.15   # Max negative adjustment to probability

    def __init__(self):
        self._news_engine   = None
        self._fg_engine     = None
        self._social_engine = None
        self._cot_reader    = None
        self._ready         = False

    def _init(self):
        if self._ready:
            return
        try:
            from sentiment.news_sentiment   import news_sentiment_engine
            from sentiment.fear_greed       import fear_greed_engine
            from sentiment.social_sentiment import social_sentiment_engine
            from data_feeds.cot_reader      import cot_reader
            self._news_engine   = news_sentiment_engine
            self._fg_engine     = fear_greed_engine
            self._social_engine = social_sentiment_engine
            self._cot_reader    = cot_reader
            self._ready = True
        except Exception as e:
            log.warning(f"Sentiment engine init error: {e}")

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self,
        symbol:  str,
        market:  str = "stock",
    ) -> SentimentSnapshot:
        """
        Full sentiment analysis for one symbol.

        Args:
            symbol: Instrument ticker
            market: "stock" | "forex" | "crypto"

        Returns:
            SentimentSnapshot with composite score and signal boost
        """
        self._init()

        scores       = {}
        sources      = []

        # ── News Sentiment ─────────────────────────────────────────────────
        try:
            news = self._news_engine.analyse(symbol)
            if news.article_count >= 2:
                scores["news"] = news.score
                sources.append("news")
        except Exception as e:
            log.debug(f"News sentiment failed for {symbol}: {e}")

        # ── Fear & Greed ───────────────────────────────────────────────────
        try:
            if market == "crypto":
                fg = self._fg_engine.get_crypto_fear_greed()
            else:
                fg = self._fg_engine.get_stock_fear_greed()

            # Convert F&G (0-100) to -1/+1 scale (contrarian)
            fg_score = (50 - fg.value) / 50    # Extreme fear = +1, extreme greed = -1
            scores["fear_greed"] = fg_score
            sources.append("fear_greed")
        except Exception as e:
            log.debug(f"Fear & Greed failed: {e}")

        # ── Social Sentiment ───────────────────────────────────────────────
        try:
            social = self._social_engine.analyse(symbol, market)
            if social.post_count >= 5:
                # Social is CONTRARIAN at extremes
                if social.contrarian:
                    # Flip the score — extreme bulls = bearish signal
                    scores["social"] = -social.score
                else:
                    scores["social"] = social.score * 0.5   # Dampen non-extreme social
                sources.append("social")
        except Exception as e:
            log.debug(f"Social sentiment failed for {symbol}: {e}")

        # ── COT Data (Forex only) ──────────────────────────────────────────
        if market == "forex":
            try:
                cot_bias = self._cot_reader.get_bias(symbol)
                cot_score = 0.6 if cot_bias == "bullish" else -0.6 if cot_bias == "bearish" else 0.0
                scores["cot"] = cot_score
                sources.append("cot")
            except Exception as e:
                log.debug(f"COT failed for {symbol}: {e}")

        if not scores:
            return self._empty_snapshot(symbol)

        # ── Weighted Composite ─────────────────────────────────────────────
        total_weight = sum(self.WEIGHTS[s] for s in scores)
        composite    = sum(
            scores[s] * self.WEIGHTS[s] for s in scores
        ) / total_weight if total_weight > 0 else 0.0

        confidence   = len(scores) / len(self.WEIGHTS)    # % of sources active

        if composite > 0.15:
            label = "bullish"
        elif composite < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        # Contrarian check
        social_result  = None
        is_contrarian  = False
        contrarian_dir = "neutral"
        try:
            social_result  = self._social_engine._cache.get(symbol)
            if social_result and social_result.contrarian:
                is_contrarian  = True
                contrarian_dir = social_result.contrarian_direction
        except Exception:
            pass

        # Signal boost calculation
        boost = self._compute_boost(composite, confidence, is_contrarian, contrarian_dir)

        snapshot = SentimentSnapshot(
            symbol=symbol,
            composite_score=round(composite, 4),
            composite_label=label,
            confidence=round(confidence, 3),
            news_score=round(scores.get("news", 0), 4),
            fear_greed_score=round(scores.get("fear_greed", 0), 4),
            social_score=round(scores.get("social", 0), 4),
            cot_score=round(scores.get("cot", 0), 4),
            is_contrarian=is_contrarian,
            contrarian_dir=contrarian_dir,
            signal_boost=round(boost, 4),
            sources_active=sources,
        )

        log.info(snapshot.summary())
        return snapshot

    # ── Signal Boost ──────────────────────────────────────────────────────────
    def _compute_boost(
        self,
        composite:     float,
        confidence:    float,
        is_contrarian: bool,
        contrarian_dir: str,
    ) -> float:
        """
        Compute the probability score adjustment from sentiment.

        Logic:
          - Strong confirming sentiment → +boost
          - Strong opposing sentiment  → -reduction
          - Contrarian extreme         → moderate boost in contrarian direction
        """
        if is_contrarian:
            # Contrarian: when everyone is bullish, fade them
            boost = 0.05 * confidence
            if contrarian_dir == "short":
                boost = -boost
        elif abs(composite) > 0.5 and confidence > 0.6:
            # Strong, confident signal
            boost = composite * 0.15 * confidence
        elif abs(composite) > 0.25:
            # Moderate signal
            boost = composite * 0.08 * confidence
        else:
            # Weak/neutral signal — no boost
            boost = 0.0

        return float(max(self.MAX_REDUCTION, min(self.MAX_BOOST, boost)))

    # ── Integration with Probability Scorer ───────────────────────────────────
    def apply_to_probability(
        self,
        base_probability: float,
        snapshot:         SentimentSnapshot,
        trade_direction:  str,
    ) -> float:
        """
        Apply sentiment boost/reduction to base probability score.

        Args:
            base_probability: Existing signal probability (0-1)
            snapshot:         SentimentSnapshot from analyse()
            trade_direction:  "long" | "short"

        Returns:
            Adjusted probability (still 0-1)
        """
        boost = snapshot.signal_boost

        # Only apply boost if sentiment AGREES with trade direction
        if trade_direction == "long" and snapshot.composite_label == "bearish":
            boost = -abs(boost)   # Opposing sentiment reduces probability
        elif trade_direction == "short" and snapshot.composite_label == "bullish":
            boost = -abs(boost)

        adjusted = base_probability + boost
        adjusted = max(0.0, min(1.0, adjusted))

        log.debug(
            f"Probability: {base_probability:.3f} "
            f"+ sentiment boost {boost:+.3f} = {adjusted:.3f}"
        )

        return round(adjusted, 4)

    def _empty_snapshot(self, symbol: str) -> SentimentSnapshot:
        return SentimentSnapshot(
            symbol=symbol, composite_score=0.0,
            composite_label="neutral", confidence=0.0,
            news_score=0, fear_greed_score=0,
            social_score=0, cot_score=0,
            is_contrarian=False, contrarian_dir="neutral",
            signal_boost=0.0,
        )


# Singleton
sentiment_engine = SentimentEngine()