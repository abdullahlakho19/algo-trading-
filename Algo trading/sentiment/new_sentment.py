"""
sentiment/news_sentiment.py
─────────────────────────────────────────────────────────────────────────────
Financial News Sentiment Engine.
Aggregates news from multiple sources and scores sentiment
using FinBERT — a model trained specifically on financial text.

Sources:
  - Finnhub     (free tier: 60 calls/min)
  - Alpha Vantage (free tier: 25 calls/day for news)
  - NewsAPI      (free tier: 100 calls/day)
  - Benzinga     (via Alpaca — free with account)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class NewsItem:
    """A single news article with sentiment score."""
    headline:   str
    source:     str
    published:  datetime
    url:        str
    symbol:     str
    sentiment:  float     # -1.0 (very bearish) to +1.0 (very bullish)
    confidence: float     # 0.0 – 1.0
    label:      str       # "positive" | "negative" | "neutral"


@dataclass
class NewsSentimentResult:
    """Aggregated news sentiment for a symbol."""
    symbol:         str
    score:          float     # -1.0 to +1.0 weighted average
    label:          str       # "bullish" | "bearish" | "neutral"
    confidence:     float
    article_count:  int
    bullish_count:  int
    bearish_count:  int
    neutral_count:  int
    articles:       list[NewsItem] = field(default_factory=list)
    timestamp:      datetime = field(default_factory=datetime.utcnow)

    @property
    def is_strong_signal(self) -> bool:
        return abs(self.score) >= 0.4 and self.confidence >= 0.65 and self.article_count >= 3


class NewsSentimentEngine:
    """
    Multi-source financial news sentiment engine.
    Uses FinBERT for NLP scoring when available,
    falls back to VADER/keyword scoring.
    """

    def __init__(self):
        self.finnhub_key     = os.getenv("FINNHUB_API_KEY", "")
        self.alphav_key      = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.newsapi_key     = os.getenv("NEWS_API_KEY", "")

        self._finbert        = None     # Loaded lazily
        self._vader          = None
        self._cache:         dict[str, NewsSentimentResult] = {}
        self._cache_ttl_min  = 30       # Cache for 30 minutes

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self,
        symbol: str,
        lookback_hours: int = 24,
        max_articles:   int = 20,
    ) -> NewsSentimentResult:
        """
        Fetch and score news sentiment for a symbol.

        Args:
            symbol:         Ticker (e.g. "AAPL", "BTC/USDT")
            lookback_hours: How far back to look for news
            max_articles:   Max articles to process

        Returns:
            NewsSentimentResult with aggregate sentiment
        """
        # Check cache
        cached = self._get_cache(symbol)
        if cached:
            return cached

        articles = []

        # Fetch from all available sources
        if self.finnhub_key:
            articles += self._fetch_finnhub(symbol, lookback_hours)
        if self.alphav_key:
            articles += self._fetch_alphavantage(symbol)
        if self.newsapi_key:
            articles += self._fetch_newsapi(symbol, lookback_hours)

        if not articles:
            log.debug(f"No news found for {symbol}")
            return self._empty_result(symbol)

        # Deduplicate by headline
        seen = set()
        unique = []
        for a in articles:
            key = a["headline"][:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        unique = unique[:max_articles]

        # Score each article
        scored  = []
        for art in unique:
            score, conf, label = self._score_text(art["headline"])
            scored.append(NewsItem(
                headline=art["headline"],
                source=art["source"],
                published=art.get("published", datetime.utcnow()),
                url=art.get("url", ""),
                symbol=symbol,
                sentiment=score,
                confidence=conf,
                label=label,
            ))

        # Aggregate (recency-weighted)
        result = self._aggregate(symbol, scored)
        self._set_cache(symbol, result)

        log.info(
            f"News Sentiment | {symbol} | Score: {result.score:.2f} | "
            f"{result.label} | {result.article_count} articles"
        )

        return result

    # ── Data Sources ──────────────────────────────────────────────────────────
    def _fetch_finnhub(self, symbol: str, lookback_hours: int) -> list[dict]:
        """Fetch news from Finnhub API."""
        try:
            clean  = symbol.replace("/", "").replace("USDT", "")
            end    = datetime.utcnow()
            start  = end - timedelta(hours=lookback_hours)

            url = "https://finnhub.io/api/v1/company-news"
            r   = requests.get(url, params={
                "symbol": clean,
                "from":   start.strftime("%Y-%m-%d"),
                "to":     end.strftime("%Y-%m-%d"),
                "token":  self.finnhub_key,
            }, timeout=10)

            if r.status_code != 200:
                return []

            data = r.json()
            return [
                {
                    "headline":  item.get("headline", ""),
                    "source":    item.get("source", "Finnhub"),
                    "url":       item.get("url", ""),
                    "published": datetime.utcfromtimestamp(item.get("datetime", 0)),
                }
                for item in data[:15]
                if item.get("headline")
            ]

        except Exception as e:
            log.warning(f"Finnhub fetch failed for {symbol}: {e}")
            return []

    def _fetch_alphavantage(self, symbol: str) -> list[dict]:
        """Fetch news sentiment from Alpha Vantage."""
        try:
            clean = symbol.replace("/", "").replace("USDT", "")
            url   = "https://www.alphavantage.co/query"
            r     = requests.get(url, params={
                "function": "NEWS_SENTIMENT",
                "tickers":  clean,
                "apikey":   self.alphav_key,
                "limit":    10,
            }, timeout=10)

            if r.status_code != 200:
                return []

            data = r.json().get("feed", [])
            return [
                {
                    "headline":  item.get("title", ""),
                    "source":    item.get("source", "AlphaVantage"),
                    "url":       item.get("url", ""),
                    "published": datetime.utcnow(),
                    # Alpha Vantage provides its own score — use as hint
                    "_av_score": float(item.get("overall_sentiment_score", 0)),
                }
                for item in data
                if item.get("title")
            ]

        except Exception as e:
            log.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return []

    def _fetch_newsapi(self, symbol: str, lookback_hours: int) -> list[dict]:
        """Fetch from NewsAPI.org."""
        try:
            clean = symbol.replace("/USDT", "").replace("/USD", "")
            from_dt = (datetime.utcnow() - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")

            r = requests.get("https://newsapi.org/v2/everything", params={
                "q":       f"{clean} stock OR {clean} crypto OR {clean} market",
                "from":    from_dt,
                "sortBy":  "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey":  self.newsapi_key,
            }, timeout=10)

            if r.status_code != 200:
                return []

            articles = r.json().get("articles", [])
            return [
                {
                    "headline":  a.get("title", ""),
                    "source":    a.get("source", {}).get("name", "NewsAPI"),
                    "url":       a.get("url", ""),
                    "published": datetime.utcnow(),
                }
                for a in articles
                if a.get("title") and "[Removed]" not in a.get("title", "")
            ]

        except Exception as e:
            log.warning(f"NewsAPI fetch failed for {symbol}: {e}")
            return []

    # ── NLP Scoring ───────────────────────────────────────────────────────────
    def _score_text(self, text: str) -> tuple[float, float, str]:
        """
        Score a headline using FinBERT → VADER → keyword fallback.
        Returns (score, confidence, label).
        score: -1.0 to +1.0
        """
        # Try FinBERT first (most accurate for finance)
        if self._finbert is not None:
            return self._score_finbert(text)

        # Try loading FinBERT
        try:
            from transformers import pipeline
            self._finbert = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
                device=-1,    # CPU
            )
            return self._score_finbert(text)
        except Exception:
            pass

        # Fallback: VADER
        try:
            return self._score_vader(text)
        except Exception:
            pass

        # Last resort: keyword scoring
        return self._score_keywords(text)

    def _score_finbert(self, text: str) -> tuple[float, float, str]:
        """Score with FinBERT — best for financial headlines."""
        try:
            result = self._finbert(text[:512])[0]
            scores = {r["label"]: r["score"] for r in result}

            pos = scores.get("positive", 0)
            neg = scores.get("negative", 0)
            neu = scores.get("neutral",  0)

            # Net score: positive - negative, normalised
            net_score = pos - neg
            confidence = max(pos, neg, neu)

            if pos > neg and pos > neu:
                return net_score, confidence, "positive"
            elif neg > pos and neg > neu:
                return net_score, confidence, "negative"
            return 0.0, confidence, "neutral"

        except Exception as e:
            log.warning(f"FinBERT scoring failed: {e}")
            return self._score_keywords(text)

    def _score_vader(self, text: str) -> tuple[float, float, str]:
        """Score with VADER sentiment analyser."""
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()

        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]   # -1 to +1

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        confidence = abs(compound)
        return compound, confidence, label

    def _score_keywords(self, text: str) -> tuple[float, float, str]:
        """Keyword-based fallback scorer."""
        text_lower = text.lower()

        bullish_kw = [
            "surge", "rally", "soar", "gain", "rise", "beat", "record",
            "bullish", "upgrade", "breakout", "strong", "growth", "profit",
            "outperform", "buy", "positive", "optimistic", "recovery",
        ]
        bearish_kw = [
            "crash", "drop", "fall", "plunge", "decline", "miss", "loss",
            "bearish", "downgrade", "breakdown", "weak", "recession", "debt",
            "underperform", "sell", "negative", "concern", "fear", "risk",
        ]

        bull_hits = sum(1 for k in bullish_kw if k in text_lower)
        bear_hits = sum(1 for k in bearish_kw if k in text_lower)
        total     = bull_hits + bear_hits + 1e-10

        score = (bull_hits - bear_hits) / total
        confidence = min(0.7, total / 5)    # Cap at 0.7 — keywords are imprecise

        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        return round(score, 3), round(confidence, 3), label

    # ── Aggregation ───────────────────────────────────────────────────────────
    def _aggregate(
        self, symbol: str, articles: list[NewsItem]
    ) -> NewsSentimentResult:
        """Aggregate article scores into a single sentiment result."""
        if not articles:
            return self._empty_result(symbol)

        now = datetime.utcnow()

        # Recency-weighted average (newer articles count more)
        total_weight = 0.0
        weighted_sum = 0.0

        bullish = bearish = neutral = 0

        for art in articles:
            age_hours = (now - art.published).total_seconds() / 3600
            weight    = max(0.1, 1 / (1 + age_hours / 12))    # Half-life 12 hours

            weighted_sum  += art.sentiment * art.confidence * weight
            total_weight  += art.confidence * weight

            if art.label == "positive":
                bullish += 1
            elif art.label == "negative":
                bearish += 1
            else:
                neutral += 1

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        avg_conf    = sum(a.confidence for a in articles) / len(articles)

        if final_score > 0.15:
            label = "bullish"
        elif final_score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        return NewsSentimentResult(
            symbol=symbol,
            score=round(final_score, 4),
            label=label,
            confidence=round(avg_conf, 3),
            article_count=len(articles),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            articles=articles,
        )

    # ── Cache ─────────────────────────────────────────────────────────────────
    def _get_cache(self, symbol: str) -> NewsSentimentResult | None:
        cached = self._cache.get(symbol)
        if not cached:
            return None
        age = (datetime.utcnow() - cached.timestamp).total_seconds() / 60
        return cached if age < self._cache_ttl_min else None

    def _set_cache(self, symbol: str, result: NewsSentimentResult) -> None:
        self._cache[symbol] = result

    def _empty_result(self, symbol: str) -> NewsSentimentResult:
        return NewsSentimentResult(
            symbol=symbol, score=0.0, label="neutral",
            confidence=0.0, article_count=0,
            bullish_count=0, bearish_count=0, neutral_count=0,
        )


# Singleton
news_sentiment_engine = NewsSentimentEngine()