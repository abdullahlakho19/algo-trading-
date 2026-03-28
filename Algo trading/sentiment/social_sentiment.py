"""
sentiment/social_sentiment.py
─────────────────────────────────────────────────────────────────────────────
Social Media Sentiment Engine.
Analyses Reddit posts/comments from financial subreddits
to gauge retail sentiment and detect crowd momentum shifts.

Sources:
  - Reddit (PRAW — free, no credit card)
    Subreddits: r/wallstreetbets, r/stocks, r/investing,
                r/CryptoCurrency, r/Bitcoin, r/algotrading

Key insight: Social sentiment is a CONTRARIAN indicator at extremes.
Extreme bullish Reddit = institutional distribution incoming.
Extreme bearish Reddit = potential accumulation zone.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class RedditPost:
    """A scored Reddit post."""
    title:      str
    subreddit:  str
    score:      int       # Reddit upvotes
    comments:   int
    sentiment:  float     # -1 to +1
    label:      str
    created:    datetime


@dataclass
class SocialSentimentResult:
    """Aggregated social sentiment result."""
    symbol:         str
    score:          float     # -1.0 to +1.0
    label:          str       # "bullish" | "bearish" | "neutral"
    buzz_score:     float     # 0-1 (how much is this being talked about?)
    contrarian:     bool      # True if sentiment is at extreme (contrarian signal)
    contrarian_direction: str # Direction the contrarian signal points
    post_count:     int
    total_upvotes:  int
    timestamp:      datetime = field(default_factory=datetime.utcnow)

    @property
    def contrarian_signal(self) -> str:
        """
        At extremes, social sentiment is contrarian:
        Extreme bullish retail = likely near a top → short signal
        Extreme bearish retail = likely near a bottom → long signal
        """
        if not self.contrarian:
            return "neutral"
        return self.contrarian_direction


class SocialSentimentEngine:
    """
    Reddit sentiment analysis engine using PRAW.
    """

    # Subreddits by asset class
    SUBREDDITS = {
        "stock":  ["wallstreetbets", "stocks", "investing", "StockMarket"],
        "crypto": ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets"],
        "forex":  ["Forex", "investing", "algotrading"],
        "all":    ["wallstreetbets", "stocks", "CryptoCurrency", "investing"],
    }

    # Extreme sentiment thresholds (contrarian signals)
    EXTREME_BULL  = 0.65
    EXTREME_BEAR  = -0.65

    def __init__(self):
        self.client_id     = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent    = os.getenv("REDDIT_USER_AGENT", "TradingAgent/1.0")
        self._reddit       = None
        self._cache:       dict[str, SocialSentimentResult] = {}
        self._cache_ttl    = 60    # minutes

    # ── Connection ────────────────────────────────────────────────────────────
    def _connect(self) -> bool:
        if self._reddit:
            return True
        if not self.client_id or not self.client_secret:
            log.warning("Reddit API credentials not set. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
            return False
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                read_only=True,
            )
            log.info("Reddit (PRAW) connected.")
            return True
        except Exception as e:
            log.error(f"Reddit connection failed: {e}")
            return False

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self,
        symbol:         str,
        market:         str = "stock",
        lookback_hours: int = 24,
        max_posts:      int = 50,
    ) -> SocialSentimentResult:
        """
        Analyse Reddit sentiment for a symbol.

        Args:
            symbol:         Ticker / pair (e.g. "AAPL", "BTC/USDT")
            market:         "stock" | "crypto" | "forex"
            lookback_hours: Time window to search
            max_posts:      Max posts to analyse
        """
        cached = self._get_cache(symbol)
        if cached:
            return cached

        if not self._connect():
            return self._empty_result(symbol)

        # Clean symbol for search
        clean  = self._clean_symbol(symbol)
        subs   = self.SUBREDDITS.get(market, self.SUBREDDITS["all"])

        posts  = []
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        for sub_name in subs:
            try:
                sub      = self._reddit.subreddit(sub_name)
                # Search recent posts mentioning the symbol
                for post in sub.search(clean, sort="new", time_filter="day", limit=20):
                    created = datetime.utcfromtimestamp(post.created_utc)
                    if created < cutoff:
                        continue
                    if self._mentions_symbol(post.title, clean):
                        posts.append({
                            "title":     post.title,
                            "subreddit": sub_name,
                            "score":     post.score,
                            "comments":  post.num_comments,
                            "created":   created,
                        })
            except Exception as e:
                log.debug(f"Reddit subreddit {sub_name} error: {e}")

        if not posts:
            log.debug(f"No Reddit posts found for {symbol}")
            return self._empty_result(symbol)

        # Score posts
        scored = self._score_posts(posts[:max_posts], symbol)
        result = self._aggregate(symbol, scored)

        self._set_cache(symbol, result)
        log.info(
            f"Social Sentiment | {symbol} | Score: {result.score:.2f} | "
            f"{result.label} | Buzz: {result.buzz_score:.2f} | "
            f"Contrarian: {result.contrarian}"
        )
        return result

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _score_posts(self, posts: list[dict], symbol: str) -> list[RedditPost]:
        """Score each Reddit post for sentiment."""
        scored = []
        for p in posts:
            score, label = self._keyword_score(p["title"])
            scored.append(RedditPost(
                title=p["title"],
                subreddit=p["subreddit"],
                score=p["score"],
                comments=p["comments"],
                sentiment=score,
                label=label,
                created=p["created"],
            ))
        return scored

    def _keyword_score(self, text: str) -> tuple[float, str]:
        """Financial keyword sentiment scorer for social text."""
        text_lower = text.lower()

        bullish = [
            "moon", "mooning", "bullish", "long", "buy", "calls", "breakout",
            "all time high", "ath", "pump", "rip", "squeeze", "diamond hands",
            "hodl", "accumulate", "oversold", "support", "bounce", "strong",
            "earnings beat", "upgrade", "bull run", "rally",
        ]
        bearish = [
            "crash", "dump", "bearish", "short", "puts", "breakdown", "rekt",
            "bubble", "overvalued", "sell", "scared", "worried", "capitulate",
            "overbought", "resistance", "collapse", "fraud", "bankrupt",
            "earnings miss", "downgrade", "bear market", "correction",
        ]

        bull_hits = sum(1 for k in bullish if k in text_lower)
        bear_hits = sum(1 for k in bearish if k in text_lower)
        total     = bull_hits + bear_hits + 1e-10

        score = (bull_hits - bear_hits) / total
        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        return round(score, 3), label

    # ── Aggregation ───────────────────────────────────────────────────────────
    def _aggregate(
        self, symbol: str, posts: list[RedditPost]
    ) -> SocialSentimentResult:
        if not posts:
            return self._empty_result(symbol)

        # Upvote-weighted sentiment
        total_weight = sum(max(1, p.score) for p in posts)
        weighted_sum = sum(p.sentiment * max(1, p.score) for p in posts)
        final_score  = weighted_sum / total_weight if total_weight > 0 else 0.0

        total_upvotes = sum(p.score for p in posts)
        # Buzz score: normalise post activity (0-1)
        buzz_score    = min(1.0, len(posts) / 50 * 0.5 + min(total_upvotes / 10000, 0.5))

        label = "bullish" if final_score > 0.15 else "bearish" if final_score < -0.15 else "neutral"

        # Contrarian check — extreme sentiment flips the signal
        contrarian = abs(final_score) >= self.EXTREME_BULL
        contrarian_dir = "short" if final_score >= self.EXTREME_BULL else \
                         "long"  if final_score <= self.EXTREME_BEAR else "neutral"

        return SocialSentimentResult(
            symbol=symbol,
            score=round(final_score, 4),
            label=label,
            buzz_score=round(buzz_score, 3),
            contrarian=contrarian,
            contrarian_direction=contrarian_dir,
            post_count=len(posts),
            total_upvotes=total_upvotes,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _clean_symbol(self, symbol: str) -> str:
        """Convert symbol to Reddit-searchable string."""
        return symbol.replace("/USDT", "").replace("/USD", "").replace("/", "")

    def _mentions_symbol(self, text: str, symbol: str) -> bool:
        """Check if post actually mentions the symbol."""
        pattern = r'\b' + re.escape(symbol.upper()) + r'\b'
        return bool(re.search(pattern, text.upper()))

    def _get_cache(self, symbol: str) -> SocialSentimentResult | None:
        cached = self._cache.get(symbol)
        if not cached:
            return None
        age = (datetime.utcnow() - cached.timestamp).total_seconds() / 60
        return cached if age < self._cache_ttl else None

    def _set_cache(self, symbol: str, result: SocialSentimentResult) -> None:
        self._cache[symbol] = result

    def _empty_result(self, symbol: str) -> SocialSentimentResult:
        return SocialSentimentResult(
            symbol=symbol, score=0.0, label="neutral",
            buzz_score=0.0, contrarian=False,
            contrarian_direction="neutral", post_count=0, total_upvotes=0,
        )


# Singleton
social_sentiment_engine = SocialSentimentEngine()