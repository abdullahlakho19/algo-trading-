"""
sentiment/finbert_analyzer.py
─────────────────────────────────────────────────────────────────────────────
FinBERT Analyzer — Purpose-built Financial NLP.
FinBERT is a BERT model fine-tuned on 10,000+ financial news articles.
It understands financial language FAR better than general NLP models.

Model: ProsusAI/finbert (HuggingFace — completely free)
Accuracy on financial headlines: ~85% vs ~65% for VADER

Usage:
  - Standalone text scoring
  - Batch scoring of headline arrays
  - Confidence-filtered scoring
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class FinBERTScore:
    """Single text sentiment score from FinBERT."""
    text:       str
    positive:   float    # Probability 0-1
    negative:   float
    neutral:    float
    label:      str      # Dominant class
    net_score:  float    # positive - negative (-1 to +1)
    confidence: float    # Max class probability


class FinBERTAnalyzer:
    """
    FinBERT-based financial text sentiment analyzer.
    Lazy-loads the model on first use (downloads ~400MB once).
    Caches model in memory after first load.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self._pipeline = None
        self._loaded   = False

    def load(self) -> bool:
        """
        Load FinBERT model. Downloads on first call (~400MB).
        Returns True if successful.
        """
        if self._loaded:
            return True
        try:
            from transformers import pipeline
            log.info("Loading FinBERT model (first time: ~400MB download)...")
            self._pipeline = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                return_all_scores=True,
                device=-1,      # CPU — set to 0 for GPU if available
                truncation=True,
                max_length=512,
            )
            self._loaded = True
            log.info("FinBERT model loaded successfully.")
            return True
        except Exception as e:
            log.error(f"FinBERT load failed: {e}")
            log.info("Install with: pip install transformers torch")
            return False

    # ── Single Text Scoring ───────────────────────────────────────────────────
    def score(self, text: str) -> Optional[FinBERTScore]:
        """
        Score a single financial text string.

        Args:
            text: Headline or sentence to analyse

        Returns:
            FinBERTScore or None if model not available
        """
        if not self._loaded and not self.load():
            return None

        try:
            result = self._pipeline(text[:512])[0]
            scores = {r["label"]: r["score"] for r in result}

            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)

            label      = max(scores, key=scores.get)
            net_score  = pos - neg
            confidence = max(pos, neg, neu)

            return FinBERTScore(
                text=text[:100],
                positive=round(pos, 4),
                negative=round(neg, 4),
                neutral=round(neu, 4),
                label=label,
                net_score=round(net_score, 4),
                confidence=round(confidence, 4),
            )

        except Exception as e:
            log.warning(f"FinBERT scoring error: {e}")
            return None

    # ── Batch Scoring ─────────────────────────────────────────────────────────
    def score_batch(
        self,
        texts:          list[str],
        min_confidence: float = 0.6,
    ) -> list[FinBERTScore]:
        """
        Score multiple texts efficiently.
        Filters out low-confidence results.

        Args:
            texts:          List of headlines/sentences
            min_confidence: Skip results below this confidence

        Returns:
            List of FinBERTScore objects
        """
        if not self._loaded and not self.load():
            return []

        results = []
        # Batch in chunks of 16 for memory efficiency
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = [t[:512] for t in texts[i:i + batch_size]]
            try:
                batch_results = self._pipeline(batch)
                for text, result in zip(batch, batch_results):
                    scores = {r["label"]: r["score"] for r in result}
                    pos  = scores.get("positive", 0.0)
                    neg  = scores.get("negative", 0.0)
                    neu  = scores.get("neutral",  0.0)
                    conf = max(pos, neg, neu)

                    if conf >= min_confidence:
                        results.append(FinBERTScore(
                            text=text[:100],
                            positive=round(pos, 4),
                            negative=round(neg, 4),
                            neutral=round(neu, 4),
                            label=max(scores, key=scores.get),
                            net_score=round(pos - neg, 4),
                            confidence=round(conf, 4),
                        ))
            except Exception as e:
                log.warning(f"Batch scoring error: {e}")

        return results

    # ── Aggregate Scoring ─────────────────────────────────────────────────────
    def aggregate_score(
        self,
        texts:          list[str],
        weights:        list[float] = None,
    ) -> dict:
        """
        Score multiple texts and return aggregated sentiment.
        Optionally weight each text (e.g. by upvotes or recency).

        Returns:
            dict with aggregate score, label, confidence, and breakdown
        """
        scored = self.score_batch(texts)
        if not scored:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0, "count": 0}

        w = weights[:len(scored)] if weights else [1.0] * len(scored)
        total_w = sum(w) + 1e-10

        weighted_score = sum(s.net_score * wi for s, wi in zip(scored, w)) / total_w
        avg_confidence = np.mean([s.confidence for s in scored])

        positive_pct = sum(1 for s in scored if s.label == "positive") / len(scored)
        negative_pct = sum(1 for s in scored if s.label == "negative") / len(scored)

        if weighted_score > 0.15:
            label = "bullish"
        elif weighted_score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "score":        round(float(weighted_score), 4),
            "label":        label,
            "confidence":   round(float(avg_confidence), 4),
            "count":        len(scored),
            "positive_pct": round(positive_pct, 3),
            "negative_pct": round(negative_pct, 3),
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton
finbert_analyzer = FinBERTAnalyzer()