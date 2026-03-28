"""
quant/probability_scorer.py
─────────────────────────────────────────────────────────────────────────────
Probability Scorer — Jane Street philosophy.
Every trade is a probability problem, not a prediction.
Combines all signals into a single probability score (0–1).
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class ProbabilityScore:
    """Final probability score for a trade setup."""
    symbol:        str
    direction:     str
    final_score:   float     # 0.0 – 1.0
    passed:        bool
    components:    dict = field(default_factory=dict)
    confluence:    int  = 0   # Number of confirming signals
    rejection:     str  = ""

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{self.symbol} | {self.direction} | "
            f"Score: {self.final_score:.1%} | "
            f"Confluence: {self.confluence} | {status}"
        )


class ProbabilityScorer:
    """
    Combines multiple signal sources into a weighted probability score.

    Components & Weights:
        Ensemble ML signal      30%
        Market structure (BoS)  20%
        Volume profile level    15%
        Market regime           15%
        Multi-TF alignment      10%
        Session priority        10%
    """

    WEIGHTS = {
        "ensemble_ml":      0.30,
        "market_structure": 0.20,
        "volume_profile":   0.15,
        "market_regime":    0.15,
        "mtf_alignment":    0.10,
        "session":          0.10,
    }

    def score(
        self,
        ensemble_prob:       float,
        structure_confidence: float,
        regime_confidence:   float,
        vp_confluence:       float,
        mtf_alignment:       float,
        session_priority:    str,
        direction:           str,
        symbol:              str,
    ) -> ProbabilityScore:
        """
        Compute final probability score from all components.

        All input values should be in range 0.0 – 1.0 except:
            session_priority: "highest" | "high" | "low" | "off_hours"
            direction:        "long" | "short"
        """
        session_score = {
            "highest":   1.0,
            "high":      0.75,
            "low":       0.40,
            "off_hours": 0.20,
            "closed":    0.0,
        }.get(session_priority, 0.5)

        components = {
            "ensemble_ml":      ensemble_prob,
            "market_structure": structure_confidence,
            "volume_profile":   vp_confluence,
            "market_regime":    regime_confidence,
            "mtf_alignment":    abs(mtf_alignment),
            "session":          session_score,
        }

        # Weighted sum
        final_score = sum(
            components[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS
        )
        final_score = round(min(1.0, max(0.0, final_score)), 4)

        # Count confluences (components above 0.6)
        confluence = sum(1 for v in components.values() if v >= 0.6)

        passed = (
            final_score >= config.signal.MIN_PROBABILITY
            and confluence >= config.signal.MIN_CONFLUENCE
        )

        rejection = ""
        if not passed:
            if final_score < config.signal.MIN_PROBABILITY:
                rejection = f"Score too low: {final_score:.1%} < {config.signal.MIN_PROBABILITY:.0%}"
            elif confluence < config.signal.MIN_CONFLUENCE:
                rejection = f"Confluence too low: {confluence} < {config.signal.MIN_CONFLUENCE}"

        result = ProbabilityScore(
            symbol=symbol,
            direction=direction,
            final_score=final_score,
            passed=passed,
            components=components,
            confluence=confluence,
            rejection=rejection,
        )

        log.info(result.summary())
        return result


# Singleton
probability_scorer = ProbabilityScorer()