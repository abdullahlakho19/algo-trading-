"""
strategies/confirmation.py
─────────────────────────────────────────────────────────────────────────────
Multi-Confluence Signal Confirmation Engine.
Aggregates signals from all intelligence layers and confirms
that a minimum number of independent signals agree.

Jane Street rule: "No trade without proven confluence."
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class ConfirmationResult:
    """Output of multi-confluence check."""
    passed:       bool
    confluence:   int
    required:     int
    signals:      dict = field(default_factory=dict)   # name → bool
    direction:    str  = "neutral"
    summary:      str  = ""


class ConfirmationEngine:
    """
    Checks that a minimum number of independent signals confirm
    the same directional bias before allowing trade execution.

    Signal sources checked:
    1. Market Structure (BoS/CHoCH)
    2. Volume Profile (price at key level)
    3. Market Regime (trending in right direction)
    4. Momentum Convergence (indicators aligned)
    5. Order Flow (aggressive side confirms)
    6. Multi-TF Alignment (macro aligns with entry)
    7. Session Priority (high-priority window)
    """

    def __init__(self):
        self.required = config.signal.MIN_CONFLUENCE

    def check(
        self,
        direction:        str,
        structure_signal: str,    # "bullish" | "bearish" | "neutral"
        vp_at_level:      bool,
        regime_ok:        bool,
        momentum_signal:  str,
        order_flow:       str,
        mtf_aligned:      bool,
        session_priority: str,
    ) -> ConfirmationResult:
        """
        Run all confluence checks and return pass/fail result.
        """
        is_long = direction == "long"

        signals = {
            "market_structure":  (structure_signal == "bullish" and is_long) or
                                 (structure_signal == "bearish" and not is_long),
            "volume_profile":    vp_at_level,
            "market_regime":     regime_ok,
            "momentum":          (momentum_signal == "bullish" and is_long) or
                                 (momentum_signal == "bearish" and not is_long),
            "order_flow":        (order_flow == "bullish" and is_long) or
                                 (order_flow == "bearish" and not is_long),
            "mtf_alignment":     mtf_aligned,
            "session":           session_priority in ("high", "highest"),
        }

        confluence = sum(1 for v in signals.values() if v)
        passed     = confluence >= self.required

        active  = [k for k, v in signals.items() if v]
        missing = [k for k, v in signals.items() if not v]

        summary = (
            f"✅ {confluence}/{self.required} confirmations | "
            f"Active: {active}" if passed else
            f"❌ Only {confluence}/{self.required} | Missing: {missing}"
        )

        if passed:
            log.info(f"Confirmation PASSED | {direction} | {summary}")
        else:
            log.debug(f"Confirmation FAILED | {direction} | {summary}")

        return ConfirmationResult(
            passed=passed,
            confluence=confluence,
            required=self.required,
            signals=signals,
            direction=direction,
            summary=summary,
        )


# Singleton
confirmation_engine = ConfirmationEngine()