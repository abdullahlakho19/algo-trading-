"""
strategies/missed_trade.py
─────────────────────────────────────────────────────────────────────────────
Missed Trade Protocol — Core Discipline Rule.
If the entry price has moved beyond the valid zone, the trade is
CANCELLED. The bot never chases price. Ever.

"A missed trade is not a lost trade. A chased trade often is." 
─────────────────────────────────────────────────────────────────────────────
"""

from core.logger import get_logger

log = get_logger(__name__)


class MissedTradeProtocol:
    """
    Enforces the no-chase rule.
    Validates that price is still within the valid entry zone
    before execution. If missed → cancel, log, move on.
    """

    def __init__(self, max_slippage_pct: float = 0.002):
        """
        Args:
            max_slippage_pct: Max acceptable distance from ideal entry (0.2%)
        """
        self.max_slippage_pct = max_slippage_pct
        self._missed_count    = 0
        self._chased_count    = 0   # Always 0 — we never chase

    def is_entry_valid(
        self,
        signal_price:  float,    # Price when signal was generated
        current_price: float,    # Current live price
        direction:     str,      # "long" | "short"
    ) -> tuple[bool, str]:
        """
        Check if the current price is still within the valid entry zone.

        Returns:
            (is_valid, reason)
        """
        if signal_price <= 0 or current_price <= 0:
            return False, "invalid_prices"

        deviation = (current_price - signal_price) / signal_price

        if direction == "long":
            # For long: if price has moved UP too much, we missed the entry
            if deviation > self.max_slippage_pct:
                self._missed_count += 1
                reason = (
                    f"Entry missed: price moved up {deviation:.3%} from signal. "
                    f"Max allowed: {self.max_slippage_pct:.3%}. NO CHASE."
                )
                log.warning(f"⏭️ MISSED TRADE | {reason}")
                return False, reason

        elif direction == "short":
            # For short: if price has moved DOWN too much, we missed the entry
            if deviation < -self.max_slippage_pct:
                self._missed_count += 1
                reason = (
                    f"Entry missed: price moved down {abs(deviation):.3%} from signal. "
                    f"Max allowed: {self.max_slippage_pct:.3%}. NO CHASE."
                )
                log.warning(f"⏭️ MISSED TRADE | {reason}")
                return False, reason

        return True, "entry_valid"

    def log_skip(self, symbol: str, reason: str) -> None:
        """Log that a trade was skipped — not a loss, just skipped."""
        self._missed_count += 1
        log.info(f"Trade skipped: {symbol} | {reason} | Total missed: {self._missed_count}")

    @property
    def missed_count(self) -> int:
        return self._missed_count

    def get_status(self) -> dict:
        return {
            "missed_trades":  self._missed_count,
            "chased_trades":  0,   # Always zero — we never chase
            "max_slip_pct":   self.max_slippage_pct,
        }


# Singleton
missed_trade_protocol = MissedTradeProtocol()