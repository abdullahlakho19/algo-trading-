"""
ai_ml/signal_decay.py
─────────────────────────────────────────────────────────────────────────────
Signal Decay Tracker — Renaissance Technologies philosophy.
Tracks how long each signal type remains valid before it expires.
A signal that was valid 30 minutes ago may no longer be relevant.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class SignalRecord:
    """A recorded signal with its creation time and state."""
    signal_id:    str
    symbol:       str
    direction:    str
    created_at:   datetime
    probability:  float
    timeframe:    str
    is_valid:     bool = True
    expired_at:   datetime = None


class SignalDecayTracker:
    """
    Monitors all active signals and invalidates them when they expire.

    Key rule: If an entry was missed → signal is immediately cancelled.
    Signals also expire after MAX_SIGNAL_AGE_MINUTES regardless.
    """

    def __init__(self):
        self.max_age_minutes = config.signal.MAX_SIGNAL_AGE_MINUTES
        self._signals: dict[str, SignalRecord] = {}

    def register(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        probability: float,
        timeframe: str,
    ) -> None:
        """Register a new signal for decay tracking."""
        self._signals[signal_id] = SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            created_at=datetime.utcnow(),
            probability=probability,
            timeframe=timeframe,
        )
        log.debug(f"Signal registered: {signal_id} | {symbol} | {direction}")

    def is_valid(self, signal_id: str) -> bool:
        """Check if a signal is still within its valid window."""
        record = self._signals.get(signal_id)
        if not record or not record.is_valid:
            return False

        age = (datetime.utcnow() - record.created_at).total_seconds() / 60
        if age > self.max_age_minutes:
            self._expire(signal_id, "age_expired")
            return False

        return True

    def invalidate(self, signal_id: str, reason: str = "missed_entry") -> None:
        """Manually invalidate a signal (e.g. entry price was missed)."""
        self._expire(signal_id, reason)
        log.info(f"Signal {signal_id} invalidated: {reason} — no chase rule applied.")

    def _expire(self, signal_id: str, reason: str) -> None:
        record = self._signals.get(signal_id)
        if record:
            record.is_valid  = False
            record.expired_at = datetime.utcnow()
            log.debug(f"Signal {signal_id} expired: {reason}")

    def cleanup_old(self) -> None:
        """Remove signals older than 2x max age."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.max_age_minutes * 2)
        self._signals = {
            sid: rec for sid, rec in self._signals.items()
            if rec.created_at > cutoff
        }

    def get_active_signals(self) -> list[SignalRecord]:
        self.cleanup_old()
        return [r for r in self._signals.values() if self.is_valid(r.signal_id)]

    def get_status(self) -> dict:
        active = self.get_active_signals()
        return {
            "total_tracked": len(self._signals),
            "active":        len(active),
            "expired":       len(self._signals) - len(active),
            "max_age_min":   self.max_age_minutes,
        }


# Singleton
signal_decay_tracker = SignalDecayTracker()