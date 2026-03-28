"""
core/market_clock.py
─────────────────────────────────────────────────────────────────────────────
24/7 market clock — handles scheduling, timing, and heartbeat.
Knows when markets open/close and when sessions overlap.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, time
import pytz
from core.logger import get_logger
from config import config

log = get_logger(__name__)

UTC = pytz.UTC


class MarketClock:
    """
    Central clock for all time-based decisions.
    All times internally handled in UTC.
    """

    def __init__(self):
        self.tz_utc   = pytz.UTC
        self.tz_ny    = pytz.timezone("America/New_York")
        self.tz_lon   = pytz.timezone("Europe/London")
        self.tz_tokyo = pytz.timezone("Asia/Tokyo")

    # ── Current Time ──────────────────────────────────────────────────────────
    def now_utc(self) -> datetime:
        return datetime.now(self.tz_utc)

    def now_ny(self) -> datetime:
        return datetime.now(self.tz_ny)

    def now_london(self) -> datetime:
        return datetime.now(self.tz_lon)

    # ── Session Detection ─────────────────────────────────────────────────────
    def current_sessions(self) -> list[str]:
        """Returns list of currently active sessions."""
        now    = self.now_utc()
        hour   = now.hour
        minute = now.minute
        t      = hour * 60 + minute   # minutes since midnight UTC

        active = []

        # Asian: 00:00–09:00 UTC
        if 0 <= t < 540:
            active.append("asian")

        # London: 08:00–17:00 UTC
        if 480 <= t < 1020:
            active.append("london")

        # New York: 13:00–22:00 UTC
        if 780 <= t < 1320:
            active.append("new_york")

        # Overlap: 13:00–17:00 UTC (London + NY)
        if 780 <= t < 1020:
            active.append("overlap")

        if not active:
            active.append("off_hours")

        return active

    def session_priority(self) -> str:
        """Returns priority level of current market conditions."""
        sessions = self.current_sessions()
        if "overlap" in sessions:
            return "highest"
        if "london" in sessions or "new_york" in sessions:
            return "high"
        if "asian" in sessions:
            return "low"
        return "closed"

    def is_high_priority_session(self) -> bool:
        return self.session_priority() in ("high", "highest")

    # ── Stock Market Hours ────────────────────────────────────────────────────
    def is_us_market_open(self) -> bool:
        """Check if US stock market is currently open (Mon-Fri, 9:30-16:00 ET)."""
        now   = self.now_ny()
        day   = now.weekday()    # 0=Mon, 6=Sun
        if day >= 5:
            return False
        open_time  = time(9, 30)
        close_time = time(16, 0)
        return open_time <= now.time() <= close_time

    def is_weekend(self) -> bool:
        return self.now_utc().weekday() >= 5

    # ── Forex Hours ───────────────────────────────────────────────────────────
    def is_forex_open(self) -> bool:
        """
        Forex trades 24/5 — closed from Friday 22:00 UTC to Sunday 22:00 UTC.
        """
        now = self.now_utc()
        day = now.weekday()
        hour = now.hour

        # Friday after 22:00 UTC
        if day == 4 and hour >= 22:
            return False
        # All of Saturday
        if day == 5:
            return False
        # Sunday before 22:00 UTC
        if day == 6 and hour < 22:
            return False
        return True

    def is_crypto_open(self) -> bool:
        """Crypto is always open — 24/7/365."""
        return True

    # ── Time Until Events ─────────────────────────────────────────────────────
    def minutes_until_ny_open(self) -> float:
        """Minutes until US market opens. Returns 0 if already open."""
        if self.is_us_market_open():
            return 0.0
        now    = self.now_ny()
        open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now >= open_t:
            # Next day open
            from datetime import timedelta
            open_t += timedelta(days=1)
        delta = open_t - now
        return delta.total_seconds() / 60

    # ── Heartbeat ─────────────────────────────────────────────────────────────
    def log_status(self) -> None:
        """Log current clock status — called periodically by the agent."""
        sessions = self.current_sessions()
        priority = self.session_priority()
        log.info(
            f"Clock Status | UTC: {self.now_utc().strftime('%H:%M:%S')} | "
            f"Sessions: {sessions} | Priority: {priority} | "
            f"US Market: {'OPEN' if self.is_us_market_open() else 'CLOSED'} | "
            f"Forex: {'OPEN' if self.is_forex_open() else 'CLOSED'}"
        )

    def get_status(self) -> dict:
        return {
            "utc_time":      self.now_utc().isoformat(),
            "sessions":      self.current_sessions(),
            "priority":      self.session_priority(),
            "us_market":     self.is_us_market_open(),
            "forex_open":    self.is_forex_open(),
            "crypto_open":   self.is_crypto_open(),
            "is_weekend":    self.is_weekend(),
        }


# Singleton
market_clock = MarketClock()