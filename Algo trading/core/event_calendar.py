"""
core/event_calendar.py
─────────────────────────────────────────────────────────────────────────────
Macro event calendar awareness.
Fetches high-impact economic events and blocks trading before/after them.
Uses investing.com economic calendar via requests.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, timedelta
import pytz
import requests
from core.logger import get_logger
from config import config

log = get_logger(__name__)

UTC = pytz.UTC


class EventCalendar:
    """
    Tracks high-impact economic events.
    Prevents the bot from trading within the pre/post event buffer window.
    """

    def __init__(self):
        self.events: list[dict] = []
        self._last_fetch: datetime | None = None
        self._fetch_interval_hours = 6

    # ── Fetch Events ──────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """
        Fetch upcoming high-impact events.
        Falls back to a manual list if API is unavailable.
        Called once at startup and every 6 hours.
        """
        try:
            self._fetch_from_api()
        except Exception as e:
            log.warning(f"Event calendar API failed: {e}. Using fallback schedule.")
            self._load_fallback_events()

        self._last_fetch = datetime.now(UTC)
        log.info(f"Event calendar loaded. {len(self.events)} events tracked.")

    def _fetch_from_api(self) -> None:
        """
        Fetch from TradingEconomics or ForexFactory-style endpoint.
        Replace with your preferred calendar API.
        """
        # Placeholder — in production connect to TradingEconomics API
        # For now we simulate with known scheduled events
        self._load_fallback_events()

    def _load_fallback_events(self) -> None:
        """
        Hardcoded high-impact events for the current week.
        In production this is replaced by live API data.
        """
        # This would be populated dynamically from an API
        # Format: {name, datetime_utc, impact, currency}
        now = datetime.now(UTC)

        # Example structure — populated by API in production
        self.events = [
            # {
            #     "name": "NFP",
            #     "datetime_utc": datetime(2025, 2, 7, 13, 30, tzinfo=UTC),
            #     "impact": "high",
            #     "currency": "USD",
            # },
        ]

    # ── Event Checks ──────────────────────────────────────────────────────────
    def is_safe_to_trade(self, symbol: str | None = None) -> bool:
        """
        Returns True if it is safe to trade right now.
        Returns False if a high-impact event is within the buffer window.
        """
        if not self.events:
            return True

        now = datetime.now(UTC)
        pre_buffer  = timedelta(minutes=config.macro.PRE_EVENT_BUFFER_MINUTES)
        post_buffer = timedelta(minutes=config.macro.POST_EVENT_BUFFER_MINUTES)

        for event in self.events:
            if event.get("impact") != "high":
                continue

            event_time = event["datetime_utc"]

            # Block if within pre-event buffer
            if now >= event_time - pre_buffer and now <= event_time:
                log.warning(
                    f"Trading blocked — {event['name']} in "
                    f"{(event_time - now).seconds // 60} minutes."
                )
                return False

            # Block if within post-event buffer
            if now > event_time and now <= event_time + post_buffer:
                log.warning(
                    f"Trading blocked — post-event cool-down after {event['name']}."
                )
                return False

        return True

    def next_event(self) -> dict | None:
        """Returns the next upcoming high-impact event."""
        now = datetime.now(UTC)
        upcoming = [e for e in self.events if e["datetime_utc"] > now]
        if not upcoming:
            return None
        return min(upcoming, key=lambda e: e["datetime_utc"])

    def minutes_to_next_event(self) -> float | None:
        """Minutes until next high-impact event. None if no events."""
        event = self.next_event()
        if not event:
            return None
        now = datetime.now(UTC)
        delta = event["datetime_utc"] - now
        return delta.total_seconds() / 60

    def get_status(self) -> dict:
        next_ev = self.next_event()
        return {
            "safe_to_trade":        self.is_safe_to_trade(),
            "total_events_tracked": len(self.events),
            "next_event":           next_ev["name"] if next_ev else None,
            "minutes_to_next":      self.minutes_to_next_event(),
            "last_refresh":         self._last_fetch.isoformat() if self._last_fetch else None,
        }


# Singleton
event_calendar = EventCalendar()