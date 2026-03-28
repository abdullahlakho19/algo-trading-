"""
core/session_detector.py
─────────────────────────────────────────────────────────────────────────────
Trading Session Detector.
Identifies which global market sessions are active and provides
session-specific trading guidance for each instrument type.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, time
import pytz
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)

UTC     = pytz.UTC
NY_TZ   = pytz.timezone("America/New_York")
LON_TZ  = pytz.timezone("Europe/London")
TOK_TZ  = pytz.timezone("Asia/Tokyo")
SYD_TZ  = pytz.timezone("Australia/Sydney")


@dataclass
class SessionInfo:
    """Details about a specific trading session."""
    name:          str
    is_active:     bool
    priority:      str        # "highest" | "high" | "medium" | "low" | "closed"
    local_time:    str        # Current time in that timezone
    opens_in_min:  float      # Minutes until session opens (0 if active)
    closes_in_min: float      # Minutes until session closes (0 if inactive)
    best_pairs:    list[str]  # Most liquid pairs in this session
    volatility:    str        # Expected volatility level


@dataclass
class SessionSnapshot:
    """Complete snapshot of all active sessions."""
    utc_time:       str
    active_sessions: list[str]
    priority:       str
    asian:          SessionInfo
    london:         SessionInfo
    new_york:       SessionInfo
    overlap:        SessionInfo
    sydney:         SessionInfo
    recommended_action: str
    best_instruments:   list[str] = field(default_factory=list)


class SessionDetector:
    """
    Detects and characterises all active trading sessions.
    Provides actionable guidance on when and what to trade.
    """

    # Session windows in UTC (hour, minute)
    SESSIONS = {
        "sydney":   {"open": (21, 0),  "close": (6,  0),  "next_day_close": True},
        "asian":    {"open": (0,  0),  "close": (9,  0),  "next_day_close": False},
        "london":   {"open": (8,  0),  "close": (17, 0),  "next_day_close": False},
        "new_york": {"open": (13, 0),  "close": (22, 0),  "next_day_close": False},
        "overlap":  {"open": (13, 0),  "close": (17, 0),  "next_day_close": False},  # London/NY
    }

    BEST_PAIRS = {
        "sydney":   ["AUD/USD", "AUD/JPY", "NZD/USD", "AUD/NZD"],
        "asian":    ["USD/JPY", "EUR/JPY", "AUD/JPY", "USD/CNH", "GBP/JPY"],
        "london":   ["EUR/USD", "GBP/USD", "EUR/GBP", "USD/CHF", "EUR/JPY"],
        "new_york": ["EUR/USD", "USD/CAD", "GBP/USD", "USD/JPY", "SPY", "QQQ"],
        "overlap":  ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP"],
    }

    VOLATILITY = {
        "sydney":   "low",
        "asian":    "low-medium",
        "london":   "high",
        "new_york": "high",
        "overlap":  "very_high",
    }

    def get_snapshot(self) -> SessionSnapshot:
        """Get complete session snapshot for right now."""
        now     = datetime.now(UTC)
        utc_str = now.strftime("%H:%M:%S UTC")

        asian   = self._analyse_session("asian",    now)
        london  = self._analyse_session("london",   now)
        new_york= self._analyse_session("new_york", now)
        overlap = self._analyse_session("overlap",  now)
        sydney  = self._analyse_session("sydney",   now)

        active = [s for s in [asian, london, new_york, overlap, sydney]
                  if s.is_active]
        active_names = [s.name for s in active]

        # Priority
        if overlap.is_active:
            priority = "highest"
            action   = "PRIME WINDOW — Highest probability setups. Focus on EUR/USD, GBP/USD, USD/JPY"
        elif london.is_active or new_york.is_active:
            priority = "high"
            action   = "ACTIVE SESSION — Good conditions. Trade with full attention."
        elif asian.is_active:
            priority = "medium"
            action   = "ASIAN SESSION — Lower volatility. Focus on JPY pairs. Tighter ranges."
        elif sydney.is_active:
            priority = "low"
            action   = "SYDNEY SESSION — Quiet markets. AUD/NZD pairs only. Reduce size."
        else:
            priority = "closed"
            action   = "MARKETS CLOSED — No Forex trading. Crypto only. Reduce position sizes."

        # Best instruments right now
        best = []
        if overlap.is_active:
            best = self.BEST_PAIRS["overlap"]
        elif london.is_active:
            best = self.BEST_PAIRS["london"]
        elif new_york.is_active:
            best = self.BEST_PAIRS["new_york"]
        elif asian.is_active:
            best = self.BEST_PAIRS["asian"]
        # Always add crypto (24/7)
        if "BTC/USDT" not in best:
            best = best[:4] + ["BTC/USDT", "ETH/USDT"]

        snapshot = SessionSnapshot(
            utc_time=utc_str,
            active_sessions=active_names,
            priority=priority,
            asian=asian,
            london=london,
            new_york=new_york,
            overlap=overlap,
            sydney=sydney,
            recommended_action=action,
            best_instruments=best,
        )

        log.debug(
            f"Sessions: {active_names} | Priority: {priority} | "
            f"Best: {best[:3]}"
        )
        return snapshot

    def _analyse_session(self, name: str, now: datetime) -> SessionInfo:
        """Analyse a single session."""
        cfg         = self.SESSIONS[name]
        open_h, open_m  = cfg["open"]
        close_h, close_m= cfg["close"]

        t_min = now.hour * 60 + now.minute

        open_min  = open_h  * 60 + open_m
        close_min = close_h * 60 + close_m

        # Handle sessions that cross midnight (Sydney)
        if cfg.get("next_day_close"):
            is_active = t_min >= open_min or t_min < close_min
        else:
            is_active = open_min <= t_min < close_min

        # Minutes until open / close
        if is_active:
            opens_in = 0.0
            if cfg.get("next_day_close") and t_min >= open_min:
                closes_in = float((24*60 - t_min) + close_min)
            else:
                closes_in = float(close_min - t_min)
        else:
            closes_in = 0.0
            if t_min < open_min:
                opens_in = float(open_min - t_min)
            else:
                opens_in = float((24*60 - t_min) + open_min)

        # Local times
        tz_map = {
            "sydney":   SYD_TZ,
            "asian":    TOK_TZ,
            "london":   LON_TZ,
            "new_york": NY_TZ,
            "overlap":  NY_TZ,
        }
        local_time = now.astimezone(tz_map[name]).strftime("%H:%M")

        priority = self.VOLATILITY.get(name, "medium") if is_active else "closed"

        return SessionInfo(
            name=name,
            is_active=is_active,
            priority=priority,
            local_time=local_time,
            opens_in_min=round(opens_in, 1),
            closes_in_min=round(closes_in, 1),
            best_pairs=self.BEST_PAIRS.get(name, []),
            volatility=self.VOLATILITY.get(name, "medium"),
        )

    def is_high_priority(self) -> bool:
        snap = self.get_snapshot()
        return snap.priority in ("highest", "high")

    def current_sessions(self) -> list[str]:
        snap = self.get_snapshot()
        return snap.active_sessions

    def best_instruments_now(self) -> list[str]:
        return self.get_snapshot().best_instruments

    def minutes_to_london_open(self) -> float:
        now = datetime.now(UTC)
        t   = now.hour * 60 + now.minute
        open_min = 8 * 60
        if t < open_min:
            return float(open_min - t)
        return float((24 * 60 - t) + open_min)

    def minutes_to_overlap(self) -> float:
        now = datetime.now(UTC)
        t   = now.hour * 60 + now.minute
        overlap_min = 13 * 60
        if t < overlap_min:
            return float(overlap_min - t)
        return float((24 * 60 - t) + overlap_min)


# Singleton
session_detector = SessionDetector()