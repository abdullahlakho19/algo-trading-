"""
risk/circuit_breaker.py
─────────────────────────────────────────────────────────────────────────────
Circuit Breaker — Standalone Risk Protection Module.
Automatically pauses all trading when loss thresholds are breached.
Protects capital from runaway drawdowns and model failures.

Triggers:
  - Daily loss > 3%    → pause until next trading day
  - Weekly loss > 7%   → pause until next week
  - Consecutive losses > 5 → pause for 2 hours
  - Drawdown from peak > 10% → pause until manually reset
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class CircuitBreakerStatus:
    """Current circuit breaker state."""
    is_tripped:         bool
    reason:             str
    tripped_at:         datetime | None
    resumes_at:         datetime | None
    daily_pnl_pct:      float
    weekly_pnl_pct:     float
    consecutive_losses: int
    peak_equity:        float
    current_drawdown:   float
    trades_today:       int


class CircuitBreaker:
    """
    Multi-level circuit breaker protecting the trading agent from
    catastrophic losses and model failures.
    """

    def __init__(self):
        # Thresholds from config
        self.daily_limit        = config.risk.DAILY_LOSS_LIMIT      # 3%
        self.weekly_limit       = config.risk.WEEKLY_LOSS_LIMIT     # 7%
        self.max_consec_losses  = 5
        self.max_drawdown       = 0.10   # 10% from peak

        # State
        self._is_tripped        = False
        self._reason            = ""
        self._tripped_at:       datetime | None = None
        self._resumes_at:       datetime | None = None

        # Running P&L trackers
        self._daily_pnl:        float = 0.0
        self._weekly_pnl:       float = 0.0
        self._consecutive_losses: int = 0
        self._peak_equity:      float = 0.0
        self._current_equity:   float = 0.0
        self._trades_today:     int   = 0

        # Reset schedule
        self._last_daily_reset: datetime = datetime.utcnow().replace(hour=0, minute=0)
        self._last_weekly_reset: datetime = datetime.utcnow()

    # ── Core Methods ──────────────────────────────────────────────────────────
    def record_trade(self, pnl: float, equity: float) -> None:
        """
        Record a completed trade P&L and check all thresholds.
        Call after every trade closes.
        """
        self._auto_reset()

        pnl_pct = pnl / max(equity, 1)

        self._daily_pnl  += pnl_pct
        self._weekly_pnl += pnl_pct
        self._trades_today += 1

        # Track equity peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
        self._current_equity = equity

        # Consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check all breakers
        self._check_all(equity)

    def _check_all(self, equity: float) -> None:
        """Check all circuit breaker conditions."""
        if self._is_tripped:
            return

        # Daily loss limit
        if self._daily_pnl < -self.daily_limit:
            self._trip(
                reason=f"Daily loss limit: {self._daily_pnl:.2%} (limit {-self.daily_limit:.2%})",
                resume_in_hours=self._hours_until_day_reset(),
            )
            return

        # Weekly loss limit
        if self._weekly_pnl < -self.weekly_limit:
            self._trip(
                reason=f"Weekly loss limit: {self._weekly_pnl:.2%} (limit {-self.weekly_limit:.2%})",
                resume_in_hours=self._hours_until_week_reset(),
            )
            return

        # Consecutive losses
        if self._consecutive_losses >= self.max_consec_losses:
            self._trip(
                reason=f"{self._consecutive_losses} consecutive losses — model likely malfunctioning",
                resume_in_hours=2,
            )
            return

        # Drawdown from peak
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd >= self.max_drawdown:
                self._trip(
                    reason=f"Max drawdown: {dd:.2%} from peak ${self._peak_equity:,.2f}",
                    resume_in_hours=24,
                )

    def _trip(self, reason: str, resume_in_hours: float = 24) -> None:
        """Trip the circuit breaker."""
        self._is_tripped  = True
        self._reason      = reason
        self._tripped_at  = datetime.utcnow()
        self._resumes_at  = datetime.utcnow() + timedelta(hours=resume_in_hours)

        log.critical(
            f"\n{'='*60}\n"
            f"🔴 CIRCUIT BREAKER TRIPPED\n"
            f"   Reason:  {reason}\n"
            f"   Resumes: {self._resumes_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"{'='*60}"
        )

    # ── Auto-Reset Logic ──────────────────────────────────────────────────────
    def _auto_reset(self) -> None:
        """Check if timed resets should fire."""
        now = datetime.utcnow()

        # Check if resume time has passed (timed auto-reset)
        if self._is_tripped and self._resumes_at and now >= self._resumes_at:
            self._is_tripped = False
            self._reason     = ""
            log.info("Circuit breaker auto-reset after pause window.")

        # Daily reset at 00:00 UTC
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._last_daily_reset < today_midnight:
            self._daily_pnl         = 0.0
            self._consecutive_losses = 0
            self._trades_today       = 0
            self._last_daily_reset   = today_midnight
            # Auto-reset daily-triggered breaker
            if self._is_tripped and "Daily" in self._reason:
                self._is_tripped = False
                log.info("Circuit breaker daily reset.")

        # Weekly reset on Monday 00:00 UTC
        if now.weekday() == 0 and self._last_weekly_reset.weekday() != 0:
            self._weekly_pnl        = 0.0
            self._last_weekly_reset = now
            if self._is_tripped and "Weekly" in self._reason:
                self._is_tripped = False
                log.info("Circuit breaker weekly reset.")

    def manual_reset(self) -> None:
        """Manually reset the circuit breaker. Use with caution."""
        self._is_tripped         = False
        self._reason             = ""
        self._tripped_at         = None
        self._resumes_at         = None
        self._consecutive_losses = 0
        log.warning("Circuit breaker manually reset.")

    # ── Status ────────────────────────────────────────────────────────────────
    @property
    def is_tripped(self) -> bool:
        self._auto_reset()
        return self._is_tripped

    @property
    def reason(self) -> str:
        return self._reason

    def get_status(self) -> CircuitBreakerStatus:
        self._auto_reset()
        dd = 0.0
        if self._peak_equity > 0 and self._current_equity > 0:
            dd = (self._peak_equity - self._current_equity) / self._peak_equity

        return CircuitBreakerStatus(
            is_tripped=self._is_tripped,
            reason=self._reason,
            tripped_at=self._tripped_at,
            resumes_at=self._resumes_at,
            daily_pnl_pct=round(self._daily_pnl, 4),
            weekly_pnl_pct=round(self._weekly_pnl, 4),
            consecutive_losses=self._consecutive_losses,
            peak_equity=round(self._peak_equity, 2),
            current_drawdown=round(dd, 4),
            trades_today=self._trades_today,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _hours_until_day_reset(self) -> float:
        now = datetime.utcnow()
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        return (midnight - now).total_seconds() / 3600

    def _hours_until_week_reset(self) -> float:
        now  = datetime.utcnow()
        days = (7 - now.weekday()) % 7 or 7
        next_mon = (now + timedelta(days=days)).replace(hour=0, minute=0, second=0)
        return (next_mon - now).total_seconds() / 3600


# Singleton
circuit_breaker = CircuitBreaker()