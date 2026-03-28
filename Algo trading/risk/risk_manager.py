"""
risk/risk_manager.py
─────────────────────────────────────────────────────────────────────────────
Portfolio-Level Risk Management Engine.
Combines position sizing, correlation checking, circuit breakers,
and pre-trade validation into a single risk gateway.

Every trade MUST pass through this module before execution.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class TradeProposal:
    """A proposed trade before risk approval."""
    symbol:        str
    direction:     str       # "long" | "short"
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    probability:   float     # From signal engine (0-1)
    market:        str       # "stock" | "forex" | "crypto"
    timeframe:     str
    signal_source: str = ""


@dataclass
class RiskApproval:
    """Result of risk evaluation for a trade proposal."""
    approved:          bool
    position_size:     float      # Units / contracts / lots
    position_value:    float      # Dollar value of position
    risk_amount:       float      # Dollar amount at risk
    risk_pct:          float      # % of portfolio at risk
    reward_amount:     float      # Potential dollar reward
    risk_reward:       float      # Risk/reward ratio
    rejection_reasons: list[str] = field(default_factory=list)
    warnings:          list[str]  = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        reasons = " | ".join(self.rejection_reasons) if self.rejection_reasons else "—"
        return (
            f"{status} | Size: {self.position_size:.4f} | "
            f"Risk: ${self.risk_amount:.2f} ({self.risk_pct:.2%}) | "
            f"R/R: 1:{self.risk_reward:.1f} | {reasons}"
        )


class CircuitBreaker:
    """
    Auto-pauses the bot when loss thresholds are breached.
    Protects capital from runaway drawdowns.
    """

    def __init__(self):
        self.daily_loss_limit   = config.risk.DAILY_LOSS_LIMIT
        self.weekly_loss_limit  = config.risk.WEEKLY_LOSS_LIMIT
        self._daily_pnl:  float = 0.0
        self._weekly_pnl: float = 0.0
        self._paused:     bool  = False
        self._pause_reason: str = ""

    def update_pnl(self, pnl: float) -> None:
        """Update running P&L. Called after each trade closes."""
        self._daily_pnl  += pnl
        self._weekly_pnl += pnl
        self._check_limits()

    def _check_limits(self) -> None:
        if self._daily_pnl < -self.daily_loss_limit:
            self._paused = True
            self._pause_reason = (
                f"Daily loss limit breached: {self._daily_pnl:.2%}"
            )
            log.warning(f"🔴 CIRCUIT BREAKER TRIGGERED: {self._pause_reason}")

        elif self._weekly_pnl < -self.weekly_loss_limit:
            self._paused = True
            self._pause_reason = (
                f"Weekly loss limit breached: {self._weekly_pnl:.2%}"
            )
            log.warning(f"🔴 CIRCUIT BREAKER TRIGGERED: {self._pause_reason}")

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_pnl = 0.0
        if self._paused and "Daily" in self._pause_reason:
            self._paused = False
            self._pause_reason = ""
            log.info("Circuit breaker reset for new day.")

    def reset_weekly(self) -> None:
        """Call at start of each trading week."""
        self._weekly_pnl = 0.0
        self._paused = False
        self._pause_reason = ""
        log.info("Circuit breaker reset for new week.")

    @property
    def is_active(self) -> bool:
        return self._paused

    @property
    def reason(self) -> str:
        return self._pause_reason

    def get_status(self) -> dict:
        return {
            "paused":       self._paused,
            "reason":       self._pause_reason,
            "daily_pnl":    self._daily_pnl,
            "weekly_pnl":   self._weekly_pnl,
            "daily_limit":  -self.daily_loss_limit,
            "weekly_limit": -self.weekly_loss_limit,
        }


class RiskManager:
    """
    Portfolio-level risk management gateway.
    Every trade proposal must pass through evaluate() before execution.
    """

    def __init__(self, portfolio_value: float = 100_000.0):
        self.portfolio_value     = portfolio_value
        self.max_risk_per_trade  = config.risk.MAX_RISK_PER_TRADE
        self.max_portfolio_risk  = config.risk.MAX_PORTFOLIO_RISK
        self.max_open_positions  = config.risk.MAX_OPEN_POSITIONS
        self.min_risk_reward     = config.risk.MIN_RISK_REWARD
        self.max_correlation     = config.risk.MAX_CORRELATION

        self.circuit_breaker     = CircuitBreaker()
        self.open_positions:     list[dict] = []
        self._total_risk_used:   float = 0.0

    # ── Main Gate ─────────────────────────────────────────────────────────────
    def evaluate(self, proposal: TradeProposal) -> RiskApproval:
        """
        Full risk evaluation of a trade proposal.
        Returns RiskApproval — if not approved, trade is rejected.

        This is the FINAL gate before execution.
        """
        rejection_reasons = []
        warnings = []

        # ── Gate 1: Circuit Breaker ────────────────────────────────────────
        if self.circuit_breaker.is_active:
            return RiskApproval(
                approved=False,
                position_size=0, position_value=0,
                risk_amount=0, risk_pct=0,
                reward_amount=0, risk_reward=0,
                rejection_reasons=[f"Circuit breaker active: {self.circuit_breaker.reason}"],
            )

        # ── Gate 2: Max Open Positions ─────────────────────────────────────
        if len(self.open_positions) >= self.max_open_positions:
            rejection_reasons.append(
                f"Max open positions reached: {len(self.open_positions)}"
            )

        # ── Gate 3: Compute Risk/Reward ────────────────────────────────────
        sl_distance = abs(proposal.entry_price - proposal.stop_loss)
        tp_distance = abs(proposal.take_profit - proposal.entry_price)

        if sl_distance == 0:
            return RiskApproval(
                approved=False, position_size=0, position_value=0,
                risk_amount=0, risk_pct=0, reward_amount=0, risk_reward=0,
                rejection_reasons=["Stop-loss distance is zero"],
            )

        rr_ratio = tp_distance / sl_distance

        if rr_ratio < self.min_risk_reward:
            rejection_reasons.append(
                f"R/R too low: 1:{rr_ratio:.1f} (min 1:{self.min_risk_reward})"
            )

        # ── Gate 4: Position Sizing ────────────────────────────────────────
        risk_per_trade_dollar = self.portfolio_value * self.max_risk_per_trade
        position_size         = risk_per_trade_dollar / sl_distance

        # Adjust for volatility
        if config.risk.VOLATILITY_SCALE:
            position_size = self._volatility_scale(position_size, proposal)

        position_value = position_size * proposal.entry_price
        risk_amount    = position_size * sl_distance
        risk_pct       = risk_amount / self.portfolio_value
        reward_amount  = position_size * tp_distance

        # ── Gate 5: Portfolio Risk Cap ─────────────────────────────────────
        projected_risk = self._total_risk_used + risk_pct
        if projected_risk > self.max_portfolio_risk:
            rejection_reasons.append(
                f"Portfolio risk cap exceeded: {projected_risk:.2%} > {self.max_portfolio_risk:.2%}"
            )

        # ── Gate 6: Correlation Check ──────────────────────────────────────
        corr_warning = self._check_correlation(proposal)
        if corr_warning:
            warnings.append(corr_warning)
            position_size *= 0.5   # Halve size for correlated trades
            log.warning(f"Correlated trade — size halved: {proposal.symbol}")

        approved = len(rejection_reasons) == 0

        approval = RiskApproval(
            approved=approved,
            position_size=round(position_size, 6),
            position_value=round(position_value, 2),
            risk_amount=round(risk_amount, 2),
            risk_pct=round(risk_pct, 4),
            reward_amount=round(reward_amount, 2),
            risk_reward=round(rr_ratio, 2),
            rejection_reasons=rejection_reasons,
            warnings=warnings,
        )

        log.info(f"{proposal.symbol} | {approval.summary()}")

        if approved:
            self._total_risk_used += risk_pct

        return approval

    # ── Position Management ───────────────────────────────────────────────────
    def register_open_position(self, trade: dict) -> None:
        """Register a newly opened position."""
        self.open_positions.append(trade)

    def close_position(self, symbol: str, pnl: float) -> None:
        """Close a position and update circuit breaker."""
        self.open_positions = [p for p in self.open_positions if p["symbol"] != symbol]
        risk_freed = pnl / self.portfolio_value
        self._total_risk_used = max(0.0, self._total_risk_used - risk_freed)
        self.circuit_breaker.update_pnl(pnl / self.portfolio_value)

    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value (called after each trade or regularly)."""
        self.portfolio_value = value

    # ── SL/TP Calculation ─────────────────────────────────────────────────────
    def calculate_sl_tp(
        self,
        entry: float,
        direction: str,
        atr: float,
        atr_mult_sl: float = 1.5,
        atr_mult_tp: float = 3.0,
    ) -> tuple[float, float]:
        """
        Calculate mathematically placed stop-loss and take-profit using ATR.

        Args:
            entry:       Entry price
            direction:   "long" or "short"
            atr:         Current Average True Range
            atr_mult_sl: ATR multiplier for stop-loss
            atr_mult_tp: ATR multiplier for take-profit (must give >= 1:2 R/R)

        Returns:
            (stop_loss, take_profit)
        """
        if direction == "long":
            sl = entry - atr * atr_mult_sl
            tp = entry + atr * atr_mult_tp
        else:
            sl = entry + atr * atr_mult_sl
            tp = entry - atr * atr_mult_tp

        return round(sl, 6), round(tp, 6)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _volatility_scale(self, size: float, proposal: TradeProposal) -> float:
        """Reduce position size in high-volatility environments."""
        # Simplified — in production uses actual IV data
        return size * 0.8 if proposal.market == "crypto" else size

    def _check_correlation(self, proposal: TradeProposal) -> str | None:
        """
        Check if this trade is correlated with existing open positions.
        Returns a warning string if correlated, None if clean.
        """
        open_symbols = [p["symbol"] for p in self.open_positions]
        open_markets = [p.get("market") for p in self.open_positions]

        # Simple heuristic: same market = potentially correlated
        if proposal.market in open_markets and len(open_markets) >= 2:
            return (
                f"Potential correlation with existing {proposal.market} positions. "
                f"Size reduced."
            )
        return None

    # ── Status ────────────────────────────────────────────────────────────────
    def get_status(self) -> dict:
        return {
            "portfolio_value":   self.portfolio_value,
            "open_positions":    len(self.open_positions),
            "total_risk_used":   round(self._total_risk_used, 4),
            "available_risk":    round(self.max_portfolio_risk - self._total_risk_used, 4),
            "circuit_breaker":   self.circuit_breaker.get_status(),
        }


# Singleton — initialised with default portfolio value
# Update with actual account balance at runtime
risk_manager = RiskManager(portfolio_value=100_000.0)