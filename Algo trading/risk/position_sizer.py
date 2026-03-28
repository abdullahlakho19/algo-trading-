"""
risk/position_sizer.py
─────────────────────────────────────────────────────────────────────────────
Position Sizer — Volatility & Correlation Adjusted.
Computes the correct position size for each trade using
multiple institutional sizing methods.

Methods:
  1. Fixed Fractional    — Risk fixed % of capital per trade
  2. ATR-based           — Size based on volatility (ATR)
  3. Kelly Criterion     — Optimal sizing based on win rate & payoff
  4. Volatility Parity   — Equal volatility contribution per position
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class SizeResult:
    """Position sizing output."""
    symbol:         str
    method:         str
    units:          float    # Number of units/shares/contracts
    value:          float    # Dollar value of position
    risk_amount:    float    # Dollar amount at risk
    risk_pct:       float    # % of portfolio at risk
    size_reduction: float    # Reduction factor applied (1.0 = no reduction)
    notes:          str = ""


class PositionSizer:
    """
    Multi-method position sizer.
    Default method: ATR-based fixed fractional (most robust).
    """

    def __init__(self):
        self.max_risk_pct    = config.risk.MAX_RISK_PER_TRADE    # 1%
        self.max_position_pct = 0.20    # Max 20% of portfolio in one position
        self.vol_target      = 0.15     # 15% annualised vol target per position

    # ── Primary Method: ATR-Based ─────────────────────────────────────────────
    def size_atr(
        self,
        symbol:        str,
        entry:         float,
        stop_loss:     float,
        portfolio_val: float,
        atr:           float,
        volatility_adj: bool = True,
    ) -> SizeResult:
        """
        ATR-based position sizing.
        Risk a fixed % of portfolio; size determined by stop distance.
        """
        sl_dist    = abs(entry - stop_loss)
        if sl_dist == 0:
            sl_dist = atr * 1.5

        risk_dollar = portfolio_val * self.max_risk_pct
        units       = risk_dollar / sl_dist

        # Volatility adjustment
        reduction = 1.0
        if volatility_adj:
            atr_norm = atr / entry
            if atr_norm > 0.03:       # High vol — reduce
                reduction = 0.7
            elif atr_norm > 0.05:     # Very high — reduce more
                reduction = 0.5
            elif atr_norm > 0.08:     # Extreme — minimal
                reduction = 0.3
            units *= reduction

        # Cap at max position size
        max_units = (portfolio_val * self.max_position_pct) / entry
        units     = min(units, max_units)

        return SizeResult(
            symbol=symbol,
            method="atr_fixed_fractional",
            units=round(units, 6),
            value=round(units * entry, 2),
            risk_amount=round(units * sl_dist, 2),
            risk_pct=round((units * sl_dist) / portfolio_val, 4),
            size_reduction=reduction,
            notes=f"SL dist: {sl_dist:.5f} | ATR: {atr:.5f}",
        )

    # ── Kelly Criterion ───────────────────────────────────────────────────────
    def size_kelly(
        self,
        symbol:        str,
        entry:         float,
        stop_loss:     float,
        take_profit:   float,
        portfolio_val: float,
        win_rate:      float,
        kelly_fraction: float = 0.25,    # Quarter-Kelly for safety
    ) -> SizeResult:
        """
        Kelly Criterion position sizing.
        Optimal fraction based on win rate and risk/reward.
        """
        sl_dist = abs(entry - stop_loss)
        tp_dist = abs(take_profit - entry)

        if sl_dist == 0:
            return SizeResult(symbol, "kelly", 0, 0, 0, 0, 0, "Zero SL distance")

        rr    = tp_dist / sl_dist
        b     = rr        # Payoff ratio
        p     = win_rate
        q     = 1 - p

        # Kelly formula: f* = (b*p - q) / b
        kelly = (b * p - q) / b
        kelly = max(0, kelly)    # Never negative (don't short)

        # Apply Kelly fraction (quarter-Kelly by default for safety)
        f       = kelly * kelly_fraction
        f       = min(f, self.max_risk_pct * 3)    # Hard cap

        risk_dollar = portfolio_val * f
        units       = risk_dollar / sl_dist

        return SizeResult(
            symbol=symbol,
            method=f"kelly_{kelly_fraction}x",
            units=round(units, 6),
            value=round(units * entry, 2),
            risk_amount=round(units * sl_dist, 2),
            risk_pct=round(f, 4),
            size_reduction=kelly_fraction,
            notes=f"Full Kelly: {kelly:.2%} | Applied: {f:.2%}",
        )

    # ── Volatility Parity ─────────────────────────────────────────────────────
    def size_vol_parity(
        self,
        symbol:         str,
        entry:          float,
        portfolio_val:  float,
        realised_vol:   float,   # Annualised
    ) -> SizeResult:
        """
        Volatility parity sizing — equal risk contribution regardless of volatility.
        """
        if realised_vol <= 0:
            return SizeResult(symbol, "vol_parity", 0, 0, 0, 0, 0, "Zero volatility")

        # Target: each position contributes equally to portfolio vol
        target_vol_dollar = portfolio_val * self.vol_target
        pos_value         = target_vol_dollar / realised_vol
        pos_value         = min(pos_value, portfolio_val * self.max_position_pct)
        units             = pos_value / entry

        return SizeResult(
            symbol=symbol,
            method="vol_parity",
            units=round(units, 6),
            value=round(pos_value, 2),
            risk_amount=round(pos_value * realised_vol / np.sqrt(252), 2),
            risk_pct=round(pos_value / portfolio_val, 4),
            size_reduction=1.0,
            notes=f"Realised vol: {realised_vol:.2%}",
        )

    # ── Correlation Adjustment ────────────────────────────────────────────────
    def apply_correlation_reduction(
        self,
        size_result:    SizeResult,
        correlation:    float,
    ) -> SizeResult:
        """
        Reduce size when correlated positions already exist.
        correlation: 0.0 (uncorrelated) to 1.0 (perfectly correlated)
        """
        if correlation < config.risk.MAX_CORRELATION:
            return size_result

        # Reduce size proportionally to correlation
        reduction = 1 - (correlation - 0.5)   # 0.7 corr → 0.8x, 1.0 corr → 0.5x
        reduction = max(0.3, reduction)

        size_result.units         *= reduction
        size_result.value         *= reduction
        size_result.risk_amount   *= reduction
        size_result.risk_pct      *= reduction
        size_result.size_reduction *= reduction
        size_result.notes         += f" | Corr reduction: {reduction:.2f}x"
        return size_result

    # ── Convenience ───────────────────────────────────────────────────────────
    def size(
        self,
        symbol:        str,
        entry:         float,
        stop_loss:     float,
        take_profit:   float,
        portfolio_val: float,
        atr:           float,
        win_rate:      float = 0.65,
        correlation:   float = 0.0,
    ) -> SizeResult:
        """
        Default sizing — uses ATR method then applies Kelly cap and correlation adj.
        """
        result = self.size_atr(symbol, entry, stop_loss, portfolio_val, atr)
        kelly  = self.size_kelly(symbol, entry, stop_loss, take_profit,
                                 portfolio_val, win_rate)

        # Take the minimum of ATR and Kelly size (conservative)
        if kelly.units > 0:
            result.units        = min(result.units, kelly.units)
            result.value        = round(result.units * entry, 2)
            result.risk_amount  = round(result.units * abs(entry - stop_loss), 2)
            result.risk_pct     = round(result.risk_amount / portfolio_val, 4)
            result.notes       += f" | Kelly cap applied"

        # Correlation adjustment
        if correlation > 0:
            result = self.apply_correlation_reduction(result, correlation)

        log.debug(
            f"Size | {symbol} | Units: {result.units:.4f} | "
            f"Value: ${result.value:,.2f} | Risk: {result.risk_pct:.2%}"
        )
        return result


# Singleton
position_sizer = PositionSizer()