"""
backtesting/performance_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Performance Analyzer.
Computes institutional-grade performance metrics from any trade history.
Used for both backtest evaluation and live paper trading assessment.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class PerformanceReport:
    """Complete performance report."""
    total_trades:    int
    win_rate:        float
    loss_rate:       float
    profit_factor:   float
    sharpe_ratio:    float
    sortino_ratio:   float
    calmar_ratio:    float
    max_drawdown:    float
    max_drawdown_dur: int     # Bars in max drawdown
    avg_win:         float
    avg_loss:        float
    win_loss_ratio:  float
    expectancy:      float    # Expected return per trade
    total_pnl:       float
    total_pnl_pct:   float
    best_trade:      float
    worst_trade:     float
    avg_bars_held:   float
    consecutive_wins:  int    # Max consecutive wins
    consecutive_losses: int   # Max consecutive losses
    grade:           str      # A / B / C / D / F

    def summary(self) -> str:
        return (
            f"Grade: {self.grade} | "
            f"WR: {self.win_rate:.1%} | "
            f"PF: {self.profit_factor:.2f} | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"MaxDD: {self.max_drawdown:.2%} | "
            f"P&L: ${self.total_pnl:,.2f}"
        )


class PerformanceAnalyzer:
    """Computes full performance analytics from trade history."""

    def analyse(
        self,
        trades: list,
        initial_capital: float = 100_000.0,
    ) -> PerformanceReport:
        """
        Compute full performance report.
        trades: list of ClosedTrade or BacktestTrade objects with .pnl attribute
        """
        if not trades:
            log.warning("No trades to analyse.")
            return self._empty_report()

        pnl_list = [t.pnl for t in trades]
        pnl_s    = pd.Series(pnl_list)

        wins   = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        win_rate      = len(wins) / len(pnl_list)
        avg_win       = np.mean(wins)  if wins   else 0
        avg_loss      = abs(np.mean(losses)) if losses else 1e-10
        win_loss_ratio = avg_win / avg_loss

        gross_profit  = sum(wins)  if wins   else 0
        gross_loss    = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        # Expectancy (expected $ per trade)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Equity curve
        equity = initial_capital + pnl_s.cumsum()

        # Sharpe (annualised, assuming daily trades)
        sharpe = float(pnl_s.mean() / (pnl_s.std() + 1e-10) * (252 ** 0.5))

        # Sortino (only penalises downside)
        downside = pnl_s[pnl_s < 0].std() + 1e-10
        sortino  = float(pnl_s.mean() / downside * (252 ** 0.5))

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        max_dd      = float(drawdown.min())

        # Max drawdown duration
        is_dd    = drawdown < 0
        dd_dur   = 0
        max_dur  = 0
        for in_dd in is_dd:
            if in_dd:
                dd_dur += 1
                max_dur = max(max_dur, dd_dur)
            else:
                dd_dur = 0

        # Calmar
        annual_return = float(pnl_s.sum() / initial_capital)
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Consecutive streaks
        max_cons_win  = self._max_streak(pnl_list, positive=True)
        max_cons_loss = self._max_streak(pnl_list, positive=False)

        # Average bars held
        bars_held = [getattr(t, "bars_held", getattr(t, "duration_min", 1)) for t in trades]
        avg_bars  = np.mean(bars_held)

        # Grade
        grade = self._grade(win_rate, profit_factor, sharpe, max_dd)

        total_pnl     = float(pnl_s.sum())
        total_pnl_pct = total_pnl / initial_capital

        report = PerformanceReport(
            total_trades=len(trades),
            win_rate=round(win_rate, 4),
            loss_rate=round(1 - win_rate, 4),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown=round(max_dd, 4),
            max_drawdown_dur=max_dur,
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            win_loss_ratio=round(win_loss_ratio, 2),
            expectancy=round(expectancy, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 4),
            best_trade=round(max(pnl_list), 2),
            worst_trade=round(min(pnl_list), 2),
            avg_bars_held=round(avg_bars, 1),
            consecutive_wins=max_cons_win,
            consecutive_losses=max_cons_loss,
            grade=grade,
        )

        log.info(f"Performance: {report.summary()}")
        return report

    def _max_streak(self, pnl_list: list, positive: bool) -> int:
        max_s, cur = 0, 0
        for p in pnl_list:
            if (positive and p > 0) or (not positive and p < 0):
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s

    def _grade(
        self, win_rate: float, pf: float, sharpe: float, max_dd: float
    ) -> str:
        score = sum([
            win_rate  >= 0.75,
            win_rate  >= 0.60,
            pf        >= 2.0,
            pf        >= 1.5,
            sharpe    >= 2.0,
            sharpe    >= 1.5,
            abs(max_dd) <= 0.05,
            abs(max_dd) <= 0.10,
        ])
        if score >= 7: return "A"
        if score >= 5: return "B"
        if score >= 3: return "C"
        if score >= 1: return "D"
        return "F"

    def _empty_report(self) -> PerformanceReport:
        return PerformanceReport(
            total_trades=0, win_rate=0, loss_rate=0, profit_factor=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_dur=0, avg_win=0, avg_loss=0,
            win_loss_ratio=0, expectancy=0, total_pnl=0, total_pnl_pct=0,
            best_trade=0, worst_trade=0, avg_bars_held=0,
            consecutive_wins=0, consecutive_losses=0, grade="F",
        )


# Singleton
performance_analyzer = PerformanceAnalyzer()