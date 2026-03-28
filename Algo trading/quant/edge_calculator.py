"""
quant/edge_calculator.py
─────────────────────────────────────────────────────────────────────────────
Statistical Edge Calculator — Renaissance Technologies philosophy.
A strategy only gets traded if it has a PROVEN mathematical edge.
Hope is not a strategy. Math is.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class EdgeResult:
    """Statistical edge measurement output."""
    has_edge:      bool
    edge_score:    float       # Expected value per trade (positive = edge)
    win_rate:      float
    avg_win:       float
    avg_loss:      float
    expectancy:    float       # Kelly-style expected return
    t_statistic:   float       # t-test statistic
    p_value:       float       # Statistical significance
    is_significant: bool       # p < 0.05
    sample_size:   int
    sharpe:        float
    notes:         str = ""

    def summary(self) -> str:
        sig = "✅ SIGNIFICANT" if self.is_significant else "⚠️ NOT SIGNIFICANT"
        edge = "✅ EDGE EXISTS" if self.has_edge else "❌ NO EDGE"
        return (
            f"{edge} | {sig} | "
            f"E[R]={self.expectancy:.4f} | "
            f"WR={self.win_rate:.1%} | "
            f"p={self.p_value:.4f} | n={self.sample_size}"
        )


class EdgeCalculator:
    """
    Measures statistical edge from a trade sample.
    Uses t-test to determine if returns are significantly positive.
    """

    def __init__(self, min_samples: int = 30, min_p_value: float = 0.05):
        self.min_samples = min_samples
        self.min_p_value = min_p_value

    def calculate(self, returns: list[float]) -> EdgeResult:
        """
        Calculate statistical edge from a list of trade returns.

        Args:
            returns: List of trade returns as decimals (e.g. 0.02 = 2%)

        Returns:
            EdgeResult with significance test and edge metrics
        """
        if len(returns) < self.min_samples:
            log.warning(f"Insufficient samples: {len(returns)} < {self.min_samples}")
            return self._no_edge(f"Only {len(returns)} samples (min {self.min_samples})")

        r = np.array(returns)

        wins   = r[r > 0]
        losses = r[r < 0]

        win_rate = len(wins)  / len(r)
        avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        # Expectancy (expected return per trade)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # One-sample t-test: is mean return significantly > 0?
        t_stat, p_value = stats.ttest_1samp(r, popmean=0)

        # One-tailed test (we only care if mean is positive)
        p_one_tail     = p_value / 2 if t_stat > 0 else 1.0
        is_significant = p_one_tail < self.min_p_value and t_stat > 0

        # Sharpe
        sharpe = float(r.mean() / (r.std() + 1e-10) * (252 ** 0.5))

        has_edge = (
            is_significant
            and expectancy > 0
            and win_rate > 0.5
            and sharpe > 0.5
        )

        edge_score = expectancy * (1 if is_significant else 0.5)

        result = EdgeResult(
            has_edge=has_edge,
            edge_score=round(edge_score, 6),
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 6),
            avg_loss=round(avg_loss, 6),
            expectancy=round(expectancy, 6),
            t_statistic=round(float(t_stat), 4),
            p_value=round(float(p_one_tail), 6),
            is_significant=is_significant,
            sample_size=len(r),
            sharpe=round(sharpe, 2),
        )

        log.info(f"Edge calculation: {result.summary()}")
        return result

    def calculate_from_trades(self, trades: list) -> EdgeResult:
        """Calculate edge from a list of trade objects with .pnl_pct attribute."""
        returns = [t.pnl_pct for t in trades if hasattr(t, "pnl_pct")]
        return self.calculate(returns)

    def _no_edge(self, reason: str) -> EdgeResult:
        return EdgeResult(
            has_edge=False, edge_score=0, win_rate=0,
            avg_win=0, avg_loss=0, expectancy=0,
            t_statistic=0, p_value=1.0, is_significant=False,
            sample_size=0, sharpe=0, notes=reason,
        )


# Singleton
edge_calculator = EdgeCalculator()