"""
quant/monte_carlo.py
─────────────────────────────────────────────────────────────────────────────
Monte Carlo Simulation Engine.
Runs thousands of scenario simulations to stress-test trades and
compute probabilistic risk/reward outcomes before execution.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class MonteCarloResult:
    """Output of a Monte Carlo simulation run."""
    num_simulations:    int
    expected_return:    float      # Mean expected return
    std_return:         float      # Standard deviation of returns
    var_95:             float      # Value at Risk (95% confidence)
    var_99:             float      # Value at Risk (99% confidence)
    cvar_95:            float      # Conditional VaR (Expected Shortfall)
    prob_profit:        float      # Probability of positive return
    prob_loss_gt_sl:    float      # Probability of loss exceeding stop-loss
    max_simulated_gain: float
    max_simulated_loss: float
    passed:             bool       # True if trade passes risk criteria

    def summary(self) -> str:
        return (
            f"MC({self.num_simulations} runs) | "
            f"E[R]={self.expected_return:.2%} | "
            f"VaR95={self.var_95:.2%} | "
            f"P(profit)={self.prob_profit:.1%} | "
            f"{'✓ PASSED' if self.passed else '✗ FAILED'}"
        )


class MonteCarloEngine:
    """
    Monte Carlo simulation for trade risk assessment.
    
    Runs N simulations of possible price paths and computes
    the probability distribution of outcomes.
    """

    def __init__(self):
        self.num_simulations = config.monte_carlo.NUM_SIMULATIONS
        self.confidence      = config.monte_carlo.CONFIDENCE_LEVEL
        self.time_horizon    = config.monte_carlo.TIME_HORIZON_DAYS

    # ── Pre-Trade Simulation ──────────────────────────────────────────────────
    def simulate_trade(
        self,
        entry_price:  float,
        stop_loss:    float,
        take_profit:  float,
        df:           pd.DataFrame,
        hold_candles: int = 20,
    ) -> MonteCarloResult:
        """
        Simulate the outcome of a specific trade N times.

        Args:
            entry_price:  Planned entry price
            stop_loss:    Stop loss level
            take_profit:  Take profit level
            df:           Historical OHLCV (for return distribution)
            hold_candles: Max candles to hold position

        Returns:
            MonteCarloResult with full probability breakdown
        """
        returns = self._compute_log_returns(df)
        mu      = float(returns.mean())
        sigma   = float(returns.std())

        sl_pct = (stop_loss - entry_price) / entry_price    # Negative for long
        tp_pct = (take_profit - entry_price) / entry_price  # Positive for long

        outcomes = []
        for _ in range(self.num_simulations):
            outcome = self._simulate_path(mu, sigma, sl_pct, tp_pct, hold_candles)
            outcomes.append(outcome)

        outcomes = np.array(outcomes)
        return self._compute_stats(outcomes, sl_pct, tp_pct)

    def _simulate_path(
        self,
        mu: float,
        sigma: float,
        sl_pct: float,
        tp_pct: float,
        hold_candles: int,
    ) -> float:
        """
        Simulate one price path.
        Returns the return (%) when SL/TP hit or time expires.
        """
        cumulative = 0.0
        for _ in range(hold_candles):
            daily_return = np.random.normal(mu, sigma)
            cumulative  += daily_return

            # Check stop-loss hit
            if sl_pct < 0 and cumulative <= sl_pct:
                return sl_pct
            if sl_pct > 0 and cumulative >= sl_pct:
                return sl_pct

            # Check take-profit hit
            if tp_pct > 0 and cumulative >= tp_pct:
                return tp_pct
            if tp_pct < 0 and cumulative <= tp_pct:
                return tp_pct

        return cumulative

    def _compute_stats(
        self,
        outcomes: np.ndarray,
        sl_pct: float,
        tp_pct: float,
    ) -> MonteCarloResult:
        """Compute statistics from simulation outcomes."""
        sorted_outcomes = np.sort(outcomes)
        n = len(outcomes)

        var_95_idx  = int(n * (1 - self.confidence))
        var_99_idx  = int(n * 0.01)

        var_95  = float(sorted_outcomes[var_95_idx])
        var_99  = float(sorted_outcomes[var_99_idx])
        cvar_95 = float(sorted_outcomes[:var_95_idx].mean()) if var_95_idx > 0 else var_95

        prob_profit   = float((outcomes > 0).mean())
        prob_loss_sl  = float((outcomes <= sl_pct * 1.1).mean())

        # Pass criteria:
        # - Probability of profit > 50%
        # - VaR 95 loss < 3x stop loss
        # - Probability of hitting SL < 40%
        passed = (
            prob_profit > 0.50
            and var_95 > sl_pct * 3
            and prob_loss_sl < 0.40
        )

        return MonteCarloResult(
            num_simulations=self.num_simulations,
            expected_return=float(outcomes.mean()),
            std_return=float(outcomes.std()),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            prob_profit=prob_profit,
            prob_loss_gt_sl=prob_loss_sl,
            max_simulated_gain=float(outcomes.max()),
            max_simulated_loss=float(outcomes.min()),
            passed=passed,
        )

    # ── Portfolio Simulation ──────────────────────────────────────────────────
    def simulate_portfolio(
        self,
        returns_history: pd.DataFrame,
        num_days: int = 252,
    ) -> dict:
        """
        Simulate full portfolio equity curve over num_days.
        returns_history: DataFrame of daily returns per asset.
        """
        port_returns = returns_history.mean(axis=1)
        mu    = float(port_returns.mean())
        sigma = float(port_returns.std())

        all_paths = []
        for _ in range(self.num_simulations):
            path = [1.0]
            for _ in range(num_days):
                r = np.random.normal(mu, sigma)
                path.append(path[-1] * (1 + r))
            all_paths.append(path)

        paths_arr = np.array(all_paths)
        final_vals = paths_arr[:, -1]

        return {
            "mean_final_value":  float(final_vals.mean()),
            "median_final_value": float(np.median(final_vals)),
            "var_95_portfolio":  float(np.percentile(final_vals, 5)),
            "prob_profit_port":  float((final_vals > 1.0).mean()),
            "worst_case":        float(final_vals.min()),
            "best_case":         float(final_vals.max()),
            "num_days":          num_days,
            "num_simulations":   self.num_simulations,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _compute_log_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns from close prices."""
        return np.log(df["close"] / df["close"].shift(1)).dropna()


# Singleton
monte_carlo_engine = MonteCarloEngine()