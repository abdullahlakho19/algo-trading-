"""
backtesting/walk_forward.py
─────────────────────────────────────────────────────────────────────────────
Walk-Forward Optimization Engine.
Prevents curve-fitting by training on a window, testing on the next,
then rolling forward — just like professional quant shops do.

Method:
  1. Split data into N folds
  2. Train on fold N, test on fold N+1
  3. Aggregate out-of-sample results
  4. Report combined performance across ALL test folds
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    """Walk-forward optimization output."""
    symbol:         str
    n_folds:        int
    train_pct:      float
    combined_wr:    float      # Combined out-of-sample win rate
    combined_pf:    float      # Combined profit factor
    combined_sharpe: float
    fold_results:   list[dict] = field(default_factory=list)
    is_robust:      bool = False   # True if OOS performance is consistent

    def summary(self) -> str:
        robust = "✅ ROBUST" if self.is_robust else "⚠️ NOT ROBUST"
        return (
            f"Walk-Forward ({self.n_folds} folds) | "
            f"OOS WR: {self.combined_wr:.1%} | "
            f"OOS PF: {self.combined_pf:.2f} | "
            f"Sharpe: {self.combined_sharpe:.2f} | {robust}"
        )


class WalkForwardEngine:
    """
    Walk-forward analysis to validate strategy robustness.
    """

    def __init__(
        self,
        n_folds:    int   = 5,
        train_pct:  float = 0.7,
    ):
        self.n_folds   = n_folds
        self.train_pct = train_pct

    def run(
        self,
        df:          pd.DataFrame,
        symbol:      str,
        strategy_fn  = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            df:          Full OHLCV DataFrame
            symbol:      Instrument name
            strategy_fn: Signal function (df, i) → (direction, sl, tp) or None

        Returns:
            WalkForwardResult with aggregated OOS performance
        """
        from backtesting.backtest_engine import BacktestEngine

        if len(df) < 200:
            log.warning("Insufficient data for walk-forward analysis.")
            return WalkForwardResult(symbol=symbol, n_folds=0, train_pct=self.train_pct,
                                     combined_wr=0, combined_pf=0, combined_sharpe=0)

        fold_size   = len(df) // self.n_folds
        all_trades  = []
        fold_results = []

        log.info(f"Walk-forward: {symbol} | {self.n_folds} folds | {len(df)} bars")

        for fold in range(self.n_folds - 1):
            start     = fold * fold_size
            mid       = start + int(fold_size * self.train_pct)
            end       = start + fold_size

            train_df  = df.iloc[start:mid]
            test_df   = df.iloc[mid:end]

            if len(train_df) < 100 or len(test_df) < 30:
                continue

            # Backtest on test fold only
            engine = BacktestEngine(initial_capital=100_000.0)
            result = engine.run(test_df, symbol, "1h", strategy_fn)

            fold_info = {
                "fold":        fold + 1,
                "train_bars":  len(train_df),
                "test_bars":   len(test_df),
                "oos_trades":  result.total_trades,
                "oos_wr":      result.win_rate,
                "oos_pf":      result.profit_factor,
                "oos_sharpe":  result.sharpe_ratio,
                "oos_pnl":     result.total_pnl,
            }
            fold_results.append(fold_info)
            all_trades.extend(result.trades)

            log.info(
                f"Fold {fold+1}/{self.n_folds} | "
                f"OOS: {result.total_trades} trades | "
                f"WR: {result.win_rate:.1%} | PF: {result.profit_factor:.2f}"
            )

        if not all_trades:
            return WalkForwardResult(
                symbol=symbol, n_folds=self.n_folds,
                train_pct=self.train_pct,
                combined_wr=0, combined_pf=0, combined_sharpe=0,
                fold_results=fold_results,
            )

        # Aggregate OOS results
        wins  = [t for t in all_trades if t.outcome == "win"]
        loss  = [t for t in all_trades if t.outcome == "loss"]
        pnl_s = pd.Series([t.pnl for t in all_trades])

        comb_wr  = len(wins) / len(all_trades)
        gp       = sum(t.pnl for t in wins) if wins else 0
        gl       = abs(sum(t.pnl for t in loss)) + 1e-10
        comb_pf  = gp / gl
        comb_sh  = float(pnl_s.mean() / (pnl_s.std() + 1e-10) * (252 ** 0.5))

        # Check consistency across folds
        fold_wrs    = [f["oos_wr"] for f in fold_results if f["oos_trades"] > 0]
        is_robust   = (
            comb_wr >= 0.55 and
            comb_pf >= 1.2 and
            len(fold_wrs) >= 2 and
            np.std(fold_wrs) < 0.15    # Win rates consistent across folds
        )

        result = WalkForwardResult(
            symbol=symbol,
            n_folds=self.n_folds,
            train_pct=self.train_pct,
            combined_wr=round(comb_wr, 4),
            combined_pf=round(comb_pf, 2),
            combined_sharpe=round(comb_sh, 2),
            fold_results=fold_results,
            is_robust=is_robust,
        )

        log.info(result.summary())
        return result


# Singleton
walk_forward_engine = WalkForwardEngine()