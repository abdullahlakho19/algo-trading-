"""
intelligence/correlation_matrix.py
─────────────────────────────────────────────────────────────────────────────
Cross-Asset Correlation Matrix Engine — BlackRock / JP Morgan philosophy.
Tracks real-time correlations between all instruments.
Used by the risk manager to prevent over-concentration in correlated trades.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from core.logger import get_logger
from config import config

log = get_logger(__name__)


class CorrelationMatrix:
    """
    Computes and maintains a live rolling correlation matrix
    across all instruments in the universe.
    """

    def __init__(self, window: int = 30):
        self.window   = window
        self._returns: dict[str, pd.Series] = {}
        self._matrix:  pd.DataFrame = pd.DataFrame()

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, symbol: str, df: pd.DataFrame) -> None:
        """Update correlation data for one instrument."""
        if df.empty or len(df) < self.window:
            return
        self._returns[symbol] = df["close"].pct_change().dropna().tail(self.window * 2)

    def update_all(self, universe: dict[str, pd.DataFrame]) -> None:
        """Update all instruments at once."""
        for symbol, df in universe.items():
            self.update(symbol, df)
        self._recompute()

    def _recompute(self) -> None:
        """Recompute the full correlation matrix."""
        if len(self._returns) < 2:
            return
        df = pd.DataFrame(self._returns)
        df = df.dropna()
        if len(df) >= self.window:
            self._matrix = df.tail(self.window).corr()

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_correlation(self, sym_a: str, sym_b: str) -> float | None:
        """Return correlation between two symbols (0 = uncorrelated, 1 = perfect)."""
        if self._matrix.empty:
            return None
        if sym_a not in self._matrix or sym_b not in self._matrix:
            return None
        return round(float(self._matrix.loc[sym_a, sym_b]), 4)

    def get_correlated_pairs(
        self, threshold: float = None
    ) -> list[tuple[str, str, float]]:
        """Return all pairs with correlation above threshold."""
        if self._matrix.empty:
            return []
        threshold = threshold or config.risk.MAX_CORRELATION
        pairs = []
        cols = self._matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = abs(float(self._matrix.iloc[i, j]))
                if corr >= threshold:
                    pairs.append((cols[i], cols[j], round(corr, 4)))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def are_correlated(self, sym_a: str, sym_b: str) -> bool:
        """True if two symbols exceed the correlation threshold."""
        corr = self.get_correlation(sym_a, sym_b)
        if corr is None:
            return False
        return abs(corr) >= config.risk.MAX_CORRELATION

    def get_diversification_score(self, symbols: list[str]) -> float:
        """
        Score 0-1: how diversified is a set of positions?
        1.0 = completely uncorrelated, 0.0 = all correlated.
        """
        if len(symbols) < 2 or self._matrix.empty:
            return 1.0
        corrs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                c = self.get_correlation(symbols[i], symbols[j])
                if c is not None:
                    corrs.append(abs(c))
        if not corrs:
            return 1.0
        avg_corr = np.mean(corrs)
        return round(1 - avg_corr, 3)

    def get_matrix(self) -> pd.DataFrame:
        return self._matrix.copy()

    def summary(self) -> dict:
        return {
            "symbols_tracked": len(self._returns),
            "matrix_ready":    not self._matrix.empty,
            "correlated_pairs": len(self.get_correlated_pairs()),
        }


# Singleton
correlation_matrix = CorrelationMatrix(window=30)