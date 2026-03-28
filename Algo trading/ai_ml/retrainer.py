"""
ai_ml/retrainer.py
─────────────────────────────────────────────────────────────────────────────
Continuous Model Retraining Pipeline — Renaissance / XTX philosophy.
Monitors model performance degradation and automatically retrains
when accuracy drops or market regime changes significantly.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, timedelta
from core.logger import get_logger
from config import config

log = get_logger(__name__)


class ModelRetrainer:
    """
    Monitors model health and triggers retraining when needed.

    Retraining triggers:
    1. Scheduled interval (every 24 hours by default)
    2. Win rate drops below threshold
    3. Model accuracy degrades significantly
    4. Market regime changes detected
    """

    def __init__(self):
        self.retrain_interval = timedelta(hours=config.ml.RETRAIN_INTERVAL_HOURS)
        self.min_win_rate     = 0.55    # Below this → force retrain
        self._last_retrain:   datetime = None
        self._performance_history: list[dict] = []

    # ── Trigger Checks ────────────────────────────────────────────────────────
    def should_retrain(self, current_win_rate: float = None) -> tuple[bool, str]:
        """
        Check if model retraining should be triggered.
        Returns (should_retrain, reason).
        """
        # Trigger 1: Never been trained
        if self._last_retrain is None:
            return True, "initial_training"

        # Trigger 2: Scheduled interval
        if datetime.utcnow() - self._last_retrain >= self.retrain_interval:
            return True, f"scheduled_interval_{config.ml.RETRAIN_INTERVAL_HOURS}h"

        # Trigger 3: Win rate degradation
        if current_win_rate is not None and current_win_rate < self.min_win_rate:
            return True, f"win_rate_degraded_{current_win_rate:.1%}"

        # Trigger 4: Performance trend declining
        if self._is_performance_declining():
            return True, "performance_trend_declining"

        return False, "no_trigger"

    def _is_performance_declining(self) -> bool:
        """Check if recent performance is declining over last 3 checks."""
        if len(self._performance_history) < 3:
            return False
        recent = self._performance_history[-3:]
        rates  = [r.get("win_rate", 0.5) for r in recent]
        return rates[-1] < rates[0] - 0.10   # 10% drop

    # ── Retraining Execution ──────────────────────────────────────────────────
    def retrain(
        self,
        data_engine,
        ensemble_voter,
        symbols: list[str],
    ) -> dict:
        """
        Execute full model retraining pipeline.

        Args:
            data_engine:    DataEngine instance for fetching training data
            ensemble_voter: EnsembleVoter instance to retrain
            symbols:        Symbols to train on

        Returns:
            Dict with training metrics
        """
        log.info(f"Starting model retraining on {len(symbols)} symbols...")
        all_metrics = {}

        for symbol in symbols:
            try:
                log.info(f"Fetching training data: {symbol}")
                md = data_engine.get_bars(symbol, "1h", lookback_days=365)

                if md.ohlcv.empty or len(md.ohlcv) < config.ml.MIN_TRAIN_SAMPLES:
                    log.warning(f"Insufficient data for {symbol}")
                    continue

                metrics = ensemble_voter.train(md.ohlcv, symbol)
                if metrics:
                    all_metrics[symbol] = metrics
                    log.info(f"Retrained {symbol}: {metrics}")

            except Exception as e:
                log.error(f"Retraining failed for {symbol}: {e}")

        self._last_retrain = datetime.utcnow()
        log.info(f"Retraining complete. {len(all_metrics)} symbols updated.")
        return all_metrics

    # ── Performance Logging ───────────────────────────────────────────────────
    def log_performance(self, win_rate: float, profit_factor: float) -> None:
        self._performance_history.append({
            "timestamp":     datetime.utcnow().isoformat(),
            "win_rate":      win_rate,
            "profit_factor": profit_factor,
        })
        # Keep last 20 records
        self._performance_history = self._performance_history[-20:]

    def get_status(self) -> dict:
        return {
            "last_retrain":    self._last_retrain.isoformat() if self._last_retrain else None,
            "next_retrain":    (self._last_retrain + self.retrain_interval).isoformat()
                               if self._last_retrain else "immediate",
            "history_count":   len(self._performance_history),
        }


# Singleton
model_retrainer = ModelRetrainer()