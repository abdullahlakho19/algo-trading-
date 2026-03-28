"""
reporting/trade_logger.py
─────────────────────────────────────────────────────────────────────────────
Trade Logger.
Maintains a real-time structured audit trail of every
trade decision — entry, exit, P&L, and reason.
─────────────────────────────────────────────────────────────────────────────
"""

import json
from datetime import datetime
from pathlib import Path
from core.logger import get_logger
from config import config

log = get_logger(__name__)


class TradeLogger:
    """Logs every trade event to structured JSON and human-readable formats."""

    def __init__(self):
        self.log_dir = config.paths.logs
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log: list[dict] = []

    def log_signal(self, signal) -> None:
        """Log when a signal is generated."""
        entry = {
            "event":       "signal_generated",
            "timestamp":   datetime.utcnow().isoformat(),
            "symbol":      signal.symbol,
            "direction":   signal.direction,
            "probability": signal.probability,
            "entry_price": signal.entry_price,
            "sl":          signal.stop_loss,
            "tp":          signal.take_profit,
            "timeframe":   signal.timeframe,
            "regime":      signal.regime,
        }
        self._append(entry)
        log.bind(TRADE=True).info(f"SIGNAL | {signal.summary()}")

    def log_trade_open(self, symbol: str, direction: str, qty: float,
                       entry: float, sl: float, tp: float) -> None:
        entry_rec = {
            "event":     "trade_opened",
            "timestamp": datetime.utcnow().isoformat(),
            "symbol":    symbol, "direction": direction,
            "qty": qty, "entry": entry, "sl": sl, "tp": tp,
        }
        self._append(entry_rec)
        log.bind(TRADE=True).info(
            f"OPENED | {symbol} {direction.upper()} {qty:.4f} @ {entry:.5f}"
        )

    def log_trade_close(self, trade) -> None:
        rec = {
            "event":      "trade_closed",
            "timestamp":  datetime.utcnow().isoformat(),
            "symbol":     trade.symbol,
            "direction":  trade.direction,
            "pnl":        trade.pnl,
            "outcome":    trade.outcome,
            "exit_reason": trade.exit_reason,
        }
        self._append(rec)
        emoji = "✅" if trade.outcome == "win" else "❌"
        log.bind(TRADE=True).info(
            f"CLOSED {emoji} | {trade.symbol} | P&L: ${trade.pnl:.2f} | "
            f"{trade.exit_reason.upper()}"
        )

    def log_rejection(self, symbol: str, reason: str) -> None:
        self._append({
            "event":     "trade_rejected",
            "timestamp": datetime.utcnow().isoformat(),
            "symbol":    symbol, "reason": reason,
        })

    def _append(self, record: dict) -> None:
        self._trade_log.append(record)
        # Write to JSON file
        today    = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"trades_{today}.json"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            log.warning(f"Trade log write failed: {e}")

    def get_recent(self, n: int = 20) -> list[dict]:
        return self._trade_log[-n:]

    def get_all(self) -> list[dict]:
        return self._trade_log.copy()


# Singleton
trade_logger = TradeLogger()