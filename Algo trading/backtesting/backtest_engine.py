"""
backtesting/backtest_engine.py
─────────────────────────────────────────────────────────────────────────────
Core Backtesting Engine.
Simulates the trading agent historically to validate strategy
performance before risking any capital.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class BacktestTrade:
    """A trade record from backtesting."""
    symbol:       str
    direction:    str
    entry_price:  float
    exit_price:   float
    stop_loss:    float
    take_profit:  float
    entry_idx:    int
    exit_idx:     int
    pnl:          float
    pnl_pct:      float
    outcome:      str       # "win" | "loss" | "breakeven"
    exit_reason:  str       # "tp" | "sl" | "timeout"
    bars_held:    int


@dataclass
class BacktestResult:
    """Full backtest results."""
    symbol:          str
    timeframe:       str
    total_trades:    int
    wins:            int
    losses:          int
    win_rate:        float
    total_pnl:       float
    total_pnl_pct:   float
    max_drawdown:    float
    sharpe_ratio:    float
    profit_factor:   float
    avg_win:         float
    avg_loss:        float
    avg_bars_held:   float
    trades:          list[BacktestTrade] = field(default_factory=list)
    equity_curve:    list[float] = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    Iterates bar by bar and simulates signal generation + execution.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_per_trade:  float = 0.01,
        max_hold_bars:   int   = 50,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade  = risk_per_trade
        self.max_hold_bars   = max_hold_bars

    # ── Main Backtest ─────────────────────────────────────────────────────────
    def run(
        self,
        df:        pd.DataFrame,
        symbol:    str,
        timeframe: str = "1h",
        strategy_fn = None,
    ) -> BacktestResult:
        """
        Run backtest on OHLCV data.

        Args:
            df:          Full OHLCV DataFrame (at least 200 bars)
            symbol:      Instrument name
            timeframe:   Timeframe string
            strategy_fn: Optional custom signal function (df, i) → (direction, sl, tp)
                         If None, uses built-in SMC strategy

        Returns:
            BacktestResult with all performance metrics
        """
        if len(df) < 100:
            log.warning(f"Insufficient data for backtest: {len(df)} bars")
            return self._empty_result(symbol, timeframe)

        log.info(f"Running backtest: {symbol} | {timeframe} | {len(df)} bars")

        capital      = self.initial_capital
        equity_curve = [capital]
        trades       = []
        open_trade   = None

        # Start from bar 50 (need history for indicators)
        for i in range(50, len(df)):
            current = df.iloc[i]
            current_price = float(current["close"])

            # Check open trade SL/TP
            if open_trade:
                exit_price, exit_reason = self._check_exit(open_trade, current)
                if exit_price:
                    pnl   = self._compute_pnl(open_trade, exit_price)
                    pnl_abs = pnl * open_trade["size"] * open_trade["entry"]
                    capital += pnl_abs

                    outcome = "win" if pnl_abs > 0 else "loss" if pnl_abs < 0 else "breakeven"
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        direction=open_trade["direction"],
                        entry_price=open_trade["entry"],
                        exit_price=exit_price,
                        stop_loss=open_trade["sl"],
                        take_profit=open_trade["tp"],
                        entry_idx=open_trade["entry_idx"],
                        exit_idx=i,
                        pnl=round(pnl_abs, 2),
                        pnl_pct=round(pnl, 4),
                        outcome=outcome,
                        exit_reason=exit_reason,
                        bars_held=i - open_trade["entry_idx"],
                    ))
                    open_trade = None

            # Generate new signal (no pyramiding)
            if open_trade is None:
                signal = self._generate_signal(df, i, strategy_fn)
                if signal:
                    direction, sl, tp = signal
                    risk_amt = capital * self.risk_per_trade
                    sl_dist  = abs(current_price - sl)
                    size     = risk_amt / sl_dist if sl_dist > 0 else 0

                    if size > 0:
                        open_trade = {
                            "direction": direction,
                            "entry":     current_price,
                            "sl":        sl,
                            "tp":        tp,
                            "size":      size,
                            "entry_idx": i,
                        }

            equity_curve.append(capital)

        # Force-close any open trade at end
        if open_trade:
            last_price = float(df.iloc[-1]["close"])
            pnl        = self._compute_pnl(open_trade, last_price)
            pnl_abs    = pnl * open_trade["size"] * open_trade["entry"]
            capital   += pnl_abs
            trades.append(BacktestTrade(
                symbol=symbol, direction=open_trade["direction"],
                entry_price=open_trade["entry"], exit_price=last_price,
                stop_loss=open_trade["sl"], take_profit=open_trade["tp"],
                entry_idx=open_trade["entry_idx"], exit_idx=len(df) - 1,
                pnl=round(pnl_abs, 2), pnl_pct=round(pnl, 4),
                outcome="win" if pnl_abs > 0 else "loss",
                exit_reason="end_of_data", bars_held=len(df) - 1 - open_trade["entry_idx"],
            ))

        return self._compute_results(
            symbol, timeframe, trades, equity_curve, capital
        )

    # ── Signal Generation ─────────────────────────────────────────────────────
    def _generate_signal(
        self,
        df:          pd.DataFrame,
        i:           int,
        strategy_fn,
    ):
        """Generate a signal at bar i. Returns (direction, sl, tp) or None."""
        if strategy_fn:
            return strategy_fn(df, i)

        # Default: simple EMA crossover for baseline testing
        window = df.iloc[:i+1]
        close  = window["close"]
        if len(close) < 50:
            return None

        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        prev_ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-2])
        prev_ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-2])

        current_price = float(df.iloc[i]["close"])
        atr = self._atr(window)

        # Bullish crossover
        if prev_ema20 < prev_ema50 and ema20 > ema50:
            sl = current_price - atr * 1.5
            tp = current_price + atr * 3.0
            return ("long", sl, tp)

        # Bearish crossover
        if prev_ema20 > prev_ema50 and ema20 < ema50:
            sl = current_price + atr * 1.5
            tp = current_price - atr * 3.0
            return ("short", sl, tp)

        return None

    # ── Exit Logic ────────────────────────────────────────────────────────────
    def _check_exit(self, trade: dict, candle) -> tuple[float, str] | tuple[None, None]:
        """Check if SL or TP was hit on this candle. Returns (exit_price, reason)."""
        low   = float(candle["low"])
        high  = float(candle["high"])
        bars_held = 1   # Will be corrected in main loop

        if trade["direction"] == "long":
            if low <= trade["sl"]:
                return trade["sl"], "sl"
            if high >= trade["tp"]:
                return trade["tp"], "tp"
        else:
            if high >= trade["sl"]:
                return trade["sl"], "sl"
            if low <= trade["tp"]:
                return trade["tp"], "tp"

        return None, None

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _compute_pnl(self, trade: dict, exit_price: float) -> float:
        """Compute P&L as percentage return."""
        if trade["direction"] == "long":
            return (exit_price - trade["entry"]) / trade["entry"]
        return (trade["entry"] - exit_price) / trade["entry"]

    def _compute_results(
        self,
        symbol: str, timeframe: str,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        final_capital: float,
    ) -> BacktestResult:
        if not trades:
            return self._empty_result(symbol, timeframe)

        wins   = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]

        win_rate     = len(wins) / len(trades)
        total_pnl    = final_capital - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital

        pnl_series   = pd.Series([t.pnl for t in trades])
        sharpe       = (pnl_series.mean() / (pnl_series.std() + 1e-10)) * (252 ** 0.5)

        gross_profit = sum(t.pnl for t in wins)  if wins   else 0
        gross_loss   = abs(sum(t.pnl for t in losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        # Max drawdown from equity curve
        eq  = pd.Series(equity_curve)
        max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())

        avg_win  = gross_profit / len(wins)  if wins   else 0
        avg_loss = gross_loss   / len(losses) if losses else 0

        result = BacktestResult(
            symbol=symbol, timeframe=timeframe,
            total_trades=len(trades), wins=len(wins), losses=len(losses),
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 4),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            avg_bars_held=round(sum(t.bars_held for t in trades) / len(trades), 1),
            trades=trades,
            equity_curve=equity_curve,
        )

        log.info(
            f"Backtest: {symbol} | {len(trades)} trades | "
            f"Win: {win_rate:.1%} | P&L: ${total_pnl:,.2f} | "
            f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2%}"
        )

        return result

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

    def _empty_result(self, symbol: str, timeframe: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, timeframe=timeframe, total_trades=0,
            wins=0, losses=0, win_rate=0, total_pnl=0, total_pnl_pct=0,
            max_drawdown=0, sharpe_ratio=0, profit_factor=0,
            avg_win=0, avg_loss=0, avg_bars_held=0,
        )


# Singleton
backtest_engine = BacktestEngine()