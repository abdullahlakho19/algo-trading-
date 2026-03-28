"""
execution/paper_simulator.py
─────────────────────────────────────────────────────────────────────────────
Paper Trading Simulator.
Simulates order execution with realistic fills, slippage, and commissions.
Tracks full portfolio P&L, open positions, and trade history.
Zero real money — 100% safe to use for validation.
─────────────────────────────────────────────────────────────────────────────
"""

import uuid
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class SimulatedOrder:
    """A simulated order in paper trading mode."""
    order_id:    str
    symbol:      str
    side:        str           # "buy" | "sell"
    qty:         float
    order_type:  str           # "market" | "limit"
    limit_price: Optional[float]
    status:      str = "pending"   # pending | filled | cancelled | rejected
    filled_price: Optional[float] = None
    filled_at:   Optional[datetime] = None
    slippage:    float = 0.0


@dataclass
class SimulatedPosition:
    """An open position in paper trading."""
    symbol:      str
    direction:   str           # "long" | "short"
    qty:         float
    entry_price: float
    stop_loss:   float
    take_profit: float
    market:      str
    opened_at:   datetime = field(default_factory=datetime.utcnow)
    current_price: float = 0.0
    unrealised_pnl: float = 0.0


@dataclass
class ClosedTrade:
    """A completed trade record."""
    trade_id:    str
    symbol:      str
    direction:   str
    qty:         float
    entry_price: float
    exit_price:  float
    stop_loss:   float
    take_profit: float
    pnl:         float
    pnl_pct:     float
    outcome:     str           # "win" | "loss" | "breakeven"
    duration_min: float
    opened_at:   datetime
    closed_at:   datetime
    market:      str
    exit_reason: str           # "tp_hit" | "sl_hit" | "manual" | "timeout"


class PaperSimulator:
    """
    Full paper trading simulator with realistic execution.
    
    Simulates:
    - Market and limit order fills
    - Bid/ask spread slippage
    - Position tracking with live P&L
    - Automatic SL/TP monitoring
    - Commission simulation
    """

    COMMISSION_PCT = 0.001    # 0.1% per trade (realistic for most brokers)
    SLIPPAGE_BPS   = 3        # 3 basis points slippage on market orders

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital  = initial_capital
        self.cash             = initial_capital
        self.open_positions:  dict[str, SimulatedPosition] = {}
        self.closed_trades:   list[ClosedTrade] = []
        self.orders:          list[SimulatedOrder] = []
        self._trade_count     = 0

    # ── Order Placement ───────────────────────────────────────────────────────
    def place_order(
        self,
        symbol:      str,
        side:        str,
        qty:         float,
        order_type:  str        = "market",
        limit_price: float      = None,
        stop_loss:   float      = None,
        take_profit: float      = None,
        current_price: float    = None,
        market:      str        = "stock",
        direction:   str        = "long",
    ) -> SimulatedOrder:
        """
        Place a simulated order.

        Args:
            symbol:       Instrument symbol
            side:         "buy" or "sell"
            qty:          Number of units / shares / contracts
            order_type:   "market" or "limit"
            limit_price:  Required if order_type="limit"
            stop_loss:    Auto-set SL level
            take_profit:  Auto-set TP level
            current_price: Current market price (for market orders)
            market:       "stock" | "forex" | "crypto"
            direction:    "long" | "short"

        Returns:
            SimulatedOrder with fill details
        """
        order_id = str(uuid.uuid4())[:8].upper()

        # Validate
        if qty <= 0:
            log.warning(f"Invalid order quantity: {qty}")
            return SimulatedOrder(
                order_id=order_id, symbol=symbol, side=side, qty=qty,
                order_type=order_type, limit_price=limit_price, status="rejected"
            )

        # Check buying power
        if side == "buy" and current_price:
            cost = qty * current_price * (1 + self.COMMISSION_PCT)
            if cost > self.cash:
                log.warning(f"Insufficient cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
                return SimulatedOrder(
                    order_id=order_id, symbol=symbol, side=side, qty=qty,
                    order_type=order_type, limit_price=limit_price, status="rejected"
                )

        # Simulate fill
        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
        )

        if order_type == "market" and current_price:
            fill_price = self._apply_slippage(current_price, side)
            order.filled_price = fill_price
            order.filled_at    = datetime.utcnow()
            order.slippage     = abs(fill_price - current_price)
            order.status       = "filled"

            # Update cash
            commission = qty * fill_price * self.COMMISSION_PCT
            if side == "buy":
                self.cash -= (qty * fill_price + commission)
            else:
                self.cash += (qty * fill_price - commission)

            # Register position
            if side == "buy":
                self._open_position(
                    symbol, direction, qty, fill_price,
                    stop_loss or 0.0, take_profit or 0.0, market
                )
            else:
                self._close_position(symbol, fill_price, "manual")

            log.info(
                f"📋 PAPER ORDER FILLED | {symbol} {side.upper()} {qty:.4f} "
                f"@ {fill_price:.4f} | Commission: ${commission:.2f} | "
                f"Order: {order_id}"
            )

        elif order_type == "limit":
            order.status = "pending"
            log.info(f"📋 PAPER LIMIT ORDER PLACED | {symbol} @ {limit_price}")

        self.orders.append(order)
        return order

    # ── SL/TP Monitoring ──────────────────────────────────────────────────────
    def update_positions(self, price_feed: dict[str, float]) -> list[ClosedTrade]:
        """
        Update all open positions with latest prices.
        Check if any SL or TP has been hit.
        Called every tick / bar close.

        Args:
            price_feed: {symbol: current_price}

        Returns:
            List of trades that were automatically closed
        """
        auto_closed = []

        for symbol, position in list(self.open_positions.items()):
            price = price_feed.get(symbol)
            if not price:
                continue

            position.current_price   = price
            position.unrealised_pnl  = self._compute_pnl(position, price)

            # Check TP / SL for long positions
            if position.direction == "long":
                if position.take_profit > 0 and price >= position.take_profit:
                    trade = self._close_position(symbol, position.take_profit, "tp_hit")
                    if trade:
                        auto_closed.append(trade)
                        log.info(f"✅ TP HIT | {symbol} @ {position.take_profit:.4f} | P&L: ${trade.pnl:.2f}")

                elif position.stop_loss > 0 and price <= position.stop_loss:
                    trade = self._close_position(symbol, position.stop_loss, "sl_hit")
                    if trade:
                        auto_closed.append(trade)
                        log.warning(f"🛑 SL HIT | {symbol} @ {position.stop_loss:.4f} | P&L: ${trade.pnl:.2f}")

            # Check TP / SL for short positions
            elif position.direction == "short":
                if position.take_profit > 0 and price <= position.take_profit:
                    trade = self._close_position(symbol, position.take_profit, "tp_hit")
                    if trade:
                        auto_closed.append(trade)
                        log.info(f"✅ TP HIT (SHORT) | {symbol} @ {position.take_profit:.4f}")

                elif position.stop_loss > 0 and price >= position.stop_loss:
                    trade = self._close_position(symbol, position.stop_loss, "sl_hit")
                    if trade:
                        auto_closed.append(trade)
                        log.warning(f"🛑 SL HIT (SHORT) | {symbol} @ {position.stop_loss:.4f}")

        return auto_closed

    # ── Position Helpers ──────────────────────────────────────────────────────
    def _open_position(
        self, symbol: str, direction: str, qty: float,
        fill_price: float, sl: float, tp: float, market: str
    ) -> None:
        self.open_positions[symbol] = SimulatedPosition(
            symbol=symbol, direction=direction, qty=qty,
            entry_price=fill_price, stop_loss=sl,
            take_profit=tp, market=market,
        )

    def _close_position(
        self, symbol: str, exit_price: float, reason: str
    ) -> Optional[ClosedTrade]:
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return None

        commission = pos.qty * exit_price * self.COMMISSION_PCT
        raw_pnl    = self._compute_pnl(pos, exit_price) - commission

        if raw_pnl > 0:
            outcome = "win"
        elif raw_pnl < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"

        closed_at = datetime.utcnow()
        duration  = (closed_at - pos.opened_at).total_seconds() / 60

        trade = ClosedTrade(
            trade_id=f"T{self._trade_count:05d}",
            symbol=symbol,
            direction=pos.direction,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl=round(raw_pnl, 2),
            pnl_pct=round(raw_pnl / (pos.qty * pos.entry_price), 4),
            outcome=outcome,
            duration_min=round(duration, 1),
            opened_at=pos.opened_at,
            closed_at=closed_at,
            market=pos.market,
            exit_reason=reason,
        )

        self.cash += pos.qty * exit_price - commission
        self.closed_trades.append(trade)
        self._trade_count += 1

        return trade

    def _compute_pnl(self, pos: SimulatedPosition, current_price: float) -> float:
        if pos.direction == "long":
            return (current_price - pos.entry_price) * pos.qty
        else:
            return (pos.entry_price - current_price) * pos.qty

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply realistic slippage to market orders."""
        slip = price * (self.SLIPPAGE_BPS / 10000)
        return price + slip if side == "buy" else price - slip

    # ── Portfolio Metrics ─────────────────────────────────────────────────────
    @property
    def equity(self) -> float:
        """Total equity = cash + unrealised P&L."""
        unrealised = sum(p.unrealised_pnl for p in self.open_positions.values())
        return self.cash + unrealised

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.initial_capital

    def get_performance(self) -> dict:
        """Compute full performance metrics."""
        trades = self.closed_trades
        if not trades:
            return {"message": "No closed trades yet."}

        wins   = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]

        win_rate     = len(wins) / len(trades) if trades else 0
        avg_win      = sum(t.pnl for t in wins)  / len(wins)  if wins   else 0
        avg_loss     = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else float("inf")

        pnl_series = pd.Series([t.pnl for t in trades])
        sharpe     = (pnl_series.mean() / pnl_series.std() * (252 ** 0.5)) if len(pnl_series) > 1 else 0

        return {
            "total_trades":    len(trades),
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate":        round(win_rate, 4),
            "avg_win":         round(avg_win, 2),
            "avg_loss":        round(avg_loss, 2),
            "profit_factor":   round(profit_factor, 2),
            "total_pnl":       round(self.total_pnl, 2),
            "total_pnl_pct":   round(self.total_pnl_pct, 4),
            "equity":          round(self.equity, 2),
            "sharpe_ratio":    round(sharpe, 2),
            "open_positions":  len(self.open_positions),
        }

    def get_status(self) -> dict:
        return {
            "mode":              "PAPER",
            "initial_capital":   self.initial_capital,
            "equity":            round(self.equity, 2),
            "cash":              round(self.cash, 2),
            "total_pnl":         round(self.total_pnl, 2),
            "total_pnl_pct":     round(self.total_pnl_pct, 4),
            "open_positions":    len(self.open_positions),
            "closed_trades":     len(self.closed_trades),
        }


# Singleton
paper_simulator = PaperSimulator(initial_capital=100_000.0)