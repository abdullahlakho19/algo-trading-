"""
execution/order_manager.py
─────────────────────────────────────────────────────────────────────────────
Order Manager.
Tracks the full lifecycle of every order and position.
Acts as the single source of truth for what the agent owns.
─────────────────────────────────────────────────────────────────────────────
"""

import uuid
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class ManagedOrder:
    """Full lifecycle order record."""
    order_id:     str
    symbol:       str
    side:         str
    qty:          float
    order_type:   str
    status:       str = "pending"
    filled_price: Optional[float] = None
    stop_loss:    Optional[float] = None
    take_profit:  Optional[float] = None
    created_at:   datetime = field(default_factory=datetime.utcnow)
    filled_at:    Optional[datetime] = None
    closed_at:    Optional[datetime] = None
    pnl:          float = 0.0
    market:       str = "stock"
    direction:    str = "long"


class OrderManager:
    """
    Central order and position tracker.
    Maintains state for all open orders and positions.
    """

    def __init__(self):
        self._orders:    dict[str, ManagedOrder] = {}
        self._positions: dict[str, ManagedOrder] = {}

    def register_order(
        self,
        symbol: str, side: str, qty: float,
        order_type: str, stop_loss: float = None,
        take_profit: float = None, market: str = "stock",
        direction: str = "long",
    ) -> str:
        """Register a new order. Returns order_id."""
        order_id = str(uuid.uuid4())[:8].upper()
        self._orders[order_id] = ManagedOrder(
            order_id=order_id, symbol=symbol, side=side,
            qty=qty, order_type=order_type,
            stop_loss=stop_loss, take_profit=take_profit,
            market=market, direction=direction,
        )
        log.debug(f"Order registered: {order_id} | {symbol} {side}")
        return order_id

    def mark_filled(self, order_id: str, filled_price: float) -> None:
        order = self._orders.get(order_id)
        if not order:
            return
        order.status       = "filled"
        order.filled_price = filled_price
        order.filled_at    = datetime.utcnow()
        self._positions[order.symbol] = order
        log.info(f"Order filled: {order_id} | {order.symbol} @ {filled_price:.5f}")

    def mark_closed(self, symbol: str, exit_price: float) -> Optional[float]:
        """Mark position as closed. Returns P&L."""
        pos = self._positions.pop(symbol, None)
        if not pos:
            return None
        pos.status    = "closed"
        pos.closed_at = datetime.utcnow()

        if pos.direction == "long":
            pos.pnl = (exit_price - pos.filled_price) * pos.qty
        else:
            pos.pnl = (pos.filled_price - exit_price) * pos.qty

        log.info(f"Position closed: {symbol} | P&L: ${pos.pnl:.2f}")
        return pos.pnl

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order:
            order.status = "cancelled"
            log.info(f"Order cancelled: {order_id}")

    def get_open_positions(self) -> list[ManagedOrder]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[ManagedOrder]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_status(self) -> dict:
        return {
            "open_orders":    sum(1 for o in self._orders.values() if o.status == "pending"),
            "open_positions": len(self._positions),
            "total_orders":   len(self._orders),
        }


# Singleton
order_manager = OrderManager()