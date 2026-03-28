"""
execution/alpaca_executor.py
─────────────────────────────────────────────────────────────────────────────
Alpaca Live Execution Engine — Stocks & Forex.
Handles order placement, cancellation, and monitoring in both
paper and live trading modes.
─────────────────────────────────────────────────────────────────────────────
"""

import time
from alpaca.trading.client    import TradingClient
from alpaca.trading.requests  import (
    MarketOrderRequest, LimitOrderRequest,
    StopLossRequest, TakeProfitRequest,
    GetOrdersRequest, ClosePositionRequest
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
)
from core.logger import get_logger
from config import config

log = get_logger(__name__)


class AlpacaExecutor:
    """
    Alpaca Markets live order execution.
    Supports market, limit orders with bracket SL/TP.
    """

    def __init__(self):
        self._client  = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self._client = TradingClient(
                api_key=config.alpaca.API_KEY,
                secret_key=config.alpaca.SECRET_KEY,
                paper=config.alpaca.PAPER_MODE,
            )
            acct = self._client.get_account()
            self._connected = True
            log.info(
                f"Alpaca Executor connected | "
                f"Mode: {'PAPER' if config.alpaca.PAPER_MODE else 'LIVE'} | "
                f"Equity: ${float(acct.equity):,.2f}"
            )
            return True
        except Exception as e:
            log.error(f"Alpaca Executor connection failed: {e}")
            return False

    # ── Order Placement ───────────────────────────────────────────────────────
    def place_market_order(
        self,
        symbol:    str,
        side:      str,       # "buy" | "sell"
        qty:       float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> dict | None:
        """Place a market order with optional bracket SL/TP."""
        if not self._connected:
            self.connect()

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        try:
            sl  = StopLossRequest(stop_price=round(stop_loss, 4)) if stop_loss else None
            tp  = TakeProfitRequest(limit_price=round(take_profit, 4)) if take_profit else None

            req = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=order_side,
                time_in_force=TimeInForce.DAY,
                stop_loss=sl,
                take_profit=tp,
            )

            order = self._client.submit_order(req)
            log.info(
                f"ORDER SUBMITTED | {symbol} {side.upper()} {qty:.4f} "
                f"MARKET | ID: {order.id}"
            )
            return {
                "order_id":  str(order.id),
                "symbol":    symbol,
                "side":      side,
                "qty":       qty,
                "type":      "market",
                "status":    order.status.value,
            }

        except Exception as e:
            log.error(f"Order submission failed: {symbol} {side}: {e}")
            return None

    def place_limit_order(
        self,
        symbol:      str,
        side:        str,
        qty:         float,
        limit_price: float,
        stop_loss:   float = None,
        take_profit: float = None,
    ) -> dict | None:
        """Place a limit order."""
        if not self._connected:
            self.connect()

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        try:
            sl = StopLossRequest(stop_price=round(stop_loss, 4)) if stop_loss else None
            tp = TakeProfitRequest(limit_price=round(take_profit, 4)) if take_profit else None

            req = LimitOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=order_side,
                limit_price=round(limit_price, 4),
                time_in_force=TimeInForce.GTC,
                stop_loss=sl,
                take_profit=tp,
            )

            order = self._client.submit_order(req)
            log.info(
                f"LIMIT ORDER | {symbol} {side.upper()} {qty:.4f} "
                f"@ {limit_price:.4f} | ID: {order.id}"
            )
            return {
                "order_id":    str(order.id),
                "symbol":      symbol,
                "side":        side,
                "qty":         qty,
                "limit_price": limit_price,
                "type":        "limit",
                "status":      order.status.value,
            }

        except Exception as e:
            log.error(f"Limit order failed: {symbol}: {e}")
            return None

    # ── Order Management ──────────────────────────────────────────────────────
    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        try:
            self._client.cancel_order_by_id(order_id)
            log.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            log.error(f"Cancel failed: {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> None:
        if not self._connected:
            return
        try:
            self._client.cancel_orders()
            log.info("All open orders cancelled.")
        except Exception as e:
            log.error(f"Cancel all failed: {e}")

    def close_position(self, symbol: str) -> bool:
        if not self._connected:
            return False
        try:
            self._client.close_position(symbol)
            log.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            log.error(f"Close position failed {symbol}: {e}")
            return False

    def close_all_positions(self) -> None:
        if not self._connected:
            return
        try:
            self._client.close_all_positions(cancel_orders=True)
            log.warning("ALL POSITIONS CLOSED.")
        except Exception as e:
            log.error(f"Close all failed: {e}")

    # ── Status ────────────────────────────────────────────────────────────────
    def get_open_positions(self) -> list[dict]:
        if not self._connected:
            return []
        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol":    p.symbol,
                    "qty":       float(p.qty),
                    "side":      p.side.value,
                    "avg_entry": float(p.avg_entry_price),
                    "market_val": float(p.market_value),
                    "unrealised_pnl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception:
            return []

    def get_account(self) -> dict | None:
        if not self._connected:
            self.connect()
        try:
            a = self._client.get_account()
            return {
                "equity":       float(a.equity),
                "cash":         float(a.cash),
                "buying_power": float(a.buying_power),
                "paper":        config.alpaca.PAPER_MODE,
            }
        except Exception:
            return None

    def is_connected(self) -> bool:
        return self._connected


# Singleton
alpaca_executor = AlpacaExecutor()