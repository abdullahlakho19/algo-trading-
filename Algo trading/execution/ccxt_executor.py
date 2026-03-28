"""
execution/ccxt_executor.py
─────────────────────────────────────────────────────────────────────────────
CCXT Crypto Execution Engine — Binance.
Handles live/testnet order placement, cancellation,
and position management for all CCXT-supported exchanges.
─────────────────────────────────────────────────────────────────────────────
"""

import ccxt
import time
from dataclasses import dataclass
from typing import Optional
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class CCXTOrder:
    """A placed CCXT order record."""
    order_id:    str
    symbol:      str
    side:        str
    qty:         float
    order_type:  str
    status:      str
    filled_price: Optional[float] = None
    fee:         float = 0.0


class CCXTExecutor:
    """
    Crypto order execution via CCXT (Binance).
    Supports market, limit orders with stop-loss simulation.
    Testnet by default.
    """

    def __init__(self):
        self.exchange_id  = config.markets.CRYPTO_EXCHANGE
        self.api_key      = config.binance.API_KEY
        self.secret_key   = config.binance.SECRET_KEY
        self.testnet      = config.binance.TESTNET
        self._exchange    = None
        self._connected   = False
        self._open_orders: dict[str, CCXTOrder] = {}

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            cls = getattr(ccxt, self.exchange_id)
            self._exchange = cls({
                "apiKey":           self.api_key,
                "secret":           self.secret_key,
                "enableRateLimit":  True,
                "options":          {"defaultType": "spot"},
            })
            if self.testnet:
                self._exchange.set_sandbox_mode(True)
            self._exchange.load_markets()
            self._connected = True
            mode = "TESTNET" if self.testnet else "LIVE"
            log.info(f"CCXT Executor connected | {self.exchange_id.upper()} | {mode}")
            return True
        except Exception as e:
            log.error(f"CCXT Executor connection failed: {e}")
            return False

    # ── Order Placement ───────────────────────────────────────────────────────
    def place_market_order(
        self,
        symbol: str,
        side:   str,
        qty:    float,
    ) -> CCXTOrder | None:
        """Place a market order."""
        if not self._connected:
            self.connect()
        try:
            order = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=qty,
            )
            result = CCXTOrder(
                order_id=str(order["id"]),
                symbol=symbol, side=side, qty=qty,
                order_type="market",
                status=order.get("status", "closed"),
                filled_price=float(order.get("average") or order.get("price") or 0),
                fee=float(order.get("fee", {}).get("cost", 0)),
            )
            self._open_orders[result.order_id] = result
            log.info(f"CCXT MARKET | {symbol} {side.upper()} {qty:.6f} @ {result.filled_price:.4f}")
            return result
        except Exception as e:
            log.error(f"CCXT market order failed: {symbol} {side}: {e}")
            return None

    def place_limit_order(
        self,
        symbol:      str,
        side:        str,
        qty:         float,
        limit_price: float,
    ) -> CCXTOrder | None:
        """Place a limit order."""
        if not self._connected:
            self.connect()
        try:
            order = self._exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=qty,
                price=limit_price,
            )
            result = CCXTOrder(
                order_id=str(order["id"]),
                symbol=symbol, side=side, qty=qty,
                order_type="limit",
                status=order.get("status", "open"),
                filled_price=None,
            )
            self._open_orders[result.order_id] = result
            log.info(f"CCXT LIMIT | {symbol} {side.upper()} {qty:.6f} @ {limit_price:.4f}")
            return result
        except Exception as e:
            log.error(f"CCXT limit order failed: {symbol}: {e}")
            return None

    def place_stop_order(
        self,
        symbol:     str,
        side:       str,
        qty:        float,
        stop_price: float,
    ) -> CCXTOrder | None:
        """
        Place a stop-loss order.
        Note: Binance spot uses 'stop_market' type.
        """
        if not self._connected:
            self.connect()
        try:
            order = self._exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=side,
                amount=qty,
                params={"stopPrice": stop_price},
            )
            result = CCXTOrder(
                order_id=str(order["id"]),
                symbol=symbol, side=side, qty=qty,
                order_type="stop_market",
                status=order.get("status", "open"),
            )
            self._open_orders[result.order_id] = result
            log.info(f"CCXT STOP | {symbol} {side.upper()} {qty:.6f} stop @ {stop_price:.4f}")
            return result
        except Exception as e:
            log.warning(f"Stop order failed (may not be supported on testnet): {e}")
            return None

    # ── Order Management ──────────────────────────────────────────────────────
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if not self._connected:
            return False
        try:
            self._exchange.cancel_order(order_id, symbol)
            self._open_orders.pop(order_id, None)
            log.info(f"CCXT order cancelled: {order_id}")
            return True
        except Exception as e:
            log.error(f"CCXT cancel failed: {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str = None) -> None:
        if not self._connected:
            return
        try:
            if symbol:
                self._exchange.cancel_all_orders(symbol)
            else:
                for sym in set(o.symbol for o in self._open_orders.values()):
                    self._exchange.cancel_all_orders(sym)
            self._open_orders.clear()
            log.info("All CCXT orders cancelled.")
        except Exception as e:
            log.error(f"CCXT cancel all failed: {e}")

    # ── Account & Balance ─────────────────────────────────────────────────────
    def get_balance(self) -> dict:
        if not self._connected:
            self.connect()
        try:
            bal = self._exchange.fetch_balance()
            return {k: v for k, v in bal["total"].items() if v > 0}
        except Exception as e:
            log.error(f"CCXT balance fetch failed: {e}")
            return {}

    def get_positions(self) -> list[dict]:
        """Get all non-zero balances as positions."""
        bal = self.get_balance()
        positions = []
        for coin, amount in bal.items():
            if coin in ("USDT", "BUSD", "USD"):
                continue
            symbol = f"{coin}/USDT"
            try:
                ticker = self._exchange.fetch_ticker(symbol)
                price  = float(ticker["last"])
                positions.append({
                    "symbol":    symbol,
                    "qty":       amount,
                    "price":     price,
                    "value":     amount * price,
                })
            except Exception:
                positions.append({"symbol": symbol, "qty": amount, "price": 0, "value": 0})
        return positions

    def get_latest_price(self, symbol: str) -> float | None:
        if not self._connected:
            self.connect()
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception:
            return None

    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> dict:
        return {
            "connected":  self._connected,
            "exchange":   self.exchange_id,
            "testnet":    self.testnet,
            "open_orders": len(self._open_orders),
        }


# Singleton
ccxt_executor = CCXTExecutor()