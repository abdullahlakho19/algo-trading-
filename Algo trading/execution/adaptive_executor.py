"""
execution/adaptive_executor.py
─────────────────────────────────────────────────────────────────────────────
Adaptive Execution Engine — XTX Markets philosophy.
Splits large orders intelligently to minimise market impact and slippage.
Uses TWAP/VWAP-inspired logic for order scheduling.

For paper trading: routes to paper_simulator.
For live trading:  routes to alpaca_executor or ccxt_executor.
─────────────────────────────────────────────────────────────────────────────
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class ExecutionPlan:
    """Execution plan for a single order."""
    symbol:       str
    direction:    str
    total_qty:    float
    total_value:  float
    slices:       list[dict] = field(default_factory=list)   # Each slice to execute
    method:       str = "single"   # "single" | "twap" | "vwap"
    estimated_slippage_bps: float = 0.0


@dataclass
class ExecutionResult:
    """Result of adaptive order execution."""
    symbol:       str
    direction:    str
    requested_qty: float
    filled_qty:   float
    avg_fill_price: float
    slippage_bps: float
    slices_executed: int
    success:      bool
    notes:        str = ""


class AdaptiveExecutor:
    """
    Intelligently executes orders to minimise slippage.
    
    Logic:
    - Small orders (<$10k): single market order
    - Medium orders ($10k-$100k): split into 3-5 limit orders
    - Large orders (>$100k): TWAP over execution window
    """

    # Slippage thresholds in basis points
    SMALL_ORDER_BPS  = 3
    MEDIUM_ORDER_BPS = 8
    LARGE_ORDER_BPS  = 15

    def __init__(self):
        self._paper_sim  = None
        self._alpaca     = None
        self._ccxt       = None
        self._ready      = False

    def _init(self):
        if self._ready:
            return
        from execution.paper_simulator import paper_simulator
        from execution.alpaca_executor import alpaca_executor
        from execution.ccxt_executor   import ccxt_executor
        self._paper_sim = paper_simulator
        self._alpaca    = alpaca_executor
        self._ccxt      = ccxt_executor
        self._ready     = True

    # ── Main Entry ────────────────────────────────────────────────────────────
    def execute(
        self,
        symbol:        str,
        direction:     str,      # "long" | "short"
        qty:           float,
        current_price: float,
        stop_loss:     float,
        take_profit:   float,
        market:        str = "stock",
        paper:         bool = True,
    ) -> ExecutionResult:
        """
        Execute an order adaptively.

        Args:
            symbol:        Instrument ticker
            direction:     "long" or "short"
            qty:           Number of units
            current_price: Current market price
            stop_loss:     SL level
            take_profit:   TP level
            market:        "stock" | "crypto" | "forex"
            paper:         True = paper sim, False = live

        Returns:
            ExecutionResult with fill details
        """
        self._init()

        order_value = qty * current_price
        side        = "buy" if direction == "long" else "sell"

        # Determine execution method based on order size
        plan = self._build_plan(symbol, direction, qty, current_price, order_value)
        log.info(
            f"Adaptive Execution | {symbol} {direction.upper()} {qty:.4f} "
            f"(${order_value:,.2f}) | Method: {plan.method}"
        )

        if paper:
            return self._execute_paper(plan, current_price, stop_loss, take_profit, market)
        else:
            return self._execute_live(plan, current_price, stop_loss, take_profit, market)

    # ── Execution Plan Builder ────────────────────────────────────────────────
    def _build_plan(
        self,
        symbol:      str,
        direction:   str,
        qty:         float,
        price:       float,
        value:       float,
    ) -> ExecutionPlan:
        """Determine optimal execution strategy."""
        if value < 10_000:
            # Small: single order
            method   = "single"
            slices   = [{"qty": qty, "delay_sec": 0}]
            slip_bps = self.SMALL_ORDER_BPS

        elif value < 100_000:
            # Medium: 3 slices over 30 seconds
            n        = 3
            method   = "multi_limit"
            per_qty  = qty / n
            slices   = [{"qty": per_qty, "delay_sec": i * 10} for i in range(n)]
            slip_bps = self.MEDIUM_ORDER_BPS

        else:
            # Large: TWAP over 5 minutes
            n        = 5
            method   = "twap"
            per_qty  = qty / n
            slices   = [{"qty": per_qty, "delay_sec": i * 60} for i in range(n)]
            slip_bps = self.LARGE_ORDER_BPS

        return ExecutionPlan(
            symbol=symbol,
            direction=direction,
            total_qty=qty,
            total_value=value,
            slices=slices,
            method=method,
            estimated_slippage_bps=slip_bps,
        )

    # ── Paper Execution ───────────────────────────────────────────────────────
    def _execute_paper(
        self,
        plan:          ExecutionPlan,
        price:         float,
        stop_loss:     float,
        take_profit:   float,
        market:        str,
    ) -> ExecutionResult:
        """Execute all slices in paper simulator."""
        filled_qty   = 0.0
        filled_val   = 0.0
        side         = "buy" if plan.direction == "long" else "sell"
        slices_done  = 0

        for i, sl in enumerate(plan.slices):
            # Simulate slippage
            slip_factor = 1 + (plan.estimated_slippage_bps / 10000) * (1 if side == "buy" else -1)
            fill_price  = price * slip_factor

            order = self._paper_sim.place_order(
                symbol=plan.symbol,
                side=side,
                qty=sl["qty"],
                order_type="market",
                stop_loss=stop_loss if i == 0 else None,
                take_profit=take_profit if i == 0 else None,
                current_price=fill_price,
                market=market,
                direction=plan.direction,
            )

            if order and order.status == "filled":
                filled_qty  += sl["qty"]
                filled_val  += sl["qty"] * (order.filled_price or fill_price)
                slices_done += 1

            # Wait between slices (in live mode — skip in paper)
            # In paper mode we skip delays for speed

        avg_fill = filled_val / max(filled_qty, 1e-10)
        slip     = abs(avg_fill - price) / price * 10000   # in bps

        return ExecutionResult(
            symbol=plan.symbol,
            direction=plan.direction,
            requested_qty=plan.total_qty,
            filled_qty=round(filled_qty, 6),
            avg_fill_price=round(avg_fill, 6),
            slippage_bps=round(slip, 2),
            slices_executed=slices_done,
            success=filled_qty > 0,
            notes=f"Paper | {plan.method} | {slices_done}/{len(plan.slices)} slices",
        )

    # ── Live Execution ────────────────────────────────────────────────────────
    def _execute_live(
        self,
        plan:       ExecutionPlan,
        price:      float,
        stop_loss:  float,
        take_profit: float,
        market:     str,
    ) -> ExecutionResult:
        """Execute in live market."""
        side        = "buy" if plan.direction == "long" else "sell"
        filled_qty  = 0.0
        filled_val  = 0.0
        slices_done = 0

        executor = self._ccxt if market == "crypto" else self._alpaca

        for i, sl in enumerate(plan.slices):
            if sl["delay_sec"] > 0 and i > 0:
                time.sleep(sl["delay_sec"])

            if market == "crypto":
                order = executor.place_market_order(plan.symbol, side, sl["qty"])
            else:
                order = executor.place_market_order(
                    plan.symbol, side, sl["qty"],
                    stop_loss=stop_loss if i == 0 else None,
                    take_profit=take_profit if i == 0 else None,
                )

            if order:
                fill_price   = getattr(order, "filled_price", price) or price
                filled_qty  += sl["qty"]
                filled_val  += sl["qty"] * fill_price
                slices_done += 1
                log.info(f"Slice {i+1}/{len(plan.slices)} filled: {sl['qty']:.4f} @ {fill_price:.5f}")

        avg_fill = filled_val / max(filled_qty, 1e-10)
        slip     = abs(avg_fill - price) / price * 10000

        return ExecutionResult(
            symbol=plan.symbol, direction=plan.direction,
            requested_qty=plan.total_qty, filled_qty=round(filled_qty, 6),
            avg_fill_price=round(avg_fill, 6), slippage_bps=round(slip, 2),
            slices_executed=slices_done, success=filled_qty > 0,
            notes=f"Live | {plan.method} | {slices_done}/{len(plan.slices)} slices",
        )

    def get_status(self) -> dict:
        return {"ready": self._ready, "method": "adaptive"}


# Singleton
adaptive_executor = AdaptiveExecutor()