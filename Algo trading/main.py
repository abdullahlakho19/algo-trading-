"""
main.py
─────────────────────────────────────────────────────────────────────────────
INSTITUTIONAL TRADING AGENT — Master Entry Point

Starts and orchestrates the entire system:
    1. Initialises all modules
    2. Loads market data
    3. Trains ML models (if needed)
    4. Runs the continuous trading loop
    5. Monitors positions and circuit breakers

Usage:
    python main.py                   # Start agent (paper mode)
    python main.py --mode backtest   # Run backtest
    python main.py --mode train      # Train ML models only
    python main.py --symbols AAPL MSFT  # Override watchlist
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import argparse
import signal as os_signal
from datetime import datetime

# ── Setup path ────────────────────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logger import setup_logger, get_logger
from config import config, TradingMode

# Initialise logger before anything else
setup_logger()
log = get_logger(__name__)


# ── Graceful Shutdown ─────────────────────────────────────────────────────────
_running = True

def _handle_shutdown(signum, frame):
    global _running
    log.warning("Shutdown signal received. Stopping agent gracefully...")
    _running = False

os_signal.signal(os_signal.SIGINT,  _handle_shutdown)
os_signal.signal(os_signal.SIGTERM, _handle_shutdown)


# ── Agent Core ────────────────────────────────────────────────────────────────
class TradingAgent:
    """
    Master trading agent orchestrator.
    Coordinates data, analysis, signals, risk, and execution.
    """

    def __init__(self, mode: str = "paper", symbols: list[str] = None):
        self.mode     = mode
        self.symbols  = symbols or (
            config.markets.STOCKS[:3]   # Start with a subset for initial testing
            + config.markets.CRYPTO[:2]
        )
        self.running  = False

        log.info("=" * 70)
        log.info("   INSTITUTIONAL TRADING AGENT — STARTING UP")
        log.info("=" * 70)
        log.info(f"   Mode:    {mode.upper()}")
        log.info(f"   Symbols: {self.symbols}")
        log.info(f"   Markets: Stocks + Crypto + Forex")
        log.info("=" * 70)

    # ── Initialisation ────────────────────────────────────────────────────────
    def initialise(self) -> bool:
        """Initialise all system modules."""
        log.info("Initialising modules...")

        try:
            # Data engine
            from core.data_engine import data_engine
            self.data_engine = data_engine
            log.info("✅ Data engine ready.")

            # Market clock
            from core.market_clock import market_clock
            self.clock = market_clock
            self.clock.log_status()
            log.info("✅ Market clock ready.")

            # Event calendar
            from core.event_calendar import event_calendar
            event_calendar.refresh()
            self.event_calendar = event_calendar
            log.info("✅ Event calendar loaded.")

            # Signal engine
            from strategies.signal_engine import signal_engine
            self.signal_engine = signal_engine
            log.info("✅ Signal engine ready.")

            # Risk manager
            from risk.risk_manager import risk_manager
            self.risk_manager = risk_manager
            log.info("✅ Risk manager ready.")

            # Paper simulator
            from execution.paper_simulator import paper_simulator
            self.simulator = paper_simulator
            log.info(f"✅ Paper simulator ready. Capital: ${self.simulator.equity:,.2f}")

            # Excel exporter
            from reporting.excel_exporter import excel_exporter
            self.exporter = excel_exporter
            log.info("✅ Excel exporter ready.")

            # Ensemble voter
            from ai_ml.ensemble_voter import ensemble_voter
            self.ensemble = ensemble_voter
            log.info(f"✅ Ensemble voter ready. Trained: {ensemble_voter.is_trained}")

            log.info("All modules initialised successfully.")
            return True

        except Exception as e:
            log.error(f"Initialisation failed: {e}")
            return False

    # ── Training ──────────────────────────────────────────────────────────────
    def train_models(self) -> None:
        """Train all ML models on historical data."""
        log.info("Starting model training phase...")

        for symbol in self.symbols:
            log.info(f"Fetching training data: {symbol}")
            df = self.data_engine.get_bars(
                symbol, timeframe="1h", lookback_days=365
            ).ohlcv

            if df.empty or len(df) < config.ml.MIN_TRAIN_SAMPLES:
                log.warning(f"Insufficient data for {symbol}, skipping.")
                continue

            metrics = self.ensemble.train(df, symbol)
            if metrics:
                log.info(f"✅ {symbol} trained: {metrics}")

        log.info("Model training complete.")

    # ── Main Trading Loop ─────────────────────────────────────────────────────
    def run(self) -> None:
        """Main continuous trading loop."""
        if not self.initialise():
            log.error("Failed to initialise. Exiting.")
            return

        # Train if models not loaded
        if not self.ensemble.is_trained:
            log.info("No pre-trained models found. Training now...")
            self.train_models()

        log.info("Starting trading loop...")
        self.running = True
        scan_interval = 60   # Seconds between full market scans

        while _running and self.running:
            try:
                loop_start = datetime.utcnow()

                # ── Clock & Session Check ──────────────────────────────────
                self.clock.log_status()
                sessions = self.clock.current_sessions()

                if "off_hours" in sessions and "crypto" not in [
                    self.data_engine._detect_market(s) for s in self.symbols
                ]:
                    log.info("Markets closed. Sleeping 5 minutes.")
                    time.sleep(300)
                    continue

                # ── Macro Event Check ──────────────────────────────────────
                if not self.event_calendar.is_safe_to_trade():
                    log.info("Macro event buffer active. Skipping scan.")
                    time.sleep(120)
                    continue

                # ── Circuit Breaker Check ──────────────────────────────────
                if self.risk_manager.circuit_breaker.is_active:
                    log.warning(f"Circuit breaker active: {self.risk_manager.circuit_breaker.reason}")
                    time.sleep(300)
                    continue

                # ── Market Scan ────────────────────────────────────────────
                self._scan_all_symbols()

                # ── Update Positions ───────────────────────────────────────
                self._update_positions()

                # ── Periodic Reports ───────────────────────────────────────
                self._periodic_reporting()

                # ── Sleep until next scan ──────────────────────────────────
                elapsed = (datetime.utcnow() - loop_start).total_seconds()
                sleep_time = max(0, scan_interval - elapsed)
                log.debug(f"Loop complete in {elapsed:.1f}s. Next scan in {sleep_time:.0f}s.")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Loop error: {e}", exc_info=True)
                time.sleep(30)

        self._shutdown()

    # ── Symbol Scan ───────────────────────────────────────────────────────────
    def _scan_all_symbols(self) -> None:
        """Scan all symbols for trade opportunities."""
        log.info(f"Scanning {len(self.symbols)} symbols...")

        for symbol in self.symbols:
            try:
                market = self.data_engine._detect_market(symbol)

                # Skip stocks if US market closed
                if market == "stock" and not self.clock.is_us_market_open():
                    continue

                # Fetch all timeframes
                data = self.data_engine.get_all_timeframes_ohlcv(
                    symbol, lookback_days=200
                )

                if not data:
                    log.debug(f"No data for {symbol}")
                    continue

                # Run signal engine
                signal = self.signal_engine.analyse(symbol, data, market)

                if signal:
                    self._process_signal(signal)

            except Exception as e:
                log.error(f"Error scanning {symbol}: {e}")

    # ── Signal Processing ─────────────────────────────────────────────────────
    def _process_signal(self, signal) -> None:
        """Process a trade signal through risk and execution."""
        log.info(f"Signal received: {signal.summary()}")

        # Risk evaluation
        from risk.risk_manager import TradeProposal
        proposal = TradeProposal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            probability=signal.probability,
            market=signal.market,
            timeframe=signal.timeframe,
        )

        # Update portfolio value from simulator
        self.risk_manager.update_portfolio_value(self.simulator.equity)
        approval = self.risk_manager.evaluate(proposal)

        if not approval.approved:
            log.info(f"Trade rejected by risk manager: {approval.rejection_reasons}")
            return

        # Execute in paper simulator
        side = "buy" if signal.direction == "long" else "sell"
        order = self.simulator.place_order(
            symbol=signal.symbol,
            side=side,
            qty=approval.position_size,
            order_type="market",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_price=signal.entry_price,
            market=signal.market,
            direction=signal.direction,
        )

        if order.status == "filled":
            log.info(
                f"✅ TRADE EXECUTED | {signal.symbol} {side.upper()} "
                f"{approval.position_size:.4f} @ {order.filled_price:.5f} | "
                f"Risk: ${approval.risk_amount:.2f}"
            )
            self.risk_manager.register_open_position({
                "symbol": signal.symbol,
                "market": signal.market,
                "direction": signal.direction,
            })

    # ── Position Updates ──────────────────────────────────────────────────────
    def _update_positions(self) -> None:
        """Update open positions with latest prices and check SL/TP."""
        if not self.simulator.open_positions:
            return

        price_feed = {}
        for symbol in self.simulator.open_positions:
            price = self.data_engine.get_latest_price(symbol)
            if price:
                price_feed[symbol] = price

        if price_feed:
            closed = self.simulator.update_positions(price_feed)
            for trade in closed:
                self.risk_manager.close_position(trade.symbol, trade.pnl)
                log.info(
                    f"Position closed: {trade.symbol} | "
                    f"P&L: ${trade.pnl:.2f} | {trade.exit_reason}"
                )

    # ── Reporting ─────────────────────────────────────────────────────────────
    def _periodic_reporting(self) -> None:
        """Export reports and log performance periodically."""
        trades = self.simulator.closed_trades
        if not trades:
            return

        # Log performance every 10 closed trades
        if len(trades) % 10 == 0:
            perf = self.simulator.get_performance()
            log.info(
                f"Performance Update | "
                f"Trades: {perf['total_trades']} | "
                f"Win Rate: {perf['win_rate']:.1%} | "
                f"P&L: ${perf['total_pnl']:,.2f} | "
                f"Sharpe: {perf['sharpe_ratio']:.2f}"
            )
            # Export Excel
            self.exporter.export(
                closed_trades=trades,
                open_positions=list(self.simulator.open_positions.values()),
                portfolio_value=self.simulator.equity,
                initial_capital=self.simulator.initial_capital,
            )

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        """Clean shutdown — export final report."""
        log.info("Shutting down trading agent...")

        perf = self.simulator.get_performance()
        log.info(f"Final Performance: {perf}")

        if self.simulator.closed_trades:
            path = self.exporter.export(
                closed_trades=self.simulator.closed_trades,
                portfolio_value=self.simulator.equity,
                initial_capital=self.simulator.initial_capital,
                filename="final_report.xlsx",
            )
            log.info(f"Final report exported: {path}")

        log.info("Agent shutdown complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Institutional Trading Agent")
    parser.add_argument(
        "--mode",
        choices=["paper", "backtest", "train", "live"],
        default="paper",
        help="Trading mode",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override symbol watchlist",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital (paper mode)",
    )
    return parser.parse_args()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    agent = TradingAgent(
        mode=args.mode,
        symbols=args.symbols,
    )

    if args.mode == "train":
        agent.initialise()
        agent.train_models()

    elif args.mode == "paper":
        agent.run()

    elif args.mode == "live":
        log.warning("=" * 50)
        log.warning("  ⚠️  LIVE MODE — REAL MONEY AT RISK  ⚠️")
        log.warning("  Only activate after paper trading")
        log.warning("  proves consistent 75%+ win rate.")
        log.warning("=" * 50)
        confirm = input("Type 'I UNDERSTAND' to proceed with live trading: ")
        if confirm == "I UNDERSTAND":
            config.mode.ACTIVE = TradingMode.LIVE
            agent.run()
        else:
            log.info("Live mode cancelled.")

    elif args.mode == "backtest":
        log.info("Backtest mode — coming in Phase 9.")