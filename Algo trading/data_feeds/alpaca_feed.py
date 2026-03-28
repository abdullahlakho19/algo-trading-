"""
data_feeds/alpaca_feed.py
─────────────────────────────────────────────────────────────────────────────
Alpaca Markets data feed — Stocks and Forex.
Handles both REST (historical) and WebSocket (live streaming) connections.
Supports paper and live trading modes.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import (
    StockBarsRequest, StockLatestQuoteRequest, StockLatestBarRequest
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.common.exceptions import APIError
from core.logger import get_logger
from config import config

log = get_logger(__name__)


# ── Timeframe Mapping ─────────────────────────────────────────────────────────
TF_MAP = {
    "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
    "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
    "1d":  TimeFrame(1,  TimeFrameUnit.Day),
}


class AlpacaFeed:
    """
    Alpaca Markets data feed.
    Provides historical OHLCV data and live streaming quotes.
    """

    def __init__(self):
        self.api_key    = config.alpaca.API_KEY
        self.secret_key = config.alpaca.SECRET_KEY
        self.paper_mode = config.alpaca.PAPER_MODE

        # REST clients
        self._hist_client:    Optional[StockHistoricalDataClient] = None
        self._trading_client: Optional[TradingClient]             = None
        self._stream:         Optional[StockDataStream]           = None

        self._connected = False

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """Initialise REST clients. Returns True on success."""
        try:
            self._hist_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_mode,
            )
            self._connected = True
            mode = "PAPER" if self.paper_mode else "LIVE"
            log.info(f"Alpaca Feed connected. Mode: {mode}")
            return True
        except Exception as e:
            log.error(f"Alpaca connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._stream:
            self._stream.stop()
        self._connected = False
        log.info("Alpaca Feed disconnected.")

    # ── Historical Data ───────────────────────────────────────────────────────
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 90,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a stock symbol.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (UTC)
        """
        if not self._connected:
            self.connect()

        tf = TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        start = datetime.utcnow() - timedelta(days=lookback_days)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                limit=limit,
                feed="iex",       # Use IEX for free tier; switch to "sip" for paid
            )
            bars = self._hist_client.get_stock_bars(request)
            df   = bars.df

            if df.empty:
                log.warning(f"No data returned for {symbol} | {timeframe}")
                return pd.DataFrame()

            # Normalise column names
            df = df.rename(columns={
                "open": "open", "high": "high",
                "low": "low",   "close": "close", "volume": "volume"
            })

            # Drop multi-index if present (Alpaca returns (symbol, timestamp))
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)

            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()

            log.debug(f"Fetched {len(df)} bars | {symbol} | {timeframe}")
            return df[["open", "high", "low", "close", "volume"]]

        except APIError as e:
            log.error(f"Alpaca API error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Unexpected error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_all_timeframes(
        self,
        symbol: str,
        lookback_days: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for ALL timeframes for a given symbol.
        Returns dict: {timeframe_str: DataFrame}
        """
        result = {}
        for tf in config.timeframes.ALL:
            df = self.get_bars(symbol, timeframe=tf, lookback_days=lookback_days)
            if not df.empty:
                result[tf] = df
            time.sleep(0.2)    # Rate limiting
        return result

    # ── Latest Quote ──────────────────────────────────────────────────────────
    def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest mid price for a symbol."""
        if not self._connected:
            self.connect()
        try:
            req   = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self._hist_client.get_stock_latest_quote(req)
            snap  = quote[symbol]
            return (snap.bid_price + snap.ask_price) / 2
        except Exception as e:
            log.error(f"Failed to get latest price for {symbol}: {e}")
            return None

    def get_latest_bar(self, symbol: str) -> dict | None:
        """Get the latest OHLCV bar."""
        if not self._connected:
            self.connect()
        try:
            req = StockLatestBarRequest(symbol_or_symbols=symbol)
            bar = self._hist_client.get_stock_latest_bar(req)
            b   = bar[symbol]
            return {
                "open":   b.open,
                "high":   b.high,
                "low":    b.low,
                "close":  b.close,
                "volume": b.volume,
                "time":   b.timestamp,
            }
        except Exception as e:
            log.error(f"Failed to get latest bar for {symbol}: {e}")
            return None

    # ── Account Info ──────────────────────────────────────────────────────────
    def get_account(self) -> dict | None:
        """Return account equity, buying power, and portfolio value."""
        if not self._trading_client:
            self.connect()
        try:
            acct = self._trading_client.get_account()
            return {
                "equity":        float(acct.equity),
                "buying_power":  float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "cash":          float(acct.cash),
                "paper":         self.paper_mode,
            }
        except Exception as e:
            log.error(f"Failed to get account info: {e}")
            return None

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        if not self._trading_client:
            self.connect()
        try:
            positions = self._trading_client.get_all_positions()
            return [
                {
                    "symbol":    p.symbol,
                    "qty":       float(p.qty),
                    "side":      p.side.value,
                    "avg_entry": float(p.avg_entry_price),
                    "market_val": float(p.market_value),
                    "unrealised_pl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return []

    # ── Live Stream ───────────────────────────────────────────────────────────
    def start_stream(
        self,
        symbols: list[str],
        on_bar: Callable,
        on_quote: Optional[Callable] = None,
    ) -> None:
        """
        Start WebSocket stream for live bar/quote data.
        on_bar and on_quote are async callback functions.
        """
        self._stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.secret_key,
            feed="iex",
        )

        self._stream.subscribe_bars(on_bar, *symbols)
        if on_quote:
            self._stream.subscribe_quotes(on_quote, *symbols)

        log.info(f"Starting Alpaca WebSocket stream for: {symbols}")
        self._stream.run()

    def stop_stream(self) -> None:
        if self._stream:
            self._stream.stop()
            log.info("Alpaca WebSocket stream stopped.")

    # ── Status ────────────────────────────────────────────────────────────────
    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> dict:
        return {
            "connected":  self._connected,
            "paper_mode": self.paper_mode,
            "api_key":    self.api_key[:8] + "..." if self.api_key else "NOT SET",
        }


# Singleton
alpaca_feed = AlpacaFeed()