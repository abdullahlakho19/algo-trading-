"""
data_feeds/ccxt_feed.py
─────────────────────────────────────────────────────────────────────────────
Crypto data feed via CCXT library (Binance).
Supports historical OHLCV, live orderbook, and ticker streaming.
Testnet by default — switch to live when ready.
─────────────────────────────────────────────────────────────────────────────
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from core.logger import get_logger
from config import config

log = get_logger(__name__)


# ── Timeframe Mapping ─────────────────────────────────────────────────────────
CCXT_TF_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}


class CCXTFeed:
    """
    Crypto market data feed via CCXT (Binance).
    Handles OHLCV, orderbook depth, and ticker data.
    """

    def __init__(self):
        self.exchange_id = config.markets.CRYPTO_EXCHANGE
        self.api_key     = config.binance.API_KEY
        self.secret_key  = config.binance.SECRET_KEY
        self.testnet     = config.binance.TESTNET
        self.exchange    = None
        self._connected  = False

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """Initialise CCXT exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange  = exchange_class({
                "apiKey":    self.api_key,
                "secret":    self.secret_key,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                },
            })

            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                log.info("CCXT Binance connected. Mode: TESTNET")
            else:
                log.info("CCXT Binance connected. Mode: LIVE")

            # Load markets
            self.exchange.load_markets()
            self._connected = True
            return True

        except Exception as e:
            log.error(f"CCXT connection failed: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False
        log.info("CCXT Feed disconnected.")

    # ── Historical OHLCV ─────────────────────────────────────────────────────
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a crypto pair.
        symbol format: "BTC/USDT"

        Returns DataFrame: open, high, low, close, volume (UTC index)
        """
        if not self._connected:
            self.connect()

        ccxt_tf = CCXT_TF_MAP.get(timeframe)
        if ccxt_tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        since = int(
            (datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000
        )

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=ccxt_tf,
                since=since,
                limit=limit,
            )

            if not ohlcv:
                log.warning(f"No OHLCV data returned for {symbol} | {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()

            log.debug(f"Fetched {len(df)} bars | {symbol} | {timeframe}")
            return df

        except ccxt.NetworkError as e:
            log.error(f"CCXT network error for {symbol}: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            log.error(f"CCXT exchange error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Unexpected CCXT error for {symbol}: {e}")
            return pd.DataFrame()

    def get_all_timeframes(
        self, symbol: str, lookback_days: int = 200
    ) -> dict[str, pd.DataFrame]:
        """Fetch all timeframes for a crypto symbol."""
        result = {}
        for tf in config.timeframes.ALL:
            df = self.get_bars(symbol, timeframe=tf, lookback_days=lookback_days)
            if not df.empty:
                result[tf] = df
            time.sleep(self.exchange.rateLimit / 1000 if self.exchange else 0.5)
        return result

    # ── Ticker ────────────────────────────────────────────────────────────────
    def get_ticker(self, symbol: str) -> dict | None:
        """Get latest ticker for a symbol."""
        if not self._connected:
            self.connect()
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol":    symbol,
                "bid":       ticker["bid"],
                "ask":       ticker["ask"],
                "last":      ticker["last"],
                "volume_24h": ticker["baseVolume"],
                "change_24h": ticker["percentage"],
                "timestamp": pd.to_datetime(ticker["timestamp"], unit="ms", utc=True),
            }
        except Exception as e:
            log.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> float | None:
        ticker = self.get_ticker(symbol)
        if ticker:
            return (ticker["bid"] + ticker["ask"]) / 2
        return None

    # ── Order Book ────────────────────────────────────────────────────────────
    def get_orderbook(self, symbol: str, depth: int = 20) -> dict | None:
        """
        Fetch order book depth for microstructure analysis.
        Returns bids and asks with cumulative volume.
        """
        if not self._connected:
            self.connect()
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=depth)

            bids = pd.DataFrame(ob["bids"], columns=["price", "size"])
            asks = pd.DataFrame(ob["asks"], columns=["price", "size"])

            total_bid_vol = bids["size"].sum()
            total_ask_vol = asks["size"].sum()

            # Order book imbalance: positive = more buyers, negative = more sellers
            imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

            return {
                "symbol":        symbol,
                "bids":          bids,
                "asks":          asks,
                "best_bid":      ob["bids"][0][0] if ob["bids"] else None,
                "best_ask":      ob["asks"][0][0] if ob["asks"] else None,
                "spread":        ob["asks"][0][0] - ob["bids"][0][0] if ob["bids"] and ob["asks"] else None,
                "bid_volume":    total_bid_vol,
                "ask_volume":    total_ask_vol,
                "imbalance":     round(imbalance, 4),
                "timestamp":     pd.Timestamp.utcnow(),
            }
        except Exception as e:
            log.error(f"Failed to get orderbook for {symbol}: {e}")
            return None

    # ── Account & Positions ───────────────────────────────────────────────────
    def get_balance(self) -> dict | None:
        """Get account balances for all currencies."""
        if not self._connected:
            self.connect()
        try:
            balance = self.exchange.fetch_balance()
            return {k: v for k, v in balance["total"].items() if v > 0}
        except Exception as e:
            log.error(f"Failed to get balance: {e}")
            return None

    # ── Status ────────────────────────────────────────────────────────────────
    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> dict:
        return {
            "connected":   self._connected,
            "exchange":    self.exchange_id,
            "testnet":     self.testnet,
            "api_key":     self.api_key[:8] + "..." if self.api_key else "NOT SET",
        }


# Singleton
ccxt_feed = CCXTFeed()