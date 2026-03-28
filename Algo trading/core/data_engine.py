"""
core/data_engine.py
─────────────────────────────────────────────────────────────────────────────
Master data engine — the central hub for all market data.
Routes data requests to the correct feed (Alpaca / CCXT / yFinance),
normalises everything into a unified format, and serves it to
the intelligence layer.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class MarketData:
    """
    Unified market data object — standard format used by ALL layers.
    Every feed normalises into this structure.
    """
    symbol:     str
    market:     str           # "stock" | "forex" | "crypto"
    timeframe:  str
    ohlcv:      pd.DataFrame  # columns: open, high, low, close, volume
    timestamp:  pd.Timestamp  = field(default_factory=pd.Timestamp.utcnow)
    bid:        Optional[float] = None
    ask:        Optional[float] = None
    spread:     Optional[float] = None
    orderbook:  Optional[dict]  = None

    @property
    def last_close(self) -> float:
        return float(self.ohlcv["close"].iloc[-1])

    @property
    def last_high(self) -> float:
        return float(self.ohlcv["high"].iloc[-1])

    @property
    def last_low(self) -> float:
        return float(self.ohlcv["low"].iloc[-1])

    @property
    def last_volume(self) -> float:
        return float(self.ohlcv["volume"].iloc[-1])

    @property
    def is_valid(self) -> bool:
        return (
            not self.ohlcv.empty
            and len(self.ohlcv) >= config.timeframes.MIN_CANDLES
        )


class DataEngine:
    """
    Master data router and normaliser.
    
    Detects the market type for each symbol and routes to the
    correct data feed. Returns unified MarketData objects.
    """

    def __init__(self):
        # Lazy imports to avoid circular dependencies
        self._alpaca_feed  = None
        self._ccxt_feed    = None
        self._yfinance_feed = None
        self._feeds_ready  = False

    def _init_feeds(self):
        """Lazy initialise all feeds."""
        if self._feeds_ready:
            return
        from data_feeds.alpaca_feed   import alpaca_feed
        from data_feeds.ccxt_feed     import ccxt_feed
        from data_feeds.yfinance_feed import yfinance_feed
        self._alpaca_feed   = alpaca_feed
        self._ccxt_feed     = ccxt_feed
        self._yfinance_feed = yfinance_feed
        self._feeds_ready   = True

    # ── Market Type Detection ─────────────────────────────────────────────────
    def _detect_market(self, symbol: str) -> str:
        if symbol in config.markets.CRYPTO or "/" in symbol and "USD" in symbol and len(symbol) > 8:
            return "crypto"
        if symbol in config.markets.FOREX or "=X" in symbol:
            return "forex"
        return "stock"

    # ── Primary Data Fetch ────────────────────────────────────────────────────
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 200,
        live: bool = False,
    ) -> MarketData:
        """
        Fetch OHLCV bars for any symbol on any market.
        Automatically routes to correct feed.

        Args:
            symbol:       e.g. "AAPL", "EUR/USD", "BTC/USDT"
            timeframe:    "1m" to "1d"
            lookback_days: history depth
            live:         if True, use live feeds; else use yFinance

        Returns:
            Normalised MarketData object
        """
        self._init_feeds()
        market = self._detect_market(symbol)

        try:
            if live and market == "stock":
                df = self._alpaca_feed.get_bars(symbol, timeframe, lookback_days)
            elif live and market == "crypto":
                df = self._ccxt_feed.get_bars(symbol, timeframe, lookback_days=lookback_days)
            else:
                # Default: yFinance for all (backtesting + paper mode)
                df = self._yfinance_feed.get_bars(symbol, timeframe, lookback_days)

            # Validate
            if df.empty:
                log.warning(f"Empty data for {symbol} | {timeframe}")
                return MarketData(symbol=symbol, market=market,
                                  timeframe=timeframe, ohlcv=pd.DataFrame())

            df = self._validate_and_clean(df)

            return MarketData(
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                ohlcv=df,
                timestamp=pd.Timestamp.utcnow(),
            )

        except Exception as e:
            log.error(f"DataEngine error fetching {symbol} | {timeframe}: {e}")
            return MarketData(symbol=symbol, market=market,
                              timeframe=timeframe, ohlcv=pd.DataFrame())

    def get_all_timeframes(
        self,
        symbol: str,
        lookback_days: int = 200,
        live: bool = False,
    ) -> dict[str, MarketData]:
        """
        Fetch all timeframes for a symbol.
        Returns dict: {timeframe: MarketData}
        """
        result = {}
        for tf in config.timeframes.ALL:
            md = self.get_bars(symbol, tf, lookback_days, live)
            if md.is_valid:
                result[tf] = md
        log.info(f"Loaded {len(result)} timeframes for {symbol}")
        return result

    def get_all_timeframes_ohlcv(
        self,
        symbol: str,
        lookback_days: int = 200,
        live: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Convenience method — returns raw DataFrames instead of MarketData objects.
        Used by TimeframeScanner.
        """
        all_tf = self.get_all_timeframes(symbol, lookback_days, live)
        return {tf: md.ohlcv for tf, md in all_tf.items()}

    def get_universe(
        self,
        live: bool = False,
        lookback_days: int = 200,
    ) -> dict[str, dict[str, MarketData]]:
        """
        Fetch all configured symbols across all markets.
        Returns {symbol: {timeframe: MarketData}}
        """
        universe   = {}
        all_symbols = (
            config.markets.STOCKS
            + config.markets.FOREX
            + config.markets.CRYPTO
        )

        for symbol in all_symbols:
            log.info(f"Loading universe data: {symbol}")
            universe[symbol] = self.get_all_timeframes(symbol, lookback_days, live)

        log.info(f"Universe loaded: {len(universe)} symbols.")
        return universe

    # ── Latest Prices ─────────────────────────────────────────────────────────
    def get_latest_price(self, symbol: str) -> float | None:
        """Get current market price for any symbol."""
        self._init_feeds()
        market = self._detect_market(symbol)
        try:
            if market == "stock":
                return self._alpaca_feed.get_latest_price(symbol)
            elif market == "crypto":
                return self._ccxt_feed.get_latest_price(symbol)
            else:
                # Forex — use yFinance latest
                df = self._yfinance_feed.get_bars(symbol, "1m", lookback_days=1)
                return float(df["close"].iloc[-1]) if not df.empty else None
        except Exception as e:
            log.error(f"Failed to get latest price for {symbol}: {e}")
            return None

    # ── Data Quality ──────────────────────────────────────────────────────────
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        - Remove NaN rows
        - Ensure OHLC relationships are valid (high >= low, etc.)
        - Remove duplicate indices
        - Sort by timestamp
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                log.warning(f"Missing column: {col}")
                return pd.DataFrame()

        # Drop duplicates and NaNs
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=required_cols)
        df = df.sort_index()

        # Validate OHLC relationships
        invalid_mask = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["volume"] < 0)
        )
        if invalid_mask.sum() > 0:
            log.warning(f"Removed {invalid_mask.sum()} invalid OHLC rows.")
            df = df[~invalid_mask]

        return df

    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """Return data quality report for a DataFrame."""
        if df.empty:
            return {"valid": False, "reason": "Empty DataFrame"}

        gaps    = df.index.to_series().diff().dt.total_seconds().dropna()
        nan_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))

        return {
            "valid":       len(df) >= config.timeframes.MIN_CANDLES,
            "row_count":   len(df),
            "nan_pct":     round(nan_pct * 100, 2),
            "date_from":   df.index[0].isoformat(),
            "date_to":     df.index[-1].isoformat(),
            "avg_gap_sec": round(gaps.mean(), 1) if not gaps.empty else None,
            "max_gap_sec": round(gaps.max(), 1) if not gaps.empty else None,
        }


# Singleton
data_engine = DataEngine()