"""
data_feeds/yfinance_feed.py
─────────────────────────────────────────────────────────────────────────────
yFinance data feed — free historical OHLCV data.
Primary source for backtesting across Stocks, Forex, and Crypto.
No API key required.
─────────────────────────────────────────────────────────────────────────────
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from core.logger import get_logger
from config import config

log = get_logger(__name__)


# ── Symbol Mapping ─────────────────────────────────────────────────────────────
# yFinance uses different symbol formats for Forex and Crypto
FOREX_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "GBP/JPY": "GBPJPY=X",
}

CRYPTO_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
    "XRP/USDT": "XRP-USD",
    "ADA/USDT": "ADA-USD",
}

YF_TF_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "1h",   # yFinance doesn't have 4h; we resample from 1h
    "1d":  "1d",
}


class YFinanceFeed:
    """
    yFinance historical data feed.
    Handles stocks, forex, and crypto with local caching.
    """

    def __init__(self):
        self.cache_dir = config.paths.historical
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Symbol Resolution ─────────────────────────────────────────────────────
    def _resolve_symbol(self, symbol: str) -> str:
        """Convert internal symbol format to yFinance format."""
        if symbol in FOREX_MAP:
            return FOREX_MAP[symbol]
        if symbol in CRYPTO_MAP:
            return CRYPTO_MAP[symbol]
        return symbol   # Stocks use ticker directly

    # ── Fetch Data ────────────────────────────────────────────────────────────
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 365,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.

        Args:
            symbol:       Internal symbol (e.g. "EUR/USD", "AAPL", "BTC/USDT")
            timeframe:    Timeframe string ("1m", "5m", "15m", "1h", "4h", "1d")
            lookback_days: How many days of history to fetch
            use_cache:    Load from disk cache if available

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        yf_symbol = self._resolve_symbol(symbol)
        yf_tf     = YF_TF_MAP.get(timeframe, "1d")
        resample_4h = (timeframe == "4h")

        # Check cache first
        if use_cache:
            cached = self._load_cache(symbol, timeframe)
            if cached is not None and not cached.empty:
                log.debug(f"Cache hit: {symbol} | {timeframe} ({len(cached)} bars)")
                return cached

        start = datetime.utcnow() - timedelta(days=lookback_days)

        try:
            ticker = yf.Ticker(yf_symbol)
            df     = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                interval=yf_tf,
                auto_adjust=True,
                prepost=False,
            )

            if df.empty:
                log.warning(f"yFinance returned empty data for {symbol} | {timeframe}")
                return pd.DataFrame()

            # Normalise columns
            df = df.rename(columns={
                "Open": "open", "High": "high",
                "Low": "low",   "Close": "close", "Volume": "volume"
            })
            df = df[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index().dropna()

            # Resample to 4H if requested
            if resample_4h:
                df = self._resample_to_4h(df)

            # Cache to disk
            if use_cache:
                self._save_cache(df, symbol, timeframe)

            log.debug(f"yFinance fetched {len(df)} bars | {symbol} | {timeframe}")
            return df

        except Exception as e:
            log.error(f"yFinance error for {symbol}: {e}")
            return pd.DataFrame()

    def get_all_timeframes(
        self, symbol: str, lookback_days: int = 365
    ) -> dict[str, pd.DataFrame]:
        """Fetch all timeframes for a symbol."""
        result = {}
        for tf in config.timeframes.ALL:
            # Intraday data limited to 60 days on yFinance free tier
            lb = min(lookback_days, 60) if tf in ["1m", "5m", "15m"] else lookback_days
            df = self.get_bars(symbol, timeframe=tf, lookback_days=lb)
            if not df.empty:
                result[tf] = df
        return result

    def get_multiple_symbols(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        lookback_days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch multiple symbols."""
        result = {}
        for sym in symbols:
            df = self.get_bars(sym, timeframe=timeframe, lookback_days=lookback_days)
            if not df.empty:
                result[sym] = df
        return result

    # ── Resampling ────────────────────────────────────────────────────────────
    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H data to 4H OHLCV."""
        return df.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

    # ── Caching ───────────────────────────────────────────────────────────────
    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _save_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        try:
            path = self._cache_path(symbol, timeframe)
            df.to_parquet(path)
        except Exception as e:
            log.warning(f"Failed to cache {symbol} | {timeframe}: {e}")

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Load cached data if it exists and is less than 1 hour old."""
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return None
        # Only use cache if file is fresh (< 1 hour)
        age_seconds = (datetime.utcnow().timestamp() - path.stat().st_mtime)
        if age_seconds > 3600:
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def clear_cache(self) -> None:
        """Clear all cached data files."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        log.info("yFinance cache cleared.")


# Singleton
yfinance_feed = YFinanceFeed()