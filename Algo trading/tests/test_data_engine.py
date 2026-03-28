"""tests/test_data_engine.py — Unit tests for the data engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from core.data_engine import DataEngine, MarketData


@pytest.fixture
def engine():
    return DataEngine()


@pytest.fixture
def sample_ohlcv():
    idx  = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    data = {
        "open":   np.random.uniform(100, 110, 200),
        "high":   np.random.uniform(110, 120, 200),
        "low":    np.random.uniform(90,  100, 200),
        "close":  np.random.uniform(100, 110, 200),
        "volume": np.random.randint(1000, 10000, 200).astype(float),
    }
    # Fix OHLC relationships
    df = pd.DataFrame(data, index=idx)
    df["high"]  = df[["open", "close"]].max(axis=1) + np.random.uniform(0.5, 2, 200)
    df["low"]   = df[["open", "close"]].min(axis=1) - np.random.uniform(0.5, 2, 200)
    return df


class TestMarketDetection:
    def test_detects_stock(self, engine):
        assert engine._detect_market("AAPL")    == "stock"
        assert engine._detect_market("MSFT")    == "stock"
        assert engine._detect_market("SPY")     == "stock"

    def test_detects_crypto(self, engine):
        assert engine._detect_market("BTC/USDT") == "crypto"
        assert engine._detect_market("ETH/USDT") == "crypto"

    def test_detects_forex(self, engine):
        assert engine._detect_market("EUR/USD")  == "forex"
        assert engine._detect_market("GBP/USD")  == "forex"


class TestDataValidation:
    def test_clean_data_passes(self, engine, sample_ohlcv):
        result = engine._validate_and_clean(sample_ohlcv)
        assert not result.empty
        assert len(result) == len(sample_ohlcv)

    def test_removes_nan_rows(self, engine, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.iloc[5, 3] = float("nan")    # Corrupt one close
        result = engine._validate_and_clean(df)
        assert len(result) == len(sample_ohlcv) - 1

    def test_removes_invalid_ohlc(self, engine, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.iloc[0, 1] = df.iloc[0, 2] - 1    # high < low (invalid)
        result = engine._validate_and_clean(df)
        assert len(result) < len(sample_ohlcv)

    def test_removes_duplicates(self, engine, sample_ohlcv):
        df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:5]])
        result = engine._validate_and_clean(df)
        assert len(result) == len(sample_ohlcv)

    def test_quality_report(self, engine, sample_ohlcv):
        report = engine.validate_data_quality(sample_ohlcv)
        assert report["valid"] is True
        assert report["row_count"] == 200
        assert report["nan_pct"] == 0.0


class TestMarketData:
    def test_market_data_properties(self, sample_ohlcv):
        md = MarketData(symbol="AAPL", market="stock",
                        timeframe="1h", ohlcv=sample_ohlcv)
        assert md.last_close > 0
        assert md.last_high >= md.last_low
        assert md.is_valid is True

    def test_empty_market_data_invalid(self):
        md = MarketData(symbol="AAPL", market="stock",
                        timeframe="1h", ohlcv=pd.DataFrame())
        assert md.is_valid is False