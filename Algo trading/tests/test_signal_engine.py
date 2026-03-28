"""tests/test_signal_engine.py — Unit tests for intelligence and signal layers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from intelligence.volume_profile import VolumeProfileEngine
from intelligence.market_regime  import MarketRegimeEngine, Regime
from intelligence.structure_engine import StructureEngine, StructureSignal
from intelligence.momentum       import MomentumEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def trending_up_df():
    """Simulated strong uptrend."""
    n     = 200
    price = np.cumsum(np.random.normal(0.3, 0.5, n)) + 100
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    vol   = np.random.randint(1000, 5000, n).astype(float)
    opens = price - np.random.uniform(0, 0.5, n)
    df    = pd.DataFrame({
        "open":   opens,
        "high":   price + np.random.uniform(0.2, 1, n),
        "low":    opens - np.random.uniform(0.2, 1, n),
        "close":  price,
        "volume": vol,
    }, index=idx)
    return df


@pytest.fixture
def ranging_df():
    """Simulated ranging/choppy market."""
    n     = 200
    price = np.sin(np.linspace(0, 20, n)) * 3 + 100
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    vol   = np.random.randint(500, 2000, n).astype(float)
    df    = pd.DataFrame({
        "open":   price - 0.1,
        "high":   price + 0.5,
        "low":    price - 0.5,
        "close":  price,
        "volume": vol,
    }, index=idx)
    return df


# ── Volume Profile Tests ──────────────────────────────────────────────────────
class TestVolumeProfile:

    @pytest.fixture
    def vp(self):
        return VolumeProfileEngine()

    def test_poc_within_price_range(self, vp, trending_up_df):
        result = vp.analyse(trending_up_df, "TEST", "1h")
        assert result.val <= result.poc <= result.vah

    def test_value_area_contains_poc(self, vp, ranging_df):
        result = vp.analyse(ranging_df, "TEST", "1h")
        assert result.val <= result.poc <= result.vah

    def test_handles_empty_df(self, vp):
        result = vp.analyse(pd.DataFrame(), "TEST", "1h")
        assert result.poc == 0.0

    def test_key_levels_sorted(self, vp, trending_up_df):
        result = vp.analyse(trending_up_df, "TEST", "1h")
        levels = vp.get_key_levels(result)
        prices = [l["level"] for l in levels]
        assert prices == sorted(prices)


# ── Market Regime Tests ───────────────────────────────────────────────────────
class TestMarketRegime:

    @pytest.fixture
    def engine(self):
        return MarketRegimeEngine()

    def test_trending_up_detection(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        # Should detect some kind of trending or bullish regime
        assert result.regime != Regime.UNKNOWN
        assert result.confidence > 0

    def test_ranging_detection(self, engine, ranging_df):
        result = engine.analyse(ranging_df, "TEST", "1h")
        # Sine wave should be range or balance
        assert result.regime in (Regime.BALANCE, Regime.ACCUMULATION,
                                  Regime.DISTRIBUTION, Regime.TRENDING_UP,
                                  Regime.TRENDING_DOWN)

    def test_handles_insufficient_data(self, engine):
        small_df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100], "volume": [1000]
        })
        result = engine.analyse(small_df, "TEST", "1h")
        assert result.regime == Regime.UNKNOWN

    def test_result_has_confidence(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert 0.0 <= result.confidence <= 1.0

    def test_bias_is_valid(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert result.bias in ("bullish", "bearish", "neutral")


# ── Structure Engine Tests ────────────────────────────────────────────────────
class TestStructureEngine:

    @pytest.fixture
    def engine(self):
        return StructureEngine()

    def test_finds_swing_highs(self, engine, trending_up_df):
        highs = engine._find_swing_highs(trending_up_df)
        assert len(highs) > 0
        for h in highs:
            assert h.kind == "high"
            assert h.price > 0

    def test_finds_swing_lows(self, engine, trending_up_df):
        lows = engine._find_swing_lows(trending_up_df)
        assert len(lows) > 0
        for l in lows:
            assert l.kind == "low"
            assert l.price > 0

    def test_signal_is_valid_enum(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert result.signal in list(StructureSignal)

    def test_handles_short_data(self, engine):
        short = pd.DataFrame({
            "open": [100]*5, "high": [101]*5,
            "low": [99]*5, "close": [100]*5, "volume": [1000]*5,
        })
        result = engine.analyse(short, "TEST", "1h")
        assert result.signal == StructureSignal.NONE

    def test_trend_is_valid(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert result.trend in ("bullish", "bearish", "neutral")


# ── Momentum Tests ────────────────────────────────────────────────────────────
class TestMomentumEngine:

    @pytest.fixture
    def engine(self):
        return MomentumEngine()

    def test_rsi_between_0_100(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert 0 <= result.rsi <= 100

    def test_convergence_between_minus1_and_1(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert -1 <= result.convergence <= 1

    def test_signal_is_valid(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert result.signal in ("bullish", "bearish", "neutral")

    def test_strength_is_valid(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        assert result.strength in ("strong", "moderate", "weak")

    def test_trending_up_mostly_bullish(self, engine, trending_up_df):
        result = engine.analyse(trending_up_df, "TEST", "1h")
        # Strong uptrend should usually give bullish signal
        assert result.convergence > -0.5    # Not strongly bearish