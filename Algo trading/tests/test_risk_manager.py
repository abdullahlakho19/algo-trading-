"""tests/test_risk_manager.py — Unit tests for risk management."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from risk.risk_manager import RiskManager, TradeProposal, RiskApproval
from risk.circuit_breaker import CircuitBreaker
from risk.position_sizer import PositionSizer
from risk.sl_tp_engine import SLTPEngine


# ── RiskManager Tests ─────────────────────────────────────────────────────────
class TestRiskManager:

    @pytest.fixture
    def rm(self):
        return RiskManager(portfolio_value=100_000.0)

    @pytest.fixture
    def good_trade(self):
        return TradeProposal(
            symbol="AAPL", direction="long",
            entry_price=150.0, stop_loss=147.0,
            take_profit=156.0, probability=0.78,
            market="stock", timeframe="1h",
        )

    def test_approves_valid_trade(self, rm, good_trade):
        result = rm.evaluate(good_trade)
        assert result.approved is True
        assert result.position_size > 0
        assert result.risk_pct <= 0.01    # Max 1%

    def test_rejects_low_rr(self, rm):
        bad = TradeProposal(
            symbol="AAPL", direction="long",
            entry_price=150.0, stop_loss=149.5,
            take_profit=150.5,    # 1:1 R/R — too low
            probability=0.8, market="stock", timeframe="1h",
        )
        result = rm.evaluate(bad)
        assert result.approved is False
        assert any("R/R" in r for r in result.rejection_reasons)

    def test_rejects_zero_sl(self, rm):
        bad = TradeProposal(
            symbol="AAPL", direction="long",
            entry_price=150.0, stop_loss=150.0,
            take_profit=160.0,
            probability=0.8, market="stock", timeframe="1h",
        )
        result = rm.evaluate(bad)
        assert result.approved is False

    def test_risk_amount_correct(self, rm, good_trade):
        result = rm.evaluate(good_trade)
        # Risk = size × SL distance
        sl_dist = abs(good_trade.entry_price - good_trade.stop_loss)
        expected_risk = result.position_size * sl_dist
        assert abs(result.risk_amount - expected_risk) < 0.01

    def test_circuit_breaker_blocks_trade(self, rm, good_trade):
        rm.circuit_breaker._is_tripped = True
        rm.circuit_breaker._reason = "Test trip"
        result = rm.evaluate(good_trade)
        assert result.approved is False
        assert "Circuit breaker" in result.rejection_reasons[0]
        rm.circuit_breaker._is_tripped = False

    def test_calculates_sl_tp(self, rm):
        sl, tp = rm.calculate_sl_tp(
            entry=100.0, direction="long", atr=2.0
        )
        assert sl < 100.0    # SL below entry for long
        assert tp > 100.0    # TP above entry for long
        rr = (tp - 100) / (100 - sl)
        assert rr >= 1.5     # Minimum R/R


# ── CircuitBreaker Tests ──────────────────────────────────────────────────────
class TestCircuitBreaker:

    @pytest.fixture
    def cb(self):
        breaker = CircuitBreaker()
        breaker._peak_equity = 100_000.0
        return breaker

    def test_not_tripped_initially(self, cb):
        assert cb.is_tripped is False

    def test_trips_on_daily_loss(self, cb):
        cb.record_trade(pnl=-3_500.0, equity=96_500.0)    # -3.5%
        assert cb.is_tripped is True
        assert "Daily" in cb.reason

    def test_trips_on_consecutive_losses(self, cb):
        for _ in range(5):
            cb.record_trade(pnl=-100.0, equity=99_900.0)
        assert cb.is_tripped is True
        assert "consecutive" in cb.reason.lower()

    def test_manual_reset(self, cb):
        cb._is_tripped = True
        cb.manual_reset()
        assert cb.is_tripped is False

    def test_does_not_trip_on_small_loss(self, cb):
        cb.record_trade(pnl=-500.0, equity=99_500.0)    # -0.5%
        assert cb.is_tripped is False

    def test_status_report(self, cb):
        status = cb.get_status()
        assert "is_tripped" in status
        assert "daily_pnl_pct" in status
        assert "consecutive_losses" in status


# ── PositionSizer Tests ───────────────────────────────────────────────────────
class TestPositionSizer:

    @pytest.fixture
    def sizer(self):
        return PositionSizer()

    def test_atr_sizing_respects_risk_limit(self, sizer):
        result = sizer.size_atr(
            symbol="AAPL", entry=150.0, stop_loss=147.0,
            portfolio_val=100_000.0, atr=3.0,
        )
        assert result.risk_pct <= 0.01    # Max 1%
        assert result.units > 0

    def test_kelly_sizing_positive(self, sizer):
        result = sizer.size_kelly(
            symbol="AAPL", entry=150.0, stop_loss=147.0,
            take_profit=159.0, portfolio_val=100_000.0,
            win_rate=0.65,
        )
        assert result.units >= 0

    def test_size_zero_on_zero_sl(self, sizer):
        result = sizer.size_atr(
            symbol="AAPL", entry=150.0, stop_loss=150.0,
            portfolio_val=100_000.0, atr=3.0,
        )
        # Should not crash, should return something reasonable
        assert result.units >= 0

    def test_correlation_reduction(self, sizer):
        base = sizer.size_atr("AAPL", 150.0, 147.0, 100_000.0, 3.0)
        original_units = base.units
        reduced = sizer.apply_correlation_reduction(base, correlation=0.85)
        assert reduced.units < original_units


# ── SLTPEngine Tests ──────────────────────────────────────────────────────────
class TestSLTPEngine:

    @pytest.fixture
    def sltp(self):
        return SLTPEngine()

    def test_long_sl_below_entry(self, sltp):
        result = sltp.atr_based(100.0, "long", atr=2.0)
        assert result.stop_loss < result.entry
        assert result.take_profit > result.entry

    def test_short_sl_above_entry(self, sltp):
        result = sltp.atr_based(100.0, "short", atr=2.0)
        assert result.stop_loss > result.entry
        assert result.take_profit < result.entry

    def test_minimum_rr_enforced(self, sltp):
        result = sltp.atr_based(100.0, "long", atr=2.0, sl_mult=1.5, tp_mult=3.0)
        assert result.risk_reward >= 2.0

    def test_trailing_stop_moves_up_for_long(self, sltp):
        initial_sl = 95.0
        new_sl     = sltp.trailing_stop_level(
            entry=100.0, current_price=108.0, direction="long",
            initial_sl=initial_sl, atr=2.0,
        )
        assert new_sl > initial_sl    # Stop moved up

    def test_trailing_stop_never_moves_against(self, sltp):
        initial_sl = 95.0
        # Price drops — stop should not move down
        new_sl = sltp.trailing_stop_level(
            entry=100.0, current_price=97.0, direction="long",
            initial_sl=initial_sl, atr=2.0,
        )
        assert new_sl >= initial_sl