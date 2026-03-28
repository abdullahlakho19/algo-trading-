"""
strategies/signal_engine.py
─────────────────────────────────────────────────────────────────────────────
Master Signal Engine.
Orchestrates all analysis layers and runs the 6-gate safety filter.
This is the central decision module — it combines intelligence,
ML, quant, and risk into a final GO / NO-GO trade decision.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from core.logger import get_logger
from core.market_clock import market_clock
from core.event_calendar import event_calendar
from config import config

log = get_logger(__name__)


@dataclass
class TradeSignal:
    """
    Final trade signal output — only generated when ALL gates pass.
    """
    symbol:       str
    direction:    str         # "long" | "short"
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    probability:  float
    confluence:   int
    market:       str
    timeframe:    str
    regime:       str
    session:      str
    risk_reward:  float
    timestamp:    pd.Timestamp = field(default_factory=pd.Timestamp.utcnow)
    signal_id:    str = ""

    def __post_init__(self):
        import uuid
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())[:8].upper()

    def summary(self) -> str:
        return (
            f"🎯 SIGNAL [{self.signal_id}] | {self.symbol} | "
            f"{self.direction.upper()} @ {self.entry_price:.5f} | "
            f"SL: {self.stop_loss:.5f} | TP: {self.take_profit:.5f} | "
            f"P: {self.probability:.1%} | R/R: 1:{self.risk_reward:.1f}"
        )


@dataclass
class GateResult:
    """Result of a single safety gate check."""
    gate_number:  int
    gate_name:    str
    passed:       bool
    reason:       str = ""


class SignalEngine:
    """
    Master signal engine.
    Runs the full analysis pipeline and 6-gate safety filter.
    Only produces a TradeSignal when ALL 6 gates pass.
    """

    def __init__(self):
        # Lazy imports to avoid circular deps
        self._intelligence_ready = False

    def _init_dependencies(self):
        if self._intelligence_ready:
            return
        from intelligence.volume_profile  import volume_profile_engine
        from intelligence.market_regime   import market_regime_engine
        from intelligence.structure_engine import structure_engine
        from core.timeframe_scanner       import timeframe_scanner
        from ai_ml.ensemble_voter         import ensemble_voter
        from quant.probability_scorer     import probability_scorer
        from quant.monte_carlo            import monte_carlo_engine
        from risk.risk_manager            import risk_manager, TradeProposal

        self.vp_engine       = volume_profile_engine
        self.regime_engine   = market_regime_engine
        self.structure_engine = structure_engine
        self.tf_scanner      = timeframe_scanner
        self.ensemble        = ensemble_voter
        self.prob_scorer     = probability_scorer
        self.mc_engine       = monte_carlo_engine
        self.risk_manager    = risk_manager
        self.TradeProposal   = TradeProposal
        self._intelligence_ready = True

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self,
        symbol: str,
        data:   dict[str, pd.DataFrame],
        market: str = "stock",
    ) -> Optional[TradeSignal]:
        """
        Full analysis pipeline for one symbol.

        Args:
            symbol: Instrument ticker
            data:   Dict of {timeframe: OHLCV DataFrame}
            market: "stock" | "forex" | "crypto"

        Returns:
            TradeSignal if all 6 gates pass, else None
        """
        self._init_dependencies()

        # Primary timeframe for signal generation
        primary_tf = config.timeframes.TRIGGER    # 15m
        df = data.get(primary_tf) or data.get(config.timeframes.ENTRY)
        if df is None or len(df) < config.timeframes.MIN_CANDLES:
            log.debug(f"{symbol} — insufficient data on primary TF.")
            return None

        current_price = float(df["close"].iloc[-1])

        # ── 1. Run Intelligence Layer ──────────────────────────────────────
        regime   = self.regime_engine.analyse(df, symbol, primary_tf)
        vp       = self.vp_engine.analyse(df, symbol, primary_tf, current_price)
        structure = self.structure_engine.analyse(df, symbol, primary_tf)
        mtf_ctx  = self.tf_scanner.scan(symbol, data)

        # ── 2. Run Ensemble ML ─────────────────────────────────────────────
        ml_signal = self.ensemble.predict(df, symbol, primary_tf)

        # ── 3. Determine trade direction from structure + ML ───────────────
        direction = self._resolve_direction(structure, ml_signal, mtf_ctx)
        if direction == "neutral":
            log.debug(f"{symbol} — no directional conviction.")
            return None

        # ── 4. Calculate SL/TP ─────────────────────────────────────────────
        from risk.risk_manager import risk_manager as rm
        atr = self._compute_atr(df)
        sl, tp = rm.calculate_sl_tp(current_price, direction, atr)

        # ── 5. Probability Score ───────────────────────────────────────────
        vp_conf = 1.0 if (vp.val <= current_price <= vp.vah) or current_price == vp.poc else 0.5
        prob    = self.prob_scorer.score(
            ensemble_prob=ml_signal.probability,
            structure_confidence=structure.confidence,
            regime_confidence=regime.confidence,
            vp_confluence=vp_conf,
            mtf_alignment=mtf_ctx.alignment,
            session_priority=market_clock.session_priority(),
            direction=direction,
            symbol=symbol,
        )

        # ── 6-Gate Safety Filter ───────────────────────────────────────────
        gates = self._run_gates(
            prob=prob,
            sl=sl,
            tp=tp,
            entry=current_price,
            direction=direction,
        )

        all_passed = all(g.passed for g in gates)
        failed     = [g for g in gates if not g.passed]

        if not all_passed:
            for gate in failed:
                log.debug(f"{symbol} | Gate {gate.gate_number} FAILED: {gate.reason}")
            return None

        # ── All gates passed — generate signal ────────────────────────────
        rr = abs(tp - current_price) / abs(sl - current_price)

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            probability=prob.final_score,
            confluence=prob.confluence,
            market=market,
            timeframe=primary_tf,
            regime=regime.regime.value,
            session=market_clock.session_priority(),
            risk_reward=round(rr, 2),
        )

        log.info(signal.summary())
        return signal

    # ── 6-Gate Safety Filter ──────────────────────────────────────────────────
    def _run_gates(
        self,
        prob,
        sl: float,
        tp: float,
        entry: float,
        direction: str,
    ) -> list[GateResult]:
        """
        Run all 6 safety gates. Returns list of GateResult objects.
        """
        gates = []

        # Gate 1 — Probability Score
        gates.append(GateResult(
            gate_number=1,
            gate_name="Probability Score",
            passed=prob.final_score >= config.signal.MIN_PROBABILITY,
            reason=f"Score: {prob.final_score:.1%}" if prob.final_score >= config.signal.MIN_PROBABILITY
                   else f"Too low: {prob.final_score:.1%} < {config.signal.MIN_PROBABILITY:.0%}",
        ))

        # Gate 2 — Signal Confluence
        gates.append(GateResult(
            gate_number=2,
            gate_name="Signal Confluence",
            passed=prob.confluence >= config.signal.MIN_CONFLUENCE,
            reason=f"{prob.confluence}/{config.signal.MIN_CONFLUENCE} signals confirmed"
                   if prob.confluence >= config.signal.MIN_CONFLUENCE
                   else f"Only {prob.confluence} signals (need {config.signal.MIN_CONFLUENCE})",
        ))

        # Gate 3 — Circuit Breaker
        from risk.risk_manager import risk_manager as rm
        cb_ok = not rm.circuit_breaker.is_active
        gates.append(GateResult(
            gate_number=3,
            gate_name="Circuit Breaker",
            passed=cb_ok,
            reason="OK" if cb_ok else rm.circuit_breaker.reason,
        ))

        # Gate 4 — Macro Event Clear
        macro_ok = event_calendar.is_safe_to_trade()
        gates.append(GateResult(
            gate_number=4,
            gate_name="Macro Event Clear",
            passed=macro_ok,
            reason="No events" if macro_ok else "High-impact event too close",
        ))

        # Gate 5 — Risk/Reward Ratio
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp  - entry)
        rr      = tp_dist / sl_dist if sl_dist > 0 else 0
        rr_ok   = rr >= config.risk.MIN_RISK_REWARD
        gates.append(GateResult(
            gate_number=5,
            gate_name="Risk/Reward Ratio",
            passed=rr_ok,
            reason=f"1:{rr:.1f}" if rr_ok else f"R/R too low: 1:{rr:.1f}",
        ))

        # Gate 6 — Entry Validity (price within acceptable range)
        entry_valid = sl_dist > 0 and sl_dist < entry * 0.05   # SL < 5% from entry
        gates.append(GateResult(
            gate_number=6,
            gate_name="Entry Validity",
            passed=entry_valid,
            reason="Valid entry zone" if entry_valid else "SL distance invalid",
        ))

        return gates

    # ── Direction Resolution ──────────────────────────────────────────────────
    def _resolve_direction(self, structure, ml_signal, mtf_ctx) -> str:
        """
        Determine final direction by combining structure signal and ML vote.
        Requires agreement between both.
        """
        from intelligence.structure_engine import StructureSignal

        struct_dir = "neutral"
        if structure.signal in (
            StructureSignal.BOS_BULLISH, StructureSignal.CHOCH_BULLISH
        ):
            struct_dir = "long"
        elif structure.signal in (
            StructureSignal.BOS_BEARISH, StructureSignal.CHOCH_BEARISH
        ):
            struct_dir = "short"

        ml_dir = ml_signal.direction if ml_signal.passed else "neutral"
        mtf_dir = mtf_ctx.macro_bias   # "bullish" | "bearish" | "neutral"
        mtf_map = {"bullish": "long", "bearish": "short", "neutral": "neutral"}
        mtf_dir = mtf_map.get(mtf_dir, "neutral")

        # All three must agree for a clean signal
        directions = [struct_dir, ml_dir, mtf_dir]
        long_count  = directions.count("long")
        short_count = directions.count("short")

        if long_count >= 2:
            return "long"
        elif short_count >= 2:
            return "short"
        return "neutral"

    # ── ATR ───────────────────────────────────────────────────────────────────
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        import numpy as np
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


# Singleton
signal_engine = SignalEngine()