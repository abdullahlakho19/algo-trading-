"""
reporting/model_evaluator.py
─────────────────────────────────────────────────────────────────────────────
Model Evaluator.
Scores the health of all ML models and the overall agent.
Generates model health reports and triggers retraining alerts.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class ModelHealthReport:
    """Overall model health summary."""
    timestamp:        str
    overall_grade:    str          # A / B / C / D / F
    overall_score:    float        # 0-100
    win_rate:         float
    profit_factor:    float
    sharpe_ratio:     float
    model_accuracy:   float
    needs_retrain:    bool
    alerts:           list[str] = field(default_factory=list)
    recommendations:  list[str] = field(default_factory=list)


class ModelEvaluator:
    """Evaluates model and agent health from live trading data."""

    # Target thresholds
    TARGETS = {
        "win_rate":      0.75,
        "profit_factor": 2.0,
        "sharpe":        2.0,
        "accuracy":      0.70,
    }

    def evaluate(
        self,
        trades: list,
        model_accuracy: float = None,
    ) -> ModelHealthReport:
        """Generate full health report."""
        if not trades:
            return ModelHealthReport(
                timestamp=datetime.utcnow().isoformat(),
                overall_grade="N/A", overall_score=0,
                win_rate=0, profit_factor=0, sharpe_ratio=0,
                model_accuracy=0, needs_retrain=True,
                alerts=["No trade data available."],
            )

        wins   = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]
        pnl    = pd.Series([t.pnl for t in trades])

        win_rate = len(wins) / len(trades)
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) + 1e-10
        pf = gp / gl
        sharpe = float(pnl.mean() / (pnl.std() + 1e-10) * (252 ** 0.5))
        acc    = model_accuracy or 0.0

        # Score each component (0-25 each, total 100)
        score = sum([
            min(25, (win_rate / self.TARGETS["win_rate"]) * 25),
            min(25, (pf      / self.TARGETS["profit_factor"]) * 25),
            min(25, (max(sharpe, 0) / self.TARGETS["sharpe"]) * 25),
            min(25, (acc    / self.TARGETS["accuracy"]) * 25),
        ])

        if   score >= 85: grade = "A"
        elif score >= 70: grade = "B"
        elif score >= 55: grade = "C"
        elif score >= 40: grade = "D"
        else:             grade = "F"

        alerts = []
        recommendations = []

        if win_rate < self.TARGETS["win_rate"]:
            alerts.append(f"Win rate below target: {win_rate:.1%} < {self.TARGETS['win_rate']:.0%}")
            recommendations.append("Review signal confirmation thresholds")

        if pf < self.TARGETS["profit_factor"]:
            alerts.append(f"Profit factor below target: {pf:.2f} < {self.TARGETS['profit_factor']}")
            recommendations.append("Tighten stop-loss placement")

        if sharpe < self.TARGETS["sharpe"]:
            alerts.append(f"Sharpe below target: {sharpe:.2f}")

        needs_retrain = win_rate < 0.55 or (acc > 0 and acc < 0.55)
        if needs_retrain:
            recommendations.append("Trigger model retraining")

        report = ModelHealthReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_grade=grade,
            overall_score=round(score, 1),
            win_rate=round(win_rate, 4),
            profit_factor=round(pf, 2),
            sharpe_ratio=round(sharpe, 2),
            model_accuracy=round(acc, 4),
            needs_retrain=needs_retrain,
            alerts=alerts,
            recommendations=recommendations,
        )

        log.info(
            f"Model Health: Grade {grade} | Score: {score:.0f}/100 | "
            f"WR: {win_rate:.1%} | Alerts: {len(alerts)}"
        )
        return report


# Singleton
model_evaluator = ModelEvaluator()