"""
ai_ml/anomaly_detector.py
─────────────────────────────────────────────────────────────────────────────
Market Anomaly Detector — XTX Markets / Renaissance philosophy.
Detects when market conditions are outside historical norms.
When anomalies are detected, the agent steps back and waits.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from dataclasses import dataclass
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection output."""
    is_anomaly:  bool
    score:       float    # More negative = more anomalous
    severity:    str      # "none" | "mild" | "severe"
    reason:      str


class AnomalyDetector:
    """
    Isolation Forest anomaly detector.
    Trained on normal market conditions — flags anything unusual.
    """

    def __init__(self):
        self.model  = IsolationForest(
            n_estimators=200,
            contamination=config.ml.ANOMALY_CONTAMINATION,
            random_state=42, n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._trained = False
        self._model_path  = config.paths.models / "anomaly_detector.pkl"
        self._scaler_path = config.paths.models / "anomaly_scaler.pkl"
        self._load()

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build anomaly detection features."""
        f     = pd.DataFrame(index=df.index)
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        f["atr_norm"]   = atr / close
        f["vol_ratio"]  = vol / (vol.rolling(20).mean() + 1e-10)
        f["ret_1"]      = close.pct_change(1).abs()
        f["ret_5"]      = close.pct_change(5).abs()
        f["hl_range"]   = (high - low) / close
        f["bb_width"]   = 4 * close.rolling(20).std() / (close.rolling(20).mean() + 1e-10)

        return f.dropna()

    def train(self, df: pd.DataFrame) -> bool:
        """Train anomaly detector on historical data."""
        if len(df) < 100:
            return False
        try:
            features = self._features(df)
            self.scaler.fit(features)
            X_s = self.scaler.transform(features)
            self.model.fit(X_s)
            self._trained = True
            self._save()
            log.info("Anomaly detector trained.")
            return True
        except Exception as e:
            log.error(f"Anomaly detector training failed: {e}")
            return False

    def detect(self, df: pd.DataFrame) -> AnomalyResult:
        """Check if current market conditions are anomalous."""
        if not self._trained:
            return AnomalyResult(False, 0.0, "none", "Model not trained")

        features = self._features(df)
        if features.empty:
            return AnomalyResult(False, 0.0, "none", "Insufficient data")

        try:
            X    = features.iloc[[-1]]
            X_s  = self.scaler.transform(X)
            pred = int(self.model.predict(X_s)[0])
            score = float(self.model.score_samples(X_s)[0])

            is_anomaly = (pred == -1)
            severity   = "none"
            reason     = "Normal market conditions"

            if is_anomaly:
                if score < -0.7:
                    severity = "severe"
                    reason   = f"Severe anomaly detected (score={score:.3f}). Agent stepping back."
                else:
                    severity = "mild"
                    reason   = f"Mild anomaly detected (score={score:.3f}). Reduced position sizing."

                log.warning(f"ANOMALY: {severity} | {reason}")

            return AnomalyResult(is_anomaly, round(score, 4), severity, reason)

        except Exception as e:
            log.warning(f"Anomaly detection failed: {e}")
            return AnomalyResult(False, 0.0, "none", str(e))

    def _save(self):
        try:
            joblib.dump(self.model,  self._model_path)
            joblib.dump(self.scaler, self._scaler_path)
        except Exception as e:
            log.error(f"Anomaly model save failed: {e}")

    def _load(self):
        try:
            if self._model_path.exists() and self._scaler_path.exists():
                self.model   = joblib.load(self._model_path)
                self.scaler  = joblib.load(self._scaler_path)
                self._trained = True
                log.info("Anomaly detector loaded from disk.")
        except Exception:
            pass

    @property
    def is_trained(self) -> bool:
        return self._trained


# Singleton
anomaly_detector = AnomalyDetector()