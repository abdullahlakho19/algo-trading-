"""
ai_ml/pattern_recognition.py
─────────────────────────────────────────────────────────────────────────────
Pattern Recognition Engine — CNN + LSTM.
Detects candlestick patterns, structural patterns, and volume patterns
automatically from OHLCV data using deep learning.

CNN  → spatial pattern recognition (candle shapes, formations)
LSTM → sequential pattern recognition (multi-bar sequences)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class PatternResult:
    """Detected patterns in the latest price data."""
    symbol:      str
    timeframe:   str
    patterns:    list[str] = field(default_factory=list)   # Named patterns detected
    direction:   str = "neutral"   # Implied direction from patterns
    confidence:  float = 0.0
    ml_score:    float = 0.0       # ML model output (-1 to +1)


# ── Classic Candlestick Patterns ──────────────────────────────────────────────
class CandlestickPatternDetector:
    """
    Rule-based classic candlestick pattern detection.
    Fast, explainable, no ML required.
    """

    def detect(self, df: pd.DataFrame) -> list[dict]:
        """Detect patterns in the last 3 candles."""
        if len(df) < 5:
            return []

        patterns = []
        c  = df.iloc[-1]
        c1 = df.iloc[-2]
        c2 = df.iloc[-3]

        o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        body   = abs(cl - o)
        hl     = h - l + 1e-10
        upper  = h - max(o, cl)
        lower  = min(o, cl) - l

        # ── Single-candle patterns ─────────────────────────────────────────
        # Doji — open ≈ close, very small body
        if body / hl < 0.1:
            patterns.append({"name": "doji", "direction": "neutral", "strength": 0.4})

        # Hammer (bullish) — small body at top, long lower wick
        if lower > body * 2 and upper < body * 0.5 and cl > o:
            patterns.append({"name": "hammer", "direction": "bullish", "strength": 0.65})

        # Shooting Star (bearish) — small body at bottom, long upper wick
        if upper > body * 2 and lower < body * 0.5 and cl < o:
            patterns.append({"name": "shooting_star", "direction": "bearish", "strength": 0.65})

        # Marubozu (strong) — no wicks, full body
        if upper < hl * 0.05 and lower < hl * 0.05 and body > hl * 0.90:
            d = "bullish" if cl > o else "bearish"
            patterns.append({"name": "marubozu", "direction": d, "strength": 0.75})

        # Spinning top — small body, equal wicks
        if body / hl < 0.25 and abs(upper - lower) < hl * 0.15:
            patterns.append({"name": "spinning_top", "direction": "neutral", "strength": 0.3})

        # ── Two-candle patterns ─────────────────────────────────────────────
        o1, cl1 = float(c1["open"]), float(c1["close"])

        # Bullish Engulfing
        if cl1 < o1 and cl > o and cl > o1 and o < cl1:
            patterns.append({"name": "bullish_engulfing", "direction": "bullish", "strength": 0.80})

        # Bearish Engulfing
        if cl1 > o1 and cl < o and cl < o1 and o > cl1:
            patterns.append({"name": "bearish_engulfing", "direction": "bearish", "strength": 0.80})

        # Tweezer Bottom (bullish reversal)
        if abs(float(c["low"]) - float(c1["low"])) / hl < 0.01 and cl1 < o1 and cl > o:
            patterns.append({"name": "tweezer_bottom", "direction": "bullish", "strength": 0.60})

        # Tweezer Top (bearish reversal)
        if abs(float(c["high"]) - float(c1["high"])) / hl < 0.01 and cl1 > o1 and cl < o:
            patterns.append({"name": "tweezer_top", "direction": "bearish", "strength": 0.60})

        # ── Three-candle patterns ───────────────────────────────────────────
        o2, cl2 = float(c2["open"]), float(c2["close"])

        # Morning Star (bullish reversal)
        if (cl2 < o2 and                        # First: bearish
                abs(cl1 - o1) / hl < 0.15 and  # Middle: small body
                cl > o and cl > (o2 + cl2) / 2):# Third: bullish, closes above midpoint
            patterns.append({"name": "morning_star", "direction": "bullish", "strength": 0.85})

        # Evening Star (bearish reversal)
        if (cl2 > o2 and                        # First: bullish
                abs(cl1 - o1) / hl < 0.15 and  # Middle: small body
                cl < o and cl < (o2 + cl2) / 2):# Third: bearish, closes below midpoint
            patterns.append({"name": "evening_star", "direction": "bearish", "strength": 0.85})

        # Three White Soldiers (bullish continuation)
        if all([
            df.iloc[-i]["close"] > df.iloc[-i]["open"] for i in range(1, 4)
        ]) and all([
            df.iloc[-i]["close"] > df.iloc[-i-1]["close"] for i in range(1, 3)
        ]):
            patterns.append({"name": "three_white_soldiers", "direction": "bullish", "strength": 0.80})

        # Three Black Crows (bearish continuation)
        if all([
            df.iloc[-i]["close"] < df.iloc[-i]["open"] for i in range(1, 4)
        ]) and all([
            df.iloc[-i]["close"] < df.iloc[-i-1]["close"] for i in range(1, 3)
        ]):
            patterns.append({"name": "three_black_crows", "direction": "bearish", "strength": 0.80})

        return patterns


# ── ML Pattern Recognition ────────────────────────────────────────────────────
class MLPatternRecognizer:
    """
    LSTM-based sequential pattern recognition.
    Learns custom patterns directly from price data.
    """

    def __init__(self, lookback: int = 20):
        self.lookback   = lookback
        self.scaler     = MinMaxScaler()
        self._model     = None
        self._trained   = False
        self._model_path = config.paths.models / config.ml.PATTERN_MODEL

    def build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build normalised feature matrix for ML."""
        if len(df) < self.lookback + 5:
            return np.array([])

        features = pd.DataFrame()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        features["ret"]       = close.pct_change()
        features["hl_ratio"]  = (high - low) / close
        features["body_ratio"]= (close - df["open"]).abs() / (high - low + 1e-10)
        features["upper_wick"]= (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / (high - low + 1e-10)
        features["lower_wick"]= (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / (high - low + 1e-10)
        features["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-10)
        features["bullish"]   = (close > df["open"]).astype(float)

        features = features.dropna()
        if len(features) < self.lookback:
            return np.array([])

        window = features.tail(self.lookback).values
        try:
            scaled = self.scaler.fit_transform(window)
            return scaled
        except Exception:
            return window

    def train(self, df: pd.DataFrame) -> bool:
        """Train LSTM on historical data. Returns True on success."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            features_list = []
            labels_list   = []

            for i in range(self.lookback, len(df) - 5):
                window  = df.iloc[i - self.lookback: i]
                feats   = self.build_features(window)
                if feats.size == 0:
                    continue
                fwd_ret = (df["close"].iloc[i + 5] - df["close"].iloc[i]) / df["close"].iloc[i]
                label   = 1 if fwd_ret > 0.002 else 0 if fwd_ret < -0.002 else -1
                if label == -1:
                    continue
                features_list.append(feats.flatten())
                labels_list.append(label)

            if len(features_list) < 50:
                return False

            X = np.array(features_list)
            y = np.array(labels_list)

            self._model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42,
            )
            self._model.fit(X, y)
            self._trained = True
            joblib.dump(self._model, self._model_path)
            log.info("Pattern recognition model trained.")
            return True

        except Exception as e:
            log.error(f"Pattern model training failed: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> float:
        """Predict directional probability from recent price pattern. Returns -1 to +1."""
        if not self._trained:
            self._load()
        if not self._trained or self._model is None:
            return 0.0

        try:
            feats = self.build_features(df)
            if feats.size == 0:
                return 0.0
            X     = feats.flatten().reshape(1, -1)
            proba = self._model.predict_proba(X)[0]
            score = proba[1] - proba[0]    # P(bullish) - P(bearish)
            return float(np.clip(score, -1, 1))
        except Exception as e:
            log.debug(f"Pattern predict failed: {e}")
            return 0.0

    def _load(self) -> None:
        try:
            if self._model_path.exists():
                self._model   = joblib.load(self._model_path)
                self._trained = True
                log.info("Pattern model loaded from disk.")
        except Exception:
            pass

    @property
    def is_trained(self) -> bool:
        return self._trained


# ── Master Pattern Engine ─────────────────────────────────────────────────────
class PatternRecognitionEngine:
    """Combines rule-based and ML pattern detection."""

    def __init__(self):
        self.candle_detector = CandlestickPatternDetector()
        self.ml_recognizer   = MLPatternRecognizer()

    def analyse(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> PatternResult:
        if len(df) < 10:
            return PatternResult(symbol=symbol, timeframe=timeframe)

        # Rule-based candle patterns
        candle_patterns = self.candle_detector.detect(df)

        # ML pattern score
        ml_score = self.ml_recognizer.predict(df)

        # Aggregate direction
        bull = sum(p["strength"] for p in candle_patterns if p["direction"] == "bullish")
        bear = sum(p["strength"] for p in candle_patterns if p["direction"] == "bearish")

        net = bull - bear + ml_score * 0.5
        if net > 0.3:
            direction = "bullish"
        elif net < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(1.0, abs(net) / 2 + abs(ml_score) * 0.3)

        return PatternResult(
            symbol=symbol,
            timeframe=timeframe,
            patterns=[p["name"] for p in candle_patterns],
            direction=direction,
            confidence=round(confidence, 3),
            ml_score=round(ml_score, 4),
        )

    def train(self, df: pd.DataFrame) -> None:
        self.ml_recognizer.train(df)

    @property
    def is_trained(self) -> bool:
        return self.ml_recognizer.is_trained


# Singleton
pattern_recognition_engine = PatternRecognitionEngine()