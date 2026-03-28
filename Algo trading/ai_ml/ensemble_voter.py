"""
ai_ml/ensemble_voter.py
─────────────────────────────────────────────────────────────────────────────
Ensemble Voting Engine — Renaissance Technologies philosophy.
Multiple independent ML models must reach consensus before a
signal is considered valid. No single model can decide alone.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class EnsembleSignal:
    """Output of the ensemble voting process."""
    symbol:        str
    timeframe:     str
    direction:     str        # "long" | "short" | "neutral"
    probability:   float      # 0.0 – 1.0 combined confidence
    vote_count:    int         # How many models agreed
    total_models:  int
    agreement_pct: float       # % of models in agreement
    model_votes:   dict = field(default_factory=dict)  # model_name → vote
    passed:        bool = False
    anomaly:       bool = False   # True if anomaly detected

    @property
    def is_valid(self) -> bool:
        return (
            self.passed
            and not self.anomaly
            and self.direction != "neutral"
            and self.probability >= config.signal.MIN_PROBABILITY
        )


class FeatureEngineer:
    """Builds the ML feature matrix from OHLCV + indicator data."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from OHLCV data.
        Returns DataFrame of features for ML models.
        """
        f = pd.DataFrame(index=df.index)

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        # ── Price-based features ──────────────────────────────────────────
        f["returns_1"]    = close.pct_change(1)
        f["returns_5"]    = close.pct_change(5)
        f["returns_10"]   = close.pct_change(10)
        f["returns_20"]   = close.pct_change(20)

        # ── EMA features ──────────────────────────────────────────────────
        ema8  = close.ewm(span=8,   adjust=False).mean()
        ema21 = close.ewm(span=21,  adjust=False).mean()
        ema50 = close.ewm(span=50,  adjust=False).mean()
        ema200= close.ewm(span=200, adjust=False).mean()

        f["price_vs_ema8"]    = (close - ema8)   / ema8
        f["price_vs_ema21"]   = (close - ema21)  / ema21
        f["price_vs_ema50"]   = (close - ema50)  / ema50
        f["price_vs_ema200"]  = (close - ema200) / ema200
        f["ema8_vs_ema21"]    = (ema8  - ema21)  / ema21
        f["ema21_vs_ema50"]   = (ema21 - ema50)  / ema50

        # ── Volatility features ───────────────────────────────────────────
        atr_series = (
            pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
        )
        atr14 = atr_series.ewm(span=14, adjust=False).mean()
        f["atr_norm"]    = atr14 / close
        f["bb_width"]    = (
            (close.rolling(20).mean() + 2 * close.rolling(20).std()) -
            (close.rolling(20).mean() - 2 * close.rolling(20).std())
        ) / close.rolling(20).mean()
        f["hl_range"]    = (high - low) / close

        # ── Momentum features ─────────────────────────────────────────────
        f["rsi"]         = self._rsi(close, 14)
        f["rsi_signal"]  = f["rsi"].ewm(span=9, adjust=False).mean()
        f["roc_5"]       = close.pct_change(5)
        f["roc_10"]      = close.pct_change(10)

        # ── Volume features ───────────────────────────────────────────────
        vol_ma20 = vol.rolling(20).mean()
        f["vol_ratio"]   = vol / (vol_ma20 + 1e-10)
        f["vol_trend"]   = vol.pct_change(5)

        # ── Candle pattern features ───────────────────────────────────────
        body   = (close - df["open"]).abs()
        total  = (high - low) + 1e-10
        f["body_ratio"]  = body / total
        f["upper_wick"]  = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / total
        f["lower_wick"]  = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / total
        f["bullish_candle"] = (close > df["open"]).astype(float)

        return f.dropna()

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def build_labels(
        self, df: pd.DataFrame, forward_candles: int = 10, threshold: float = 0.002
    ) -> pd.Series:
        """
        Build target labels for supervised learning.
        1 = price goes up > threshold in next N candles
        0 = price goes down > threshold
        -1 = sideways (filtered out during training)
        """
        fwd_return = df["close"].shift(-forward_candles) / df["close"] - 1
        labels = pd.Series(index=df.index, dtype=int)
        labels[fwd_return > threshold]  = 1
        labels[fwd_return < -threshold] = 0
        labels[(fwd_return >= -threshold) & (fwd_return <= threshold)] = -1
        return labels


class EnsembleVoter:
    """
    Multi-model ensemble voting system.

    Models:
        1. RandomForest Regime Classifier
        2. XGBoost Signal Generator
        3. Logistic Regression (fast, simple baseline)
        4. Isolation Forest (anomaly detector)

    A signal is valid only when MIN_MODEL_AGREEMENT % of models agree.
    """

    def __init__(self):
        self.fe        = FeatureEngineer()
        self.scaler    = StandardScaler()
        self.models_dir = config.paths.models
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # ML models
        self.rf_model    = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        self.xgb_model   = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        )
        self.anomaly_detector = IsolationForest(
            contamination=config.ml.ANOMALY_CONTAMINATION,
            random_state=42, n_jobs=-1
        )

        self._trained    = False
        self._load_models()

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """
        Train all ensemble models on historical OHLCV data.

        Args:
            df:     Historical OHLCV DataFrame (min 500 rows)
            symbol: Symbol name for logging

        Returns:
            Dict with training metrics per model
        """
        if len(df) < config.ml.MIN_TRAIN_SAMPLES:
            log.warning(f"Insufficient training data: {len(df)} rows (min {config.ml.MIN_TRAIN_SAMPLES})")
            return {}

        log.info(f"Training ensemble models | {symbol} | {len(df)} samples")

        features = self.fe.build_features(df)
        labels   = self.fe.build_labels(df)

        # Align
        idx      = features.index.intersection(labels.index)
        features = features.loc[idx]
        labels   = labels.loc[idx]

        # Remove sideways (label == -1)
        mask     = labels != -1
        X        = features[mask]
        y        = labels[mask]

        if len(X) < 100:
            log.warning("Too few directional samples after filtering sideways.")
            return {}

        # Train/test split
        split    = int(len(X) * (1 - config.ml.TEST_SIZE))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Scale
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        metrics = {}

        # RandomForest
        self.rf_model.fit(X_train_s, y_train)
        rf_score = self.rf_model.score(X_test_s, y_test)
        metrics["random_forest_accuracy"] = round(rf_score, 4)

        # XGBoost
        self.xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False
        )
        xgb_score = self.xgb_model.score(X_test_s, y_test)
        metrics["xgboost_accuracy"] = round(xgb_score, 4)

        # Anomaly detector (unsupervised — train on all features)
        all_scaled = self.scaler.transform(features)
        self.anomaly_detector.fit(all_scaled)

        self._trained = True
        self._save_models()

        log.info(
            f"Training complete | RF: {rf_score:.2%} | XGB: {xgb_score:.2%}"
        )
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> EnsembleSignal:
        """
        Generate ensemble prediction for the latest candle.

        Args:
            df:        OHLCV DataFrame (needs sufficient history)
            symbol:    Symbol name
            timeframe: Timeframe string

        Returns:
            EnsembleSignal with direction, probability, and vote details
        """
        if not self._trained:
            log.warning("Models not trained yet. Returning neutral signal.")
            return self._neutral_signal(symbol, timeframe)

        features = self.fe.build_features(df)
        if features.empty:
            return self._neutral_signal(symbol, timeframe)

        # Use only the latest row for live prediction
        X_latest = features.iloc[[-1]]
        try:
            X_scaled = self.scaler.transform(X_latest)
        except Exception as e:
            log.warning(f"Scaler transform failed: {e}")
            return self._neutral_signal(symbol, timeframe)

        votes      = {}
        probs      = {}

        # ── RandomForest vote ──────────────────────────────────────────────
        rf_pred = int(self.rf_model.predict(X_scaled)[0])
        rf_prob = float(self.rf_model.predict_proba(X_scaled)[0][rf_pred])
        votes["random_forest"] = "long" if rf_pred == 1 else "short"
        probs["random_forest"] = rf_prob

        # ── XGBoost vote ───────────────────────────────────────────────────
        xgb_pred = int(self.xgb_model.predict(X_scaled)[0])
        xgb_prob = float(self.xgb_model.predict_proba(X_scaled)[0][xgb_pred])
        votes["xgboost"] = "long" if xgb_pred == 1 else "short"
        probs["xgboost"] = xgb_prob

        # ── Anomaly check ──────────────────────────────────────────────────
        anomaly_score = self.anomaly_detector.predict(X_scaled)[0]
        is_anomaly    = (anomaly_score == -1)

        # ── Tally votes ────────────────────────────────────────────────────
        long_votes  = sum(1 for v in votes.values() if v == "long")
        short_votes = sum(1 for v in votes.values() if v == "short")
        total       = len(votes)

        if long_votes > short_votes:
            direction   = "long"
            vote_count  = long_votes
        elif short_votes > long_votes:
            direction   = "short"
            vote_count  = short_votes
        else:
            direction   = "neutral"
            vote_count  = 0

        agreement_pct  = vote_count / total if total > 0 else 0
        avg_prob       = np.mean(list(probs.values()))

        passed = (
            agreement_pct >= config.signal.MIN_MODEL_AGREEMENT
            and avg_prob   >= config.signal.MIN_PROBABILITY
            and not is_anomaly
            and direction  != "neutral"
        )

        signal = EnsembleSignal(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            probability=round(avg_prob, 4),
            vote_count=vote_count,
            total_models=total,
            agreement_pct=round(agreement_pct, 3),
            model_votes=votes,
            passed=passed,
            anomaly=is_anomaly,
        )

        log.info(
            f"{symbol} | {timeframe} | Ensemble: {direction} "
            f"({vote_count}/{total} models) | P={avg_prob:.2%} | "
            f"{'✅ PASSED' if passed else '❌ FAILED'}"
            + (" | ⚠️ ANOMALY" if is_anomaly else "")
        )

        return signal

    # ── Model Persistence ─────────────────────────────────────────────────────
    def _save_models(self) -> None:
        try:
            joblib.dump(self.rf_model,    self.models_dir / config.ml.REGIME_MODEL)
            joblib.dump(self.xgb_model,   self.models_dir / config.ml.SIGNAL_MODEL)
            joblib.dump(self.scaler,      self.models_dir / "scaler.pkl")
            joblib.dump(self.anomaly_detector, self.models_dir / "anomaly.pkl")
            log.info("Models saved to disk.")
        except Exception as e:
            log.error(f"Failed to save models: {e}")

    def _load_models(self) -> None:
        try:
            rf_path  = self.models_dir / config.ml.REGIME_MODEL
            xgb_path = self.models_dir / config.ml.SIGNAL_MODEL
            sc_path  = self.models_dir / "scaler.pkl"
            an_path  = self.models_dir / "anomaly.pkl"

            if all(p.exists() for p in [rf_path, xgb_path, sc_path, an_path]):
                self.rf_model          = joblib.load(rf_path)
                self.xgb_model         = joblib.load(xgb_path)
                self.scaler            = joblib.load(sc_path)
                self.anomaly_detector  = joblib.load(an_path)
                self._trained          = True
                log.info("Pre-trained models loaded from disk.")
        except Exception as e:
            log.warning(f"Could not load models: {e}. Will train from scratch.")

    def _neutral_signal(self, symbol: str, timeframe: str) -> EnsembleSignal:
        return EnsembleSignal(
            symbol=symbol, timeframe=timeframe,
            direction="neutral", probability=0.0,
            vote_count=0, total_models=2,
            agreement_pct=0.0, passed=False,
        )

    @property
    def is_trained(self) -> bool:
        return self._trained


# Singleton
ensemble_voter = EnsembleVoter()