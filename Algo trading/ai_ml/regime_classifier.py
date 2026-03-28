"""
ai_ml/regime_classifier.py
─────────────────────────────────────────────────────────────────────────────
ML Regime Classifier.
Trained Random Forest that classifies market regime from features.
Works alongside the rule-based market_regime.py engine.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from core.logger import get_logger
from config import config

log = get_logger(__name__)

REGIME_LABELS = {
    0: "trending_up",
    1: "trending_down",
    2: "accumulation",
    3: "distribution",
    4: "balance",
    5: "volatile",
}
REGIME_TO_INT = {v: k for k, v in REGIME_LABELS.items()}


class RegimeClassifier:
    """
    ML-based market regime classifier.
    Trained on labelled OHLCV data using regime labels
    derived from the rule-based engine.
    """

    def __init__(self):
        self.model   = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=15, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        self.scaler  = StandardScaler()
        self._trained = False
        self._model_path  = config.paths.models / "regime_classifier.pkl"
        self._scaler_path = config.paths.models / "regime_scaler.pkl"
        self._load()

    # ── Feature Engineering ───────────────────────────────────────────────────
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build regime classification features from OHLCV."""
        f     = pd.DataFrame(index=df.index)
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        # Trend features
        ema20  = close.ewm(span=20, adjust=False).mean()
        ema50  = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        f["ema20_50"]  = (ema20 - ema50) / ema50
        f["ema50_200"] = (ema50 - ema200) / ema200
        f["price_ema20"] = (close - ema20) / ema20

        # ADX
        tr     = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr    = tr.ewm(span=14, adjust=False).mean()
        pdm    = high.diff().clip(lower=0)
        mdm    = (-low.diff()).clip(lower=0)
        pdi    = 100 * pdm.ewm(span=14, adjust=False).mean() / (atr + 1e-10)
        mdi    = 100 * mdm.ewm(span=14, adjust=False).mean() / (atr + 1e-10)
        dx     = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
        f["adx"]      = dx.ewm(span=14, adjust=False).mean()
        f["di_spread"]= pdi - mdi

        # Volatility
        f["atr_norm"]  = atr / close
        f["bb_width"]  = (
            close.rolling(20).std() * 4 / close.rolling(20).mean()
        )
        f["ret_vol"]   = close.pct_change().rolling(20).std()

        # Choppiness
        n = 14
        tr_sum = tr.rolling(n).sum()
        hl_rng = high.rolling(n).max() - low.rolling(n).min() + 1e-10
        f["choppiness"] = 100 * np.log10(tr_sum / hl_rng) / np.log10(n)

        # Volume
        f["vol_ratio"] = vol / vol.rolling(20).mean()
        f["vol_trend"] = vol.pct_change(5)

        # Momentum
        f["roc_10"]  = close.pct_change(10)
        f["roc_20"]  = close.pct_change(20)

        return f.dropna()

    # ── Label Generation ──────────────────────────────────────────────────────
    def _auto_label(self, df: pd.DataFrame) -> pd.Series:
        """
        Auto-generate regime labels from rule-based engine.
        Used to create training data.
        """
        from intelligence.market_regime import market_regime_engine
        labels = []
        step   = 10

        for i in range(60, len(df), step):
            window = df.iloc[max(0, i-100): i]
            if len(window) < 50:
                labels.append(4)    # balance as default
                continue
            result = market_regime_engine.analyse(window, "train", "1h")
            labels.append(REGIME_TO_INT.get(result.regime.value, 4))

        # Interpolate to full length
        full = pd.Series(index=range(len(df)), dtype=int)
        for i, lbl in enumerate(labels):
            start = 60 + i * step
            end   = min(start + step, len(df))
            full.iloc[start:end] = lbl
        full = full.fillna(4)
        return full

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """Train regime classifier. Returns accuracy metrics."""
        if len(df) < 300:
            return {}

        features = self._features(df)
        labels   = self._auto_label(df)
        aligned  = labels.loc[features.index].dropna()
        X        = features.loc[aligned.index]
        y        = aligned.astype(int)

        if len(X) < 100:
            return {}

        split  = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        self.scaler.fit(X_tr)
        X_tr_s = self.scaler.transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model.fit(X_tr_s, y_tr)
        acc = self.model.score(X_te_s, y_te)

        self._trained = True
        self._save()

        log.info(f"Regime classifier trained | Accuracy: {acc:.2%}")
        return {"regime_accuracy": round(acc, 4)}

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        """
        Predict current market regime.
        Returns (regime_name, confidence).
        """
        if not self._trained:
            return "unknown", 0.0

        features = self._features(df)
        if features.empty:
            return "unknown", 0.0

        try:
            X      = features.iloc[[-1]]
            X_s    = self.scaler.transform(X)
            pred   = int(self.model.predict(X_s)[0])
            proba  = float(self.model.predict_proba(X_s).max())
            regime = REGIME_LABELS.get(pred, "unknown")
            return regime, round(proba, 3)
        except Exception as e:
            log.warning(f"Regime prediction failed: {e}")
            return "unknown", 0.0

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self):
        try:
            joblib.dump(self.model,  self._model_path)
            joblib.dump(self.scaler, self._scaler_path)
        except Exception as e:
            log.error(f"Save failed: {e}")

    def _load(self):
        try:
            if self._model_path.exists() and self._scaler_path.exists():
                self.model   = joblib.load(self._model_path)
                self.scaler  = joblib.load(self._scaler_path)
                self._trained = True
                log.info("Regime classifier loaded from disk.")
        except Exception:
            pass

    @property
    def is_trained(self) -> bool:
        return self._trained


# Singleton
regime_classifier = RegimeClassifier()