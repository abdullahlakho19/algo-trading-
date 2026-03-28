"""
quant/black_scholes.py
─────────────────────────────────────────────────────────────────────────────
Black-Scholes Model implementation.
Used for options pricing, implied volatility estimation,
and measuring market fear/greed via IV levels.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class BSResult:
    """Black-Scholes pricing result."""
    call_price:  float
    put_price:   float
    delta:       float    # Rate of change of option price w.r.t. underlying
    gamma:       float    # Rate of change of delta
    theta:       float    # Time decay (per day)
    vega:        float    # Sensitivity to volatility
    rho:         float    # Sensitivity to interest rate
    iv:          float    # Implied volatility (if computed)
    d1:          float
    d2:          float


class BlackScholesModel:
    """
    Black-Scholes options pricing model.
    
    Also used to:
    - Estimate fair value of options
    - Calculate Greeks for risk assessment
    - Compute implied volatility from market prices
    - Measure market fear via IV percentile
    """

    def __init__(self):
        self.r = config.black_scholes.RISK_FREE_RATE
        self.high_iv_threshold = config.black_scholes.HIGH_IV_THRESHOLD

    # ── Core Pricing ──────────────────────────────────────────────────────────
    def price(
        self,
        S: float,     # Current underlying price
        K: float,     # Strike price
        T: float,     # Time to expiration (years)
        sigma: float, # Implied volatility (annualised)
        r: float | None = None,
    ) -> BSResult:
        """
        Compute Black-Scholes price and Greeks.

        Args:
            S:     Spot price
            K:     Strike price
            T:     Time to expiry in years (e.g. 30 days = 30/365)
            sigma: Annualised volatility (e.g. 0.20 = 20%)
            r:     Risk-free rate (defaults to config)

        Returns:
            BSResult with prices and Greeks
        """
        r = r or self.r

        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return self._zero_result()

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Prices
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Greeks
        delta_call = norm.cdf(d1)
        delta_put  = delta_call - 1
        gamma      = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta      = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365   # Per day
        vega       = S * norm.pdf(d1) * np.sqrt(T) / 100   # Per 1% vol move
        rho_call   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

        return BSResult(
            call_price=round(call_price, 6),
            put_price=round(put_price, 6),
            delta=round(delta_call, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 6),
            vega=round(vega, 6),
            rho=round(rho_call, 6),
            iv=sigma,
            d1=round(d1, 4),
            d2=round(d2, 4),
        )

    # ── Implied Volatility ────────────────────────────────────────────────────
    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: str = "call",
        r: float | None = None,
    ) -> float | None:
        """
        Compute implied volatility from market option price using Brent's method.

        Args:
            market_price: Observed market price of the option
            option_type:  "call" or "put"

        Returns:
            IV as decimal (e.g. 0.25 = 25%) or None if calculation fails
        """
        r = r or self.r

        def objective(sigma: float) -> float:
            result = self.price(S, K, T, sigma, r)
            if option_type == "call":
                return result.call_price - market_price
            return result.put_price - market_price

        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=500)
            return round(iv, 4)
        except ValueError:
            return None
        except Exception as e:
            log.warning(f"IV calculation failed: {e}")
            return None

    # ── Historical Volatility ─────────────────────────────────────────────────
    def historical_volatility(
        self,
        prices: "pd.Series",
        window: int = 30,
        annualise: bool = True,
    ) -> float:
        """
        Compute historical (realised) volatility from price series.
        Used as sigma estimate when no options data is available.
        """
        import pandas as pd
        log_returns = np.log(prices / prices.shift(1)).dropna()
        hv = log_returns.rolling(window).std().iloc[-1]
        if annualise:
            hv = hv * np.sqrt(252)
        return round(float(hv), 4)

    # ── IV Analysis ───────────────────────────────────────────────────────────
    def is_high_iv(self, sigma: float) -> bool:
        """True if implied volatility is elevated — consider reducing size."""
        return sigma >= self.high_iv_threshold

    def iv_percentile(self, iv_series: "pd.Series") -> float:
        """
        IV Rank — where is current IV relative to its 52-week range?
        0 = historically low IV, 100 = historically high IV.
        """
        current = float(iv_series.iloc[-1])
        lo = float(iv_series.min())
        hi = float(iv_series.max())
        if hi == lo:
            return 50.0
        return round((current - lo) / (hi - lo) * 100, 1)

    def analyse_volatility(
        self,
        prices: "pd.Series",
        symbol: str,
    ) -> dict:
        """
        Compute full volatility profile for a price series.
        Used by the risk engine to adjust position sizing.
        """
        hv_20  = self.historical_volatility(prices, window=20)
        hv_60  = self.historical_volatility(prices, window=60)

        # Approximate IV as 1.2x HV (rough institutional estimate without options data)
        est_iv = hv_20 * 1.2

        return {
            "symbol":   symbol,
            "hv_20":    hv_20,
            "hv_60":    hv_60,
            "est_iv":   est_iv,
            "is_high_iv": self.is_high_iv(est_iv),
            "vol_regime": "high" if est_iv > 0.30 else "normal" if est_iv > 0.15 else "low",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _zero_result(self) -> BSResult:
        return BSResult(
            call_price=0.0, put_price=0.0, delta=0.0,
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            iv=0.0, d1=0.0, d2=0.0,
        )


# Singleton
black_scholes = BlackScholesModel()