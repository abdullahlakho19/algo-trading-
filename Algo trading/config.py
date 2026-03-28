"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Institutional Trading Agent.
All API keys are loaded from environment variables (.env file).
Never hard-code keys here.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
DATA_DIR         = BASE_DIR / "data"
HISTORICAL_DIR   = DATA_DIR / "historical"
LOGS_DIR         = DATA_DIR / "logs"
REPORTS_DIR      = DATA_DIR / "reports"
MODELS_DIR       = BASE_DIR / "models"

# Ensure directories exist
for d in [HISTORICAL_DIR, LOGS_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── API Keys ──────────────────────────────────────────────────────────────────
class AlpacaConfig:
    API_KEY        = os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY     = os.getenv("ALPACA_SECRET_KEY", "")
    # Paper trading base URL (switch to live when ready)
    BASE_URL       = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    DATA_URL       = "https://data.alpaca.markets"
    PAPER_MODE     = os.getenv("ALPACA_PAPER_MODE", "true").lower() == "true"


class BinanceConfig:
    API_KEY        = os.getenv("BINANCE_API_KEY", "")
    SECRET_KEY     = os.getenv("BINANCE_SECRET_KEY", "")
    TESTNET        = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    TESTNET_URL    = "https://testnet.binance.vision"


class DatabaseConfig:
    URL            = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/trading_agent.db")


# ── Trading Mode ──────────────────────────────────────────────────────────────
class TradingMode:
    PAPER          = "paper"
    LIVE           = "live"
    BACKTEST       = "backtest"
    # Current active mode — ALWAYS start in paper
    ACTIVE         = os.getenv("TRADING_MODE", "paper")


# ── Markets & Instruments ─────────────────────────────────────────────────────
class Markets:
    # Stocks watchlist
    STOCKS = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "JPM", "SPY", "QQQ"
    ]

    # Forex pairs
    FOREX = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
        "USD/CAD", "USD/CHF", "NZD/USD", "GBP/JPY"
    ]

    # Crypto pairs
    CRYPTO = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT",
        "BNB/USDT", "XRP/USDT", "ADA/USDT"
    ]

    # Crypto exchange (via CCXT)
    CRYPTO_EXCHANGE = "binance"


# ── Timeframes ────────────────────────────────────────────────────────────────
class Timeframes:
    ALL = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Hierarchy: higher = macro context, lower = entry timing
    MACRO       = "1d"
    SWING       = "4h"
    CONTEXT     = "1h"
    TRIGGER     = "15m"
    ENTRY       = "5m"
    MICRO       = "1m"

    # Minimum candles required for analysis
    MIN_CANDLES = 100


# ── Sessions (UTC) ────────────────────────────────────────────────────────────
class Sessions:
    ASIAN   = {"start": "00:00", "end": "09:00", "priority": "low"}
    LONDON  = {"start": "08:00", "end": "17:00", "priority": "high"}
    NEW_YORK= {"start": "13:00", "end": "22:00", "priority": "high"}
    OVERLAP = {"start": "13:00", "end": "17:00", "priority": "highest"}


# ── Signal & Probability Thresholds ──────────────────────────────────────────
class SignalConfig:
    # Minimum probability score to consider a trade (0-1)
    MIN_PROBABILITY         = 0.75

    # Minimum number of independent signal confirmations
    MIN_CONFLUENCE          = 3

    # Signal decay — max age (minutes) before signal is invalidated
    MAX_SIGNAL_AGE_MINUTES  = 30

    # Ensemble — minimum models that must agree
    MIN_MODEL_AGREEMENT     = 0.66   # 2 out of 3 models


# ── Risk Management ───────────────────────────────────────────────────────────
class RiskConfig:
    # Maximum risk per trade as % of portfolio
    MAX_RISK_PER_TRADE      = 0.01   # 1%

    # Maximum total portfolio risk at any time
    MAX_PORTFOLIO_RISK      = 0.05   # 5%

    # Maximum number of simultaneous open positions
    MAX_OPEN_POSITIONS      = 10

    # Minimum risk/reward ratio required
    MIN_RISK_REWARD         = 2.0    # 1:2

    # Circuit breaker thresholds
    DAILY_LOSS_LIMIT        = 0.03   # 3% daily drawdown → pause
    WEEKLY_LOSS_LIMIT       = 0.07   # 7% weekly drawdown → pause

    # Volatility scaling — reduce size when VIX-equivalent is high
    VOLATILITY_SCALE        = True

    # Correlation threshold — reduce size for correlated trades
    MAX_CORRELATION         = 0.7


# ── Volume Profile ────────────────────────────────────────────────────────────
class VolumeProfileConfig:
    # Number of price bins for volume profile calculation
    NUM_BINS        = 100

    # Value area coverage (standard is 70%)
    VALUE_AREA_PCT  = 0.70

    # Naked POC — max sessions back to track unvisited POCs
    NAKED_POC_LOOKBACK = 20


# ── Machine Learning ──────────────────────────────────────────────────────────
class MLConfig:
    # Retrain frequency (hours)
    RETRAIN_INTERVAL_HOURS  = 24

    # Minimum training samples required
    MIN_TRAIN_SAMPLES       = 500

    # Train/test split
    TEST_SIZE               = 0.2

    # Feature lookback window (candles)
    LOOKBACK_WINDOW         = 50

    # Model file names
    REGIME_MODEL            = "regime_classifier.pkl"
    SIGNAL_MODEL            = "signal_model.pkl"
    PATTERN_MODEL           = "pattern_model.pkl"

    # Anomaly detection contamination rate
    ANOMALY_CONTAMINATION   = 0.05


# ── Monte Carlo ───────────────────────────────────────────────────────────────
class MonteCarloConfig:
    NUM_SIMULATIONS         = 1000
    CONFIDENCE_LEVEL        = 0.95   # 95% VaR
    TIME_HORIZON_DAYS       = 30


# ── Black-Scholes ─────────────────────────────────────────────────────────────
class BlackScholesConfig:
    RISK_FREE_RATE          = 0.05   # 5% annualised
    # High IV threshold — reduce position when market fear is elevated
    HIGH_IV_THRESHOLD       = 0.40   # 40% IV


# ── Execution ─────────────────────────────────────────────────────────────────
class ExecutionConfig:
    # Order type for entries
    ORDER_TYPE              = "limit"   # Use limit orders, not market

    # Max slippage tolerance (basis points)
    MAX_SLIPPAGE_BPS        = 10

    # Retry attempts if order fails
    MAX_ORDER_RETRIES       = 3

    # Delay between retries (seconds)
    RETRY_DELAY_SECONDS     = 2

    # Maximum time to wait for fill before cancelling (seconds)
    ORDER_TIMEOUT_SECONDS   = 30


# ── Macro Events ──────────────────────────────────────────────────────────────
class MacroConfig:
    # Stop trading X minutes before high-impact events
    PRE_EVENT_BUFFER_MINUTES    = 120

    # Resume X minutes after event
    POST_EVENT_BUFFER_MINUTES   = 30

    HIGH_IMPACT_EVENTS = [
        "FOMC", "NFP", "CPI", "PPI", "GDP",
        "EARNINGS", "FED_SPEAK", "ECB", "BOE", "BOJ"
    ]


# ── Reporting ─────────────────────────────────────────────────────────────────
class ReportingConfig:
    EXCEL_FILENAME          = "trading_report.xlsx"
    LOG_LEVEL               = "INFO"
    LOG_ROTATION            = "1 day"
    LOG_RETENTION           = "30 days"


# ── Dashboard ─────────────────────────────────────────────────────────────────
class DashboardConfig:
    HOST                    = "localhost"
    PORT                    = 8501
    REFRESH_INTERVAL_SEC    = 5


# ── Master Config Object ──────────────────────────────────────────────────────
class Config:
    alpaca          = AlpacaConfig()
    binance         = BinanceConfig()
    database        = DatabaseConfig()
    mode            = TradingMode()
    markets         = Markets()
    timeframes      = Timeframes()
    sessions        = Sessions()
    signal          = SignalConfig()
    risk            = RiskConfig()
    volume_profile  = VolumeProfileConfig()
    ml              = MLConfig()
    monte_carlo     = MonteCarloConfig()
    black_scholes   = BlackScholesConfig()
    execution       = ExecutionConfig()
    macro           = MacroConfig()
    reporting       = ReportingConfig()
    dashboard       = DashboardConfig()
    paths           = type("Paths", (), {
        "base":       BASE_DIR,
        "data":       DATA_DIR,
        "historical": HISTORICAL_DIR,
        "logs":       LOGS_DIR,
        "reports":    REPORTS_DIR,
        "models":     MODELS_DIR,
    })()


# Singleton instance — import this everywhere
config = Config()