"""
run.py
─────────────────────────────────────────────────────────────────────────────
INTERACTIVE LAUNCHER — Institutional Trading Agent
─────────────────────────────────────────────────────────────────────────────
The single file you run to start everything.
Guides you through selecting markets, symbols, modes, and starting the agent.

Usage:
    python run.py              → Interactive menu (recommended)
    python run.py --quick      → Skip menus, use saved config
    python run.py --dashboard  → Launch dashboard only
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path

# ── Terminal colors ───────────────────────────────────────────────────────────
R  = "\033[91m"   # Red
G  = "\033[92m"   # Green
Y  = "\033[93m"   # Yellow
B  = "\033[94m"   # Blue
M  = "\033[95m"   # Magenta
C  = "\033[96m"   # Cyan
W  = "\033[97m"   # White
DIM= "\033[2m"
BOLD="\033[1m"
RST= "\033[0m"

CONFIG_FILE = Path("agent_config.json")

# ── Default symbol lists ──────────────────────────────────────────────────────
ALL_STOCKS = {
    "US Tech":    ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"],
    "US Finance": ["JPM", "BAC", "GS", "MS", "V", "MA"],
    "US ETFs":    ["SPY", "QQQ", "IWM", "GLD", "TLT", "VIX"],
    "UK Stocks":  ["LLOY.L", "BARC.L", "BP.L", "SHEL.L"],
}

ALL_FOREX = {
    "Major Pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"],
    "Minor Pairs": ["EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD"],
    "Exotics":     ["USD/TRY", "USD/MXN", "USD/ZAR", "USD/SGD"],
}

ALL_CRYPTO = {
    "Large Cap":  ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
    "Mid Cap":    ["XRP/USDT", "ADA/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT"],
    "DeFi":       ["UNI/USDT", "AAVE/USDT", "LINK/USDT", "SNX/USDT"],
    "Meme":       ["DOGE/USDT", "SHIB/USDT"],
}


# ── Print helpers ─────────────────────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print(f"""
{Y}╔══════════════════════════════════════════════════════════════╗
║          INSTITUTIONAL TRADING AGENT — LAUNCHER              ║
║   Citadel · BlackRock · JP Morgan · Jane Street · RenTech    ║
╚══════════════════════════════════════════════════════════════╝{RST}
""")

def section(title: str):
    print(f"\n{C}{'─'*60}{RST}")
    print(f"{BOLD}{W}  {title}{RST}")
    print(f"{C}{'─'*60}{RST}\n")

def ok(msg):   print(f"  {G}✅ {msg}{RST}")
def warn(msg): print(f"  {Y}⚠️  {msg}{RST}")
def err(msg):  print(f"  {R}✗  {msg}{RST}")
def info(msg): print(f"  {C}ℹ  {msg}{RST}")

def numbered_menu(options: list, title: str = "") -> int:
    """Show a numbered menu and return selected index (0-based)."""
    if title:
        print(f"\n  {W}{title}{RST}")
    for i, opt in enumerate(options):
        print(f"    {Y}{i+1}.{RST} {opt}")
    while True:
        try:
            choice = input(f"\n  {C}Enter number (1-{len(options)}): {RST}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"  {R}Please enter a number between 1 and {len(options)}{RST}")
        except (ValueError, KeyboardInterrupt):
            print(f"  {R}Invalid input{RST}")

def checkbox_menu(options: list, title: str = "", preselect: list = None) -> list[int]:
    """
    Multi-select checkbox menu.
    Returns list of selected indices (0-based).
    """
    selected = set(preselect or [])
    if title:
        print(f"\n  {W}{title}{RST}")

    while True:
        print()
        for i, opt in enumerate(options):
            check = f"{G}[✓]{RST}" if i in selected else f"{DIM}[ ]{RST}"
            print(f"    {check} {Y}{i+1}.{RST} {opt}")
        print(f"\n    {Y}0.{RST} Done — confirm selection")
        print(f"    {Y}A.{RST} Select all")
        print(f"    {Y}N.{RST} Clear all")

        choice = input(f"\n  {C}Toggle (number) or 0 to confirm: {RST}").strip().upper()
        if choice == "0":
            if selected:
                return sorted(selected)
            print(f"  {R}Select at least one option.{RST}")
        elif choice == "A":
            selected = set(range(len(options)))
        elif choice == "N":
            selected = set()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    if idx in selected:
                        selected.discard(idx)
                    else:
                        selected.add(idx)
            except ValueError:
                pass


# ── Config management ─────────────────────────────────────────────────────────
def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    ok(f"Config saved to {CONFIG_FILE}")

def load_config() -> dict | None:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return None


# ── Step 1: Mode Selection ────────────────────────────────────────────────────
def select_mode() -> str:
    section("STEP 1 — SELECT TRADING MODE")
    modes = [
        f"{G}Paper Trading{RST}    — Simulated trades, zero real money (RECOMMENDED to start)",
        f"{Y}Backtest{RST}        — Test strategies on historical data",
        f"{B}Train Models{RST}    — Train AI/ML models only (no trading)",
        f"{R}Live Trading{RST}    — REAL MONEY — only after paper trading proves 75%+ win rate",
    ]
    idx = numbered_menu(modes, "Which mode do you want to run?")
    mode_map = {0: "paper", 1: "backtest", 2: "train", 3: "live"}
    mode = mode_map[idx]

    if mode == "live":
        print(f"\n  {R}{BOLD}⚠️  WARNING: LIVE TRADING USES REAL MONEY ⚠️{RST}")
        print(f"  {Y}Only proceed if you have:{RST}")
        print(f"    - Run paper trading for at least 2 weeks")
        print(f"    - Achieved 75%+ win rate on 100+ paper trades")
        print(f"    - Set your Alpaca account to LIVE in .env")
        confirm = input(f"\n  {R}Type 'I ACCEPT THE RISK' to proceed: {RST}").strip()
        if confirm != "I ACCEPT THE RISK":
            print(f"  {G}Smart choice. Switching to paper trading.{RST}")
            mode = "paper"

    ok(f"Mode selected: {mode.upper()}")
    return mode


# ── Step 2: Market Selection ──────────────────────────────────────────────────
def select_markets() -> list[str]:
    section("STEP 2 — SELECT MARKETS")
    market_options = [
        f"{B}Stocks{RST}          — US equities (requires Alpaca account)",
        f"{G}Forex{RST}           — Currency pairs (EUR/USD, GBP/USD etc)",
        f"{Y}Crypto{RST}          — Cryptocurrencies via Binance",
        f"{M}All Markets{RST}     — Stocks + Forex + Crypto simultaneously",
    ]
    indices = checkbox_menu(market_options, "Which markets do you want to trade?", preselect=[2])
    market_map = {0: "stocks", 1: "forex", 2: "crypto", 3: "all"}

    if 3 in indices:
        return ["stocks", "forex", "crypto"]
    return [market_map[i] for i in indices]


# ── Step 3: Symbol Selection ──────────────────────────────────────────────────
def select_symbols(markets: list[str]) -> list[str]:
    section("STEP 3 — SELECT SYMBOLS")
    all_selected = []

    for market in markets:
        print(f"\n  {W}═══ {market.upper()} ═══{RST}")

        if market == "stocks":
            categories = list(ALL_STOCKS.keys())
            cat_idx = numbered_menu(
                categories + ["Custom — enter your own tickers"],
                "Select stock category:"
            )
            if cat_idx < len(categories):
                symbols = ALL_STOCKS[categories[cat_idx]]
                opts    = symbols
                sel_idx = checkbox_menu(
                    [f"{s:<12}" for s in opts],
                    f"Select stocks from {categories[cat_idx]}:",
                    preselect=list(range(min(5, len(opts))))
                )
                all_selected += [opts[i] for i in sel_idx]
            else:
                raw = input(f"  {C}Enter ticker symbols separated by commas (e.g. AAPL,MSFT,TSLA): {RST}")
                all_selected += [s.strip().upper() for s in raw.split(",") if s.strip()]

        elif market == "forex":
            categories = list(ALL_FOREX.keys())
            cat_idx = numbered_menu(
                categories + ["Custom — enter your own pairs"],
                "Select Forex category:"
            )
            if cat_idx < len(categories):
                symbols = ALL_FOREX[categories[cat_idx]]
                sel_idx = checkbox_menu(
                    [f"{s:<12}" for s in symbols],
                    f"Select pairs from {categories[cat_idx]}:",
                    preselect=[0, 1, 2]
                )
                all_selected += [symbols[i] for i in sel_idx]
            else:
                raw = input(f"  {C}Enter pairs (e.g. EUR/USD,GBP/USD): {RST}")
                all_selected += [s.strip().upper() for s in raw.split(",") if s.strip()]

        elif market == "crypto":
            categories = list(ALL_CRYPTO.keys())
            cat_idx = numbered_menu(
                categories + ["Custom — enter your own coins"],
                "Select crypto category:"
            )
            if cat_idx < len(categories):
                symbols = ALL_CRYPTO[categories[cat_idx]]
                sel_idx = checkbox_menu(
                    [f"{s:<14}" for s in symbols],
                    f"Select coins from {categories[cat_idx]}:",
                    preselect=[0, 1]
                )
                all_selected += [symbols[i] for i in sel_idx]
            else:
                raw = input(f"  {C}Enter pairs (e.g. BTC/USDT,ETH/USDT): {RST}")
                all_selected += [s.strip().upper() for s in raw.split(",") if s.strip()]

    print(f"\n  {G}Selected {len(all_selected)} symbols:{RST}")
    for i, s in enumerate(all_selected, 1):
        print(f"    {Y}{i:2}.{RST} {s}")

    confirm = input(f"\n  {C}Confirm symbols? (y/n): {RST}").strip().lower()
    if confirm != "y":
        return select_symbols(markets)

    return all_selected


# ── Step 4: Capital ───────────────────────────────────────────────────────────
def select_capital(mode: str) -> float:
    section("STEP 4 — SET STARTING CAPITAL")
    if mode == "live":
        info("Live mode uses your actual account balance from Alpaca.")
        info("Capital here is for risk calculation only.")

    presets = [
        f"{DIM}$1,000{RST}   — Micro account (good for testing)",
        f"{W}$10,000{RST}  — Small account",
        f"{W}$50,000{RST}  — Medium account",
        f"{G}$100,000{RST} — Standard institutional paper account",
        f"{Y}Custom{RST}   — Enter your own amount",
    ]
    preset_vals = [1_000, 10_000, 50_000, 100_000, None]
    idx = numbered_menu(presets, "Select starting capital:")

    if preset_vals[idx] is not None:
        capital = preset_vals[idx]
    else:
        while True:
            try:
                capital = float(input(f"  {C}Enter capital amount ($): {RST}").replace(",", ""))
                if capital > 0:
                    break
            except ValueError:
                pass

    ok(f"Capital: ${capital:,.2f}")
    return capital


# ── Step 5: Settings ──────────────────────────────────────────────────────────
def select_settings() -> dict:
    section("STEP 5 — TRADING SETTINGS")

    settings = {}

    # Risk per trade
    risk_options = [
        f"{G}0.5%{RST}  — Ultra conservative (beginner)",
        f"{W}1.0%{RST}  — Standard (recommended)",
        f"{Y}1.5%{RST}  — Moderate",
        f"{R}2.0%{RST}  — Aggressive",
    ]
    risk_vals = [0.005, 0.01, 0.015, 0.02]
    idx = numbered_menu(risk_options, "Max risk per trade:")
    settings["risk_per_trade"] = risk_vals[idx]

    # Timeframe preference
    tf_options = [
        "Scalping  — 1m/5m   (very active, many trades)",
        "Intraday  — 15m/1h  (balanced, recommended)",
        "Swing     — 4h/1d   (fewer trades, higher quality)",
        "All       — Scan all timeframes simultaneously",
    ]
    tf_idx = numbered_menu(tf_options, "Trading timeframe preference:")
    settings["timeframe_mode"] = ["scalp", "intraday", "swing", "all"][tf_idx]

    # Sentiment
    sent_yn = input(f"\n  {C}Enable sentiment analysis? (y/n) [requires API keys]: {RST}").strip().lower()
    settings["sentiment_enabled"] = (sent_yn == "y")

    # Dashboard
    dash_yn = input(f"  {C}Launch dashboard automatically? (y/n): {RST}").strip().lower()
    settings["launch_dashboard"] = (dash_yn == "y")

    ok("Settings configured.")
    return settings


# ── Step 6: Review & Confirm ──────────────────────────────────────────────────
def review_and_confirm(cfg: dict) -> bool:
    section("STEP 6 — REVIEW CONFIGURATION")

    print(f"  {W}Mode:{RST}        {G if cfg['mode'] == 'paper' else R}{cfg['mode'].upper()}{RST}")
    print(f"  {W}Markets:{RST}     {', '.join(cfg['markets']).upper()}")
    print(f"  {W}Symbols:{RST}     {len(cfg['symbols'])} selected — {', '.join(cfg['symbols'][:5])}{'...' if len(cfg['symbols']) > 5 else ''}")
    print(f"  {W}Capital:{RST}     ${cfg['capital']:,.2f}")
    print(f"  {W}Risk/Trade:{RST}  {cfg['settings']['risk_per_trade']:.1%}")
    print(f"  {W}Timeframe:{RST}   {cfg['settings']['timeframe_mode']}")
    print(f"  {W}Sentiment:{RST}   {'Enabled' if cfg['settings']['sentiment_enabled'] else 'Disabled'}")
    print(f"  {W}Dashboard:{RST}   {'Auto-launch' if cfg['settings']['launch_dashboard'] else 'Manual'}")

    print()
    confirm = input(f"  {C}Start agent with these settings? (y/n): {RST}").strip().lower()
    return confirm == "y"


# ── Apply Settings to Config ──────────────────────────────────────────────────
def apply_settings(cfg: dict):
    """Apply user selections to config.py at runtime."""
    import importlib.util, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from config import config

    # Update symbols
    stocks = [s for s in cfg["symbols"] if "/" not in s and "USDT" not in s]
    forex  = [s for s in cfg["symbols"] if "/" in s and "USDT" not in s and len(s) <= 7]
    crypto = [s for s in cfg["symbols"] if "USDT" in s or ("/" in s and "USD" in s and len(s) > 7)]

    config.markets.STOCKS = stocks or config.markets.STOCKS
    config.markets.FOREX  = forex  or config.markets.FOREX
    config.markets.CRYPTO = crypto or config.markets.CRYPTO

    # Update risk
    config.risk.MAX_RISK_PER_TRADE = cfg["settings"]["risk_per_trade"]

    # Update mode
    config.mode.ACTIVE = cfg["mode"]

    ok("Runtime configuration applied.")


# ── Launch Agent ──────────────────────────────────────────────────────────────
def launch_agent(cfg: dict):
    """Start the trading agent with the configured settings."""
    section("LAUNCHING AGENT")

    # Apply config
    apply_settings(cfg)

    # Launch dashboard in background if requested
    if cfg["settings"]["launch_dashboard"]:
        info("Starting dashboard at http://localhost:8501 ...")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
             "--server.headless", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok("Dashboard started: http://localhost:8501")

    # Import and start agent
    from main import TradingAgent
    agent = TradingAgent(
        mode=cfg["mode"],
        symbols=cfg["symbols"],
    )

    # Update capital
    from execution.paper_simulator import paper_simulator
    paper_simulator.initial_capital = cfg["capital"]
    paper_simulator.cash            = cfg["capital"]
    from risk.risk_manager import risk_manager
    risk_manager.portfolio_value = cfg["capital"]

    print(f"\n  {G}{BOLD}🚀 AGENT STARTING...{RST}")
    print(f"  {DIM}Press Ctrl+C to stop gracefully{RST}\n")

    if cfg["mode"] == "train":
        agent.initialise()
        agent.train_models()
    elif cfg["mode"] == "backtest":
        run_backtest(cfg)
    else:
        agent.run()


# ── Backtest Runner ───────────────────────────────────────────────────────────
def run_backtest(cfg: dict):
    """Run backtests on selected symbols."""
    section("RUNNING BACKTEST")

    from core.data_engine import data_engine
    from backtesting.backtest_engine import BacktestEngine
    from backtesting.performance_analyzer import performance_analyzer
    from reporting.excel_exporter import excel_exporter

    engine = BacktestEngine(
        initial_capital=cfg["capital"],
        risk_per_trade=cfg["settings"]["risk_per_trade"],
    )

    all_results = {}
    for symbol in cfg["symbols"]:
        info(f"Backtesting: {symbol}")
        md = data_engine.get_bars(symbol, "1h", lookback_days=365)
        if md.ohlcv.empty:
            warn(f"No data for {symbol}")
            continue

        result = engine.run(md.ohlcv, symbol, "1h")
        all_results[symbol] = result

        status = G if result.win_rate >= 0.6 else R
        print(f"    {status}{symbol:<15}{RST} | "
              f"WR: {result.win_rate:.1%} | "
              f"PF: {result.profit_factor:.2f} | "
              f"P&L: ${result.total_pnl:,.2f} | "
              f"Sharpe: {result.sharpe_ratio:.2f}")

    # Summary
    if all_results:
        section("BACKTEST SUMMARY")
        all_trades = [t for r in all_results.values() for t in r.trades]
        if all_trades:
            report = performance_analyzer.analyse(all_trades, cfg["capital"])
            print(f"\n  {G}Overall Grade: {BOLD}{report.grade}{RST}")
            print(f"  Win Rate:     {report.win_rate:.1%}")
            print(f"  Sharpe:       {report.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {report.max_drawdown:.2%}")
            print(f"  Total P&L:    ${report.total_pnl:,.2f}")

            # Export
            path = excel_exporter.export(
                all_trades, filename="backtest_results.xlsx",
                initial_capital=cfg["capital"],
            )
            ok(f"Results exported: {path}")


# ── Quick Launch (saved config) ───────────────────────────────────────────────
def quick_launch():
    """Launch with last saved configuration."""
    cfg = load_config()
    if not cfg:
        err("No saved config found. Run interactive setup first.")
        sys.exit(1)

    section("QUICK LAUNCH — USING SAVED CONFIG")
    print(f"  Mode: {cfg['mode'].upper()} | Symbols: {len(cfg['symbols'])} | Capital: ${cfg['capital']:,.2f}")
    confirm = input(f"  {C}Launch with saved config? (y/n): {RST}").strip().lower()
    if confirm == "y":
        launch_agent(cfg)
    else:
        interactive_setup()


# ── Main Menu ─────────────────────────────────────────────────────────────────
def main_menu():
    """Show main menu."""
    section("MAIN MENU")
    options = [
        f"{G}New Setup{RST}         — Configure and start the agent",
        f"{B}Quick Launch{RST}      — Use last saved configuration",
        f"{C}Dashboard Only{RST}    — Open live dashboard",
        f"{Y}Train Models{RST}      — Train AI/ML models on fresh data",
        f"{M}Run Backtest{RST}      — Test strategies on historical data",
        f"{W}Check Status{RST}      — Verify API keys and dependencies",
        f"{R}Exit{RST}",
    ]
    idx = numbered_menu(options, "What do you want to do?")
    return idx


# ── Status Check ──────────────────────────────────────────────────────────────
def check_status():
    """Check API keys, dependencies, and model status."""
    section("SYSTEM STATUS CHECK")

    # Check .env
    env_file = Path(".env")
    if env_file.exists():
        ok(".env file exists")
        import os
        from dotenv import load_dotenv
        load_dotenv()

        keys = {
            "ALPACA_API_KEY":      "Alpaca (Stocks/Forex)",
            "BINANCE_API_KEY":     "Binance (Crypto)",
            "FINNHUB_API_KEY":     "Finnhub (News)",
            "ALPHA_VANTAGE_API_KEY":"Alpha Vantage",
            "REDDIT_CLIENT_ID":    "Reddit (Social)",
            "NEWS_API_KEY":        "NewsAPI",
        }
        for key, name in keys.items():
            val = os.getenv(key, "")
            if val and val != f"your_{key.lower()}_here":
                ok(f"{name}: Configured")
            else:
                warn(f"{name}: NOT SET (add to .env)")
    else:
        err(".env file missing — copy .env.example to .env and fill in API keys")

    # Check models
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    if model_files:
        ok(f"Models: {len(model_files)} trained models found")
        for m in model_files:
            print(f"    {G}✓{RST} {m.name}")
    else:
        warn("Models: No trained models found — run 'Train Models' first")

    # Check packages
    packages = ["alpaca", "ccxt", "yfinance", "sklearn", "xgboost", "streamlit", "pandas", "numpy"]
    print()
    for pkg in packages:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
            ok(f"Package: {pkg}")
        except ImportError:
            err(f"Package: {pkg} — NOT INSTALLED (run: pip install -r requirements.txt)")


# ── Interactive Setup ─────────────────────────────────────────────────────────
def interactive_setup():
    """Full guided setup."""
    clear()
    banner()

    mode     = select_mode()
    markets  = select_markets()
    symbols  = select_symbols(markets)
    capital  = select_capital(mode)
    settings = select_settings()

    cfg = {
        "mode":     mode,
        "markets":  markets,
        "symbols":  symbols,
        "capital":  capital,
        "settings": settings,
    }

    if review_and_confirm(cfg):
        save_config(cfg)
        launch_agent(cfg)
    else:
        print(f"\n  {Y}Setup cancelled. Returning to menu...{RST}")
        main()


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Institutional Trading Agent Launcher")
    parser.add_argument("--quick",     action="store_true", help="Quick launch with saved config")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard only")
    parser.add_argument("--status",    action="store_true", help="Check system status")
    args = parser.parse_args()

    clear()
    banner()

    if args.quick:
        quick_launch()
        return
    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
        return
    if args.status:
        check_status()
        return

    # Interactive menu loop
    while True:
        try:
            idx = main_menu()
            if idx == 0:
                interactive_setup()
            elif idx == 1:
                quick_launch()
            elif idx == 2:
                subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
                ok("Dashboard starting at http://localhost:8501")
                input(f"\n  {DIM}Press Enter to return to menu...{RST}")
            elif idx == 3:
                cfg = load_config() or {"symbols": ["AAPL", "BTC/USDT"], "mode": "train", "capital": 100000, "settings": {}}
                cfg["mode"] = "train"
                launch_agent(cfg)
            elif idx == 4:
                cfg = load_config() or {}
                if not cfg:
                    warn("No saved config. Running setup first.")
                    interactive_setup()
                else:
                    cfg["mode"] = "backtest"
                    launch_agent(cfg)
            elif idx == 5:
                check_status()
                input(f"\n  {DIM}Press Enter to return to menu...{RST}")
            elif idx == 6:
                print(f"\n  {Y}Goodbye.{RST}\n")
                sys.exit(0)
        except KeyboardInterrupt:
            print(f"\n\n  {Y}Interrupted. Returning to menu...{RST}")


if __name__ == "__main__":
    main()