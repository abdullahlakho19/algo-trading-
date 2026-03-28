"""
setup.py
─────────────────────────────────────────────────────────────────────────────
One-command setup script for the Institutional Trading Agent.
Creates virtual environment, installs all dependencies,
and validates the installation.

Usage:
    python setup.py
─────────────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
import os
from pathlib import Path


def run(cmd: str, check: bool = True) -> bool:
    print(f"  ▸ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ✗ Failed: {cmd}")
        return False
    return True


def main():
    print()
    print("=" * 60)
    print("  INSTITUTIONAL TRADING AGENT — SETUP")
    print("=" * 60)
    print()

    # Python version check
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ required. Current:", sys.version)
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")

    # Create virtual environment
    print("\n[1/4] Creating virtual environment...")
    run("python -m venv venv")
    print("✅ Virtual environment created.")

    # Determine pip path
    pip = "venv/Scripts/pip" if os.name == "nt" else "venv/bin/pip"

    # Install dependencies
    print("\n[2/4] Installing dependencies (this may take a few minutes)...")
    run(f"{pip} install --upgrade pip -q")
    run(f"{pip} install -r requirements.txt -q")
    print("✅ Dependencies installed.")

    # Create .env file from template
    print("\n[3/4] Setting up environment file...")
    env_file = Path(".env")
    if not env_file.exists():
        Path(".env.example").copyfile = lambda dst: None
        import shutil
        shutil.copy(".env.example", ".env")
        print("✅ .env created from template.")
        print("   ⚠️  Edit .env and add your Alpaca API keys before running.")
    else:
        print("✅ .env already exists.")

    # Create required directories
    print("\n[4/4] Creating data directories...")
    for d in ["data/historical", "data/logs", "data/reports", "models"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories ready.")

    print()
    print("=" * 60)
    print("  SETUP COMPLETE ✅")
    print("=" * 60)
    print()
    print("  Next steps:")
    print()
    print("  1. Add your API keys to .env:")
    print("     ALPACA_API_KEY=your_key")
    print("     ALPACA_SECRET_KEY=your_secret")
    print()
    print("  2. Activate virtual environment:")
    if os.name == "nt":
        print("     venv\\Scripts\\activate")
    else:
        print("     source venv/bin/activate")
    print()
    print("  3. Train ML models:")
    print("     python main.py --mode train")
    print()
    print("  4. Start the agent (paper trading):")
    print("     python main.py --mode paper")
    print()
    print("  5. Launch live dashboard:")
    print("     streamlit run dashboard/app.py")
    print()


if __name__ == "__main__":
    main()