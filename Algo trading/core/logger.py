"""
core/logger.py
─────────────────────────────────────────────────────────────────────────────
Centralised logging for the entire trading agent.
Uses loguru for structured, coloured, rotating logs.
Every module imports get_logger() from here.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from loguru import logger
from pathlib import Path
from config import config


def setup_logger() -> None:
    """Configure the global logger. Call once at startup from main.py."""
    logger.remove()  # Remove default handler

    log_file = config.paths.logs / "trading_agent_{time:YYYY-MM-DD}.log"

    # Console handler — coloured, human-readable
    logger.add(
        sys.stdout,
        level=config.reporting.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — JSON structured for audit trail
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation=config.reporting.LOG_ROTATION,
        retention=config.reporting.LOG_RETENTION,
        compression="zip",
        enqueue=True,   # Thread-safe async logging
    )

    # Separate file for trade events only
    logger.add(
        str(config.paths.logs / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        filter=lambda record: "TRADE" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="1 day",
        retention="90 days",
        enqueue=True,
    )

    logger.info("Logger initialised successfully.")


def get_logger(name: str):
    """
    Return a logger instance bound to a specific module name.
    Usage: log = get_logger(__name__)
    """
    return logger.bind(module=name)