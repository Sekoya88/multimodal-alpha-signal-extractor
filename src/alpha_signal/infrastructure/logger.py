"""logger.py — Structured logging configuration.

Provides robust logging suitable for scaling in production.
Supports standard colored logs for local dev and JSON logs for production/aggregators.
"""

import os
import sys

from loguru import logger


def setup_logging(log_level: str = "INFO", force_json: bool = False) -> None:
    """Configure loguru for the application.

    Args:
        log_level: Minimum log level to capture (e.g., "INFO", "DEBUG").
        force_json: If True, format logs as JSON. If False, checks LOG_JSON env var.
    """
    logger.remove()  # Remove default handler

    use_json = force_json or os.getenv("LOG_JSON", "false").lower() == "true"

    if use_json:
        # Production ready: JSON structured logs
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,  # Outputs JSON
            enqueue=True,    # Thread-safe async logging
            backtrace=True,  # Detailed exception info
            diagnose=False,  # Don't leak variable values in prod errors
        )
    else:
        # Development friendly: Colorful, readable logs
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            enqueue=True,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

# Export the logger object itself so other modules can just import it
__all__ = ["logger", "setup_logging"]
