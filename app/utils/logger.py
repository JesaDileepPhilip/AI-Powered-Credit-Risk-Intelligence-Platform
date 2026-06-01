"""
app/utils/logger.py — Centralised structured logging factory.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message")

All loggers share the same format and respect the LOG_LEVEL from config.
Handlers are added only once per logger name to avoid duplicate output.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger for the given name.

    Args:
        name:  Logger name, typically ``__name__`` of the calling module.
        level: Override log level (DEBUG | INFO | WARNING | ERROR | CRITICAL).
               Falls back to ``settings.log_level`` when not provided.

    Returns:
        A :class:`logging.Logger` instance with a stream handler attached.
    """
    # Lazy import to avoid circular imports during package initialisation
    from config import settings

    resolved_level_str = (level or settings.log_level).upper()
    resolved_level = getattr(logging, resolved_level_str, logging.INFO)

    logger = logging.getLogger(name)

    # Guard: only configure a logger once
    if logger.handlers:
        return logger

    logger.setLevel(resolved_level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(resolved_level)

    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt=settings.log_date_format,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
