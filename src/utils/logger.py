"""
Logger setup — single call returns a configured logger for the whole app.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


def setup_logger(log_config: Dict[str, Any]) -> logging.Logger:
    """Configure and return the root application logger.

    Args:
        log_config: The 'logging' section from the YAML config.

    Returns:
        Configured Logger instance.
    """
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger("traffic_mgmt")
    logger.setLevel(level)

    # Avoid duplicate handlers when the function is called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating file handler (optional)
    if log_config.get("log_to_file", False):
        log_dir = Path(log_config.get("log_dir", "logs/"))
        log_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = log_config.get("rotate_every_mb", 50) * 1024 * 1024
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=max_bytes,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
