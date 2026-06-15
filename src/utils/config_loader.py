"""
Config loader — reads YAML, validates required keys, returns a plain dict.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


_REQUIRED_KEYS = [
    "signaling",
    "detection",
    "audio",
    "ensemble",
    "dashboard",
    "logging",
]


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required top-level key is missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    for key in _REQUIRED_KEYS:
        if key not in config:
            raise KeyError(
                f"Required config section '{key}' is missing from {path}. "
                "Check config/default.yaml for the full reference."
            )

    return config
