import json
import logging
from typing import Any

from dotenv import dotenv_values
from rich.console import Console

CONFIG = dotenv_values(".env")
FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)
console = Console()


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Initialize a logger"""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter(FORMAT)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def print_config(config: dict):
    """Pretty print the config."""
    console.print_json(json.dumps(config, indent=4))
