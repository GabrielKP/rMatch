import json
import logging
from copy import deepcopy

from dotenv import dotenv_values
from rich.console import Console

ENV = dotenv_values(".env")
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


def print_config(config: dict, **kwargs: dict):
    """Pretty print the config."""
    config_copy = deepcopy(config)
    config_copy.update(kwargs)
    # if sub_ids > 3, replace them
    if "sub_ids" in config_copy.keys():
        if len(config_copy["sub_ids"]) > 3:
            config_copy["sub_ids"] = config_copy["sub_ids"][:3] + ["..."]
    console.print_json(json.dumps(config_copy, indent=4))


from rmatch.match import match  # noqa: E402,F401
