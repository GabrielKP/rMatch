import json
import logging
import signal
from copy import deepcopy

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)
console = Console()

matchlist_type = list[tuple[int, list[int]]]


def _sigterm_handler(_signum, _frame):
    raise KeyboardInterrupt()


try:
    signal.signal(signal.SIGTERM, _sigterm_handler)
except ValueError:
    pass


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


from rmatch.matchers import Matcher  # noqa: E402,F401
