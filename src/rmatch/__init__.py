import json
import signal
from copy import deepcopy
from typing import Callable

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()

matchlist_type = list[tuple[int, list[int]]]
parser_type = Callable[[str], tuple[set[int] | None, str]]
#  prompt_response[recall_idx][attempt] -> (prompt, raw_response, parsed_response)
prompts_responses_type = list[list[tuple[str, str, set[int] | None]]]


def _sigterm_handler(_signum, _frame):
    raise KeyboardInterrupt()


try:
    signal.signal(signal.SIGTERM, _sigterm_handler)
except ValueError:
    pass


def print_config(config: dict, **kwargs: dict):
    """Pretty print the config."""
    config_copy = deepcopy(config)
    config_copy.update(kwargs)
    # if sub_ids > 3, replace them
    if "sub_ids" in config_copy.keys():
        if len(config_copy["sub_ids"]) > 3:
            config_copy["sub_ids"] = config_copy["sub_ids"][:3] + ["..."]
    console.print_json(json.dumps(config_copy, indent=4))


from rmatch.matchers import (  # noqa: E402,F401
    MatcherAnthropic,
    MatcherCuda,
    MatcherHuggingFace,
    MatcherMac,
    MatcherOpenAI,
    get_matcher,
)
from rmatch.matching import match  # noqa: E402,F401
from rmatch.segmenters import SegmenterCuda  # noqa: E402,F401

__all__ = [
    "MatcherAnthropic",
    "MatcherCuda",
    "MatcherHuggingFace",
    "MatcherMac",
    "MatcherOpenAI",
    "get_matcher",
    "match",
    "SegmenterCuda",
]
