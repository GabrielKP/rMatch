from typing import Literal

from rmatch.matchers.matcher import Matcher
from rmatch.matchers.matcher_anthropic import MatcherAnthropic
from rmatch.matchers.matcher_huggingface import MatcherHuggingFace
from rmatch.matchers.matcher_openai import MatcherOpenAI
from rmatch.matchers.matcher_reranker import MatcherReranker

__all__ = [
    "Matcher",
    "MatcherReranker",
    "MatcherOpenAI",
    "MatcherHuggingFace",
]


def initialize_matcher(
    matcher_name: str,
    model_name: str | None,
    device: str | None = None,
    # reranker
    reranker_threshold: float | None = None,
    top_k: int | None = None,
    # huggingface
    verbose_errors: bool = False,
    quantization: Literal["4bit", "8bit"] | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    # anthropic/openai
    movie_mode: bool = False,
    dry_run: bool = False,
    # anthropic/openai/huggingface
    window_size: int = 5,
) -> Matcher:
    """Initialize the matcher."""
    if matcher_name == "reranker":
        matcher = MatcherReranker(
            model_name=model_name,
            device=device,
            threshold=reranker_threshold,
            top_k=top_k,
        )
    elif matcher_name == "openai":
        matcher = MatcherOpenAI(
            model_name=model_name,
            window_size=window_size,
            dry_run=dry_run,
        )
    elif matcher_name == "anthropic":
        matcher = MatcherAnthropic(
            model_name=model_name,
            window_size=window_size,
            dry_run=dry_run,
            movie_mode=movie_mode,
        )
    elif matcher_name == "huggingface":
        matcher = MatcherHuggingFace(
            model_name=model_name,
            verbose_errors=verbose_errors,
            quantization=quantization,
            window_size=window_size,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Invalid argument: {matcher_name=}")
    return matcher
