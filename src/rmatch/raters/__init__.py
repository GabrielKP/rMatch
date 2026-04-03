from typing import Literal

from rmatch.raters.rater import Rater
from rmatch.raters.rater_anthropic import RaterAnthropic
from rmatch.raters.rater_huggingface import RaterHuggingFace
from rmatch.raters.rater_huggingface_batched import RaterHuggingFaceBatched
from rmatch.raters.rater_openai import RaterOpenAI
from rmatch.raters.rater_reranker import RaterReranker

__all__ = [
    "Rater",
    "RaterReranker",
    "RaterOpenAI",
    "RaterHuggingFace",
    "RaterHuggingFaceBatched",
]


def initialize_rater(
    rater_name: str,
    model_name: str | None,
    device: str | None = None,
    # reranker
    reranker_threshold: float | None = None,
    top_k: int | None = None,
    # huggingface
    verbose_errors: bool = False,
    quantization: Literal["4bit", "8bit"] | None = None,
    # anthropic/openai
    movie_mode: bool = False,
    dry_run: bool = False,
    # anthropic/openai/huggingface
    window_size: int = 5,
) -> Rater:
    """Initialize the rater."""
    if rater_name == "reranker":
        rater = RaterReranker(
            model_name=model_name,
            device=device,
            threshold=reranker_threshold,
            top_k=top_k,
        )
    elif rater_name == "openai":
        rater = RaterOpenAI(
            model_name=model_name,
            window_size=window_size,
            dry_run=dry_run,
        )
    elif rater_name == "anthropic":
        rater = RaterAnthropic(
            model_name=model_name,
            window_size=window_size,
            dry_run=dry_run,
            movie_mode=movie_mode,
        )
    elif rater_name == "huggingface":
        rater = RaterHuggingFace(
            model_name=model_name,
            verbose_errors=verbose_errors,
            quantization=quantization,
            window_size=window_size,
        )
    elif rater_name == "huggingface_batched":
        rater = RaterHuggingFaceBatched(
            model_name=model_name,
            verbose_errors=verbose_errors,
            quantization=quantization,
            window_size=window_size,
        )
    else:
        raise ValueError(f"Invalid argument: {rater_name=}")
    return rater
