from typing import Literal

from rmatch.raters.rater import Rater
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
    reranker_threshold: float | None = None,
    top_k: int | None = None,
    verbose_errors: bool = False,
    quantization: Literal["4bit", "8bit"] | None = None,
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
        rater = RaterOpenAI(model_name=model_name)
    elif rater_name == "huggingface":
        rater = RaterHuggingFace(
            model_name=model_name,
            verbose_errors=verbose_errors,
            quantization=quantization,
        )
    elif rater_name == "huggingface_batched":
        rater = RaterHuggingFaceBatched(
            model_name=model_name,
            verbose_errors=verbose_errors,
            quantization=quantization,
        )
    else:
        raise ValueError(f"Invalid argument: {rater_name=}")
    return rater
