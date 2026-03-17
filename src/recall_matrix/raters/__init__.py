from recall_matrix.raters.rater import Rater
from recall_matrix.raters.rater_huggingface import RaterHuggingFace
from recall_matrix.raters.rater_openai import RaterOpenAI
from recall_matrix.raters.rater_reranker import RaterReranker

__all__ = ["Rater", "RaterReranker", "RaterOpenAI", "RaterHuggingFace"]


def initialize_rater(
    rater_name: str, model_name: str | None, device: str | None = None
) -> Rater:
    """Initialize the rater."""
    if rater_name == "reranker":
        rater = RaterReranker(model_name=model_name, device=device)
    elif rater_name == "openai":
        rater = RaterOpenAI(model_name=model_name)
    elif rater_name == "huggingface":
        rater = RaterHuggingFace(model_name=model_name)
    else:
        raise ValueError(f"Invalid argument: {rater_name=}")
    return rater
