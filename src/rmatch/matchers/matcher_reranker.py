import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from rmatch import ENV, get_logger
from rmatch.matchers.matcher import Matcher

log = get_logger(__name__)


class MatcherReranker(Matcher, matcher_name="reranker"):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
        # required for initialization
        matcher_name: str | None = None,
    ):
        self.matcher_name = "reranker"

        if model_name is None:
            self.model_name = "BAAI/bge-reranker-v2-m3"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        token = ENV.get("HF_TOKEN")

        self.model = CrossEncoder(
            self.model_name,
            device=device,
            token=token,
        )

        # get device
        self.device: torch.device = self.model.device
        log.info(f"Using device: {self.device}")

        self.threshold = threshold
        self.top_k = top_k

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, list[int]]]:
        if threshold is None:
            if self.threshold is not None:
                threshold = self.threshold
            else:
                threshold = 0.09
        if top_k is None:
            if self.top_k is not None:
                top_k = self.top_k
            else:
                top_k = 5

        matches_single_sub: list[tuple[int, list[int]]] = list()
        for idx_recall, recall_segment in enumerate(
            tqdm(recall_segments, desc="(rating recall segments)", position=1)
        ):
            matches_single_sub.append((idx_recall, list()))
            rankings = self.model.rank(recall_segment, story_segments, top_k=top_k)
            for ranking in rankings:
                score: float = ranking["score"]  #  type: ignore
                corpus_id: int = ranking["corpus_id"]  #  type: ignore
                if score > threshold:
                    matches_single_sub[idx_recall][1].append(corpus_id)

        return matches_single_sub
