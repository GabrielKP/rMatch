import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from recall_matrix import CONFIG, console, get_logger

log = get_logger(__name__)


class RRRM:
    def __init__(
        self,
        reranker_method: str,
        reranker_threshold: float | None = None,
        reranker_binary: bool = False,
        model_name: str | None = None,
        device: str | None = None,
        y_instruction: str = "",
        x_instruction: str = "",
        x_instruction_no_y: str = "",
        debug: bool = False,
    ):
        """Initialize the ReRanker Recall Matrix (RRRM) object.

        Parameters
        ----------
        reranker_method: str
            The method to use for computing the mutual information.
        reranker_threshold: float | None, default=None
            The threshold to use for reranking.
            If not specified, 0.09 will be used.
        model_name: str | None, default=None
            The name of the model to use.
            If None, the model will be taken from .env RERANKER_MODEL_NAME.
            If not specified, gpt2 will be used.
        device: str
            The device to use.
            If None, the device will be taken from .env DEVICE.
            If None, and not specified in .env, the device will be
            discovered by sentence-transformers.
        """

        self.reranker_method = reranker_method
        log.info(f"Reranker method: {reranker_method}")

        if reranker_threshold is None:
            try:
                self.reranker_threshold: float = float(
                    CONFIG.get("RERANKER_THRESHOLD", 0.09)  # type: ignore
                )
            except ValueError as err:
                log.critical("Invalid 'RERANKER_THRESHOLD' in .env")
                raise err
        else:
            self.reranker_threshold = reranker_threshold
        log.info(f"Reranker threshold: {self.reranker_threshold}")

        self.reranker_binary = reranker_binary
        log.info(f"Reranker binary: {reranker_binary}")

        if model_name is None:
            self.model_name: str = CONFIG.get(
                "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
            )  # type: ignore
        else:
            self.model_name = model_name
        log.info(f"Initializing model: {self.model_name}")

        if device is None:
            device = CONFIG.get("DEVICE", None)

        token = CONFIG.get("HF_TOKEN")

        trust_remote_code = self.model_name in [
            "nvidia/llama-embed-nemotron-8b",
            "jinaai/jina-reranker-v2-base-multilingual",
        ]
        self.model = CrossEncoder(
            self.model_name,
            device=device,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if "Qwen" in self.model_name:
            self.model.config.pad_token_id = self.model.tokenizer.eos_token_id  # type: ignore
            log.info(
                f"Added tokenizer.eos_token_id as model.pad_token_id:"
                f" {self.model.config.pad_token_id}"
            )

        # get device
        self.device: torch.device = self.model.device
        log.info(f"Using device: {self.device}")

        self.y_instruction = y_instruction
        self.x_instruction = x_instruction
        self.x_instruction_no_y = x_instruction_no_y

        self.debug = debug
        log.info(f"Debug mode: {self.debug}")

    def recall_matrix_thresholded(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        reranker_binary: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the recall matrix"""

        top_k = 5
        threshold = self.reranker_threshold

        recall_matrix = np.zeros((len(story_segments), len(recall_segments)))
        for j, R_j in enumerate(tqdm(recall_segments, desc="(RM)")):
            rankings = self.model.rank(R_j, story_segments, top_k=top_k)
            for ranking in rankings:
                score: float = ranking["score"]  #  type: ignore
                corpus_id: int = ranking["corpus_id"]  #  type: ignore
                if score > threshold:
                    if reranker_binary:
                        recall_matrix[corpus_id, j] = 1
                    else:
                        recall_matrix[corpus_id, j] = score

        return recall_matrix

    def compute_recall_matrix(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the reranker recall matrix.
        Returns recall matrix (S x R)
        S = # Story segments
        R = # Recall segments"""

        if self.reranker_method == "thresholded":
            return self.recall_matrix_thresholded(
                story_segments,
                recall_segments,
                reranker_binary=self.reranker_binary,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Invalid reranker method: {self.reranker_method}")
