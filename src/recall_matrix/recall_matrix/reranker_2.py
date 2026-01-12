import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from recall_matrix import CONFIG, console, get_logger

log = get_logger(__name__)


class RRRM2:
    def __init__(
        self,
        reranker_method: str,
        reranker_binary: bool = False,
        model_name: str | None = None,
        device: str | None = None,
        debug: bool = False,
    ):
        """Initialize the ReRanker Recall Matrix (RRRM2) object.
        Modified version to work with bigger huggingface models.

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
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        token = CONFIG.get("HF_TOKEN")

        trust_remote_code = self.model_name in [
            "nvidia/llama-embed-nemotron-8b",
            "jinaai/jina-reranker-v2-base-multilingual",
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Reranker-8B", padding_side="left"
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            .eval()
            .to(device)  # type: ignore
        )

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the"
            " requirements based on the Query and the Instruct provided."
            ' Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        self.task = (
            "Given a human recall of an event as query,"
            " only retrieve the passages from the story which exactly match the recall."
            " Be careful to not retrieve similar passages which contain similar"
            " information, but refer to different events. Only retrieve events which"
            " exactly match recall."
        )

        # get device
        self.device: torch.device = self.model.device
        log.info(f"Using device: {self.device}")

        self.debug = debug
        log.info(f"Debug mode: {self.debug}")

    def _format_instruction(self, instruction: str, query: str, doc: str) -> str:
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def _process_inputs(self, pairs) -> dict:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        for key in list(inputs.keys()):
            del inputs[key]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        del batch_scores
        torch.cuda.empty_cache()
        return scores

    def recall_matrix(
        self,
        story_segments: list[str],
        recall_segments: list[str],
    ) -> np.ndarray:
        """Compute the recall matrix"""

        recall_matrix = np.zeros((len(story_segments), len(recall_segments)))
        for j, R_j in enumerate(tqdm(recall_segments, desc="(RM)")):
            pairs = [
                self._format_instruction(self.task, R_j, story_segments[i])
                for i in range(len(story_segments))
            ]
            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)

            recall_matrix[:, j] = scores

        return recall_matrix

    def compute_recall_matrix(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the mutual information recall matrix

        Can override the mutual information method by passing it as an argument."""

        if self.reranker_method == "default":
            return self.recall_matrix(
                story_segments,
                recall_segments,
            )
        else:
            raise ValueError(f"Invalid reranker method: {self.reranker_method}")
