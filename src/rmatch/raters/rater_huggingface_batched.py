from typing import Literal

from tqdm import tqdm

from rmatch import get_logger
from rmatch.raters.rater_huggingface import RaterHuggingFace

log = get_logger(__name__)


class RaterHuggingFaceBatched(RaterHuggingFace):
    """HuggingFace rater that processes recall segments in batches.

    Inherits all prompt-building, parsing, and model-init logic from
    ``RaterHuggingFace`` and overrides ``compute_ratings_single_sub`` to
    send multiple prompts through the pipeline at once.
    """

    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        quantization: Literal["4bit", "8bit"] | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 64,
    ):
        super().__init__(
            model_name=model_name,
            window_size=window_size,
            verbose_errors=verbose_errors,
            quantization=quantization,
        )
        self.rater_name = "huggingface_batched"
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        tokenizer = self.pipe.tokenizer
        if tokenizer.pad_token_id is None:  # type: ignore
            tokenizer.pad_token_id = tokenizer.eos_token_id  # type: ignore
        tokenizer.padding_side = "left"  # type: ignore

    def compute_ratings_single_sub(  # type: ignore
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
        max_retries: int = 10,
        **kwargs,
    ) -> list[tuple[int, list[int]]]:
        if output_scores:
            raise NotImplementedError(
                "HuggingFace batched rater does not support output_scores = True"
            )

        story = self.format_story(story_segments)
        story_segs_formatted = self.format_story_segments(story_segments)
        n_story_segments = len(story_segments)

        prompts: list[list[dict[str, str]]] = []
        for idx in range(len(recall_segments)):
            if self.use_context:
                window = self.build_recall_window(
                    recall_segments, idx, self.window_size
                )
            else:
                window = None

            query = self.build_message(
                recall_segments[idx], story, story_segs_formatted, window
            )
            prompts.append([{"role": "user", "content": query}])

        results: dict[int, set[int]] = {}
        pending = list(range(len(recall_segments)))

        for attempt in range(1, max_retries + 1):
            if not pending:
                break

            pending_prompts = [prompts[i] for i in pending]
            responses = self.pipe(
                pending_prompts,
                batch_size=self.batch_size,
                return_full_text=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.pipe.tokenizer.eos_token_id,  # type: ignore
            )

            still_pending: list[int] = []
            for idx, response in tqdm(
                zip(pending, responses),
                total=len(pending),
                desc=f"(parsing attempt {attempt})",
                position=1,
                leave=False,
            ):
                parsed = self.parse(response[0]["generated_text"])  # type: ignore

                if parsed is not None and all(
                    1 <= i <= n_story_segments for i in parsed
                ):
                    results[idx] = parsed
                else:
                    still_pending.append(idx)

            if still_pending:
                log.info(
                    f"Attempt {attempt}/{max_retries}: "
                    f"{len(still_pending)}/{len(pending)} segments need retry"
                )

            pending = still_pending

        for idx in pending:
            log.warning(
                f"All {max_retries} attempts failed for segment {idx}, "
                "defaulting to no matches"
            )
            results[idx] = set()

        ratings: list[tuple[int, list[int]]] = []
        for idx in range(len(recall_segments)):
            story_indices = sorted(i - 1 for i in results[idx])
            ratings.append((idx, story_indices))

        return ratings
