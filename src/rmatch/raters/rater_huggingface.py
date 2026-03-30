import re

import torch
from tqdm import tqdm
from transformers import pipeline

from rmatch import get_logger
from rmatch.raters.rater import Rater

log = get_logger(__name__)


class RaterHuggingFace(Rater):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
    ):
        self.rater_name = "huggingface"

        self.use_context = window_size > 0
        self.window_size = window_size

        if model_name is None:
            self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        # handle devices
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            attn_impl = "sdpa"
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float32
            attn_impl = "sdpa"
        else:
            torch_dtype = torch.float32
            attn_impl = "eager"

        self.pipe = pipeline(
            task="text-generation",
            model=self.model_name,
            device_map="auto",
            dtype=torch_dtype,
            model_kwargs={"attn_implementation": attn_impl},
        )

        if torch.cuda.is_available():
            self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

    def build_message(
        self, recall_segment: str, story: str, story_segments: str, window: str | None
    ) -> str:
        if window is None:
            message = f"""This is the original story:
        {story}

        It can be broken down into the following independent pieces of information:
        {story_segments}

        Here is a single clause from a participant's recall of the story:
        "{recall_segment}"

        Which of the numbered information pieces above
        are expressed by this recall clause?
        List all applicable numbers. If none apply, return the string "<NONE>".

        Return ONLY a set of numbers in <>, for example:
        <3, 7, 12>
        """

        else:
            message = f"""This is the original story:
        {story}

        It can be broken down into the following independent pieces of information:
        {story_segments}

        Below is a window of consecutive clauses from a participant's recall.
        The TARGET clause is marked with >>> <<<.
        The other clauses are provided _only as context_.

        Recall window:
        {window}

        Which of the numbered information pieces are
        expressed by the >>> TARGET <<< clause?

        Use the surrounding clauses only to resolve references (e.g., pronouns),
        but DO NOT attribute information expressed only in neighboring clauses.
        If a numbered information piece is not explicitly expressed in the TARGET
        clause itself, do NOT include it.

        If none apply, return "<NONE>".

        Return ONLY a set of numbers in angle brackets, for example:
        <3, 7, 12>"""

        return message

    def build_recall_window(
        self, recall_segments: list[str], idx: int, window_size: int
    ) -> str:
        start = max(0, idx - window_size)
        end = min(len(recall_segments), idx + window_size + 1)

        lines = []
        for i in range(start, end):
            clause = recall_segments[i].strip()
            if i == idx:
                lines.append(f">>> {clause} <<<")
            else:
                lines.append(f"{clause}")

        return "\n".join(lines)

    def format_story_segments(self, story_segments: list[str]) -> str:
        return "\n".join(
            f"{i + 1}. {seg.strip()}" for i, seg in enumerate(story_segments)
        )

    def format_story(self, story_segments: list[str]) -> str:
        return " ".join(seg.strip() for seg in story_segments)

    def parse(self, raw: str) -> set[int]:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            log.warning(f"failed to parse model output: {raw}")
            return set()

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    def compute_ratings_single_sub(  # type: ignore
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
        **kwargs,
    ) -> list[tuple[int, list[int]]]:
        if output_scores:
            raise NotImplementedError(
                "Anthropic rater currently does not support output_scores = True"
            )

        story = self.format_story(story_segments)
        story_segs_formatted = self.format_story_segments(story_segments)
        ratings: list[tuple[int, list[int]]] = []

        for idx, recall_seg in enumerate(
            tqdm(
                recall_segments,
                desc="(rating recall segments)",
                position=1,
                leave=False,
            )
        ):
            if self.use_context:
                window = self.build_recall_window(
                    recall_segments,
                    idx,
                    self.window_size,
                )
            else:
                window = None

            query = self.build_message(recall_seg, story, story_segs_formatted, window)

            response = self.pipe(
                [{"role": "user", "content": query}],
                return_full_text=False,
                pad_token_id=self.pipe.tokenizer.eos_token_id,  # type: ignore
            )
            parsed_response = self.parse(response[0]["generated_text"])  # type: ignore

            story_indices = sorted(i - 1 for i in parsed_response)
            ratings.append((idx, story_indices))

        return ratings
