import re

import anthropic
from tqdm import tqdm

from recall_matrix import get_logger
from recall_matrix.raters.rater import Rater

log = get_logger(__name__)

ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
}


class RaterAnthropic(Rater):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        dry_run: bool = False,
    ):
        self.rater_name = "anthropic"

        if model_name is None:
            self.model_name = "claude-opus-4-6"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        self.client = anthropic.Anthropic()
        self.use_context = window_size > 0
        self.window_size = window_size
        self.usage_metrics = {"in_tokens": 0, "out_tokens": 0, "cost": 0.0}
        self.estimated_usage_metrics = {
            "est_in_tokens": 0,
            "est_out_tokens": 0,
            "est_cost": 0.0,
        }
        self.dry_run = dry_run

    def get_usage(self) -> dict | None:
        if self.dry_run:
            return self.estimated_usage_metrics
        return self.usage_metrics

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        if self.model_name not in ANTHROPIC_PRICES:
            log.warning(f"No pricing found for {self.model_name}, cost set to 0")
            return 0.0
        prices = ANTHROPIC_PRICES[self.model_name]
        return (in_tokens * prices[0] + out_tokens * prices[1]) / 1_000_000

    def _estimate_tokens(self, query: str) -> int:
        mock = self.client.messages.count_tokens(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
        )
        return mock.input_tokens

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
                disable=self.dry_run,
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

            if self.dry_run:
                parsed_response: set[int] = set()  # empty dummy
                estimated_in_tokens = self._estimate_tokens(query)
                estimated_out_tokens = self._estimate_tokens("<8, 9, 10>")
                self.estimated_usage_metrics["est_in_tokens"] += estimated_in_tokens
                self.estimated_usage_metrics["est_out_tokens"] += estimated_out_tokens
                self.estimated_usage_metrics["est_cost"] += self._calculate_cost(
                    estimated_in_tokens, estimated_out_tokens
                )
            else:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": query}],
                )

                in_tokens = response.usage.input_tokens
                out_tokens = response.usage.output_tokens
                self.usage_metrics["in_tokens"] += in_tokens
                self.usage_metrics["out_tokens"] += out_tokens
                self.usage_metrics["cost"] += self._calculate_cost(
                    in_tokens, out_tokens
                )
                parsed_response = self.parse(response.content[0].text)  # type: ignore

            story_indices = sorted(i - 1 for i in parsed_response)
            ratings.append((idx, story_indices))

        return ratings
