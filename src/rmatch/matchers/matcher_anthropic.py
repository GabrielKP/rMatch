import re

import anthropic
from tqdm import tqdm

from rmatch import ENV, get_logger
from rmatch.matchers.matcher import Matcher
from rmatch.prompt import prompt_default

log = get_logger(__name__)


ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
}


class MatcherAnthropic(Matcher, matcher_name="anthropic"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        dry_run: bool = False,
        api_key: str | None = None,
        # required for initialization
        matcher_name: str | None = None,
    ):
        self.matcher_name = "anthropic"

        if model_name is None:
            self.model_name = "claude-opus-4-6"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        if api_key is None:
            api_key = ENV.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in .env or environment variables."
                )
        self.client = anthropic.Anthropic(api_key=api_key)

        self.use_context = window_size > 0
        self.window_size = window_size
        self.usage_metrics = {"in_tokens": 0, "out_tokens": 0, "cost": 0.0}
        self.estimated_usage_metrics = {
            "est_in_tokens": 0,
            "est_out_tokens": 0,
            "est_cost": 0.0,
        }
        self.dry_run = dry_run
        if self.dry_run:
            log.info("RUNNING IN DRY RUN MODE")

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

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        max_retries: int = 10,
    ) -> list[tuple[int, list[int]]]:
        """Anthropic-based matcher.

        Parameters
        ----------
        story_segments: list[str]
            list of story segments
        recall_segments: list[str]
            list of recall segments

        Returns
        -------
        matches: list[tuple[int, list[int]]]
            list of matches:
            [
                (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..]),
                (recall_segment_id_2, [story_segment_id_z, ...]),
                ...
            ]
        """

        if len(recall_segments) == 0:
            return []

        matches: list[tuple[int, list[int]]] = list()

        for idx, recall_seg in enumerate(
            tqdm(
                recall_segments,
                desc="(rating recall segments)",
                position=1,
                leave=False,
                disable=self.dry_run,
            )
        ):
            parsed_response: set[int] = set()
            for attempt in range(1, max_retries + 1):
                prompt, parser = prompt_default(
                    story_segments, recall_segments, idx, self.window_size
                )

                if self.dry_run:
                    estimated_in_tokens = self._estimate_tokens(prompt)
                    estimated_out_tokens = self._estimate_tokens("<8, 9, 10>")
                    self.estimated_usage_metrics["est_in_tokens"] += estimated_in_tokens
                    self.estimated_usage_metrics["est_out_tokens"] += (
                        estimated_out_tokens
                    )
                    self.estimated_usage_metrics["est_cost"] += self._calculate_cost(
                        estimated_in_tokens, estimated_out_tokens
                    )
                    break
                else:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    in_tokens = response.usage.input_tokens
                    out_tokens = response.usage.output_tokens
                    self.usage_metrics["in_tokens"] += in_tokens
                    self.usage_metrics["out_tokens"] += out_tokens
                    self.usage_metrics["cost"] += self._calculate_cost(
                        in_tokens, out_tokens
                    )
                    parsed_response_ = parser(response.content[0].text)  # type: ignore
                    if parsed_response_ is not None:
                        parsed_response = parsed_response_
                        break
                log.warning(
                    f"Attempt {attempt}/{max_retries}:"
                    f" segmeent needs retry for segment {idx}"
                )
            else:
                log.warning(f"All attempts failed for segment {idx}")

            story_indices = sorted(i - 1 for i in parsed_response)
            matches.append((idx, story_indices))

        return matches
