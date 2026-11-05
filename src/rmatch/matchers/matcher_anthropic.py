import os
from collections import Counter

from rich.table import Table
from tqdm import trange

from rmatch import (
    console,
    matchlist_type,
    parser_type,
    prompts_responses_type,
)
from rmatch.matchers.matcher_base import DEFAULT_PARSER, MatcherBase

ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
}


class MatcherAnthropic(MatcherBase):
    matcher_name = "anthropic"

    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        dry_run: bool = False,
        api_key: str | None = None,
        max_retries: int | None = None,
    ):
        super().__init__()

        if model_name is None:
            self.model_name = "claude-opus-4-6"
            console.log(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            console.log(f"Initializing model: {self.model_name}")

        if max_retries is None:
            self.max_retries = 10
            console.log(f"Set max retries to {self.max_retries}")
        else:
            self.max_retries = max_retries

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in .env or environment variables."
                )
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)

        self.window_size = window_size
        self.usage_metrics = {"in_tokens": 0, "out_tokens": 0, "cost": 0.0}
        self.estimated_usage_metrics = {
            "est_in_tokens": 0,
            "est_out_tokens": 0,
            "est_cost": 0.0,
        }
        self.dry_run = dry_run
        if self.dry_run:
            console.log("RUNNING IN DRY RUN MODE")

    def get_usage(self) -> dict | None:
        if self.dry_run:
            return self.estimated_usage_metrics
        return self.usage_metrics

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        if self.model_name not in ANTHROPIC_PRICES:
            console.log(f"No pricing found for {self.model_name}, cost set to 0")
            return 0.0
        prices = ANTHROPIC_PRICES[self.model_name]
        return (in_tokens * prices[0] + out_tokens * prices[1]) / 1_000_000

    def _estimate_tokens(self, query: str) -> int:
        mock = self.client.messages.count_tokens(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
        )
        return mock.input_tokens

    def match_return_prompts_and_responses(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        prompt: str | None = None,
        parser: parser_type | None = None,
    ) -> tuple[matchlist_type, prompts_responses_type]:
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
        prompts_responses: list[list[(str, str, set[int] | None)]]
            prompts_responses[recall_idx][attempt]
                -> (prompt, raw_response, parsed_response)
        """

        if len(recall_segments) == 0:
            return [], []

        n_story_segments = len(story_segments)
        parser = parser or DEFAULT_PARSER
        matches: list[tuple[int, list[int]]] = list()
        prompts_responses_ls: prompts_responses_type = [[] for _ in recall_segments]
        n_matches = 0
        failure_reasons: list[str] = []

        for idx in trange(
            len(recall_segments),
            desc="(rating recall segments)",
            position=1,
            leave=False,
            disable=self.dry_run,
        ):
            model_prompt = self.build_model_prompt(
                story_segments=story_segments,
                recall_segments=recall_segments,
                target_idx=idx,
                prompt=prompt,
            )
            parsed_response: set[int] = set()
            for attempt in range(1, self.max_retries + 1):
                if self.dry_run:
                    estimated_in_tokens = self._estimate_tokens(model_prompt)
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
                        messages=[{"role": "user", "content": model_prompt}],
                    )

                    in_tokens = response.usage.input_tokens
                    out_tokens = response.usage.output_tokens
                    self.usage_metrics["in_tokens"] += in_tokens
                    self.usage_metrics["out_tokens"] += out_tokens
                    self.usage_metrics["cost"] += self._calculate_cost(
                        in_tokens, out_tokens
                    )
                    raw_text = response.content[0].text  # type: ignore
                    parsed_response_, failure_reason = parser(raw_text)
                    n_matches += 1
                    prompts_responses_ls[idx].append(
                        (model_prompt, raw_text, parsed_response_)
                    )
                    if parsed_response_ is not None and not all(
                        1 <= i <= n_story_segments for i in parsed_response_
                    ):
                        parsed_response_ = None
                        failure_reason = "Index out of range."
                    if parsed_response_ is not None:
                        parsed_response = parsed_response_
                        break
                    failure_reasons.append(failure_reason)
                    console.log(f"Parse failure ({failure_reason}):\n{raw_text}")
                console.log(
                    f"Attempt {attempt}/{self.max_retries}:"
                    f" segmeent needs retry for segment {idx}",
                    style="orange1",
                )
            else:
                console.log(
                    f"All attempts failed for segment {idx}",
                    style="orange1",
                )

            story_indices = sorted(i - 1 for i in parsed_response)
            matches.append((idx, story_indices))

        if n_matches > 0:
            n_success = n_matches - len(failure_reasons)
            table = Table(title=f"Parse results ({n_matches} matches)")
            table.add_column("Outcome")
            table.add_column("Count", justify="right")
            table.add_column("Proportion", justify="right")
            table.add_row("success", str(n_success), f"{n_success / n_matches:.1%}")
            for reason, count in Counter(failure_reasons).most_common():
                table.add_row(reason, str(count), f"{count / n_matches:.1%}")
            console.print(table)

        return matches, prompts_responses_ls
