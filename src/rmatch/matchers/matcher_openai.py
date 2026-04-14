import os

import tiktoken
from openai import OpenAI
from tqdm import tqdm

from rmatch import get_logger
from rmatch.matchers.matcher import Matcher
from rmatch.prompt import get_prompt_and_parser

log = get_logger(__name__)

OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.2": (1.75, 14),
    "gpt-4.1": (2, 8),
}


class MatcherOpenAI(Matcher, matcher_name="openai"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        dry_run: bool = False,
        api_key: str | None = None,
        prompt: str | None = None,
        max_retries: int | None = None,
        # required for initialization
        matcher_name: str | None = None,
    ):
        super().__init__()
        self.matcher_name = "openai"
        self.prompt = prompt

        if model_name is None:
            self.model_name = "gpt-4.1"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        if max_retries is None:
            self.max_retries = 10
            log.info(f"Set max retries to {self.max_retries}")
        else:
            self.max_retries = max_retries

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY not found in .env or environment variables."
                )
        self.client = OpenAI(api_key=api_key)
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
        if self.model_name not in OPENAI_PRICES:
            log.warning(f"No pricing found for {self.model_name}, cannot compute cost")
            return None
        return self.usage_metrics

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        if self.model_name not in OPENAI_PRICES:
            return 0.0
        prices = OPENAI_PRICES[self.model_name]
        return (in_tokens * prices[0] + out_tokens * prices[1]) / 1_000_000

    def _estimate_tokens(self, query: str) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(query))

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
    ) -> list[tuple[int, list[int]]]:
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
            for attempt in range(1, self.max_retries + 1):
                prompt, parser = get_prompt_and_parser(
                    story_segments,
                    recall_segments,
                    idx,
                    self.window_size,
                    prompt=self.prompt,
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
                    response = self.client.responses.create(
                        model=self.model_name, input=prompt
                    )

                    assert response.usage is not None
                    in_tokens = response.usage.input_tokens
                    out_tokens = response.usage.output_tokens
                    self.usage_metrics["in_tokens"] += in_tokens
                    self.usage_metrics["out_tokens"] += out_tokens
                    self.usage_metrics["cost"] += self._calculate_cost(
                        in_tokens, out_tokens
                    )
                    raw_text = response.output_text
                    parsed_response_ = parser(raw_text)
                    if match_key is not None:
                        self._log_prompt_response(
                            match_key=match_key,
                            idx_recall=idx,
                            prompt=prompt,
                            response=raw_text,
                            parsed_response=parsed_response_,
                        )
                    if parsed_response_ is not None:
                        parsed_response = parsed_response_
                        break
                log.warning(
                    f"Attempt {attempt}/{self.max_retries}:"
                    f" segmeent needs retry for segment {idx}"
                )
            else:
                log.warning(f"All attempts failed for segment {idx}")

            story_indices = sorted(i - 1 for i in parsed_response)
            matches.append((idx, story_indices))

        return matches
