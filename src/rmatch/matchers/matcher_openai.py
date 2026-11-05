import os

from tqdm import tqdm

from rmatch import (
    console,
    matchlist_type,
    parser_type,
    prompts_responses_type,
)
from rmatch.matchers.matcher_base import (
    DEFAULT_PARSER,
    MatcherBase,
    parse_response,
    print_parse_summary,
)

OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.2": (1.75, 14),
    "gpt-4.1": (2, 8),
}


class MatcherOpenAI(MatcherBase):
    matcher_name = "openai"

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
            self.model_name = "gpt-4.1"
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
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY not found in .env or environment variables."
                )
        from openai import OpenAI

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
            console.log(
                f"No pricing found for {self.model_name}, cannot compute cost",
                style="orange1",
            )
            return None
        return self.usage_metrics

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        if self.model_name not in OPENAI_PRICES:
            return 0.0
        prices = OPENAI_PRICES[self.model_name]
        return (in_tokens * prices[0] + out_tokens * prices[1]) / 1_000_000

    def _estimate_tokens(self, query: str) -> int:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(query))

    def match_return_prompts_and_responses(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        prompt: str | None = None,
        parser: parser_type | None = None,
    ) -> tuple[matchlist_type, prompts_responses_type]:
        n_story_segments = len(story_segments)
        parser = parser or DEFAULT_PARSER
        matches: list[tuple[int, list[int]]] = list()
        prompts_responses_ls: prompts_responses_type = [[] for _ in recall_segments]
        n_matches = 0
        failure_reasons: list[str] = []

        for idx, recall_seg in enumerate(
            tqdm(
                recall_segments,
                desc="(rating recall segments)",
                position=1,
                leave=False,
                disable=self.dry_run,
            )
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
                    response = self.client.responses.create(
                        model=self.model_name, input=model_prompt
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
                    parsed_response_, failure_reason = parse_response(
                        parser,
                        raw_text,
                        n_story_segments,
                        truncated=(
                            response.status == "incomplete"
                            and getattr(response.incomplete_details, "reason", None)
                            == "max_output_tokens"
                        ),
                    )
                    n_matches += 1
                    prompts_responses_ls[idx].append(
                        (model_prompt, raw_text, parsed_response_)
                    )
                    if parsed_response_ is not None:
                        parsed_response = parsed_response_
                        break
                    failure_reasons.append(failure_reason)
                console.log(
                    f"Attempt {attempt}/{self.max_retries}:"
                    f" segment needs retry for segment {idx}",
                    style="orange1",
                )
            else:
                console.log(f"All attempts failed for segment {idx}", style="orange1")

            story_indices = sorted(i - 1 for i in parsed_response)
            matches.append((idx, story_indices))

        print_parse_summary(n_matches, failure_reasons)

        return matches, prompts_responses_ls
