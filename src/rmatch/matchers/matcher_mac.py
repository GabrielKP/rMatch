import contextlib
import io
import os
from typing import Callable

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


class MatcherMac(MatcherBase):
    matcher_name = "mac"

    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        max_new_tokens: int | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
    ):
        # uses mlx to run on apple silicon
        super().__init__()

        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.max_new_tokens = max_new_tokens or 300
        self.max_retries = max_retries or 10

        console.log(f"Max new tokens: {self.max_new_tokens}")
        console.log(f"Max retries: {self.max_retries}")

        self.model_name = model_name or "unsloth/gemma-4-E4B-it-MLX-8bit"
        # unsloth/gemma-4-31B-it-MLX-8bit
        console.log(f"Initializing MLX model: {self.model_name}")

        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")

        try:
            import mlx_vlm.utils as mlx_vlm_utils  # pyright: ignore[reportMissingImports]
            from mlx_vlm import generate, load  # type: ignore
            from mlx_vlm.prompt_utils import (  # pyright: ignore[reportMissingImports]
                apply_chat_template,
            )
        except ImportError:
            raise ImportError(
                "MatcherMac requires mlx-vlm. "
                "Install rmatch with it: pip install rmatch[mac]"
            )

        self._apply_chat_template: Callable = apply_chat_template
        self._generate: Callable = generate
        self.model, self.processor = load(
            self.model_name,
            processor_config={"token": api_key} if api_key else {},
        )
        self._config = mlx_vlm_utils.load_config(self.model_name)

        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.token_per_second = list()
        self.peak_memory = 0

    def get_usage(self) -> dict | None:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "avg_token_per_second": (
                sum(self.token_per_second) / len(self.token_per_second)
            ),
            "peak_memory": self.peak_memory,
        }

    def match_return_prompts_and_responses(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        prompt: str | None = None,
        parser: parser_type | None = None,
    ) -> tuple[matchlist_type, prompts_responses_type]:
        if len(recall_segments) == 0:
            return [], []

        n_story_segments = len(story_segments)
        parser = parser or DEFAULT_PARSER
        matches: list[tuple[int, list[int]]] = []
        prompts_responses_ls: prompts_responses_type = [[] for _ in recall_segments]
        n_matches = 0
        failure_reasons: list[str] = []

        for idx, _ in enumerate(
            tqdm(
                recall_segments,
                desc="(rating recall segments)",
                position=1,
                leave=False,
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
                formatted = self._apply_chat_template(
                    self.processor,
                    self._config,
                    model_prompt,
                    num_images=0,
                )
                with contextlib.redirect_stderr(io.StringIO()):
                    response = self._generate(
                        self.model,
                        self.processor,
                        formatted,
                        image=[],
                        max_tokens=self.max_new_tokens,
                        verbose=False,
                    )
                gen_text = response.text
                # log stats
                self.total_prompt_tokens += response.prompt_tokens
                self.total_generation_tokens += response.generation_tokens
                self.token_per_second.append(response.generation_tps)
                self.peak_memory = max(self.peak_memory, response.peak_memory)

                # parse response
                parsed_response_, failure_reason = parse_response(
                    parser,
                    gen_text,
                    n_story_segments,
                    truncated=response.generation_tokens >= self.max_new_tokens,
                )
                n_matches += 1
                prompts_responses_ls[idx].append(
                    (model_prompt, gen_text, parsed_response_)
                )
                if parsed_response_ is not None:
                    parsed_response = parsed_response_
                    break
                failure_reasons.append(failure_reason)
                if attempt < self.max_retries:
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
