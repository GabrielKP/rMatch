import contextlib
import io
import os
from typing import Callable

from tqdm import tqdm

from rmatch import get_logger, matchlist_type
from rmatch.matchers.matcher_base import MatcherBase
from rmatch.prompt import get_prompt_and_parser

log = get_logger(__name__)


class MatcherMac(MatcherBase, matcher_name="mac"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        max_new_tokens: int | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        max_retries: int | None = None,
    ):
        # uses mlx to run on apple silicon
        super().__init__()
        self.matcher_name = "mac"
        self.prompt = prompt

        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.verbose_errors = verbose_errors
        self.max_new_tokens = max_new_tokens or 300
        self.max_retries = max_retries or 10

        log.info(f"Max new tokens: {self.max_new_tokens}")
        log.info(f"Max retries: {self.max_retries}")

        self.model_name = model_name or "unsloth/gemma-4-E4B-it-MLX-8bit"
        # unsloth/gemma-4-31B-it-MLX-8bit
        log.info(f"Initializing MLX model: {self.model_name}")

        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")

        try:
            import mlx_vlm.utils as mlx_vlm_utils
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
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

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
    ) -> matchlist_type:
        if len(recall_segments) == 0:
            return []

        n_story_segments = len(story_segments)
        matches: list[tuple[int, list[int]]] = []

        for idx, _ in enumerate(
            tqdm(
                recall_segments,
                desc="(rating recall segments)",
                position=1,
                leave=False,
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
                formatted = self._apply_chat_template(
                    self.processor,
                    self._config,
                    prompt,
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
                parsed_response_ = parser(gen_text)
                if match_key is not None:
                    self._log_prompt_response(
                        match_key=match_key,
                        idx_recall=idx,
                        prompt=prompt,
                        response=gen_text,
                        parsed_response=parsed_response_,
                    )
                if parsed_response_ is not None and all(
                    1 <= i <= n_story_segments for i in parsed_response_
                ):
                    parsed_response = parsed_response_
                    break
                log.warning(
                    f"Attempt {attempt}/{self.max_retries}:"
                    f" segment needs retry for segment {idx}"
                )
            else:
                log.warning(f"All attempts failed for segment {idx}")

            story_indices = sorted(i - 1 for i in parsed_response)
            matches.append((idx, story_indices))

        return matches
