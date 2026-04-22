import os

from tqdm import tqdm

from rmatch import get_logger, matchlist_type
from rmatch.matchers.matcher import Matcher
from rmatch.prompt import get_prompt_and_parser

log = get_logger(__name__)


class MatcherMLX(Matcher, matcher_name="mlx"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        max_new_tokens: int | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        max_retries: int | None = None,
        matcher_name: str | None = None,
    ):
        super().__init__()
        self.matcher_name = "mlx"
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
            import mlx_vlm
            import mlx_vlm.prompt_utils as mlx_vlm_prompt_utils
            import mlx_vlm.utils as mlx_vlm_utils
        except ImportError:
            raise ImportError(
                "mlx-vlm is required for MatcherMLX. "
                "Install it with: pip install mlx-vlm"
            )

        self._mlx_vlm = mlx_vlm
        self._mlx_vlm_prompt_utils = mlx_vlm_prompt_utils
        self.model, self.processor = mlx_vlm.load(
            self.model_name,
            processor_config={"token": api_key} if api_key else {},
        )
        self._config = mlx_vlm_utils.load_config(self.model_name)

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
    ) -> matchlist_type:
        if len(recall_segments) == 0:
            return []

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
                formatted = self._mlx_vlm_prompt_utils.apply_chat_template(
                    self.processor,
                    self._config,
                    prompt,
                    num_images=0,
                )
                gen_text = self._mlx_vlm.generate(
                    self.model,
                    self.processor,
                    formatted,
                    image=[],
                    max_tokens=self.max_new_tokens,
                    verbose=False,
                )
                parsed_response_ = parser(gen_text)
                if match_key is not None:
                    self._log_prompt_response(
                        match_key=match_key,
                        idx_recall=idx,
                        prompt=prompt,
                        response=gen_text,
                        parsed_response=parsed_response_,
                    )
                if parsed_response_ is not None:
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
