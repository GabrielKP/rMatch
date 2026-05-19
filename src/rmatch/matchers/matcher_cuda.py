import os
from typing import cast

from rmatch import get_logger
from rmatch.matchers.matcher_base import MatcherBase
from rmatch.prompt import get_prompt_and_parser

log = get_logger(__name__)


class MatcherCuda(MatcherBase, matcher_name="cuda"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        # e.g. "awq", "gptq", "fp8", or "bitsandbytes" for on-the-fly 4/8bit
        max_new_tokens: int | None = None,
        max_model_len: str | int | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        max_retries: int | None = None,
        # gpu params, see https://docs.vllm.ai/en/v0.8.0/api/offline_inference/llm.html#vllm.LLM
        tensor_parallel_size: int | None = None,  # number of GPUs to shard across
        gpu_memory_utilization: float = 0.90,
    ):
        super().__init__()
        self.matcher_name = "cuda"
        self.prompt = prompt
        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.verbose_errors = verbose_errors
        self.max_new_tokens = max_new_tokens or 1024
        self.max_retries = max_retries or 10
        self.model_name = model_name or "google/gemma-4-31B-it"

        log.info(f"Initializing vLLM model with CUDA: {self.model_name}")
        log.info(f"Recall context window size: {self.window_size}")
        log.info(f"Max new tokens: {self.max_new_tokens}")
        log.info(f"Max retries: {self.max_retries}")
        log.info(f"Verbose errors: {self.verbose_errors}")

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm is required for MatcherCuda. "
                "Install rMatch with it: pip install rMatch[cuda]"
            )

        if tensor_parallel_size is None:
            import torch

            tensor_parallel_size = max(1, torch.cuda.device_count())
            log.info(
                f"GPUs (tensor_parallel_size): {tensor_parallel_size} (Auto-detected)"
            )
        else:
            log.info(f"GPUs (tensor_parallel_size): {tensor_parallel_size}")

        log.info(f"GPU memory utilization: {gpu_memory_utilization}")
        if max_model_len == "auto":
            self.max_model_len = None
        else:
            self.max_model_len = max_model_len or 90000
        log.info(f"Max model length: {self.max_model_len}")

        hf_token = api_key or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token  # vLLM picks this up automatically

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,  # needed for Gemma-4
        )

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # type: ignore

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )

        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_cached_tokens = 0

    def get_usage(self) -> dict | None:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "total_cached_tokens": self.total_cached_tokens,
        }

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                messages,  # type: ignore[arg-type]
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
    ) -> list[tuple[int, list[int]]]:
        n_story_segments = len(story_segments)

        if not recall_segments:
            return []

        raw_prompts: list[list[dict[str, str]]] = []
        parsers = []
        for idx, _ in enumerate(recall_segments):
            prompt_text, parser = get_prompt_and_parser(
                story_segments,
                recall_segments,
                idx,
                self.window_size,
                prompt=self.prompt,
            )
            raw_prompts.append([{"role": "user", "content": prompt_text}])
            parsers.append(parser)

        # Apply chat template once upfront
        formatted_prompts = [self._apply_chat_template(p) for p in raw_prompts]

        results: dict[int, set[int]] = {}
        pending = list(range(len(recall_segments)))

        for attempt in range(1, self.max_retries + 1):
            if not pending:
                break

            pending_prompts = [formatted_prompts[i] for i in pending]

            # vLLM batches everything in one call — no manual chunking needed
            outputs = self.llm.generate(pending_prompts, self.sampling_params)

            still_pending: list[int] = []
            for idx, output in zip(pending, outputs):
                gen_text = output.outputs[0].text
                parsed = parsers[idx](gen_text)
                self.total_prompt_tokens += len(output.prompt_token_ids or [])
                self.total_generation_tokens += len(output.outputs[0].token_ids)
                self.total_cached_tokens += output.num_cached_tokens or 0

                if match_key is not None:
                    self._log_prompt_response(
                        match_key=match_key,
                        idx_recall=idx,
                        prompt=raw_prompts[idx][0]["content"],
                        response=gen_text,
                        parsed_response=parsed,
                    )

                if parsed is not None and all(
                    1 <= i <= n_story_segments for i in parsed
                ):
                    results[idx] = parsed
                else:
                    still_pending.append(idx)

            if still_pending:
                log.info(
                    f"Attempt {attempt}/{self.max_retries}: "
                    f"{len(still_pending)}/{len(pending)} segments need retry"
                )
            pending = still_pending

        for idx in pending:
            log.warning(f"All {self.max_retries} attempts failed for segment {idx}")
            results[idx] = set()

        return [
            (idx, sorted(i - 1 for i in results[idx]))
            for idx in range(len(recall_segments))
        ]
