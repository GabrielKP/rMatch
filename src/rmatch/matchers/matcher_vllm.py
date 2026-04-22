from rmatch import get_logger
from rmatch.matchers.matcher import Matcher
from rmatch.prompt import get_prompt_and_parser

log = get_logger(__name__)


# Quantization types supported as pre-quantized model formats
VLLM_QUANT_MAP: dict[str, str] = {
    "awq": "awq",
    "gptq": "gptq",
    "fp8": "fp8",
    "squeezellm": "squeezellm",
}


class MatcherVLLM(Matcher, matcher_name="vllm"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        # e.g. "awq", "gptq", "fp8", or "bitsandbytes" for on-the-fly 4/8bit
        quantization: str | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        max_retries: int | None = None,
        tensor_parallel_size: int = 1,  # number of GPUs to shard across
        gpu_memory_utilization: float = 0.90,
        matcher_name: str | None = None,
    ):
        super().__init__()
        self.matcher_name = "vllm"
        self.prompt = prompt
        self.window_size = window_size
        self.verbose_errors = verbose_errors
        self.batch_size = batch_size or 64  # vLLM handles its own internal batching,
        # but this controls your prompt chunking
        self.max_new_tokens = max_new_tokens or 300
        self.max_retries = max_retries or 10

        self.model_name = model_name or "google/gemma-4-31B-it"
        log.info(f"Initializing vLLM model: {self.model_name}")

        import os

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm is required for MatcherVLLM. "
                "Install with: pip install rMatch[cuda]"
            )

        hf_token = api_key or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token  # vLLM picks this up automatically

        self.llm = LLM(
            model=self.model_name,
            quantization=quantization,  # "awq", "gptq", "fp8", "bitsandbytes"
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,  # adjust to your context needs
            trust_remote_code=True,  # needed for Gemma-4
        )

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,  # greedy; set higher for sampling
        )

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Convert messages list to single prompt string using model's template."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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

        # Build prompts (same as before)
        raw_prompts: list[list[dict[str, str]]] = []
        parsers = []
        for idx in range(len(recall_segments)):
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
