import os
import re
from collections import defaultdict
from itertools import product
from typing import cast

from rmatch import get_logger

log = get_logger(__name__)


class CausalRaterCuda:
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        # e.g. "awq", "gptq", "fp8", or "bitsandbytes" for on-the-fly 4/8bit
        max_new_tokens: int | None = None,
        max_model_len: str | int | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
        # gpu params, see https://docs.vllm.ai/en/v0.8.0/api/offline_inference/llm.html#vllm.LLM
        tensor_parallel_size: int | None = None,  # number of GPUs to shard across
        gpu_memory_utilization: float = 0.90,
    ):
        super().__init__()
        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.verbose_errors = verbose_errors
        self.max_new_tokens = max_new_tokens or 1024
        self.max_retries = max_retries or 3
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

        # story_key -> cause idx -> effect idx -> list over attempts [
        #   (prompt, response, parsed_response)
        # ]
        self.prompt_response_log: dict[
            str, list[list[list[tuple[str, str, tuple[bool, bool]]]]]
        ] = defaultdict(list)

    def _log_prompt_response(
        self,
        story_key: str,
        idx_cause: int,
        idx_effect: int,
        prompt: str,
        response: str,
        parsed_response: tuple[bool, bool],
    ) -> None:
        # no guarantee that previous cause & effect are present, create empty slots
        # if needed
        while len(self.prompt_response_log[story_key]) <= idx_cause:
            self.prompt_response_log[story_key].append(list())
        while len(self.prompt_response_log[story_key][idx_cause]) <= idx_effect:
            self.prompt_response_log[story_key][idx_cause].append(list())

        self.prompt_response_log[story_key][idx_cause][idx_effect].append(
            (prompt, response, parsed_response)
        )

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

    def _parser(self, response: str) -> tuple[bool, bool]:
        match = re.search(r"<answer>([01])</answer>", response)
        if match:
            try:
                num = int(match.group(1))
            except ValueError:
                return False, False
            if num not in [0, 1]:
                return False, False
            return True, num == 1
        return False, False

    def _get_prompt(
        self, story: str, candidate_cause: str, candidate_effect: str
    ) -> str:
        # ruff: disable[E501]
        return f"""You are rating causal relationships between story segments in a narrative.

This is the narrative:
{story}

## Definition of causality
Segment A causes Segment B if and only if **A is necessary for B in the circumstances of the story**: if A had not occurred, given everything else in the story, B would not have occurred either.

This is a counterfactual test. Mentally remove A from the story, let events unfold from there, and ask whether B would still happen.

### What counts as causation
- **Physical causation:** A physically brings about B. ("The vase fell" -> "The vase shattered.")
- **Psychological causation:** A creates the motivation, belief, or emotion that leads to B. ("She saw the bear" -> "She ran.")
- **Enablement that is necessary in the circumstances:** A makes B possible, and in this specific story nothing else would have made B possible. ("He found the key" -> "He opened the door.")

### What does NOT count as causation
- **Mere temporal succession:** A happens before B but is not needed for B. ("He drank coffee" -> "He went to work" — going to work didn't depend on the coffee.)
- **Weak enablement:** A creates a possibility for B, but B would have occurred anyway through other story events.
- **Background conditions:** Facts true throughout the story that don't specifically produce B.
- **Reverse or simultaneous relations:** B causes A, or A and B are both effects of a common cause.

## Segments to evaluate
Segment A: {candidate_cause}
Segment B: {candidate_effect}

Does Segment A cause Segment B?

## Output
First, state your reasoning in one sentence, applying the counterfactual test: would B still occur if A were removed from the circumstances?

Then, on a new line, output your answer wrapped in tags as exactly one of:
- `<answer>1</answer>` — A causes B by the definition above.
- `<answer>0</answer>` — A does not cause B.

Do not output anything after the answer tag.

### Format example
Reasoning: [one sentence applying the counterfactual test]
<answer>1</answer>"""
        # ruff: enable[E501]

    def rate(
        self,
        story_segments: list[str],
        story_key: str | None = None,
    ) -> list[tuple[int, list[int]]]:
        """Returns a 'cause-list':
        [(cause_idx, [list of effect indices])]
        """
        n_story_segments = len(story_segments)

        story = " ".join(story_segments)
        raw_prompts: list[list[dict[str, str]]] = list()
        prompt_indices: list[tuple[int, int]] = list()
        for (idx_cause, cause), (idx_effect, effect) in product(
            enumerate(story_segments), enumerate(story_segments)
        ):
            if idx_cause == idx_effect:
                continue
            prompt_text = self._get_prompt(story, cause, effect)
            raw_prompts.append([{"role": "user", "content": prompt_text}])
            prompt_indices.append((idx_cause, idx_effect))

        # Apply chat template once upfront
        formatted_prompts = [self._apply_chat_template(p) for p in raw_prompts]

        results: dict[int, dict[int, bool]] = defaultdict(dict)
        pending = list(range(len(prompt_indices)))

        for attempt in range(1, self.max_retries + 1):
            if len(pending) == 0:
                break

            pending_prompts = [formatted_prompts[i] for i in pending]

            # vLLM batches everything in one call — no manual chunking needed
            outputs = self.llm.generate(pending_prompts, self.sampling_params)

            still_pending: list[int] = []
            for idx, output in zip(pending, outputs):
                idx_cause, idx_effect = prompt_indices[idx]

                # log
                self.total_prompt_tokens += len(output.prompt_token_ids or [])
                self.total_generation_tokens += len(output.outputs[0].token_ids)
                self.total_cached_tokens += output.num_cached_tokens or 0

                # parse
                gen_text = output.outputs[0].text
                success, is_cause = self._parser(gen_text)

                if story_key is not None:
                    self._log_prompt_response(
                        story_key=story_key,
                        idx_cause=idx_cause,
                        idx_effect=idx_effect,
                        prompt=raw_prompts[idx][0]["content"],
                        response=gen_text,
                        parsed_response=(success, is_cause),
                    )

                if success:
                    results[idx_cause][idx_effect] = is_cause
                else:
                    still_pending.append(idx)

            if still_pending:
                log.info(
                    f"Attempt {attempt}/{self.max_retries}: "
                    f"{len(still_pending)}/{len(pending)} segments need retry"
                )
            pending = still_pending

        for idx in pending:
            idx_cause, idx_effect = prompt_indices[idx]
            log.warning(
                f"All {self.max_retries} attempts failed"
                f" for pairs (cause, effect) = ({idx_cause}, {idx_effect})"
            )
            results[idx_cause][idx_effect] = False

        return [
            (
                idx_cause,
                sorted(
                    idx_effect
                    for idx_effect in results[idx_cause]
                    if results[idx_cause][idx_effect]
                ),
            )
            for idx_cause in range(n_story_segments)
        ]
