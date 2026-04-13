import os
from collections import defaultdict
from typing import Literal

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline

from rmatch import get_logger
from rmatch.matchers.matcher import Matcher
from rmatch.prompt import get_prompt_and_parser

log = get_logger(__name__)


FLASH_ATTN_INCOMPATIBLE: list[str] = [
    "gemma-4",
]


def _flash_attention_2_available() -> bool:
    """
    Conservative runtime check for FlashAttention2 support.

    Returns True only if:
    - CUDA is available
    - GPU is Ampere+ (SM80+)
    - `flash_attn` imports successfully
    """

    if not torch.cuda.is_available():
        return False

    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False

    if (major, minor) < (8, 0):
        return False

    try:
        import flash_attn  # noqa: F401 # type: ignore
    except Exception:
        return False

    return True


def _pick_attn_implementation(
    *, quantization: str | None, no_flash_attn: bool, model_name: str
) -> str:
    """
    Pick a Transformers `attn_implementation` string.

    Preference order:
    - bitsandbytes quantization: sdpa (most broadly compatible)
    - CUDA + FlashAttention2 available: flash_attention_2
    - CUDA/MPS: sdpa
    - CPU: eager
    """

    if quantization in {"4bit", "8bit"}:
        return "sdpa"

    if (
        not no_flash_attn
        and not any(
            [incompatible in model_name for incompatible in FLASH_ATTN_INCOMPATIBLE]
        )
        and _flash_attention_2_available()
    ):
        return "flash_attention_2"

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return "sdpa"

    return "eager"


class MatcherHuggingFace(Matcher, matcher_name="huggingface"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        quantization: Literal["4bit", "8bit"] | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        no_flash_attn: bool = False,
        max_retries: int | None = None,
        # required for initialization
        matcher_name: str | None = None,
    ):
        super().__init__()
        self.matcher_name = "huggingface"
        self.prompt = prompt

        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.verbose_errors = verbose_errors
        self.batch_size = batch_size or 64

        if max_retries is None:
            self.max_retries = 10
            log.info(f"Set max retries to {self.max_retries}")
        else:
            self.max_retries = max_retries
        log.info(f"Batch size: {self.batch_size}")
        self.max_new_tokens = max_new_tokens or 210
        log.info(f"Max new tokens: {self.max_new_tokens}")

        if model_name is None:
            self.model_name = "google/gemma-4-E4B-it"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")
            if api_key is None:
                raise ValueError("HF_TOKEN not found in .env or environment variables.")

        # handle devices and quantization
        model_kwargs: dict = {}

        if quantization == "4bit":
            log.info("Using 4-bit quantization (NF4)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = torch.bfloat16
        elif quantization == "8bit":
            log.info("Using 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32

        model_kwargs["attn_implementation"] = _pick_attn_implementation(
            quantization=quantization,
            no_flash_attn=no_flash_attn,
            model_name=self.model_name,
        )

        model_kwargs["device_map"] = "auto"
        self.pipe = pipeline(
            task="text-generation",
            model=self.model_name,
            dtype=torch_dtype,
            model_kwargs=model_kwargs,
            token=api_key,
        )

        if torch.cuda.is_available() and quantization is None:
            device_map = getattr(self.pipe.model, "hf_device_map", {})
            used_devices = set(v for v in device_map.values() if isinstance(v, int))
            if len(used_devices) > 1:
                log.info(
                    f"Model sharded across GPUs {sorted(used_devices)}; "
                    "skipping torch.compile"
                )
            else:
                self.pipe.model = torch.compile(  # type: ignore[assignment]
                    self.pipe.model, mode="reduce-overhead"
                )

        tokenizer = self.pipe.tokenizer
        if tokenizer.pad_token_id is None:  # type: ignore
            tokenizer.pad_token_id = tokenizer.eos_token_id  # type: ignore
        tokenizer.padding_side = "left"  # type: ignore

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
    ) -> list[tuple[int, list[int]]]:
        n_story_segments = len(story_segments)

        if len(recall_segments) == 0:
            return []

        prompts: list[list[dict[str, str]]] = []
        for idx in range(len(recall_segments)):
            prompt, parser = get_prompt_and_parser(
                story_segments,
                recall_segments,
                idx,
                self.window_size,
                prompt=self.prompt,
            )
            prompts.append([{"role": "user", "content": prompt}])

        results: dict[int, set[int]] = {}
        pending = list(range(len(recall_segments)))

        for attempt in range(1, self.max_retries + 1):
            if not pending:
                break

            pending_prompts = [prompts[i] for i in pending]
            responses = self.pipe(
                pending_prompts,
                batch_size=self.batch_size,
                return_full_text=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.pipe.tokenizer.eos_token_id,  # type: ignore
            )

            still_pending: list[int] = []
            for idx, response in tqdm(
                zip(pending, responses),
                total=len(pending),
                desc=f"(parsing attempt {attempt})",
                position=1,
                leave=False,
            ):
                gen_text = str(response[0]["generated_text"])  # type: ignore[index]
                prompt = prompts[idx][0]["content"]
                parsed = parser(gen_text)  # type: ignore

                if match_key is not None:
                    self._log_prompt_response(
                        match_key=match_key,
                        idx_recall=idx,
                        prompt=prompt,
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
            log.warning(
                f"All {self.max_retries} attempts failed for segment {idx}, "
                "defaulting to no matches"
            )
            results[idx] = set()

        ratings: list[tuple[int, list[int]]] = []
        for idx in range(len(recall_segments)):
            story_indices = sorted(i - 1 for i in results[idx])
            ratings.append((idx, story_indices))

        return ratings
