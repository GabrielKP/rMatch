from typing import Literal

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline

from rmatch import ENV, get_logger
from rmatch.matchers.matcher import Matcher
from rmatch.prompt import prompt_default

log = get_logger(__name__)


class MatcherHuggingFace(Matcher, matcher_name="huggingface"):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        quantization: Literal["4bit", "8bit"] | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 64,
        api_key: str | None = None,
        # required for initialization
        matcher_name: str | None = None,
    ):
        self.matcher_name = "huggingface"

        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.verbose_errors = verbose_errors

        if model_name is None:
            self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
            log.info(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            log.info(f"Initializing model: {self.model_name}")

        if api_key is None:
            api_key = ENV.get("HF_TOKEN")
            if api_key is None:
                raise ValueError("HF_TOKEN not found in .env file")

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
            attn_impl = "sdpa"
        elif quantization == "8bit":
            log.info("Using 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            torch_dtype = torch.bfloat16
            attn_impl = "sdpa"
        elif torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            attn_impl = "sdpa"
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float32
            attn_impl = "sdpa"
        else:
            torch_dtype = torch.float32
            attn_impl = "eager"

        model_kwargs["attn_implementation"] = attn_impl

        self.pipe = pipeline(
            task="text-generation",
            model=self.model_name,
            device_map="auto",
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
                self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        tokenizer = self.pipe.tokenizer
        if tokenizer.pad_token_id is None:  # type: ignore
            tokenizer.pad_token_id = tokenizer.eos_token_id  # type: ignore
        tokenizer.padding_side = "left"  # type: ignore

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        max_retries: int = 10,
    ) -> list[tuple[int, list[int]]]:
        n_story_segments = len(story_segments)

        if len(recall_segments) == 0:
            return []

        prompts: list[list[dict[str, str]]] = []
        for idx in range(len(recall_segments)):
            prompt, parser = prompt_default(
                story_segments, recall_segments, idx, self.window_size
            )
            prompts.append([{"role": "user", "content": prompt}])

        results: dict[int, set[int]] = {}
        pending = list(range(len(recall_segments)))

        for attempt in range(1, max_retries + 1):
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
                parsed = parser(response[0]["generated_text"])  # type: ignore

                if parsed is not None and all(
                    1 <= i <= n_story_segments for i in parsed
                ):
                    results[idx] = parsed
                else:
                    still_pending.append(idx)

            if still_pending:
                log.info(
                    f"Attempt {attempt}/{max_retries}: "
                    f"{len(still_pending)}/{len(pending)} segments need retry"
                )

            pending = still_pending

        for idx in pending:
            log.warning(
                f"All {max_retries} attempts failed for segment {idx}, "
                "defaulting to no matches"
            )
            results[idx] = set()

        ratings: list[tuple[int, list[int]]] = []
        for idx in range(len(recall_segments)):
            story_indices = sorted(i - 1 for i in results[idx])
            ratings.append((idx, story_indices))

        return ratings
