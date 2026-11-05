import os
from typing import Literal

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


class MatcherHuggingFace(MatcherBase):
    matcher_name = "huggingface"

    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        quantization: Literal["4bit", "8bit"] | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
    ):
        super().__init__()

        assert window_size >= 0, "window_size must be non-negative"
        self.window_size = window_size
        self.batch_size = batch_size or 64

        if max_retries is None:
            self.max_retries = 10
            console.log(f"Set max retries to {self.max_retries}")
        else:
            self.max_retries = max_retries
        console.log(f"Batch size: {self.batch_size}")
        self.max_new_tokens = max_new_tokens or 300
        console.log(f"Max new tokens: {self.max_new_tokens}")

        if model_name is None:
            self.model_name = "google/gemma-4-31B-it"
            console.log(f"Initializing model to default: {self.model_name}")
        else:
            self.model_name = model_name
            console.log(f"Initializing model: {self.model_name}")

        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")
            if api_key is None:
                raise ValueError("HF_TOKEN not found in .env or environment variables.")

        import torch
        from transformers import BitsAndBytesConfig, pipeline

        # handle devices and quantization
        model_kwargs: dict = {}

        if quantization == "4bit":
            console.log("Using 4-bit quantization (NF4)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            console.log("Using 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        if torch.backends.mps.is_available():
            torch_dtype = torch.float32  # MPS bfloat16 support is incomplete
        else:
            torch_dtype = "auto"  # trust the model config (CUDA, CPU, quantized)

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            model_kwargs["attn_implementation"] = "sdpa"
        else:
            model_kwargs["attn_implementation"] = "eager"

        if torch.cuda.is_available() or quantization in {"4bit", "8bit"}:
            model_kwargs["device_map"] = "auto"
            pipe_kwargs: dict = {}
        elif torch.backends.mps.is_available():
            pipe_kwargs = {"device": "mps"}
        else:
            pipe_kwargs = {"device": "cpu"}

        self.pipe = pipeline(
            task="text-generation",
            model=self.model_name,
            dtype=torch_dtype,
            model_kwargs=model_kwargs,
            token=api_key,
            **pipe_kwargs,
        )

        if torch.cuda.is_available() and quantization is None:
            device_map = getattr(self.pipe.model, "hf_device_map", {})
            used_devices = set(v for v in device_map.values() if isinstance(v, int))
            if len(used_devices) > 1:
                console.log(
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

    def match_return_prompts_and_responses(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        prompt: str | None = None,
        parser: parser_type | None = None,
    ) -> tuple[matchlist_type, prompts_responses_type]:
        n_story_segments = len(story_segments)
        parser = parser or DEFAULT_PARSER

        if len(recall_segments) == 0:
            return [], []

        prompts: list[list[dict[str, str]]] = []
        for idx in range(len(recall_segments)):
            model_prompt = self.build_model_prompt(
                story_segments=story_segments,
                recall_segments=recall_segments,
                target_idx=idx,
                prompt=prompt,
            )
            prompts.append([{"role": "user", "content": model_prompt}])

        results: dict[int, set[int]] = {}
        pending = list(range(len(recall_segments)))
        prompts_responses_ls: prompts_responses_type = [[] for _ in recall_segments]
        n_matches = 0
        failure_reasons: list[str] = []

        tokenizer = self.pipe.tokenizer
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
                model_prompt = prompts[idx][0]["content"]
                # the pipeline does not expose a finish reason, so re-tokenize the
                # output to check whether generation hit max_new_tokens
                n_gen_tokens = len(
                    tokenizer(gen_text, add_special_tokens=False)["input_ids"]  # type: ignore
                )
                parsed, failure_reason = parse_response(
                    parser,
                    gen_text,
                    n_story_segments,
                    truncated=n_gen_tokens >= self.max_new_tokens,
                )
                n_matches += 1

                prompts_responses_ls[idx].append((model_prompt, gen_text, parsed))

                if parsed is not None:
                    results[idx] = parsed
                else:
                    failure_reasons.append(failure_reason)
                    still_pending.append(idx)

            if still_pending:
                console.log(
                    f"Attempt {attempt}/{self.max_retries}: "
                    f"{len(still_pending)}/{len(pending)} segments need retry"
                )

            pending = still_pending

        for idx in pending:
            console.log(
                f"All {self.max_retries} attempts failed for segment {idx}, "
                "defaulting to no matches",
                style="orange1",
            )
            results[idx] = set()

        ratings: list[tuple[int, list[int]]] = []
        for idx in range(len(recall_segments)):
            story_indices = sorted(i - 1 for i in results[idx])
            ratings.append((idx, story_indices))

        print_parse_summary(n_matches, failure_reasons)

        return ratings, prompts_responses_ls
