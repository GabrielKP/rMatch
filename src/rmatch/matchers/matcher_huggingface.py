# ruff: noqa: E501
import re
from typing import Literal

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline

from rmatch import ENV, get_logger
from rmatch.matchers.matcher import Matcher

log = get_logger(__name__)


class MatcherHuggingFace(Matcher):
    def __init__(
        self,
        model_name: str | None = None,
        window_size: int = 5,
        verbose_errors: bool = False,
        quantization: Literal["4bit", "8bit"] | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 64,
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

        token = ENV.get("HF_TOKEN")
        if token is None:
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
            token=token,
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

    def build_message(self, story_segments: str, window: str | None) -> str:
        message = f"""You are an expert annotator for episodic memory research. Your task is to match which numbered story elements a participant's recall segment refers to.

Matching guidelines:
- Match based on semantic content, not surface wording. Recall is often paraphrased.
- A single recall segment may reference multiple story elements. List all that apply.
- Do NOT match inferences or distortions unless a specific element is clearly referenced.
- Match only what is explicitly or clearly implicitly referenced. Do not match based on vague thematic similarity.
- The surrounding recall context aids comprehension only. Match based solely on the <target> segment.

Return matching story element numbers between <> tags, or <NONE> if none match.

<EXAMPLE>
<story_elements>
1. Sarah walked into the coffee shop.
2. She ordered a latte.
3. The barista spilled the drink.
4. Sarah laughed and said it was fine.
5. Sarah sat down at the empty table across the room.
</story_elements>

<recall_context>
Sarah went to a order a coffee.
<target>She laughed when the barista spilled the drink, because she is a kind person</target>
</recall_context>
Story segments that match the <target> segment:
<3,4>
</EXAMPLE>

Now, given the actual story and context, return the matching story elements:
<story_elements>
{story_segments}
</story_elements>

<recall_context>
{window}
</recall_context>
Story segments that match the <target> segment:
"""
        return message

    def build_recall_window(
        self, recall_segments: list[str], idx: int, window_size: int
    ) -> str:
        if window_size == 0:
            return f"<target>{recall_segments[idx]}</target>"

        start = max(0, idx - window_size)
        end = min(len(recall_segments), idx + window_size + 1)

        lines = []
        for i in range(start, end):
            clause = recall_segments[i].strip()
            if i == idx:
                lines.append(f"<target>{clause}</target>")
            else:
                lines.append(f"{clause}")

        return "\n".join(lines)

    def format_story_segments(self, story_segments: list[str]) -> str:
        return "\n".join(
            f"{i + 1}. {seg.strip()}" for i, seg in enumerate(story_segments)
        )

    def parse(self, raw: str) -> set[int] | None:
        """Return matched indices, empty set for <NONE>, or None on parse failure."""

        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            if self.verbose_errors:
                log.warning(f"failed to parse model output:\n{raw}")
            return None

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    def compute_ratings_single_sub(  # type: ignore
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
        max_retries: int = 10,
        **kwargs,
    ) -> list[tuple[int, list[int]]]:
        if output_scores:
            raise NotImplementedError(
                "HuggingFace matcher does not support output_scores = True"
            )

        story_segs_formatted = self.format_story_segments(story_segments)
        n_story_segments = len(story_segments)

        prompts: list[list[dict[str, str]]] = []
        for idx in range(len(recall_segments)):
            window = self.build_recall_window(recall_segments, idx, self.window_size)

            query = self.build_message(story_segs_formatted, window)
            prompts.append([{"role": "user", "content": query}])

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
                parsed = self.parse(response[0]["generated_text"])  # type: ignore

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
