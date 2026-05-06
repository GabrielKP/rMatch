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


class Segmenter:
    def __init__(
        self,
        model_name: str | None = None,
        quantization: Literal["4bit", "8bit"] | None = None,
        api_key: str | None = None,
    ):
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
            no_flash_attn=False,
            model_name=self.model_name,
        )

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

    def segment(self, transcript_dir: str):
        with open(transcript_dir, "r") as f:
            transcript = f.read()
        prompt = f"""I have a transcript of someone describing movies they watched. Follow these steps:

1. Split the text into sentences
2. For sentences longer than 20 words, split them at natural breaks (punctuation, topic shifts, noun-verb units)
3. Verify the segmented text is word-for-word identical to the original

Output ONLY a JSON array where each element is a segment. No preamble, no explanation, no markdown code blocks. Format:
["segment 1 text here", "segment 2 text here", "segment 3 text here"]

Example input: "I watched a really interesting documentary about ocean life. It was fascinating and educational and I learned so much about marine biology and ecosystems."

Example output:
["I watched a really interesting documentary about ocean life.", "It was fascinating and educational", "and I learned so much about marine biology and ecosystems."]

Here is the transcript to segment:
{transcript}
"""
