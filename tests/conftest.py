# Heavy-module stubs must be installed BEFORE any rmatch imports.
# matcher_huggingface.py, and evaluate.py import torch,
# transformers, sentence_transformers, bitsandbytes, and other GPU packages at
# module level. Replacing them with MagicMocks lets the test suite run without
# a GPU or those packages installed.
import sys
from unittest.mock import MagicMock

_HEAVY = [
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
    "torchvision",
    "transformers",
    "transformers.utils",
    "sentence_transformers",
    "bitsandbytes",
    "flash_attn",
    "einops",
    "accelerate",
]
for _mod in _HEAVY:
    sys.modules[_mod] = MagicMock()

# torch.cuda.is_available() and torch.backends.mps.is_available() are called
# during MatcherHuggingFace.__init__. MagicMock() is truthy by default, which
# would activate torch.compile – set them explicitly to False.
sys.modules["torch"].cuda.is_available.return_value = False
sys.modules["torch"].cuda.device_count.return_value = 0
sys.modules["torch"].backends.mps.is_available.return_value = False
# dtype sentinels used as arguments to pipeline (which is itself mocked)
sys.modules["torch"].bfloat16 = "bfloat16"  # type: ignore
sys.modules["torch"].float32 = "float32"  # type: ignore

# ── Now safe to import standard library and project code ─────────────────────
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

# ── Shared test data ──────────────────────────────────────────────────────────

STORY_SEGMENTS = [
    "Alice met the Queen of Hearts.",
    "They sat down for tea.",
    "The Queen smiled and left.",
]

RECALL_SEGMENTS = [
    "Alice and the Queen drank tea.",
    "The Queen eventually left.",
]

STORY_TXT_CONTENT = "\n".join(STORY_SEGMENTS) + "\n"

STORY_JSON_CONTENT = json.dumps(
    {"segments": STORY_SEGMENTS, "segmentation_method": "scenes"}
)

RECALL_TXT_CONTENT = "\n".join(RECALL_SEGMENTS) + "\n"

RECALL_JSON_CONTENT = json.dumps(
    {"recalls": {"sub01": RECALL_SEGMENTS}, "segmentation_method": "sub_sentences"}
)


# ── File fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def story_txt(tmp_path) -> Path:
    p = tmp_path / "story.txt"
    p.write_text(STORY_TXT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def story_json(tmp_path) -> Path:
    p = tmp_path / "story.json"
    p.write_text(STORY_JSON_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def recall_txt(tmp_path) -> Path:
    p = tmp_path / "recall.txt"
    p.write_text(RECALL_TXT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def recall_json(tmp_path) -> Path:
    p = tmp_path / "recall.json"
    p.write_text(RECALL_JSON_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def recall_dir_txt(tmp_path) -> Path:
    d = tmp_path / "recalls"
    d.mkdir()
    (d / "sub01.txt").write_text("Alice and the Queen drank tea.\n", encoding="utf-8")
    (d / "sub02.txt").write_text("The Queen smiled.\n", encoding="utf-8")
    return d


@pytest.fixture
def recall_dir_json(tmp_path) -> Path:
    d = tmp_path / "recalls_json"
    d.mkdir()
    (d / "part1.json").write_text(
        json.dumps(
            {
                "recalls": {"sub01": RECALL_SEGMENTS},
                "segmentation_method": "sub_sentences",
            }
        ),
        encoding="utf-8",
    )
    (d / "part2.json").write_text(
        json.dumps(
            {
                "recalls": {"sub02": ["The Queen smiled.", "She waved goodbye."]},
                "segmentation_method": "sub_sentences",
            }
        ),
        encoding="utf-8",
    )
    return d


# ── Mock response helpers ─────────────────────────────────────────────────────


def make_anthropic_response(text: str, in_tokens: int = 50, out_tokens: int = 10):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage.input_tokens = in_tokens
    resp.usage.output_tokens = out_tokens
    return resp


def make_openai_response(text: str, in_tokens: int = 50, out_tokens: int = 10):
    resp = MagicMock()
    resp.output_text = text
    resp.usage.input_tokens = in_tokens
    resp.usage.output_tokens = out_tokens
    return resp


# ── Matcher fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_anthropic_client():
    return MagicMock()


@pytest.fixture
def anthropic_matcher(mock_anthropic_client):
    with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
        from rmatch.matchers.matcher_anthropic import MatcherAnthropic

        m = MatcherAnthropic(api_key="test-anthropic-key", max_retries=3)
    return m


@pytest.fixture
def mock_openai_client():
    return MagicMock()


@pytest.fixture
def openai_matcher(mock_openai_client):
    mock_openai_mod = MagicMock()
    mock_openai_mod.OpenAI.return_value = mock_openai_client
    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        from rmatch.matchers.matcher_openai import MatcherOpenAI

        m = MatcherOpenAI(api_key="test-openai-key", max_retries=3)
    return m


@pytest.fixture
def mock_hf_pipe():
    pipe = MagicMock()
    pipe.tokenizer.pad_token_id = None
    pipe.tokenizer.eos_token_id = 2
    pipe.tokenizer.padding_side = "right"
    return pipe


@pytest.fixture
def huggingface_matcher(mock_hf_pipe):
    sys.modules["transformers"].pipeline.return_value = mock_hf_pipe
    from rmatch.matchers.matcher_huggingface import MatcherHuggingFace

    m = MatcherHuggingFace(api_key="test-hf-token", max_retries=3)
    return m


# ── vLLM helpers & fixtures ───────────────────────────────────────────────────


def make_vllm_output(text: str):
    """Create a mock vLLM RequestOutput with a single completion."""
    output = MagicMock()
    output.outputs[0].text = text
    return output


@pytest.fixture
def mock_vllm_llm():
    llm = MagicMock()
    llm.get_tokenizer.return_value.pad_token_id = None
    llm.get_tokenizer.return_value.eos_token_id = 2
    return llm


@pytest.fixture
def vllm_matcher(mock_vllm_llm):
    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_vllm_llm
    with patch.dict("sys.modules", {"vllm": mock_vllm}):
        from rmatch.matchers.matcher_vllm import MatcherVLLM

        m = MatcherVLLM(model_name="test-model", max_retries=3)
    return m


# ── MLX helpers & fixtures ────────────────────────────────────────────────────


def make_mlx_response(
    text: str,
    prompt_tokens: int = 50,
    generation_tokens: int = 10,
    generation_tps: float = 20.0,
    peak_memory: float = 1.0,
):
    """Create a mock mlx-vlm GenerationResult."""
    resp = MagicMock()
    resp.text = text
    resp.prompt_tokens = prompt_tokens
    resp.generation_tokens = generation_tokens
    resp.generation_tps = generation_tps
    resp.peak_memory = peak_memory
    return resp


@pytest.fixture
def mock_mlx_generate():
    return MagicMock()


@pytest.fixture
def mlx_matcher(mock_mlx_generate):
    mock_mlx = MagicMock()
    mock_mlx.generate = mock_mlx_generate
    mock_mlx.load.return_value = (MagicMock(), MagicMock())
    with patch.dict(
        "sys.modules",
        {
            "mlx_vlm": mock_mlx,
            "mlx_vlm.utils": MagicMock(),
            "mlx_vlm.prompt_utils": MagicMock(),
        },
    ):
        from rmatch.matchers.matcher_mlx import MatcherMLX

        m = MatcherMLX(model_name="test-model", max_retries=3)
    return m
