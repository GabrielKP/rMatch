<h1 align="center">rMatch</h1>

<p align="center">Automated matching of recall segments to story segments.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>
</p>

Use **Claude** via API for best accuracy. Use **`google/gemma-4-31B-it`** to run fully locally, no API needed:

| Model                 | short text (N=21) | long text (N=19) | movie transcripts (N=138) |
| --------------------- | :---------------: | :--------------: | :-----------------------: |
| Claude Opus 4.6       |      _0.87_       |      _0.8_       |           _0.7_           |
| google/gemma-4-31B-it |      _0.84_       |       _?_        |          _0.67_           |

Pearson *r* with human ratings.

## Installation

In your project:
```sh
# To use with API
pip install rmatch

# To use with cuda:
pip install rmatch[cuda]

# On apple silicon:
pip install rmatch[apple]
```

## Usage

```python
from rmatch import Matcher

matcher = Matcher(matcher_name="anthropic", api_key="your_api_key")
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

* `matcher_name`: One of
    * `anthropic`/`openai`: run in the cloud.
    * `vLLM`: run on CUDA gpus (see more below).
    * `mlx`: run on apple silicon.
    * `huggingface`: run locally (fallback option).
* `api_key`: API provider key, or huggingface token for model download. You can set the key in the environment or in a `.env` (see below).


## Output format

A dictionary or JSON file with:

```json
{
  "matcher_name": "anthropic",
  "story_name": "story",
  "story_segmentation": "lines",
  "recall_segmentation": "lines",
  "matches": {
    "sub-001": [[0, [3, 7]], [1, [12]]],
    "sub-002": [[0, [1]], [1, [5, 6]]]
  }
}
```

Each entry in `matches` maps a subject ID to a list of `[recall_segment_id, [matched_story_segment_ids...]]` pairs.


# Documentation

1. [Full Python API](#full-python-api)
    - [Matcher](#matcher-main-entry-point)
    - [run_matching](#run_matching-file-level-convenience)
2. [API keys](#api-keys)
3. [Prompts](#prompts)


## Full Python API

### `Matcher` (main entry point)

```python
from rmatch import Matcher

matcher = Matcher(matcher_name="anthropic", model_name=None, **kwargs)
matches  = matcher.match(story_segments, recall_segments)
```

`Matcher(matcher_name, **kwargs)` is a factory — it returns the appropriate subclass based on `matcher_name`. All keyword arguments are forwarded to the subclass constructor.

**Constructor arguments:**

- **`model_name`** *(str)* — Override the default model. Applies to all matchers.
- **`window_size`** *(int)* — Context window radius around the target recall segment. Default: `5`. Applies to: `anthropic`, `openai`, `huggingface`.
- **`prompt_type`** *(str)* — Prompt type. Default: `"primary"`. Applies to: `anthropic`, `openai`, `huggingface`. See [Prompts](#prompts).
- **`dry_run`** *(bool)* — Estimate cost without calling the API. Applies to: `anthropic`, `openai`.
- **`api_key`** *(str)* — API key. Falls back to `.env`, then environment variables. Applies to: `anthropic`, `openai`, `huggingface`.
- **`device`** *(str)* — PyTorch device string. Applies to: `huggingface`.
- **`quantization`** *(str)* — `"4bit"` or `"8bit"`. Applies to: `huggingface`.
- **`batch_size`** *(int)* — Batch size for inference. Default: `4`. Applies to: `huggingface`.
- **`max_new_tokens`** *(int)* — Max generated tokens. Default: `64`. Applies to: `huggingface`.
- **`verbose_errors`** *(bool)* — Log raw output on parse failures. Applies to: `huggingface`.

**`matcher.match(story_segments, recall_segments)`**

- **`story_segments`** *(list[str])* — Ordered list of story segments (the ground-truth story elements).
- **`recall_segments`** *(list[str])* — Ordered list of a single participant's recall segments.

Returns `list[tuple[int, list[int]]]` — one entry per recall segment:

```python
[
    (0, [2, 5]),   # recall segment 0 matched story segments 2 and 5
    (1, []),       # recall segment 1 had no matches
    (2, [0]),      # recall segment 2 matched story segment 0
]
```

### `run_matching`

```python
from rmatch.match import run_matching

results = run_matching(
    story_file,            # Path — story .txt or .json
    recall_file,           # Path — recall file or directory
    matcher_name,          # str  — "anthropic", "openai", "huggingface"
    story_name=None,       # str | None — override auto-detected story name
    story_segmentation=None,   # str | None — override detected segmentation method
    recall_segmentation=None,  # str | None — override detected segmentation method
    overwrite=False,       # bool — overwrite existing output file
    **kwargs,              # forwarded to the Matcher constructor (model_name, window_size, etc.)
)
```

Loads story and recall files, runs matching for every subject, and saves a JSON results file. Returns the output dictionary.


### API keys

Keys are resolved in this order (first match wins):

1. **`api_key` argument** passed directly in Python (or `--api-key` on the CLI)
2. **`.env` file** in the current working directory
3. **Environment variables** already set in your shell

```sh
ANTHROPIC_API_KEY="your_api_key"   # for --matcher anthropic (default)
OPENAI_API_KEY="your_api_key"      # for --matcher openai
HF_TOKEN="your_hf_token"           # for --matcher huggingface
```

### Prompts

All LLM matchers share the same set of prompt templates. The default is `primary`.

| Prompt                    | Full story | Segmented story | Chain of thought | Notes                                                                        |
| ------------------------- | ---------- | --------------- | ---------------- | ---------------------------------------------------------------------------- |
| `primary`                 | yes        | yes             | yes              | Default; most complete prompt.                                               |
| `primary_no_story`        | no         | yes             | yes              | Useful for long stories where the full text would exceed the context window. |
| `primary_no_cot`          | yes        | yes             | no               | Ablation: removes chain-of-thought reasoning.                                |
| `primary_no_story_no_cot` | no         | yes             | no               | Ablation: minimal prompt with only segments and recall window.               |
| `secondary`               | yes        | yes             | yes              | Alternative prompt wording with XML-structured output.                       |



## Benchmarking

Requires [rBench](https://github.com/GabrielKP/rBench):

```sh
# outside of this dir
git clone git@github.com:GabrielKP/rBench.git
```

Add to `.env` or environment:
```sh
BENCHMARK_ROOT="path/to/rBench"
```

Run:
```sh
uv run src/rmatch/evaluate.py {alice,monthiversary,memsearch}
```
