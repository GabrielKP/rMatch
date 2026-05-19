<h1 align="center">rMatch</h1>

<p align="center">Automated matching of recall segments to story segments.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>
</p>

rMatch links each segment of a participant's spoken recall to the corresponding segment(s) in the original story, using a large language model. Use **Claude** via API for best accuracy, or **`google/gemma-4-31B-it`** to run fully locally:

| Model                 | short text (N=21) | long text (N=19) | movie transcripts (N=138) |
| --------------------- | :---------------: | :--------------: | :-----------------------: |
| Claude Opus 4.6       |      _0.87_       |      _0.8_       |           _0.7_           |
| google/gemma-4-31B-it |      _0.84_       |       _?_        |          _0.67_           |

Pearson *r* with human ratings.

## Table of contents

- [Table of contents](#table-of-contents)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Choosing a matcher](#choosing-a-matcher)
- [Matcher options](#matcher-options)
- [Output format](#output-format)
- [API keys](#api-keys)
- [Prompts](#prompts)
- [Batch matching from files directly](#batch-matching-from-files-directly)
- [Benchmarks](#benchmarks)
- [CLI](#cli)

## Quickstart

```python
from rmatch import MatcherAnthropic

matcher = MatcherAnthropic(api_key="your_api_key")
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

You can also put your API key in a `.env` file instead of passing it directly (see [API keys](#api-keys)).

## Installation

```sh
pip install rmatch              # API matchers + huggingface fallback
pip install rmatch[cuda]        # NVIDIA GPU (vLLM)
pip install rmatch[mac]         # Apple Silicon (MLX)
```

Requires Python 3.12 or 3.13.

## Choosing a matcher

| Matcher       | When to use                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `anthropic`   | Best accuracy; needs an Anthropic API key (default model: `claude-opus-4-6`) |
| `openai`      | Cloud alternative; needs an OpenAI API key (default: `gpt-4.1`)              |
| `cuda`        | Run locally on NVIDIA GPUs with vLLM (`pip install rmatch[cuda]`)            |
| `mac`         | Run locally on Apple Silicon (`pip install rmatch[mac]`)                     |
| `huggingface` | Portable local fallback; works on CPU, CUDA, or MPS                          |

```python
matcher = get_matcher("anthropic")   # or "openai", "mac", "cuda", "huggingface"
```

## Matcher options

All keyword arguments are passed to `get_matcher(name, **kwargs)`.

**Shared across matchers**

| Argument      | Default           | Notes                                          |
| ------------- | ----------------- | ---------------------------------------------- |
| `model_name`  | matcher-specific  | Override the default model                     |
| `prompt`      | `"primary"`       | See [Prompts](#prompts)                        |
| `max_retries` | `10`              | Retries when the model output cannot be parsed |
| `api_key`     | from `.env` / env | See [API keys](#api-keys)                      |
| `window_size` | `5`               | Recall context window                          |

**`anthropic`, `openai`**

| Argument  | Default | Notes                                     |
| --------- | ------- | ----------------------------------------- |
| `dry_run` | `False` | Estimate API cost without calling the API |

**`cuda`**

| Argument                 | Default | Notes                                                               |
| ------------------------ | ------- | ------------------------------------------------------------------- |
| `max_new_tokens`         | `300`   | Max tokens generated per segment                                    |
| `max_model_len`          | `90000` | Max sequence length (prompt + generation); lower to save GPU memory |
| `tensor_parallel_size`   | auto    | Number of GPUs to use                                               |
| `gpu_memory_utilization` | `0.90`  | Fraction of GPU memory to use                                       |
| `verbose_errors`         | `False` | Log raw output on parse failures                                    |


**`mac`**

| Argument         | Default | Notes                            |
| ---------------- | ------- | -------------------------------- |
| `window_size`    | `5`     | Recall context window            |
| `max_new_tokens` | `300`   | Max tokens generated per segment |
| `verbose_errors` | `False` | Log raw output on parse failures |


**`huggingface`**

| Argument         | Default | Notes                                 |
| ---------------- | ------- | ------------------------------------- |
| `window_size`    | `5`     | Recall context window                 |
| `quantization`   | none    | `"4bit"` or `"8bit"` to reduce memory |
| `batch_size`     | `64`    | Inference batch size                  |
| `max_new_tokens` | `300`   | Max tokens generated per segment      |
| `verbose_errors` | `False` | Log raw output on parse failures      |


**`matcher.match(story_segments, recall_segments)`**

- `story_segments` — ordered list of story segment strings (ground truth).
- `recall_segments` — ordered list of one participant's recall segment strings.

Returns one entry per recall segment in the format of `(recall_index, [story_indices])` tuples (0-based):

```python
[
    (0, [2, 5]),   # recall segment 0 -> story segments 2 and 5
    (1, []),       # recall segment 1 -> no match
    (2, [0]),      # recall segment 2 -> story segment 0
]
```

## Output format

From `matcher.match`, you get a list of `(recall_index, [story_indices])` tuples (0-based).


## API keys

Resolved in this order (first match wins):

1. `api_key` argument in Python
2. `.env` file in the working directory
3. Environment variables in your shell

```sh
ANTHROPIC_API_KEY="your_api_key"   # anthropic
OPENAI_API_KEY="your_api_key"      # openai
HF_TOKEN="your_hf_token"           # huggingface, mac, cuda (model download)
```

## Prompts

All matchers share the same prompt templates. Pass `prompt="primary_no_story"` (etc.) to `get_matcher`. Default is `primary`.

| Prompt                    | Full story | Segmented story | Chain of thought | Notes                                           |
| ------------------------- | ---------- | --------------- | ---------------- | ----------------------------------------------- |
| `primary`                 | yes        | yes             | yes              | Default; most complete prompt                   |
| `primary_no_story`        | no         | yes             | yes              | For long stories that exceed the context window |
| `primary_no_cot`          | yes        | yes             | no               | Ablation: no chain-of-thought                   |
| `primary_no_story_no_cot` | no         | yes             | no               | Minimal prompt                                  |
| `secondary`               | yes        | yes             | yes              | Alternative wording with XML output             |


## Batch matching from files directly

If your story and recalls are already saved as `.txt` or `.json` files:

```python
from pathlib import Path
from rmatch import match, MatcherCuda


matcher_gemma = MatcherCuda()

results = match(
    matcher=matcher_gemma,
    story_file=Path("story.txt"),
    recall_file=Path("recalls/"),   # file or directory of subject files
)
```

This loads all subjects, runs matching, and writes a JSON results file next to your recall data with the following format:

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

Each subject maps to a list of `[recall_segment_id, [matched_story_segment_ids...]]` pairs.


## Benchmarks

Requires [rBench](https://github.com/GabrielKP/rBench):

```sh
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

## CLI

If you prefer the command line over a Python script:

```sh
rmatch story.txt recalls/ --matcher anthropic
```

See `rmatch --help` for all options (model, prompt, window size, etc.).
