<h1 align="center">rMatch</h1>

<p align="center">Automated matching of recall segments to story segments.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>
</p>

rMatch matches each segment of a participant's recall to the corresponding segment(s) in the original story, using a large language model. Runs **fully locally** with **`google/gemma-4-31B-it`**, and with slightly higher performance in the cloud with **Claude**:

| Model                 | short text (N=21) | long text (N=19) | movie transcripts (N=138) |
| --------------------- | :---------------: | :--------------: | :-----------------------: |
| google/gemma-4-31B-it |      _0.84_       |      _0.78_      |          _0.67_           |
| Claude Opus 4.6       |      _0.87_       |      _0.8_       |           _0.7_           |

Pearson *r* with human ratings.

## Table of contents

- [Table of contents](#table-of-contents)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Matcher object](#matcher-object)
  - [MatcherCuda](#matchercuda)
  - [MatcherAnthropic / MatcherOpenai](#matcheranthropic--matcheropenai)
  - [MatcherMac](#matchermac)
  - [MatcherHuggingface](#matcherhuggingface)
- [API keys](#api-keys)
- [Prompts](#prompts)
- [Batch matching from files directly](#batch-matching-from-files-directly)
- [Benchmark results](#benchmark-results)
- [CLI](#cli)

## Quickstart

Local:
```python
from rmatch import MatcherCuda

# MatcherCuda requires nvidia GPUs
matcher = MatcherCuda()
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

## Installation

```sh
pip install rmatch              # API matchers + huggingface fallback
pip install rmatch[cuda]        # NVIDIA GPU (vLLM)
pip install rmatch[mac]         # Apple Silicon (MLX)
```

Requires Python 3.12 or 3.13.


## Matcher object

The Matcher object is the main tool to do the matching:

| Matcher              | When to use                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `MatcherCuda`        | Run locally on NVIDIA GPUs with vLLM (`pip install rmatch[cuda]`)               |
| `MatcherAnthropic`   | Best performance; Needs an Anthropic API key (default model: `claude-opus-4-6`) |
| `MatcherOpenai`      | Cloud alternative; Needs an OpenAI API key (default: `gpt-4.1`)                 |
| `MatcherMac`         | Run locally on Apple Silicon (`pip install rmatch[mac]`)                        |
| `MatcherHuggingface` | Local fallback; works on any local archtiecture, but is not resource efficient  |

Each matcher implements a function `matcher.match(story_segments, recall_segments)` that expects:

- `story_segments` — ordered list of story segments.
- `recall_segments` — ordered list of one participant's recall segment strings.

The function returns one entry per recall segment in the format of `(recall_index, [story_indices])` tuples (0-based):

```python
[
    (0, [2, 5]),   # recall segment 0 -> matches story segments 2 and 5
    (1, []),       # recall segment 1 -> no match
    (2, [0]),      # recall segment 2 -> matches story segment 0
]
```

### MatcherCuda

The main matcher to run locally - if you have nvidia gpus available. 94GB of VRAM (across multiple gpus) was enough to run `gemma-4-31B-it` without quantization.
See here for [quantized versions](https://huggingface.co/collections/unsloth/gemma-4).

```python
from rmatch import MatcherCuda

# install rmatch with `pip install rmatch[cuda]`
matcher = MatcherCuda()
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

| Argument                 | Default            | Notes                                                                                                             |
| ------------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `model_name`             | `"gemma-4-31B-it"` | Override the default model                                                                                        |
| `prompt`                 | `"primary"`        | See [Prompts](#prompts)                                                                                           |
| `max_retries`            | `10`               | Retries when the model output cannot be parsed                                                                    |
| `api_key`                | from `.env` / env  | See [API keys](#api-keys)                                                                                         |
| `window_size`            | `5`                | Recall context window                                                                                             |
| `max_new_tokens`         | `1024`             | Max tokens generated per segment                                                                                  |
| `max_model_len`          | `90000`            | Max sequence length (prompt + generation); lower to save GPU memory                                               |
| `tensor_parallel_size`   | auto               | Number of GPUs to use. Auto will use all of them.                                                                 |
| `gpu_memory_utilization` | `0.90`             | Fraction of GPU memory to use, see [vLLM](https://docs.vllm.ai/en/v0.8.0/api/offline_inference/llm.html#vllm.LLM) |
| `verbose_errors`         | `False`            | Log raw output on parse failures                                                                                  |



### MatcherAnthropic / MatcherOpenai

Use a cloud provider to do the matching. It's pretty inexpensive, and `--dry_run` will give you an approximate estimate of the cost. Usually a single recall doesn't cost more than 0.5$, even on frontier models.

```python
from rmatch import MatcherAnthropic

# install rmatch with `pip install rmatch`
matcher = MatcherAnthropic(api_key="your_api_key")
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

You can also put your API key in a `.env` file instead of passing it directly (see [API keys](#api-keys)).

| Argument      | Default                           | Notes                                          |
| ------------- | --------------------------------- | ---------------------------------------------- |
| `model_name`  | `"claude-opus-4-6"` / `"gpt-4.1"` | Override the default model                     |
| `prompt`      | `"primary"`                       | See [Prompts](#prompts)                        |
| `max_retries` | `10`                              | Retries when the model output cannot be parsed |
| `api_key`     | from `.env` / env                 | See [API keys](#api-keys)                      |
| `window_size` | `5`                               | Recall context window                          |
| `dry_run`     | `False`                           | Estimate API cost without calling the API      |


### MatcherMac

You can try running the matching on your mac with apple silicon! You'll probably need a lot of unified memory, or use a quantized model. The standard `unsloth/gemma-4-E4B-it-MLX-8bit` model should run on a mac with 24GB of unified memory.

```python
from rmatch import MatcherMac

# install rmatch with `pip install rmatch[mac]`
matcher = MatcherMac()
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

| Argument         | Default                             | Notes                                          |
| ---------------- | ----------------------------------- | ---------------------------------------------- |
| `model_name`     | `"unsloth/gemma-4-E4B-it-MLX-8bit"` | Override the default model                     |
| `prompt`         | `"primary"`                         | See [Prompts](#prompts)                        |
| `max_retries`    | `10`                                | Retries when the model output cannot be parsed |
| `api_key`        | from `.env` / env                   | See [API keys](#api-keys)                      |
| `window_size`    | `5`                                 | Recall context window                          |
| `max_new_tokens` | `300`                               | Max tokens generated per segment               |
| `verbose_errors` | `False`                             | Log raw output on parse failures               |



### MatcherHuggingface

The fallback matcher that should work on any platform. Can run the same models as the Cuda/Mac matcher, and will achieve the same matching performance, but will require considerably more computing resources and be a lot slower.

```python
from rmatch import MatcherHuggingface

# install rmatch with `pip install rmatch`
matcher = MatcherHuggingface()
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

| Argument         | Default           | Notes                                          |
| ---------------- | ----------------- | ---------------------------------------------- |
| `model_name`     | matcher-specific  | Override the default model                     |
| `prompt`         | `"primary"`       | See [Prompts](#prompts)                        |
| `max_retries`    | `10`              | Retries when the model output cannot be parsed |
| `api_key`        | from `.env` / env | See [API keys](#api-keys)                      |
| `window_size`    | `5`               | Recall context window                          |
| `quantization`   | none              | `"4bit"` or `"8bit"` to reduce memory          |
| `batch_size`     | `64`              | Inference batch size                           |
| `max_new_tokens` | `300`             | Max tokens generated per segment               |
| `verbose_errors` | `False`           | Log raw output on parse failures               |



## API keys

For MatcheAnthropic/MatcherOpenai you need the API key to access the models.
For all local matchers, you made need the API key in form of the `HF_token` - to allow you access to download a large language model from the hub.

rMatch will look for the API key in this order (first match wins):

1. `api_key` argument in Python
2. `.env` file in the working directory
3. Environment variables in your shell

```sh
ANTHROPIC_API_KEY="your_api_key"   # anthropic
OPENAI_API_KEY="your_api_key"      # openai
HF_TOKEN="your_hf_token"           # huggingface, mac, cuda (model download)
```

## Prompts

All matchers share the same prompt templates, but you can change it. Pass `prompt="primary_no_story"`. Default is `primary`.

| Prompt                    | Full story | Segmented story | Chain of thought | Notes                                           |
| ------------------------- | ---------- | --------------- | ---------------- | ----------------------------------------------- |
| `primary`                 | yes        | yes             | yes              | Default; most complete prompt                   |
| `primary_no_story`        | no         | yes             | yes              | For long stories that exceed the context window |
| `primary_no_cot`          | yes        | yes             | no               | Ablation: no chain-of-thought                   |
| `primary_no_story_no_cot` | no         | yes             | no               | Minimal prompt                                  |
| `secondary`               | yes        | yes             | yes              | Alternative wording with XML output             |


## Batch matching from files directly

If your story and recalls are saved as `.txt` or `.json` files, you can match them as a batch with `match()`.

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


## Benchmark results

| Matcher                           | Testset       |  F1   | Precision | Recall | Pearson R |
| --------------------------------- | ------------- | :---: | :-------: | :----: | :-------: |
| MatcherCuda:google/gemma-4-31B-it | alice         |  .84  |    .8     |  .89   |    .83    |
| MatcherCuda:google/gemma-4-31B-it | monthiversary |  .79  |    .72    |  .86   |    .78    |
| MatcherCuda:google/gemma-4-31B-it | memsearch     |  .67  |    .56    |  .84   |    .64    |

* `alice` (medium text) are 21 recalls with a length of ~200 words, and a story length of ~700 words.
* `monthiversary` (long text) are 19 recalls with a length of ~1000 words, and a story length of ~4700.
* `memsearch` (short movie transcripts) are 138 recalls with a length of ~140 words, and a story transcript length of ~240 words.


Replicate results by downloading [rBench](https://github.com/GabrielKP/rBench):

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
