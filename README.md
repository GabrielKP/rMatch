<h1 align="center">rMatch</h1>

<p align="center">Automatic recall & story matching tool.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>


## Quick start

### Command line

```sh
pip install rmatch

# single recall file
rmatch story.txt recall.txt --matcher anthropic

# directory of recall files (one per subject)
rmatch story.txt recalls/ --matcher anthropic

# estimate API cost without sending requests
rmatch story.txt recalls/ --matcher openai --dry-run
```

### Python API

```python
from rmatch import Matcher

matcher = Matcher(matcher_name="anthropic", api_key="your_api_key")
matches = matcher.match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

Or use `run_matching` to load files, run matching, and save results in one call:

```python
from rmatch.match import run_matching

results = run_matching(
    story_file="story.txt",
    recall_file="recalls/",
    matcher_name="anthropic",
    api_key="your_api_key",
)
```

## Setup API keys

API keys are resolved in this order (first match wins):

1. **`api_key` argument** passed directly in Python
2. **`.env` file** in the current working directory
3. **Environment variables** already set in your shell

Set them as environment variables:

```sh
export ANTHROPIC_API_KEY="your_api_key"   # for --matcher anthropic (default)
export OPENAI_API_KEY="your_api_key"      # for --matcher openai
export HF_TOKEN="your_hf_token"           # for --matcher huggingface
```

Or put a `.env` file in your working directory:

```sh
ANTHROPIC_API_KEY="your_api_key"
OPENAI_API_KEY="your_api_key"
HF_TOKEN="your_hf_token"
```

## Output format

A JSON file with:

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

---

## API / Documentation

### Input formats

**Story file** — a `.txt` or `.json` file containing the story segments to match against.

- **`.txt`**: one segment per line (blank lines are ignored).
- **`.json`**: must contain a `"segments"` array of strings. Optionally includes `"segmentation_method"`.

```json
{
  "segmentation_method": "sentences",
  "segments": [
    "The cat sat on the mat.",
    "It purred softly."
  ]
}
```

**Recall file** — a `.txt` file, a `.json` file, or a **directory** of either.

- **`.txt` file**: one recall segment per line. The filename stem is used as the subject ID.
- **`.json` file**: must contain a `"recalls"` object mapping subject IDs to segment arrays.
- **Directory**: all `.txt` or all `.json` files inside are loaded (mixing formats is not allowed). Each `.txt` file becomes one subject; `.json` files are merged.

```json
{
  "segmentation_method": "clauses",
  "recalls": {
    "sub-001": ["A cat was on a mat.", "It was purring."],
    "sub-002": ["There was a cat on something."]
  }
}
```

### CLI reference

```
rmatch STORY_FILE RECALL_FILE [options]
```

#### General options

- **`STORY_FILE`** *(positional, required)* — Path to the story `.txt` or `.json` file.
- **`RECALL_FILE`** *(positional, required)* — Path to a recall `.txt`/`.json` file or a directory of them.
- **`-M`, `--matcher`** *(str)* — Which matcher backend to use. One of: `anthropic`, `openai`, `reranker`, `huggingface`. Default: `anthropic`.
- **`-m`, `--model-name`** *(str)* — Override the matcher's default model (see defaults below).
- **`--track-emissions`** — Enable [CodeCarbon](https://github.com/mlco2/codecarbon) carbon-emissions tracking. Results are saved beside the output file.
- **`-f`, `--overwrite`** — Overwrite the output file if it already exists.

#### LLM matcher options (anthropic, openai, huggingface)

- **`--window-size`** *(int)* — Number of surrounding recall segments (before and after) to include as context for each target segment. Set to `0` to disable context. Default: `5`.
- **`--dry-run`** — *anthropic & openai only.* Estimate token usage and cost without making API calls.

#### Self-hosted / HuggingFace options

- **`-q`, `--quantization`** *(str)* — Load the model in reduced precision: `4bit` (NF4) or `8bit`. Requires `bitsandbytes`.
- **`-bs`, `--batch-size`** *(int)* — Number of prompts to process in parallel. Default: `4`.
- **`--max-new-tokens`** *(int)* — Maximum tokens the model may generate per prompt. Default: `64`.
- **`--verbose-errors`** — Print the raw model output when parsing fails. Useful for debugging prompt issues.

#### Reranker options

- **`--device`** *(str)* — PyTorch device for the reranker model (e.g. `cpu`, `cuda`, `mps`). Default: auto.
- **`--threshold`** *(float)* — Minimum similarity score for a story segment to be considered a match. Default: `0.09`.
- **`--top-k`** *(int)* — Number of top-scoring story candidates to evaluate per recall segment. Default: `5`.

### Default models

- **anthropic** — `claude-opus-4-6`
- **openai** — `gpt-4.1`
- **reranker** — `BAAI/bge-reranker-v2-m3`
- **huggingface** — `meta-llama/Llama-3.2-1B-Instruct`

### Python API

#### `Matcher` (main entry point)

```python
from rmatch import Matcher

matcher = Matcher(matcher_name="anthropic", model_name=None, **kwargs)
matches  = matcher.match(story_segments, recall_segments)
```

`Matcher(matcher_name, **kwargs)` is a factory — it returns the appropriate subclass based on `matcher_name`. All keyword arguments are forwarded to the subclass constructor.

**Constructor arguments:**

- **`model_name`** *(str)* — Override the default model. Applies to all matchers.
- **`window_size`** *(int)* — Context window radius around the target recall segment. Default: `5`. Applies to: `anthropic`, `openai`, `huggingface`.
- **`dry_run`** *(bool)* — Estimate cost without calling the API. Applies to: `anthropic`, `openai`.
- **`api_key`** *(str)* — API key. Falls back to `.env`, then environment variables. Applies to: `anthropic`, `openai`, `huggingface`.
- **`device`** *(str)* — PyTorch device string. Applies to: `reranker`.
- **`threshold`** *(float)* — Score threshold for matches. Default: `0.09`. Applies to: `reranker`.
- **`top_k`** *(int)* — Top-k candidates per recall segment. Default: `5`. Applies to: `reranker`.
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

#### `run_matching` (file-level convenience)

```python
from rmatch.match import run_matching

results = run_matching(
    story_file,            # Path — story .txt or .json
    recall_file,           # Path — recall file or directory
    matcher_name,          # str  — "anthropic", "openai", "reranker", "huggingface"
    track_emissions,       # bool — enable CodeCarbon tracking
    story_name=None,       # str | None — override auto-detected story name
    story_segmentation=None,   # str | None — override detected segmentation method
    recall_segmentation=None,  # str | None — override detected segmentation method
    overwrite=False,       # bool — overwrite existing output file
    **kwargs,              # forwarded to the Matcher constructor (model_name, window_size, etc.)
)
```

Loads story and recall files, runs matching for every subject, and saves a JSON results file. Returns the output dictionary.
