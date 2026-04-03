<h1 align="center">rMatch</h1>

<p align="center">Automatic recall & story matching tool.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>

## Installation

```sh
pip install .
```

## Setup

Create a `.env` file in the repository root with the API keys you need:

```sh
ANTHROPIC_API_KEY="your_api_key"   # for --matcher anthropic (default)
OPENAI_API_KEY="your_api_key"      # for --matcher openai
HF_TOKEN="your_hf_token"          # for --matcher huggingface
```

## Quick start

### Command line

```sh
rmatch story.txt recalls/ --matcher anthropic
```

`STORY_FILE` is a line-separated `.txt` or `.json` (with a `segments` array).
`RECALL_FILE` is a `.txt`/`.json` file or a directory of them.

### Python API

```python
from rmatch import match

ratings = match(
    story_segments=["The cat sat on the mat.", "It purred softly."],
    recall_segments=["A cat was on a mat."],
    matcher="anthropic",
)
# [(0, [0])]  — recall segment 0 matched story segment 0
```

## CLI options

| Flag | Description | Default |
|---|---|---|
| `--matcher`, `-M` | `anthropic`, `openai`, `reranker`, `huggingface` | `anthropic` |
| `--model-name`, `-m` | Override the matcher's default model | matcher default |
| `--device` | Device for `reranker`/`huggingface` | auto |
| `--quantization`, `-q` | `4bit` or `8bit` (huggingface only) | none |
| `--batch-size`, `-bs` | Batch size (huggingface only) | `4` |
| `--track-emissions` | Enable CodeCarbon emissions tracking | off |

Run `rmatch --help` for the full list.

## Output format

A JSON file with:

```json
{
  "matcher_name": "anthropic",
  "n_story_segments": 20,
  "ratings": {
    "sub-001": [[0, [3, 7]], [1, [12]]],
    "sub-002": [[0, [1]], [1, [5, 6]]]
  }
}
```

Each entry in `ratings` maps a subject ID to a list of `[recall_segment_id, [matched_story_segment_ids...]]` pairs.

## Benchmarking

Requires [rBench](https://github.com/GabrielKP/rBench):

```sh
git clone git@github.com:GabrielKP/rBench.git
```

Add to `.env`:

```sh
BENCHMARK_ROOT="path/to/rBench"
```

Run:

```sh
uv run src/rmatch/evaluate.py {alice,monthiversary,memsearch}
```
