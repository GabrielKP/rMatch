<h1 align="center">rMatch</h1>

<p align="center">Automatic recall & story matching tool.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>

## Installation

From the repository root:

```sh
pip install .
```

This installs the `rmatch` command on your PATH. Run `rmatch --help` for story/recall arguments and matcher options. Equivalent: `python -m rmatch.match`.


### API keys

1. Create an api key at the platform of your choice.
2. Create and add it to the `.env`:
```sh
ANTHROPIC_API_KEY="your_api_key"
OPENAI_API_KEY="your_api_key"
HF_TOKEN="your_hf_token"
```



## Usage
Run from the repository root:

```sh
rmatch STORY_FILE RECALL_FILE [--matcher {anthropic,reranker,openai,huggingface}] \
  [--model-name MODEL] [--device DEVICE] [--quantization {4bit,8bit}] \
  [--batch-size N] [--track-emissions]
```

Key arguments:
- `STORY_FILE` (positional): Path to the story, either line-separated `.txt` or `.json` with a `segments` array.
- `RECALL_FILE` (positional): Path to recalls, either `.txt`/`.json` file or a directory containing `.txt`/`.json` files.
- `--matcher` / `-M`: Which rating method to use: `anthropic`, `reranker`, `openai`, or `huggingface` (default: `anthropic`).
- `--model-name` / `-m`: Underlying model override for the selected matcher (default: matcher built-in default).
- `--device`: Device hint for `reranker`/`huggingface` (default: auto).
- `--quantization` / `-q`: Huggingface quantization: `4bit` or `8bit` (default: no override).
- `--batch-size` / `-bs`: Huggingface batch size (default: `4`).
- `--track-emissions`: Enable CodeCarbon emissions tracking (writes output beside the recall).


### Outputs

The output is a single json file with following fields:
* **matcher_name**: Name of rating method (e.g. 'openai' or 'reranker')/
* **story_segmentation_method**: How story was segmented for producing this rating file.
* **recall_segmentation_method**: How recalls were segmented for producing this rating file.
* **output_scores**: Whether scores were outputted for each matched recall and story segment.
* **n_story_segments**: The number of story segments
* **ratings**: a dictionary mapping 'sub-id' -> 'single_subject_ratings' list.

The dictionary may contain additional metadata (e.g. the model_name).


#### The ratings dictionary

Maps 'sub-id' -> 'single_subject_ratings'.
The 'single_subject_ratings' list is a list of tuples with recall segments and their matching story segments:
`[(recall_segment_id_1, [matched story segments...]), (recall_segment_id_2, [matched story segments...])]`

Thus, the ratings dictionary looks something like this:
```json
{
    "sub-001": [
        (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..., ]),
        (recall_segment_id_2, [story_segment_id_z, story_segment_id_a, ...,]),
        ...
    ],
    "sub-002": [
        (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..., ]),
        (recall_segment_id_2, [story_segment_id_z, story_segment_id_a, ...,]),
        ...
    ],
    ...
}
```

Note that the number of recall segments can be computed from the length of the 'single_subject_ratings' list.


## Benchmark

To benchmark rMatch:

1. Download [rBench](https://github.com/GabrielKP/rBench):
```sh
git clone git@github.com:GabrielKP/rBench.git
```
2. Edit `.env` to point towards rBench root:
```sh
BENCHMARK_ROOT="path/to/rBench"
```
3. Run `uv run src/rmatch/evaluate.py`

Usage:
```sh
usage: evaluate.py [-h] [-rr] [--benchmark-root BENCHMARK_ROOT] [-M {anthropic,reranker,openai,huggingface}] [-m MODEL_NAME] [--device DEVICE] [--dry-run] [--window-size WINDOW_SIZE] [-n N_REPEATS]
                   [-q {4bit,8bit}] [-bs BATCH_SIZE] [--max-new-tokens MAX_NEW_TOKENS] [--verbose-errors] [--track-emissions]
                   {alice,monthiversary,memsearch}
```

_Good luck on your adventures._
