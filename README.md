<h1 align="center">recall_matrix</h1>

<p align="center">Automatic recall & story matching tool.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="" src="https://img.shields.io/badge/code-Python-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
<a href="https://docs.astral.sh/uv/"><img alt="packaging framework: uv" src="https://img.shields.io/badge/packaging-uv-lightblue?logo=uv"></a>
<a href="https://pre-commit.com/"><img alt="pre-commit" src="https://img.shields.io/badge/tool-Pre%20Commit-yellow?logo=Pre-Commit"></a>

## Usage

```sh
# 1. clone
git clone git@github.com:GabrielKP/recall_matrix.git
cd recall_matrix

# 2. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
uv run src/recall_matrix/rate_binary.py --sub-ids sub-001

# 3. Run rating code

# single subject
uv run src/recall_matrix/rate_binary.py --story_name pieman --sub_ids sub-001
# all subjects
uv run src/recall_matrix/rate_binary.py --story_name pieman
```

### Inputs

To rate your own stories and recalls, they need to be in a specific format in the `data/stories-and-recalls` directory.

Each story directory is organized as follows (this examnple is for pieman):
```sh
data/stories-and-recalls/pieman/
├── ratings
│   └── *.json
├── recalls
│   ├── sentences
│   │   ├── sub-001.txt
│   │   ├── sub-002.txt
│   │   ├── ...
│   │   └── sub-116.txt
│   └── ...
├── transcripts
│   ├── lda_hmm.txt
│   ├── sentences.txt
│   ├── sentences_corrected.txt
│   └── ...
└── ...
```

* **recalls**: contains subdirectories with the name of the `recall_segmentation_method` (in the example, "sentences"). The subdirectories contain .txt files for each subject in which each line is a new segment.
* **transcripts**: contains .txt files with the story transcript. The filename refers to the `story_segmentation_method` (in the example, "lda_hmm", "sentences", and "sentences_corrected")/
* **ratings**: contains the rating outputs

Each story-directory can contain additional directories with other information (e.g. "plots", "causality"), but these are not required for recall rating.

#### Step by step instructions

1. Create dir with your storyname in `data/stories_and_recalls`
2. Place you transcript in `data/stories_and_recalls/storyname/transcripts/<story_segmentation_method>.txt` - this file has to have a segmented story, each line is a new segment (e.g. an event, or sentences).
3. Place your recall transcripts in `data/stories_and_recalls/storyname/recalls/<recall_segmentation_method/sub-*.txt`  - separate file for each subject. Again, each line is a new segment (e.g. an event).


### Outputs

The output is a single json file in `data/stories_and_recalls/storyname/ratings`.
It contains important metadata fields:
* **rater_name**: Name of rating method (e.g. 'openai' or 'reranker')/
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


## Development setup

```sh
# clone
git clone git@github.com:GabrielKP/recall_matrix.git
cd recall_matrix

# install uv (https://docs.astral.sh/uv/getting-started/installation/) to install package & dependencies

# set up pre-commit
uv run pre-commit install

# Now you can run the project code!
uv run src/recall_matrix/test_recall_matrix.py -m reranker
```

## Data downloads


### filmfest


### cyoa (private)

1. Download monthiversary data into `downloads/cyoa/monthiversary` so that you have `downloads/cyoa/monthiversary/3_pasv`
2. Download alice data into `downloads/cyoa/alice` so that you have `downloads/cyoa/alice/3_pasv`
3. Run `python scripts/import_cyoa`


### memsearch (private)

1. Download the folders "Completed Scene-Matched Files" and "Trimmed Movies Annotations" and unzip them into the directory `downloads/memsearch`
2. Run `uv run scripts/import_memsearch.py`
