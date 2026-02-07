# Recall matrix


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
