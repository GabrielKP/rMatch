# Recall matrix

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

### cyoa

1. Download monthiversary data into `downloads/cyoa/monthiversary` so that you have `downloads/cyoa/monthiversary/3_pasv`
2. Download alice data into `downloads/cyoa/alice` so that you have `downloads/cyoa/alice/3_pasv`
3. Run `python scripts/import_cyoa`
