# Recall matrix

## Development setup

```sh
# clone
git clone git@github.com:GabrielKP/recall_matrix.git
cd recall_matrix

# use poetry (https://python-poetry.org/docs/) to install package & dependencies
poetry install

# set up pre-commit
poetry run pre-commit install

# Now you can run the project code!
poetry run python src/recall_matrix/main.py
```

## Data downloads

### cyoa

1. Download monthiversary data into `downloads/cyoa/monthiversary` so that you have `downloads/cyoa/monthiversary/3_pasv`
2. Download alice data into `downloads/cyoa/alice` so that you have `downloads/cyoa/alice/3_pasv`
3. Run `python scripts/import_cyoa`
