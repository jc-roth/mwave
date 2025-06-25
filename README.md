## Building the package

### Building in development mode
To build the package in development mode execute
```bash
python -m pip install -e .
```

### Building a distributable

```bash
poetry build
```

### Building documentation

First spawn a shell in the poetry environment by calling `poetry shell` from the root project directory. Next call `poetry install --with dev` to install the package with development dependencies (this includes sphinx). Then from the `docs` directory call

```bash
sphinx-build -b html . _build
```

### Testing the package
To test the package execute
```bash
pytest
```
