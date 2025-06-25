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

### Building an obfuscated distributable
First generate obfuscated code using `pyarmor`
```bash
pyarmor gen -r -i src/mwave -O srcobf
```
Then modify `pyproject.toml` so that
```python
packages = [{include = 'mwave', from ='src'}]
```
becomes
```python
packages = [{include = 'mwave', from ='srcobf'}]
```
Then you can run
```bash
poetry build
```
as usual!

Note that this doesn't seem to be working on Apple Silicon.

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