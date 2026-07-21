# `mwave`

A package for exploring matterwave interferometer geometries and for numerically solving the Bloch Hamiltonian that describes Bragg diffraction and Bloch oscillations.

## Installation

Install via

```bash
pip install git+https://github.com/jc-roth/mwave@v3.1.0
```

## Documentation

Located on readthedocs [here](https://mwave.readthedocs.io/latest/quickstart.html).

## Building the package

### Building in development mode
To build the package in development mode execute
```bash
uv sync
```

### Building a distributable

```bash
uv build
```

### Building documentation

Install the package with development dependencies, then build the docs:

```bash
uv sync --extra dev
uv run sphinx-build -b html docs docs/_build
```

### Testing the package
Install the package with development dependencies, then run the tests:
```bash
uv sync --extra dev
uv run pytest
```
