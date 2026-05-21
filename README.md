# contours2mesh

Register polynomial contours to 3D meshes.
Use in order rigid, affine, polynomial, or nonrigid registration to align contours, and marching cubes to mesh them.

## Features

- `python scripts/npz_registration.py`: example to load npz files and perform registration.

## Installation

### Using `uv` (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
You can test your changes by running the tests with `pytest`:

```bash
uv sync --extra dev
uv run pre-commit install
python -m pytest tests/
```

## Citation

OHBM 2026 paper: TODO