# polymorpheo

**polymorpheo** is a Python library for registering series of 2D contours and aligning them to 3D meshes. It supports rigid, affine, polynomial, and diffeomorphic deformable registration, with multi-neighbor propagation schemes and JAX-based autodiff optimization.

## Features

- Registration of 2D contour series: rigid, affine, polynomial, deformable (SVF framework)
- Multi-neighbor propagation: Gauss-Seidel and Jacobi schemes, independent+average or simultaneous formulations
- 3D registration of contour-derived meshes to reference surfaces
- Pluggable energy functions: point-to-point, point-to-plane, gradient displacement, ALAP regularization
- Diffeomorphic transformations via stationary velocity fields (SVF) with Runge-Kutta integration
- Fast optimization via JAX autodiff and Optax

## Installation

### Using `uv` (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

## Usage

```python
import polymorpheo

# Load contours
io_obj = polymorpheo.io(datadir, names, spacing, npts=100, npts_min=5)
polylines, z_coords = io_obj.load()

# Register slices
reg = polymorpheo.register_slices("deformable", propag="jacobi", multi="simultaneous")
polylines_reg, transfos = reg.compute(polylines)

# Bridge to 3D mesh
mesh = polymorpheo.bridge_contours(polylines_reg, z_coords)
```

To get started, run the toy script which exercises the full pipeline on synthetic data:

```bash
python scripts/toy_registration.py
```

The input NPZ must contain a single key `registered_contours`: an object array of length `nslices`, where each element is either `None` (no tissue on that section) or a list of `(N, 2)` arrays — one per contour, in pixel coordinates. See `polymorpheo/data/sample_contours.npz` for a concrete example.

To align a real contour series from an NPZ file, use `align_slices.py`:

```bash
python scripts/align_slices.py path/to/contours.npz --spacing 0.1 0.1 1.25 --plot
```

This runs the full rigid → affine → deformable pipeline and writes `contours_aligned.npz` next to the input file.
`--plot` opens a before/after 3D view in the browser. `--mesh` additionally exports a surface mesh as OBJ.

To register an aligned micro contour series onto a 3D MRI surface mesh, use `micro2mri.py`:

```bash
python scripts/micro2mri.py path/to/micro_contours.npz path/to/mri_mesh.ply --plot
```

This runs the 2D slice-alignment pipeline, then a cube-init → affine → coarse-to-fine
deformable 3D registration onto the MRI mesh, and writes `micro_contours_deformed.npz`
(the final deformed micro points and simplices) next to the input file.
`--plot` shows before/after 3D overlays and loss curves for each stage.

For a complete scripted example on real NPZ contour data, see `scripts/npz_registration.py`.

## Contributing

Contributions are welcome. Open an issue or submit a pull request.
Run the tests with:

```bash
uv sync --extra dev
uv run pre-commit install
python -m pytest tests/
```

## Citation

If you use this software, please cite:

Legouhy A. et al., *Methods for the alignment of histological slice series for 3D reconstruction without reference*, OHBM 2026.

```bibtex
@inproceedings{legouhy2026polymorpheo,
  title     = {Methods for the alignment of histological slice series for 3D reconstruction without reference},
  author    = {Legouhy, Antoine and Mart{\'{i}}nez-Anh{\'{o}}m, Kevin and Maikranz, Erik and Caporal, Cl{\'{e}}ment and Traut, Nicolas and Heuer, Katja and Toro, Roberto},
  booktitle = {Annual Meeting of the Organization for Human Brain Mapping (OHBM)},
  year      = {2026},
}
```
