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
io_obj = polymorpheo.io_micro(datadir, names, spacing, npts=100, npts_min=5)
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

A contour series comes in either of two layouts, and the loader takes both.

A **single NPZ** holds the whole series under one key `registered_contours`: an object array of
length `nslices`, where each element is either `None` (no tissue on that section) or a list of
`(N, 2)` arrays — one per contour, in pixel coordinates. See `polymorpheo/data/sample_contours.npz`
for a concrete example. All its contours share one label.

A **directory** holds one `slice_%04d.npz` per section, each with a `contours` object array of
`(N, 2)` arrays and a parallel `names` array giving the region each one belongs to. The region
names become the labels, so a point is only ever matched to a point of the same region — grey
matter to grey matter, white to white. `fetch_micro.py` writes this layout from a MicroDraw
project. The slice index is read from the file name, not from the position in the directory.

Going from a raw contour series to a surface takes two steps: aligning the slices to each other,
then registering the aligned stack onto a reference surface.

To align a real contour series, in either layout, use `align_slices.py`:

```bash
python scripts/align_slices.py path/to/contours out/stack.obj out/chains.pkl \
                               --spacing 0.1 0.1 1.25 --plot
```

This runs the full rigid → affine → deformable pipeline and writes the two outputs it is given:
the aligned contours stacked in 3D, as points and edges carrying the name of the region each one
belongs to, and the per-slice transform chains. The extension of the first picks its format, npz,
vtp or obj, of which only the npz also carries the slice positions.
`--fix-slices` keeps given slices still, as raw indices into the input NPZ.
`--plot` opens a before/after 3D view in the browser. `--mesh DIR` additionally bridges the
contours into a surface mesh, one OBJ per region.

To register a surface onto another one, use `reg_surf.py`. The aligned stack is such a surface,
so this is also how a contour series reaches an MRI mesh:

```bash
python scripts/reg_surf.py --ref path/to/mri_mesh.obj \
                           --mov path/to/contours_aligned_stack.obj \
                           --out out/registered.obj --transfo out/chain.pkl --plot
```

This runs cube-init → rigid → affine → coarse-to-fine deformable registration, writing the two
outputs it is given plus what each stage produced beside the first, under the same name with the
stage appended. A stack keeps its edges, so it can be written to NPZ, OBJ or VTP, the formats that
can hold them; only the NPZ carries the slice positions alongside.

Both surfaces name their regions, and the ones they name alike are paired: a point is only ever
matched to a point of the region paired with its own, so rename them beforehand
(`polymorpheo.relabel_surf`) if the two namings differ. A region the other surface has no
counterpart for is a passenger, carried by the transform without being fitted, and it is left out
of the deformable stages entirely rather than padding their distance matrices.
`--plot` shows before/after 3D overlays and loss curves for each stage.

`apply_serial_transfo.py` replays those chains on other contours, points or ellipsoids sharing the
same slice layout, without re-running any registration:

```bash
python scripts/apply_serial_transfo.py --transfos-2d out/contours_transfos2d.pkl \
                                        --transfos-3d out/..._transfos.pkl \
                                        --pts path/to/cells.csv --outdir out/
```

Either chain can be given on its own, to act in plane only or on the stack only.

`--images` takes it further and resamples the slice images themselves into the geometry of another
image, typically the MRI the contours were registered onto:

```bash
python scripts/apply_serial_transfo.py --transfos-2d out/chains2d.pkl \
                                        --transfos-3d out/chain.pkl \
                                        --images path/to/slices --geom mri.nii.gz \
                                        --out-image out/slices_in_mri.nii.gz
```

Each slice is warped in plane by its own chain, which leaves the series stacked on a regular grid,
and that stack is resampled into the given geometry through the 3D chain. Both are pull
operations, so the chains are evaluated the other way round, which for a stationary velocity field
is just integrating it backwards. Evaluating a chain costs around 13 us a point, far too much for
every voxel of a volume, so it is sampled on a coarser grid, `--field-step` voxels apart, and sitk
interpolates the displacement in between.

`micro2mri.py` still runs both steps in one go, writing a single combined transform pickle that
`apply_serial_transfo.py` also accepts as its positional argument:

```bash
python scripts/micro2mri.py path/to/micro_contours.npz path/to/mri_mesh.ply --plot
```

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
