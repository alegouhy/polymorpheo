import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np

import polymorpheo
import polymorpheo.energy as energy
import polymorpheo.plots as plots
import polymorpheo.utils as utils
from polymorpheo.log import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align a series of 2D histological slice contours for 3D reconstruction. "
            "Outputs the aligned contours as NPZ (same format as input). "
            "Optionally also outputs a 3D surface mesh."
        )
    )
    parser.add_argument("input", type=Path,
                        help="Input NPZ file containing the slice contour series.")
    parser.add_argument("--outdir", "-o", type=Path, default=None,
                        help="Output directory (default: same directory as input).")
    parser.add_argument("--spacing", "-s", type=float, nargs=3,
                        default=[0.1, 0.1, 1.25], metavar=("SX", "SY", "SZ"),
                        help="Pixel/voxel spacing in x, y, z (default: 0.1 0.1 1.25).")
    parser.add_argument("--npts", type=int, default=100,
                        help="Number of points per contour after resampling (default: 100).")
    parser.add_argument("--npts-min", type=int, default=5, dest="npts_min",
                        help="Minimum number of points to keep a contour (default: 5).")
    parser.add_argument("--no-deformable", action="store_true",
                        help="Skip deformable registration (rigid + affine only).")
    parser.add_argument("--propag", choices=["jacobi", "gs"], default="jacobi",
                        help="Propagation scheme (default: jacobi).")
    parser.add_argument("--multi", choices=["simultaneous", "independent_avg"],
                        default="simultaneous",
                        help="Multi-neighbor formulation (default: simultaneous).")
    parser.add_argument("--mesh", action="store_true",
                        help="Also output a 3D surface mesh as OBJ.")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Show a before/after 3D plot of the contour stack.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-stage registration progress.")
    return parser.parse_args()


def save_aligned_npz(polylines, slice_idx, spacing, nslice_orig, out_path):
    """Save aligned polylines back to the registered_contours NPZ format."""
    registered_contours = np.empty(nslice_orig, dtype=object)

    for polyline, z_idx in zip(polylines, slice_idx):
        pts, simps = np.array(polyline[0]), np.array(polyline[1])
        contours = utils.contours2opts(pts, simps, closed_only=False)
        registered_contours[z_idx] = [c / spacing[:2] for c in contours]

    np.savez(out_path, registered_contours=registered_contours)


def main():
    args = parse_args()
    configure_logging(level=logging.INFO if args.verbose else logging.WARNING)

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir.resolve() if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    name = input_path.stem
    spacing = np.array(args.spacing)
    propag = args.propag
    multi = args.multi
    verbose = args.verbose
    bidir = True

    io_obj = polymorpheo.io(
        datadir=str(input_path.parent),
        names=[name],
        spacing=spacing,
        npts=args.npts,
        npts_min=args.npts_min,
    )

    polylines, z_coords = io_obj.load()
    polylines_raw = polylines

    reg = polymorpheo.register_slices(
        "rigid", propag=propag, multi=multi,
        init="centroid", bidir=bidir,
        xlim=io_obj.xlim, ylim=io_obj.ylim, verbose=verbose,
    )
    polylines, _ = reg.compute(polylines)

    reg = polymorpheo.register_slices(
        "affine", propag=propag, multi=multi,
        init="identity", bidir=bidir,
        xlim=io_obj.xlim, ylim=io_obj.ylim, verbose=verbose,
    )
    polylines, _ = reg.compute(polylines)

    if not args.no_deformable:
        fit_fun = energy.point2point(agg="mean", bidir=bidir)
        regul_fun = energy.grad_disp(l_norm=2)
        reg = polymorpheo.register_slices(
            "deformable", propag=propag, multi=multi,
            fit_fun=fit_fun, regul_fun=regul_fun,
            niter=1, icp_niter=50, lr=1e-2, wreg=5e-1, sigma=1e-1,
            int_steps=16, tol=1e-5,
            xlim=io_obj.xlim, ylim=io_obj.ylim, verbose=verbose,
        )
        polylines, _ = reg.compute(polylines)

    npz_out = outdir / f"{name}_aligned.npz"
    save_aligned_npz(polylines, io_obj.slice_idx, spacing, io_obj.nslice, npz_out)
    print(f"Saved: {npz_out}")

    if args.plot:
        plots.plot_contour_stack([polylines_raw, polylines], z_coords, labels=["raw", "aligned"])

    if args.mesh:
        mesher = polymorpheo.bridge_contours(thr_conn=0.3, sealed=True)
        meshes = mesher.compute(polylines, z_coords)
        io_obj.save(meshes, str(outdir), suffix="aligned")
        print(f"Saved: {outdir / (name + '_aligned.obj')}")


if __name__ == "__main__":
    main()
