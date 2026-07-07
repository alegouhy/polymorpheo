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
import polymorpheo.plot as plot
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


def save_aligned_npz(polylines, z_coords, spacing, nslice_orig, out_path):
    """Save aligned polylines back to the registered_contours NPZ format."""
    registered_contours = np.empty(nslice_orig, dtype=object)

    for polyline, z in zip(polylines, z_coords):
        z_idx = int(round(z / spacing[2]))
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

    nslice_orig = len(np.load(input_path, allow_pickle=True)["registered_contours"])

    io_obj = polymorpheo.io(
        datadir=str(input_path.parent),
        names=[name],
        spacing=spacing,
        npts=args.npts,
        npts_min=args.npts_min,
    )

    polylines, z_coords = io_obj.load()
    polylines_raw = polylines

    polylines = polymorpheo.register_contour_slices(
        polylines, io_obj.xlim, io_obj.ylim,
        propag=propag, multi=multi, bidir=bidir,
        no_deformable=args.no_deformable, verbose=verbose,
    )

    npz_out = outdir / f"{name}_aligned.npz"
    save_aligned_npz(polylines, z_coords, spacing, nslice_orig, npz_out)
    print(f"Saved: {npz_out}")

    if args.plot:
        plot.plot_contour_stack([polylines_raw, polylines], z_coords, labels=["raw", "aligned"])

    if args.mesh:
        mesher = polymorpheo.bridge_contours(thr_conn=0.3, sealed=True)
        meshes = mesher.compute(polylines, z_coords)
        io_obj.save(meshes, str(outdir), suffix="aligned")
        print(f"Saved: {outdir / (name + '_aligned.obj')}")


if __name__ == "__main__":
    main()
