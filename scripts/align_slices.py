import argparse
import logging
import os
import pickle
import sys
import time
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

STACK_EXT = (".npz", ".vtp", ".obj")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align a series of 2D histological slice contours for 3D reconstruction. "
            "Outputs the stacked 3D contour surface and the per-slice transform chains. "
            "Optionally also outputs a 3D surface mesh."
        )
    )
    parser.add_argument("input", type=Path,
                        help="Slice contour series: a directory of per-slice NPZ files, whose "
                             "region names label the contours, or a single NPZ holding the whole "
                             "series under 'registered_contours'.")
    parser.add_argument("stack", type=Path,
                        help="Output for the aligned contours stacked in 3D. Its extension picks "
                             "the format: npz, vtp or obj, all of which keep the edges and the "
                             "name of the region each contour belongs to, and of which only the "
                             "npz also carries the slice positions.")
    parser.add_argument("transfos", type=Path,
                        help="Output for the per-slice transform chains, as a pickle.")
    parser.add_argument("--spacing", "-s", type=float, nargs=3,
                        default=[0.1, 0.1, 1.25], metavar=("SX", "SY", "SZ"),
                        help="Pixel/voxel spacing in x, y, z (default: 0.1 0.1 1.25).")
    parser.add_argument("--npts", type=int, default=100,
                        help="Number of points per contour after resampling (default: 100).")
    parser.add_argument("--npts-min", type=int, default=5, dest="npts_min",
                        help="Minimum number of points to keep a contour (default: 5). Never "
                             "below 3, which is what it takes to close a ring.")
    parser.add_argument("--fix-slices", type=int, nargs="+", default=None, dest="fix_slices", metavar="IDX",
                        help="Slices to keep fixed during registration, as raw indices into the "
                             "input series (the same numbering io_micro.load reports on the left "
                             "of 'idx: N -> M'). Indices holding no contour are reported and "
                             "skipped.")
    parser.add_argument("--no-deformable", action="store_true",
                        help="Skip deformable registration (rigid + affine only).")
    parser.add_argument("--propag", choices=["jacobi", "gs"], default="jacobi",
                        help="Propagation scheme (default: jacobi).")
    parser.add_argument("--multi", choices=["simultaneous", "independent_avg"],
                        default="simultaneous",
                        help="Multi-neighbor formulation (default: simultaneous).")
    parser.add_argument("--regions", type=str, nargs="+", default=None, metavar="NAME",
                        help="Regions to keep, in the order they are labelled. The names travel "
                             "with the output, so this does not decide what they are paired with "
                             "later. Default: every region found, sorted.")
    parser.add_argument("--mesh", type=Path, default=None, metavar="DIR",
                        help="Also bridge the contours into a 3D surface mesh, one OBJ per region, "
                             "written to this directory.")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Show a before/after 3D plot of the contour stack.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-stage registration progress.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(level=logging.INFO if args.verbose else logging.WARNING)

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    stack_out, pkl_out = args.stack.resolve(), args.transfos.resolve()
    ext = stack_out.suffix.lower()
    if ext not in STACK_EXT:
        print(f"Error: {ext} holds no contour stack, name the output "
              f"{'/'.join(e.lstrip('.') for e in STACK_EXT)}.", file=sys.stderr)
        sys.exit(1)
    for out in (stack_out, pkl_out, args.mesh):
        if out is not None:
            (out if out is args.mesh else out.parent).mkdir(parents=True, exist_ok=True)

    name = input_path.stem
    spacing = np.array(args.spacing)
    propag = args.propag
    multi = args.multi
    verbose = args.verbose
    bidir = True


    # --- load ---what would it take to work also for deformable

    io_obj = polymorpheo.io_micro(datadir=str(input_path.parent),
                                  names=[name],
                                  spacing=spacing,
                                  npts=args.npts,
                                  npts_min=args.npts_min,
                                  regions=args.regions)

    polylines_raw, z_coords = io_obj.load()

    # --fix-slices is given in raw slice indices, but register_slices addresses polylines by their
    # position in the loaded series, which skips slices holding no contour. Map one to the other
    # here so the raw numbering is the only one a caller ever sees, and let a caller name slices
    # without knowing which of them hold a contour.
    fixed = None
    if args.fix_slices is not None:
        dense = {raw: i for i, raw in enumerate(io_obj.slice_idx)}
        missing = [z for z in args.fix_slices if z not in dense]
        if missing:
            print(f"Warning: slice(s) {missing} hold no contour or are out of range, so they are "
                  f"not in the series and stay free (kept: {io_obj.slice_idx[0]}.."
                  f"{io_obj.slice_idx[-1]}).", file=sys.stderr)
        fixed = [dense[z] for z in args.fix_slices if z in dense] or None


    # --- registration ---

    print('\nBetween slices registration (contour to contour):')

    polylines = polylines_raw
    chains_2d = [[] for _ in range(len(polylines_raw))]

    print("  - rigid...", end="\n" if verbose else " ", flush=True)
    t = time.time()
    reg = polymorpheo.register_slices("rigid", propag=propag, multi=multi,
                                      init="centroid", bidir=bidir,
                                      xlim=io_obj.xlim, ylim=io_obj.ylim, verbose=verbose)
    polylines, transfos = reg.compute(polylines, fixed=fixed)
    for chain, ts in zip(chains_2d, transfos):
        chain.extend(ts)
    print(f"done in {time.time() - t:.2f} s.")

    print("  - affine...", end="\n" if verbose else " ", flush=True)
    t = time.time()
    reg = polymorpheo.register_slices(
        "affine", propag=propag, multi=multi,
        init="identity", bidir=bidir,
        xlim=io_obj.xlim, ylim=io_obj.ylim, verbose=verbose,
    )
    polylines, transfos = reg.compute(polylines, fixed=fixed)
    for chain, ts in zip(chains_2d, transfos):
        chain.extend(ts)
    print(f"done in {time.time() - t:.2f} s.")

    if not args.no_deformable:
        print("  - deformable...", end="\n" if verbose else " ", flush=True)
        t = time.time()
        fit_fun = energy.point2point(agg="mean", bidir=bidir)
        regul_fun = energy.grad_disp(l_norm=2)
        reg = polymorpheo.register_slices("deformable", propag=propag, multi=multi,
                                          fit_fun=fit_fun, regul_fun=regul_fun,
                                          niter=1, icp_niter=50, lr=1e-2, wreg=5e-1, sigma=1e-1,
                                          int_steps=16, tol=1e-5,
                                          xlim=io_obj.xlim, ylim=io_obj.ylim, verbose=verbose)
        polylines, transfos = reg.compute(polylines, fixed=fixed)
        for chain, ts in zip(chains_2d, transfos):
            chain.extend(ts)
        print(f"done in {time.time() - t:.2f} s.")


    # --- save ---


    # The stacked contours are the moving surface reg_surf.py registers onto a reference mesh:
    # 3D points with edge simplices and one label per point.
    pts_stack, simps_stack = utils.polylines_2d_3d(polylines, 2, z_coords)
    # the regions travel with the stack under their own names, so reg_surf can pair each of them
    # with its counterpart on the reference surface
    labs_stack = np.concatenate([polyline[3] for polyline in polylines])
    utils.write_surf(str(stack_out), pts_stack, simps_stack, labs=labs_stack,
                     labels=io_obj.labels, z_coords=np.array(z_coords))
    print(f"Saved: {stack_out}")

    with open(pkl_out, "wb") as f:
        pickle.dump({"z_coords": np.array(z_coords), "transfos_2d": chains_2d}, f)
    print(f"Saved: {pkl_out}")

    if args.mesh is not None:
        mesher = polymorpheo.bridge_contours(thr_conn=0.3, sealed=True)
        meshes = mesher.compute(polylines, z_coords)
        io_obj.save(meshes, str(args.mesh.resolve()), suffix="aligned")
        print(f"Saved: {args.mesh.resolve()}/<region>_aligned.obj")

    if args.plot:
        plots.plot_contour_stack([polylines_raw, polylines], z_coords, labels=["raw", "aligned"])


if __name__ == "__main__":
    main()
