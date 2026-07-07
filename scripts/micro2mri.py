import argparse
import logging
import os
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
import polymorpheo.plot as plot
import polymorpheo.register as register
import polymorpheo.utils as utils
from polymorpheo.log import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Register a series of aligned 2D histological slice contours (microscopy) "
            "onto a 3D MRI surface mesh. "
            "Outputs the final deformed micro points and simplices as NPZ."
        )
    )
    parser.add_argument("micro_input", type=Path,
                        help="Input NPZ file containing the micro slice contour series.")
    parser.add_argument("mri_input", type=Path,
                        help="Input MRI surface mesh (.ply, .obj or .vtp).")
    parser.add_argument("--outdir", "-o", type=Path, required=True,
                        help="Output directory.")
    parser.add_argument("--spacing", "-s", type=float, nargs=3,
                        default=[0.1, 0.1, 1.25], metavar=("SX", "SY", "SZ"),
                        help="Pixel/voxel spacing in x, y, z (default: 0.1 0.1 1.25).")
    parser.add_argument("--npts", type=int, default=100,
                        help="Number of points per contour after resampling (default: 100).")
    parser.add_argument("--npts-min", type=int, default=5, dest="npts_min",
                        help="Minimum number of points to keep a contour (default: 5).")
    parser.add_argument("--propag", choices=["jacobi", "gs"], default="jacobi",
                        help="Propagation scheme for the 2D slice registration (default: jacobi).")
    parser.add_argument("--multi", choices=["simultaneous", "independent_avg"],
                        default="simultaneous",
                        help="Multi-neighbor formulation for the 2D slice registration (default: simultaneous).")
    parser.add_argument("--no-deformable", action="store_true",
                        help="Skip the 2D deformable slice-refinement step (rigid + affine only).")
    parser.add_argument("--no-deformable-3d", action="store_true",
                        help="Skip the coarse-to-fine 3D deformable registration (cube-init + affine only).")
    parser.add_argument("--icp-niter3d", type=int, default=50,
                        help="Iterations for the 3D affine registration (default: 50).")
    parser.add_argument("--lr3d", type=float, nargs="+", default=[1e-2, 1e-2, 1e-3],
                        help="Learning rate schedule for the 3D deformable stages (default: 1e-2 1e-2 1e-3).")
    parser.add_argument("--wreg3d", type=float, nargs="+", default=[5e-1, 3e-1, 1e-1],
                        help="Regularization weight schedule for the 3D deformable stages (default: 5e-1 3e-1 1e-1).")
    parser.add_argument("--sigma3d", type=float, nargs="+", default=[5e-1, 1e-1, 5e-2],
                        help="Kernel sigma schedule for the 3D deformable stages (default: 5e-1 1e-1 5e-2).")
    parser.add_argument("--cpts-ratio3d", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                        help="Control-point ratio schedule for the 3D deformable stages (default: 0.05 0.1 0.2).")
    parser.add_argument("--niter3d", type=int, default=50,
                        help="Iterations per 3D deformable stage (default: 50).")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Show before/after 3D overlays and loss curves for each registration stage.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-stage registration progress.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(level=logging.INFO if args.verbose else logging.WARNING)

    micro_path = args.micro_input.resolve()
    if not micro_path.exists():
        print(f"Error: file not found: {micro_path}", file=sys.stderr)
        sys.exit(1)

    mri_path = args.mri_input.resolve()
    if not mri_path.exists():
        print(f"Error: file not found: {mri_path}", file=sys.stderr)
        sys.exit(1)

    schedule = [args.lr3d, args.wreg3d, args.sigma3d, args.cpts_ratio3d]
    if len({len(s) for s in schedule}) > 1:
        print("Error: --lr3d, --wreg3d, --sigma3d and --cpts-ratio3d must have the same length.", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    name = micro_path.stem
    spacing = np.array(args.spacing)
    verbose = args.verbose
    bidir = True


    # --- load ---

    micro_io = polymorpheo.io(
        datadir=str(micro_path.parent),
        names=[name],
        spacing=spacing,
        npts=args.npts,
        npts_min=args.npts_min,
    )

    polylines_raw, z_coords = micro_io.load()

    mri_poly = utils.read_vtkpoly(str(mri_path))
    pts_mri, simps_mri = utils.vtkpoly2mesh(mri_poly)[:2]
    normals_mri = utils.normals_mesh(pts_mri, simps_mri)


    # --- micro 2D slice registration ---

    polylines_defo = polymorpheo.register_contour_slices(
        polylines_raw, micro_io.xlim, micro_io.ylim,
        propag=args.propag, multi=args.multi, bidir=bidir,
        no_deformable=args.no_deformable, plot=False, verbose=verbose,
    )

    pts_micro, simps_micro = utils.polylines_2d_3d(polylines_defo, 2, z_coords)


    # --- micro -> mri 3D registration ---

    mesh_mri = pts_mri, None, normals_mri, None

    if args.plot:
        fig = plot.plot_obj(pts_mri, simps_mri)
        fig = plot.plot_obj(pts_micro, simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
        fig.show()

    pts_micro_init, _, _ = register.init_affcube(pts_mri, pts_micro, verbose=verbose)
    print("cube init  -  dist:", utils.chamfer(pts_micro_init, pts_mri))
    if args.plot:
        fig = plot.plot_obj(pts_mri, simps_mri)
        fig = plot.plot_obj(pts_micro_init, simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
        fig.show()
    mesh_micro = pts_micro_init, simps_micro, None, None

    reg_aff3d = register.reg_linear(niter=args.icp_niter3d, transfo="affine", verbose=verbose)
    t = time.time()
    _, mesh_micro = reg_aff3d.compute(mesh_mri, mesh_micro)
    print("affine  -  dist:", utils.chamfer(mesh_micro[0], pts_mri), ",  time:", time.time() - t)
    if args.plot:
        fig = plot.plot_obj(pts_mri, simps_mri)
        fig = plot.plot_obj(mesh_micro[0], simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
        fig.show()

    if not args.no_deformable_3d:
        for lr, wreg, sigma, cpts_ratio in zip(*schedule):
            fit_fun = energy.point2plane(agg="mean", alpha=-2, scale=0.01, bidir=True)
            regul_fun = energy.alap(transfo="similarity", l_norm=2)
            regul_fun.set_neighs(mesh_micro[1], mesh_micro[0].shape[0])
            reg_defo = register.reg_deformable(
                niter=args.niter3d, fit_fun=fit_fun, regul_fun=regul_fun,
                lr=lr, wreg=wreg, sigma=sigma, int_steps=8, rk=2, cpts_ratio=cpts_ratio,
                verbose=verbose,
            )
            t = time.time()
            _, mesh_micro, loss = reg_defo.compute(mesh_mri, mesh_micro)
            print("deformable  -  dist:", utils.chamfer(mesh_micro[0], pts_mri), ",  time:", time.time() - t)
            if args.plot:
                fig = plot.plot_obj(pts_mri, simps_mri)
                fig = plot.plot_obj(mesh_micro[0], simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
                fig.show()


    # --- save ---

    npz_out = outdir / f"{name}_deformed.npz"
    np.savez(npz_out, pts=np.array(mesh_micro[0]), simps=np.array(simps_micro))
    print(f"Saved: {npz_out}")

    if args.plot:
        fig = plot.plot_obj(pts_mri, simps_mri)
        fig = plot.plot_obj(mesh_micro[0], simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
        fig.show()


if __name__ == "__main__":
    main()
