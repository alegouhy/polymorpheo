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
import vtk

import polymorpheo.energy as energy
import polymorpheo.register as register
import polymorpheo.transfo as transfo_ops
import polymorpheo.utils as utils
from polymorpheo.log import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Symmetrise a surface by registering it onto its mirrored copy and applying "
                     "half of the resulting transform, which lands the mesh midway between itself "
                     "and its mirror.")
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input surface: .ply, .obj or .vtp.")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output symmetrised surface, same formats.")
    parser.add_argument("--axis", "-ax", type=int, required=True, choices=[0, 1, 2],
                        help="Axis to mirror along.")
    parser.add_argument("--center", type=str, default="centroid", choices=["centroid", "origin"],
                        help="Mirror about the centroid or about the origin (default: centroid).")
    parser.add_argument("--chop", type=str, default=None, choices=["pos", "neg"],
                        help="Optional finish: cut at the mirror plane, keep the side on the positive "
                             "or negative half of the axis, and glue its mirror image to it. Exactly "
                             "symmetric, but the mesh is retriangulated at the seam so the vertex "
                             "correspondence with the input is lost.")

    parser.add_argument("--no-rigid", action="store_true",
                        help="Skip the rigid ICP stage.")
    parser.add_argument("--no-affine", action="store_true",
                        help="Skip the affine ICP stage.")
    parser.add_argument("--icp-niter", type=int, default=50,
                        help="Iterations for the rigid and affine ICP stages (default: 50).")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-2, 1e-2, 1e-3],
                        help="Learning rate schedule for the deformable stages (default: 1e-2 1e-2 1e-3).")
    parser.add_argument("--wreg", type=float, nargs="+", default=[5e-1, 3e-1, 1e-1],
                        help="Regularization weight schedule (default: 5e-1 3e-1 1e-1).")
    parser.add_argument("--sigma", type=float, nargs="+", default=[3e-1, 1e-1, 5e-2],
                        help="Kernel sigma schedule (default: 3e-1 1e-1 5e-2).")
    parser.add_argument("--cpts-ratio", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                        dest="cpts_ratio",
                        help="Control-point ratio schedule (default: 0.05 0.1 0.2).")
    parser.add_argument("--niter", type=int, default=50,
                        help="Iterations per deformable stage (default: 50).")
    parser.add_argument("--int-steps", type=int, default=8, dest="int_steps",
                        help="Integration steps of the deformable transforms (default: 8).")

    parser.add_argument("--fit-ncpts", type=int, default=2000, dest="fit_ncpts",
                        help="Control points of the velocity field fitted to the whole chain (default: 2000).")
    parser.add_argument("--fit-sigma", type=float, default=5e-2, dest="fit_sigma",
                        help="Kernel sigma of that field, as a fraction of the mesh amplitude (default: 5e-2).")
    parser.add_argument("--fit-niter", type=int, default=300, dest="fit_niter",
                        help="Iterations of the velocity fit (default: 300).")
    parser.add_argument("--fit-lr", type=float, default=1e-2, dest="fit_lr",
                        help="Learning rate of the velocity fit (default: 1e-2).")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-stage registration progress.")

    return parser.parse_args()


def load_surf(path):
    pts, simps = utils.vtkpoly2mesh(utils.read_vtkpoly(str(path)))[:2]

    return np.asarray(pts, dtype=float), np.asarray(simps, dtype=np.int32)


def save_surf(pts, simps, path):
    utils.write_vtkpoly(utils.vtkpoly(np.array(pts), np.array(simps, dtype=np.int32)), str(path))
    print(f"saved: {path}")


def mirror_pos(pts, axis, center):
    return float(pts[:, axis].mean()) if center == "centroid" else 0.0


def mirror(pts, simps, axis, pos):
    # Mirroring reverses the triangle winding, so it is flipped back to keep the normals outwards.
    pts_mir = np.array(pts)
    pts_mir[:, axis] = 2 * pos - pts_mir[:, axis]

    return pts_mir, simps[:, ::-1]


def asymmetry(pts, simps, axis, pos):
    return float(utils.chamfer(pts, mirror(pts, simps, axis, pos)[0]))


def chop_glue(pts, simps, axis, pos, side, tol=1e-6):
    # Cut at the mirror plane and glue the mirror image of the kept side onto it.
    poly = utils.vtkpoly(np.array(pts), np.array(simps, dtype=np.int32))
    poly_pos, poly_neg = utils.split_vtkpoly(poly, pos, axis)

    # The clipper can leave non triangular cells behind, which vtkpoly2mesh does not read.
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly_pos if side == "pos" else poly_neg)
    tri.Update()

    pts_half, simps_half = utils.vtkpoly2mesh(tri.GetOutput())[:2]
    pts_half = np.asarray(pts_half, dtype=float)

    # Snap the cut onto the plane, so that both seams hold the very same points and weld.
    seam = np.abs(pts_half[:, axis] - pos) < tol * np.abs(pts_half).max()
    pts_half[seam, axis] = pos

    pts_mir, simps_mir = mirror(pts_half, simps_half, axis, pos)

    pts_glue = np.concatenate([pts_half, pts_mir])
    simps_glue = np.concatenate([simps_half, simps_mir + len(pts_half)])

    return utils.clean_mesh(pts_glue, simps_glue)


def main():
    args = parse_args()
    configure_logging(level=logging.INFO if args.verbose else logging.WARNING)

    in_path = args.input.resolve()
    if not in_path.exists():
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    schedule = [args.lr, args.wreg, args.sigma, args.cpts_ratio]
    if len({len(s) for s in schedule}) > 1:
        print("Error: --lr, --wreg, --sigma and --cpts-ratio must have the same length.", file=sys.stderr)
        sys.exit(1)

    verbose = args.verbose


    # --- load and mirror ---

    pts, simps = load_surf(in_path)
    pos = mirror_pos(pts, args.axis, args.center)
    pts_mir, simps_mir = mirror(pts, simps, args.axis, pos)

    mesh_mov = pts, simps, None, None
    mesh_ref = pts_mir, simps_mir, utils.normals_mesh(pts_mir, simps_mir), None

    print(f"\n{in_path.name}: {pts.shape[0]} pts, {simps.shape[0]} faces")
    print(f"mirrored along axis {args.axis} about the {args.center} ({pos:.4f})")
    print(f"asymmetry before: {asymmetry(pts, simps, args.axis, pos):.4f}")


    # --- registration onto the mirrored copy ---

    print("\nRegistration onto the mirrored surface:")

    for stage, skip in (("rigid", args.no_rigid), ("affine", args.no_affine)):
        if skip:
            continue

        print(f"  - {stage}...", end="\n" if verbose else " ", flush=True)
        t = time.time()
        reg = register.reg_linear(niter=args.icp_niter, transfo=stage, init="identity",
                                  bidir=True, verbose=verbose)
        _, mesh_mov = reg.compute(mesh_ref, mesh_mov)
        print(f"done in {time.time() - t:.2f} s, dist: {utils.chamfer(mesh_mov[0], pts_mir):.4f}")

    for lr, wreg, sigma, cpts_ratio in zip(*schedule):
        fit_fun = energy.point2plane(agg="mean", alpha=-2, scale=0.01, bidir=True)
        regul_fun = energy.alap(transfo="similarity", l_norm=2)
        regul_fun.set_neighs(mesh_mov[1], mesh_mov[0].shape[0])

        print(f"  - deformable (sigma={sigma})...", end="\n" if verbose else " ", flush=True)
        t = time.time()
        reg = register.reg_deformable(niter=args.niter, fit_fun=fit_fun, regul_fun=regul_fun,
                                      lr=lr, wreg=wreg, sigma=sigma, int_steps=args.int_steps,
                                      rk=2, cpts_ratio=cpts_ratio, verbose=verbose)
        _, mesh_mov, _ = reg.compute(mesh_ref, mesh_mov)
        print(f"done in {time.time() - t:.2f} s, dist: {utils.chamfer(mesh_mov[0], pts_mir):.4f}")


    # --- half of the transform ---

    # The whole chain is refitted as a single stationary velocity field, whose square root is simply
    # half the velocity: exp(v / 2) sends the mesh midway between itself and its mirror.
    print("\nFitting the velocity field:")

    disp = np.array(mesh_mov[0]) - pts
    amp = float(np.abs(pts - pts.mean(axis=0)).max())
    _, cpts = utils.farthest_point_sampling(pts, min(args.fit_ncpts, pts.shape[0]))

    poly = transfo_ops.polytransfo(sigma=args.fit_sigma * amp, int_steps=args.int_steps, rk=2)
    poly.set_params(cpts)

    t = time.time()
    poly, losses = transfo_ops.fit_velocity(poly, pts, disp, max_iter=args.fit_niter, lr=args.fit_lr)
    pts_fit = np.array(poly.transform(pts))
    err = np.linalg.norm(pts_fit - (pts + disp), axis=1)
    print(f"  {len(cpts)} cpts, sigma {poly.sigma:.4f}, done in {time.time() - t:.2f} s")
    print(f"  residual to the chain: mean {err.mean():.4f} max {err.max():.4f}, "
          f"for a displacement of {np.linalg.norm(disp, axis=1).mean():.4f}")

    poly.theta_trans = poly.theta_trans / 2
    pts_sym = np.array(poly.transform(pts))

    print(f"\nasymmetry after: {asymmetry(pts_sym, simps, args.axis, pos):.4f}")


    # --- save ---

    if args.chop is not None:
        pts_sym, simps = chop_glue(pts_sym, simps, args.axis, pos, args.chop)
        print(f"chopped and glued: {pts_sym.shape[0]} pts, {simps.shape[0]} faces, "
              f"asymmetry {asymmetry(pts_sym, simps, args.axis, pos):.4e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_surf(pts_sym, simps, args.output)


if __name__ == "__main__":
    main()
