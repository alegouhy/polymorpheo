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

import polymorpheo.energy as energy
import polymorpheo.plots as plots
import polymorpheo.register as register
import polymorpheo.transfo as transfo_ops
import polymorpheo.utils as utils
from polymorpheo.log import configure_logging

MESH_EXT = (".ply", ".obj", ".vtp")
STACK_EXT = ".npz"
# a contour stack carries edges, which only these can hold: ply keeps polygons only
STACK_OUT_EXT = (STACK_EXT, ".obj", ".vtp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Register a moving surface onto a reference surface through up to four "
                     "sequential stages: cube init, rigid, affine and deformable. "
                     "Outputs the deformed moving surface and the transformation chain.")
    )
    parser.add_argument("--ref", type=Path, required=True,
                        help="Reference (fixed) surface: .ply, .obj, .vtp or .npz with 'pts'/'simps'.")
    parser.add_argument("--mov", type=Path, required=True,
                        help="Moving surface: .ply, .obj, .vtp or .npz with 'pts'/'simps'. A stack "
                             "of aligned slice contours, as align_slices.py writes it, is one of "
                             "these: its edge simplices are kept and the outputs are written to "
                             "npz, obj or vtp, the formats that can hold them.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output for the moving surface once registered. Its extension picks "
                             "the format, which has to be one the moving surface can be written "
                             "to. What each stage produced on the way is written beside it, under "
                             "the same name with the stage appended.")
    parser.add_argument("--transfo", type=Path, required=True,
                        help="Output for the transform chain, as a pickle.")

    parser.add_argument("--no-cube", action="store_true",
                        help="Skip the brute-force cube initialization.")
    parser.add_argument("--no-rigid", action="store_true",
                        help="Skip the rigid ICP stage.")
    parser.add_argument("--no-affine", action="store_true",
                        help="Skip the affine ICP stage.")
    parser.add_argument("--no-deformable", action="store_true",
                        help="Skip the coarse-to-fine deformable stage.")
    parser.add_argument("--no-cube-scale", action="store_true",
                        help="Do not rescale to the reference bounding box during cube init.")
    parser.add_argument("--no-cube-reflection", action="store_true",
                        help="Do not allow reflections during cube init.")
    parser.add_argument("--iso-cube", action="store_true",
                        help="Isotropic scaling during cube init.")
    parser.add_argument("--no-normals", action="store_false", dest="use_normals",
                        help="Point-to-point fit in the deformable stage, instead of point-to-plane on the reference normals.")

    parser.add_argument("--icp-niter", type=int, default=50,
                        help="Iterations for the rigid and affine ICP stages (default: 50).")
    parser.add_argument("--rigid-init", choices=["identity", "centroids", "similarity", "ellipsoid"],
                        default="centroids",
                        help="Initialization of the rigid stage (default: centroid).")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-2, 1e-2, 1e-3],
                        help="Learning rate schedule for the deformable stages (default: 1e-2 1e-2 1e-3).")
    parser.add_argument("--wreg", type=float, nargs="+", default=[5e-1, 3e-1, 1e-1],
                        help="Regularization weight schedule (default: 5e-1 3e-1 1e-1).")
    parser.add_argument("--sigma", type=float, nargs="+", default=[5e-1, 1e-1, 5e-2],
                        help="Kernel sigma schedule (default: 5e-1 1e-1 5e-2).")
    parser.add_argument("--cpts-ratio", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                        dest="cpts_ratio",
                        help="Control-point ratio schedule (default: 0.05 0.1 0.2).")
    parser.add_argument("--niter", type=int, default=50,
                        help="Iterations per deformable stage (default: 50).")
    parser.add_argument("--int-steps", type=int, default=8, dest="int_steps",
                        help="Integration steps of the deformable transform (default: 8).")

    parser.add_argument("--plot", "-p", action="store_true",
                        help="Show before/after 3D overlays and loss curves for each stage.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-stage registration progress.")

    return parser.parse_args()


def load_surf(path):
    # Returns the regions and the slice positions too when the file carries them, so a contour
    # stack keeps them through the registration and stays readable as a stack afterwards.
    return utils.read_surf(str(path))


def pair_regions(labs_ref, labels_ref, labs_mov, labels_mov):
    """Give the regions the two surfaces name alike one label value, and every other one its own.

    Both surfaces carry the names of their regions, and those are what has to agree: rename them
    beforehand if they do not. The label values themselves are private to this run, the energies
    only ever asking whether two of them are equal.
    """
    new_ref, new_mov, matched = {}, {}, []
    ref_of = {utils.obj_name(name): i + 1 for i, name in enumerate(labels_ref)}

    for lab, name in enumerate(labels_mov, start=1):
        if utils.obj_name(name) in ref_of:
            value = len(matched) + 1
            new_ref[ref_of[utils.obj_name(name)]] = value
            new_mov[lab] = value
            matched.append(name)

    free = len(matched)
    for table, labels in ((new_ref, labels_ref), (new_mov, labels_mov)):
        for lab in range(1, len(labels) + 1):
            if lab not in table:
                free += 1
                table[lab] = free

    def remap(labs, table):
        return np.array([table.get(int(lab), 0) for lab in labs], dtype=np.int32)

    return remap(labs_ref, new_ref), remap(labs_mov, new_mov), matched


def is_stack(simps):
    # Contour stacks carry edges, meshes carry polygons: 2-column simplices have no face to take a
    # normal from and no surface format to be written to.
    return simps.shape[1] == 2


def show(pts_ref, simps_ref, pts_mov, simps_mov, title):
    fig = plots.plot_obj(pts_ref, simps_ref, show_points=False, title=title)
    fig = plots.plot_obj(pts_mov, simps_mov, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5),
                         show_points=False, fig=fig)
    fig.show()


def moved_pts(transfos, pts, mesh_mov, fitted):
    # the registration only moved the points it fitted, so the rest are caught up with the chain
    if fitted is None or fitted.all():
        return np.array(mesh_mov[0])

    return np.array(transfo_ops.apply_transfo_chain(transfos, pts))


def save_surf(pts, simps, moved, suffix=None, indent="", labs=None, labels=None, z_coords=None):
    # what a stage produced goes beside the registered surface, under the same name and format
    out = moved.with_name(f"{moved.stem}_{suffix}{moved.suffix}") if suffix else moved
    utils.write_surf(str(out), pts, simps, labs=labs, labels=labels, z_coords=z_coords)
    print(f"{indent}saved: {out}")


def main():
    args = parse_args()
    configure_logging(level=logging.INFO if args.verbose else logging.WARNING)

    ref_path, mov_path = args.ref.resolve(), args.mov.resolve()
    for path in (ref_path, mov_path):
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    schedule = [args.lr, args.wreg, args.sigma, args.cpts_ratio]
    if not args.no_deformable and len({len(s) for s in schedule}) > 1:
        print("Error: --lr, --wreg, --sigma and --cpts-ratio must have the same length.", file=sys.stderr)
        sys.exit(1)

    moved_out, pkl_out = args.out.resolve(), args.transfo.resolve()
    moved_out.parent.mkdir(parents=True, exist_ok=True)
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    verbose = args.verbose


    # --- load ---

    pts_ref, simps_ref, labs_ref, labels_ref, _ = load_surf(ref_path)
    pts_mov, simps_mov, labs_mov, labels_mov, z_coords = load_surf(mov_path)

    # Regions restrict the correspondences: a point is only ever matched to a point of the region
    # paired with its own, and a region with no counterpart is matched to nothing at all.
    labs_out = labs_mov  # what the file came in with, so the outputs keep its own label values
    named = labels_ref is not None and labels_mov is not None
    if not named:
        if labels_ref is not None or labels_mov is not None:
            print(f"Warning: only the {'reference' if labels_ref else 'moving'} surface names its "
                  f"regions, so they are ignored and every point may match any other.",
                  file=sys.stderr)
        labs_ref = labs_mov = None
    else:
        labs_ref, labs_mov, matched = pair_regions(labs_ref, labels_ref, labs_mov, labels_mov)

    if args.use_normals and is_stack(simps_ref):
        print("Error: the reference holds edges, not faces, so it carries no normals: give a "
              "surface mesh or --no-normals.", file=sys.stderr)
        sys.exit(1)

    # A contour stack keeps its edges, so it can only be written to a format that holds them, and
    # only the npz carries the slice positions alongside.
    ext = moved_out.suffix.lower()
    holds = STACK_OUT_EXT if is_stack(simps_mov) else MESH_EXT
    if ext not in holds:
        what = "a contour stack, whose edges" if is_stack(simps_mov) else "a surface mesh, which"
        print(f"Error: the moving input is {what} {ext} cannot hold, name the output "
              f"{'/'.join(e.lstrip('.') for e in holds)}.", file=sys.stderr)
        sys.exit(1)

    normals_ref = utils.normals_mesh(pts_ref, simps_ref) if args.use_normals else None
    mesh_ref = pts_ref, simps_ref, normals_ref, labs_ref
    mesh_mov = pts_mov, simps_mov, None, labs_mov

    # A point whose region the other surface does not have is never a correspondence. The linear
    # stages already leave those out, nearest_neighbors taking the labels one at a time, but the
    # deformable ones pad a full distance matrix with them, so they are dropped before those and
    # the chain is applied to the whole surface to write it out.
    fitted = None
    if named and matched:
        paired = np.arange(1, len(matched) + 1)
        fitted = np.isin(labs_mov, paired)

    unit_ref = "edges" if is_stack(simps_ref) else "faces"
    unit_mov = "edges" if is_stack(simps_mov) else "faces"
    print(f"\nReference: {ref_path.name} ({pts_ref.shape[0]} pts, {simps_ref.shape[0]} {unit_ref})")
    print(f"Moving:    {mov_path.name} ({pts_mov.shape[0]} pts, {simps_mov.shape[0]} {unit_mov})")
    if named:
        free = [name for name in labels_mov if name not in matched]
        if not matched:
            print(f"Warning: the two surfaces name no region alike, so nothing is fitted. The "
                  f"moving one has {', '.join(labels_mov)} and the reference "
                  f"{', '.join(labels_ref)}.", file=sys.stderr)
        print("regions:   paired " + (", ".join(matched) or "none")
              + (f" | carried but not fitted: {', '.join(free)}" if free else ""))
    print(f"initial dist: {utils.chamfer(pts_mov, pts_ref):.4f}")
    print(f"deformable fit: {'point-to-plane' if args.use_normals else 'point-to-point'}")
    if args.plot:
        show(pts_ref, simps_ref, pts_mov, simps_mov, "raw")


    # --- registration ---

    print("\nSurface to surface registration:")
    transfos = []
    losses = []

    if not args.no_cube:

        print("  - cube init...", end="\n" if verbose else " ", flush=True)
        t = time.time()

        pts_cube, lin, trans = register.init_affcube(pts_ref, mesh_mov[0],
                                                     iso_scale=not args.iso_cube, do_scale=not args.no_cube_scale,
                                                     no_reflection=args.no_cube_reflection, verbose=verbose)

        mesh_mov = pts_cube, mesh_mov[1], None, mesh_mov[3]
        cube_transfo = transfo_ops.affine()
        cube_transfo.set_params(lin, trans)
        transfos.append(cube_transfo)
        pts_stage = np.array(mesh_mov[0])
        print(f"done in {time.time() - t:.2f} s, dist: {utils.chamfer(pts_stage, pts_ref):.4f}")
        save_surf(pts_stage, simps_mov, moved_out, "cube", indent="    ",
                  labs=labs_out, labels=labels_mov, z_coords=z_coords)

        if args.plot:
            show(mesh_ref[0], mesh_ref[1], mesh_mov[0], mesh_mov[1], "cube init")

    for stage, skip in (("rigid", args.no_rigid), ("affine", args.no_affine)):

        if skip: continue

        print(f"  - {stage}...", end="\n" if verbose else " ", flush=True)
        t = time.time()

        reg = register.reg_linear(niter=args.icp_niter, transfo=stage,
                                  init=args.rigid_init if stage == "rigid" else "identity",
                                  bidir=True, verbose=verbose)

        transfo, mesh_mov = reg.compute(mesh_ref, mesh_mov)
        transfos.append(transfo)
        pts_stage = np.array(mesh_mov[0])
        print(f"done in {time.time() - t:.2f} s, dist: {utils.chamfer(pts_stage, pts_ref):.4f}")
        save_surf(pts_stage, simps_mov, moved_out, stage, indent="    ",
                      labs=labs_out, labels=labels_mov, z_coords=z_coords)

        if args.plot:
            show(mesh_ref[0], mesh_ref[1], mesh_mov[0], mesh_mov[1], stage)

    if not args.no_deformable:

        if fitted is not None and not fitted.all():
            mesh_ref = utils.extract_polyline(mesh_ref, np.isin(labs_ref, paired))
            mesh_mov = utils.extract_polyline(mesh_mov, fitted)
            print(f"    fitting {int(fitted.sum())} of {fitted.size} points, carrying the rest")

        for i, (lr, wreg, sigma, cpts_ratio) in enumerate(zip(*schedule)):

            fit_fun = energy.point2plane(agg="mean", alpha=-2, scale=0.01, bidir=True)
            regul_fun = energy.alap(transfo="similarity", l_norm=2)
            regul_fun.set_neighs(mesh_mov[1], mesh_mov[0].shape[0])
            # regul_fun = energy.grad_disp(l_norm=2)
            print(f"  - deformable (sigma={sigma})...", end="\n" if verbose else " ", flush=True)
            t = time.time()

            reg = register.reg_deformable(niter=args.niter, fit_fun=fit_fun, regul_fun=regul_fun,
                                          lr=lr, wreg=wreg, sigma=sigma, int_steps=args.int_steps, rk=2,
                                          cpts_ratio=cpts_ratio, verbose=verbose)

            transfo, mesh_mov, loss = reg.compute(mesh_ref, mesh_mov)
            transfos.append(transfo)
            losses.append(loss)
            pts_stage = moved_pts(transfos, pts_mov, mesh_mov, fitted)
            print(f"done in {time.time() - t:.2f} s, dist: {utils.chamfer(pts_stage, pts_ref):.4f}")
            save_surf(pts_stage, simps_mov, moved_out, f"deformable{i}", indent="    ",
                      labs=labs_out, labels=labels_mov, z_coords=z_coords)

            if args.plot:
                show(mesh_ref[0], mesh_ref[1], mesh_mov[0], mesh_mov[1], f"deformable sigma={sigma}")

    final_pts = moved_pts(transfos, pts_mov, mesh_mov, fitted)
    print(f"\nfinal dist: {utils.chamfer(final_pts, pts_ref):.4f}")


    # --- save ---

    save_surf(final_pts, simps_mov, moved_out, labs=labs_out, labels=labels_mov,
              z_coords=z_coords)

    with open(pkl_out, "wb") as f:
        pickle.dump({"transfos": transfos}, f)
    print(f"saved: {pkl_out}")

    if args.plot and losses:
        import matplotlib.pyplot as plt

        for i, loss in enumerate(losses):
            plt.subplot(1, len(losses), i + 1)
            plt.plot(loss)
            plt.title(f"sigma={args.sigma[i]}", fontsize=8)
        plt.suptitle("deformable energy")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
