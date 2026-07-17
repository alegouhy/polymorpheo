import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np

import polymorpheo
import polymorpheo.utils as utils
from polymorpheo.transfo import (apply_transfo_chain, apply_transfo_chain_ellipsoids,
                                 apply_transfo_chain_jacobian)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a transform chain saved by micro2mri.py to a contour series and/or to a set of "
            "points or ellipsoids sharing the same slice/z layout, without re-running registration."
        )
    )
    parser.add_argument("transfos", type=Path,
                        help="Transform chain pickle produced by micro2mri.py (<name>_transfos.pkl).")
    parser.add_argument("--contours", type=Path, default=None,
                        help="NPZ file containing a contour series to transform "
                             "(same slice/z layout as the original micro2mri.py input). "
                             "Optional if --pts is given.")
    parser.add_argument("--pts", type=Path, default=None,
                        help="Points to transform, as a single (npts, 2) or (npts, 3) array in a "
                             ".csv, .npy or .npz file, in pixel/voxel units. In 3D the third "
                             "column is the raw slice index; in 2D --slice says which slice they "
                             "lie on.")
    parser.add_argument("--covs", "-c", type=Path, default=None,
                        help="Covariances of the --pts, turning them into ellipsoids: a single "
                             "(n, 3) or (n, 6) array packed as the upper triangle, row major, "
                             "i.e. [xx, xy, yy] in 2D and [xx, xy, xz, yy, yz, zz] in 3D. An "
                             "(n, 3) covariance on 3D points is a flat ellipse in the slice plane.")
    parser.add_argument("--slice", type=int, default=None, dest="slice_idx", metavar="IDX",
                        help="Raw slice index the 2D --pts lie on. Required for (npts, 2) points.")
    parser.add_argument("--orientation-only", action="store_true", dest="orientation_only",
                        help="Reorient the ellipsoids with PPD, preserving their eigenvalues, "
                             "instead of transporting the full covariance.")
    parser.add_argument("--chunk-size", type=int, default=None, dest="chunk_size", metavar="N",
                        help="Transform the --pts in blocks of at most N at a time to cap peak "
                             "memory (default: all at once). Does not change the result.")
    parser.add_argument("--outdir", "-o", type=Path, required=True,
                        help="Output directory. Results mirror the --pts format: give a .csv and "
                             "you get <pts>_deformed.csv and <covs>_deformed.csv, otherwise a "
                             "single <pts>_deformed.npz holding both arrays.")
    parser.add_argument("--spacing", "-s", type=float, nargs=3,
                        default=[0.1, 0.1, 1.25], metavar=("SX", "SY", "SZ"),
                        help="Pixel/voxel spacing in x, y, z (default: 0.1 0.1 1.25). "
                             "Must match the spacing used when the transforms were computed.")
    parser.add_argument("--npts", type=int, default=None,
                        help="Resample each contour to this many points before transforming "
                             "(default: no resampling). Contour input only.")
    parser.add_argument("--npts-min", type=int, default=5, dest="npts_min",
                        help="Minimum number of points to keep a contour (default: 5). "
                             "Contour input only.")
    return parser.parse_args()


def fail(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def check_file(path):
    if not path.exists():
        fail(f"file not found: {path}")

    return path


def pts_header(ndims):
    return ",".join("xyz"[:ndims])


def covs_header(ncomp):
    # Built from the same np.triu_indices as utils.pack_sym, so the labels cannot drift from the
    # packing order: [xx, xy, yy] in 2D, [xx, xy, xz, yy, yz, zz] in 3D.
    ndims = int((np.sqrt(8 * ncomp + 1) - 1) / 2)
    i, j = np.triu_indices(ndims)

    return ",".join("xyz"[a] + "xyz"[b] for a, b in zip(i, j))


def save_csv(path, arr, header):
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    print(f"Saved: {path}")


def match_slices(z_phys, z_coords_ref):
    # The saved z_coords are physical (raw slice index * sz, see polymorpheo.io.load) and index the
    # transform chains densely: slices holding no contour at all are absent from the chain.

    chain_idx = np.full(len(z_phys), -1, dtype=int)
    for i, z in enumerate(z_phys):
        match = np.nonzero(np.isclose(z_coords_ref, z))[0]
        if match.size:
            chain_idx[i] = match[0]

    return chain_idx


def transform_contours(input_path, outdir, args, spacing, z_coords_ref, transfos_2d, transfos_3d):
    name = input_path.stem

    io_obj = polymorpheo.io(
        datadir=str(input_path.parent),
        names=[name],
        spacing=spacing,
        npts=args.npts,
        npts_min=args.npts_min,
    )
    polylines, z_coords = io_obj.load()

    polylines_moved = []
    z_matched = []
    for polyline, z in zip(polylines, z_coords):
        match = np.nonzero(np.isclose(z_coords_ref, z))[0]
        if match.size == 0:
            print(f"Warning: no matching transform for slice z={z}, skipping.", file=sys.stderr)
            continue

        pts2d, simps2d, normals2d, labs2d = polyline
        pts2d_moved = np.array(apply_transfo_chain(transfos_2d[int(match[0])], np.array(pts2d)))
        polylines_moved.append((pts2d_moved, simps2d, normals2d, labs2d))
        z_matched.append(z)

    pts3d, simps3d = utils.polylines_2d_3d(polylines_moved, 2, z_matched)
    pts3d_final = np.array(apply_transfo_chain(transfos_3d, pts3d))

    npz_out = outdir / f"{name}_deformed.npz"
    np.savez(npz_out, pts=pts3d_final, simps=np.array(simps3d))
    print(f"Saved: {npz_out}")


def transform_2d(pts, covs, chain_2d, orientation_only):
    # 2D points on a single slice: the 2D chain is the whole pipeline.

    if covs is None:
        return np.array(apply_transfo_chain(chain_2d, pts)), None

    pts_moved, covs_moved = apply_transfo_chain_ellipsoids(
        chain_2d, pts, covs, orientation_only=orientation_only)

    return np.array(pts_moved), np.array(covs_moved)


def transform_3d(pts, covs, chain_idx, transfos_2d, transfos_3d, orientation_only):
    # Slice-stacked points: the per-slice 2D chain acts in plane, then the 3D chain.

    z_phys = pts[:, 2]

    if covs is None:
        xy_moved = np.empty((len(pts), 2))
        for idx in np.unique(chain_idx):
            grp = chain_idx == idx
            xy_moved[grp] = np.array(apply_transfo_chain(transfos_2d[int(idx)], pts[grp, :2]))
        return np.array(apply_transfo_chain(transfos_3d, np.c_[xy_moved, z_phys])), None

    # A 2D covariance embeds as a rank 2 matrix: transporting it keeps that null direction, so the
    # ellipse stays flat while the 3D chain is free to turn it out of plane.
    if covs.shape[-1] == 2:
        covs_3d = np.zeros((len(covs), 3, 3))
        covs_3d[:, :2, :2] = covs
        covs = covs_3d

    xy_moved = np.empty((len(pts), 2))
    jac_2d = np.empty((len(pts), 2, 2))
    for idx in np.unique(chain_idx):
        grp = chain_idx == idx
        moved, jac = apply_transfo_chain_jacobian(transfos_2d[int(idx)], pts[grp, :2])
        xy_moved[grp] = np.array(moved)
        jac_2d[grp] = np.array(jac)

    # The 2D chain maps (x, y, z) -> (T(x, y), z), so its 3D Jacobian is jac_2d with a unit z
    # row/column: the chain is piecewise in z and contributes no d/dz term. Lifting it is what
    # carries the xz/yz terms of a full 3D covariance through the slice transform.
    jac = np.zeros((len(pts), 3, 3))
    jac[:, :2, :2] = jac_2d
    jac[:, 2, 2] = 1.0

    pts_final, jac_3d = apply_transfo_chain_jacobian(transfos_3d, np.c_[xy_moved, z_phys])
    jac = np.array(jac_3d) @ jac

    return np.array(pts_final), np.array(utils.transform_ellipsoids(jac, covs, orientation_only))


def run_chunked(n, chunk_size, transform):
    # Push the rows through in blocks of at most chunk_size so the per-point JAX buffers never span
    # the whole input at once. Each point is transformed independently of the others, so the chunk
    # boundaries do not change the result. transform(sl) returns (pts, covs-or-None) for the rows in
    # sl; the pieces concatenate back in order.
    if not chunk_size or chunk_size >= n:
        return transform(slice(0, n))

    pts_parts, covs_parts = [], []
    for start in range(0, n, chunk_size):
        pts_part, covs_part = transform(slice(start, min(start + chunk_size, n)))
        pts_parts.append(pts_part)
        covs_parts.append(covs_part)
    print(f"Transformed {n} points in {len(pts_parts)} chunks of up to {chunk_size}.")

    covs_out = None if covs_parts[0] is None else np.concatenate(covs_parts)
    return np.concatenate(pts_parts), covs_out


def transform_pts(pts_path, covs_path, outdir, args, spacing, z_coords_ref, transfos_2d, transfos_3d):
    name = pts_path.stem

    try:
        pts = polymorpheo.load_pts(pts_path, spacing)
        covs = polymorpheo.load_covs(covs_path, spacing) if covs_path else None
    except ValueError as e:
        fail(str(e))

    ndims = pts.shape[1]
    if covs is not None:
        if len(covs) != len(pts):
            fail(f"got {len(covs)} covariances for {len(pts)} points.")
        if covs.shape[-1] > ndims:
            fail("3D covariances need 3D points: the 2D chain cannot transport them.")

        npd = int(np.sum(np.any(np.linalg.eigvalsh(covs) <= 0, axis=-1)))
        if npd:
            print(f"Warning: {npd} covariance(s) are not positive definite.", file=sys.stderr)

    if ndims == 2:
        if args.slice_idx is None:
            fail("2D points need --slice to say which slice they lie on.")
        chain_idx = match_slices(np.array([args.slice_idx * spacing[2]]), z_coords_ref)
        if chain_idx[0] < 0:
            fail(f"no matching transform for slice {args.slice_idx}.")
        chain_2d = transfos_2d[int(chain_idx[0])]
        pts_final, covs_final = run_chunked(
            len(pts), args.chunk_size,
            lambda s: transform_2d(pts[s], None if covs is None else covs[s],
                                   chain_2d, args.orientation_only))
    else:
        if args.slice_idx is not None:
            fail("--slice only applies to 2D points; 3D points carry their slice in column 3.")
        chain_idx = match_slices(pts[:, 2], z_coords_ref)
        keep = chain_idx >= 0
        if not np.all(keep):
            missing = np.unique(pts[~keep, 2] / spacing[2])
            print(f"Warning: no matching transform for slice(s) {missing.tolist()}, skipping "
                  f"{int(np.sum(~keep))} point(s). Column 3 must be a raw slice index.",
                  file=sys.stderr)
        pts, chain_idx = pts[keep], chain_idx[keep]
        covs = covs[keep] if covs is not None else None
        pts_final, covs_final = run_chunked(
            len(pts), args.chunk_size,
            lambda s: transform_3d(pts[s], None if covs is None else covs[s],
                                   chain_idx[s], transfos_2d, transfos_3d, args.orientation_only))

    pts_final = np.asarray(pts_final, dtype=float)
    covs_final = None if covs_final is None else np.asarray(utils.pack_sym(covs_final), dtype=float)

    # The output mirrors the format the points came in as. A csv holds one array, so the
    # covariances land beside the points rather than in the same file.
    if pts_path.suffix == ".csv":
        save_csv(outdir / f"{name}_deformed.csv", pts_final, pts_header(pts_final.shape[1]))
        if covs_final is not None:
            save_csv(outdir / f"{covs_path.stem}_deformed.csv", covs_final,
                     covs_header(covs_final.shape[1]))
    else:
        out = {"pts": pts_final}
        if covs_final is not None:
            out["covs"] = covs_final
        npz_out = outdir / f"{name}_deformed.npz"
        np.savez(npz_out, **out)
        print(f"Saved: {npz_out}")


def main():
    args = parse_args()

    if args.covs is not None and args.pts is None:
        fail("--covs needs --pts.")
    if args.chunk_size is not None and args.chunk_size <= 0:
        fail("--chunk-size must be a positive integer.")
    if args.contours is None and args.pts is None:
        fail("nothing to transform, provide --contours and/or --pts.")

    input_path = check_file(args.contours.resolve()) if args.contours else None
    pts_path = check_file(args.pts.resolve()) if args.pts else None
    covs_path = check_file(args.covs.resolve()) if args.covs else None
    transfos_path = check_file(args.transfos.resolve())

    if input_path and pts_path and input_path.stem == pts_path.stem:
        fail(f"'{input_path.stem}' is used by both inputs, their outputs would collide.")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    spacing = np.array(args.spacing)

    with open(transfos_path, "rb") as f:
        t = pickle.load(f)
    z_coords_ref = np.asarray(t["z_coords"])
    transfos_2d = t["transfos_2d"]
    transfos_3d = t["transfos_3d"]

    if input_path:
        transform_contours(input_path, outdir, args, spacing, z_coords_ref, transfos_2d, transfos_3d)

    if pts_path:
        transform_pts(pts_path, covs_path, outdir, args, spacing, z_coords_ref, transfos_2d,
                      transfos_3d)


if __name__ == "__main__":
    main()
