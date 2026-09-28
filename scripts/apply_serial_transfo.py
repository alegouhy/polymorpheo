import argparse
import os
import pickle
import sys
from pathlib import Path

from tqdm.asyncio import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import SimpleITK as sitk

import polymorpheo
import polymorpheo.utils as utils
from polymorpheo.transfo import apply_transfo_chain, apply_transfo_chain_ellipsoids, apply_transfo_chain_jacobian


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply saved transform chains to a contour series and/or to a set of points or "
            "ellipsoids sharing the same slice/z layout, without re-running registration. The "
            "chains come either from align_slices.py and reg_surf.py (--transfos-2d, "
            "--transfos-3d, one or both) or from a single micro2mri.py pickle."
        )
    )
    parser.add_argument("transfos", type=Path, nargs="?", default=None,
                        help="Combined transform chain pickle produced by micro2mri.py "
                             "(<name>_transfos.pkl). Mutually exclusive with --transfos-2d "
                             "and --transfos-3d.")
    parser.add_argument("--transfos-2d", type=Path, default=None, dest="transfos_2d",
                        help="Per-slice in-plane transform chains from align_slices.py "
                             "(<name>_transfos2d.pkl).")
    parser.add_argument("--transfos-3d", type=Path, default=None, dest="transfos_3d",
                        help="Stack-to-surface transform chain from reg_surf.py "
                             "(<name>_transfos.pkl).")
    parser.add_argument("--contours", type=Path, default=None,
                        help="NPZ file containing a contour series to transform "
                             "(same slice/z layout as the series the chains were computed on). "
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
    parser.add_argument("--chunk-size", type=int, default=1000, dest="chunk_size", metavar="N",
                        help="Transform the --pts in blocks of at most N at a time to cap peak "
                             "memory (default: %(default)s). Does not change the result.")
    parser.add_argument("--images", type=Path, default=None, metavar="DIR",
                        help="Directory of slice images, one per slice as fetch_micro writes them, "
                             "to resample through the chains. Needs --geom and --out-image.")
    parser.add_argument("--geom", type=Path, default=None, metavar="IMAGE",
                        help="Image whose geometry the resampled slices are written into: its "
                             "size, spacing, origin and direction are the ones of the output.")
    parser.add_argument("--out-image", type=Path, default=None, dest="out_image", metavar="PATH",
                        help="Output for the resampled slices, in any format sitk writes.")
    parser.add_argument("--field-step", type=int, default=4, dest="field_step", metavar="N",
                        help="Sample the 3D displacement field every N voxels of the output "
                             "geometry, sitk interpolating it in between (default: 4). The chain "
                             "costs about 13 us a point, so a step of 1 is minutes of work for a "
                             "field the transform is smooth enough not to need.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output for the transformed contours, required with --contours. Its "
                             "extension picks the format: npz, vtp or obj, all of which keep the "
                             "edges and the name of the region each contour belongs to, and of "
                             "which only the npz also carries the slice positions.")
    parser.add_argument("--outdir", "-o", type=Path, default=None,
                        help="Output directory for the --pts results, which mirror its format: "
                             "give a .csv and you get <pts>_deformed.csv and <covs>_deformed.csv, "
                             "otherwise a single <pts>_deformed.npz holding both arrays.")
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


def transform_contours(input_path, out_path, args, spacing, z_coords_ref, transfos_2d,
                       transfos_3d):
    name = input_path.stem

    io_obj = polymorpheo.io_micro(
        datadir=str(input_path.parent),
        names=[name],
        spacing=spacing,
        npts=args.npts,
        npts_min=args.npts_min,
    )
    polylines, z_coords = io_obj.load()

    if transfos_2d is None:
        # Nothing acts in plane: the series is stacked as it comes and only the 3D chain applies.
        polylines_moved, z_matched = polylines, z_coords
    else:
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

    # the regions come through under their own names, so the deformed contours stay as readable as
    # the series they were replayed from
    labs = np.concatenate([polyline[3] for polyline in polylines_moved])
    utils.write_surf(str(out_path), pts3d_final, simps3d, labs=labs, labels=io_obj.labels,
                     z_coords=np.array(z_matched))
    print(f"Saved: {out_path}")


def displacement(chain, pts, invert=True):
    # what sitk asks of a transform, a displacement from each point rather than its image
    return np.array(apply_transfo_chain(chain, pts, invert=invert)) - pts


def grid_pts(image, step=1):
    # the physical point of every voxel of an image, or of every step-th one along each axis
    size = [len(range(0, n, step)) for n in image.GetSize()]
    idx = np.stack(np.meshgrid(*[np.arange(n) * step for n in size], indexing="ij"), axis=-1)
    origin = np.array(image.TransformIndexToPhysicalPoint([0] * image.GetDimension()))
    # one index along each axis, the step being carried by the indices themselves
    axes = np.array([np.array(image.TransformIndexToPhysicalPoint(
        [1 if k == d else 0 for k in range(image.GetDimension())])) - origin
        for d in range(image.GetDimension())])

    return size, origin + idx.reshape(-1, image.GetDimension()) @ axes


def field_transfo(pts, disp, size, image, step):
    # a displacement field sampled on a grid of its own, which sitk interpolates over the output.
    # The points run fastest along the last axis, where an array sitk reads runs fastest along the
    # first, so the axes are reversed on the way in.
    disp = disp.reshape(*size, -1).transpose(*range(len(size))[::-1], len(size))
    field = sitk.GetImageFromArray(np.ascontiguousarray(disp, dtype=np.float64), isVector=True)
    field.SetOrigin(image.GetOrigin())
    field.SetDirection(image.GetDirection())
    field.SetSpacing([sp * step for sp in image.GetSpacing()])

    return sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))


def resample(image, ref, transfo, background):
    return sitk.Resample(image, ref, transfo, sitk.sitkLinear, background, image.GetPixelID())


def slice_image(arr, spacing):
    # the slice as sitk sees it: a 2D image whose physical coordinates are the ones the chains
    # were computed in, pixels scaled by the spacing with the origin on the first one
    image = sitk.GetImageFromArray(arr.astype(np.float32))
    image.SetSpacing([float(spacing[0]), float(spacing[1])])

    return image


def transform_images(imgdir, geom_path, out_path, args, spacing, z_coords_ref, transfos_2d,
                     transfos_3d):
    """Resample a set of slice images through the chains into the geometry of another image.

    Each slice is warped in plane by its own chain, which leaves the series stacked on a regular
    grid, and that stack is then resampled into the given geometry through the 3D chain. Both are
    pull operations, so the chains are evaluated the other way round, and sitk does the sampling
    from a displacement field rather than the chain being asked for every voxel.
    """
    images = polymorpheo.load_img(str(imgdir))
    if not images:
        fail(f"{imgdir} holds no image.")
    nchan = images[0].shape[2] if images[0].ndim == 3 else 1
    background = float(np.median([np.median(im) for im in images[:5]]))

    # the chains are held for the slices that carried a contour, at their position in the stack
    slice_of = {int(round(z / spacing[2])): i for i, z in enumerate(z_coords_ref)}
    kept = sorted(z for z in slice_of if z < len(images))
    if not kept:
        fail("no slice of the series has both an image and a transform.")
    print(f"Warping {len(kept)} slices in plane...", flush=True)

    aligned = np.full((len(kept), images[kept[0]].shape[0], images[kept[0]].shape[1], nchan),
                      background, dtype=np.float32)
    for k, z in enumerate(kept):
        arr = images[z]
        arr = arr[..., None] if arr.ndim == 2 else arr
        plane = slice_image(arr[..., 0], spacing)
        size, pts = grid_pts(plane)
        transfo = field_transfo(pts, displacement(transfos_2d[slice_of[z]], pts), size, plane, 1)
        for c in range(nchan):
            moved = resample(slice_image(arr[..., c], spacing), plane, transfo, background)
            aligned[k, ..., c] = sitk.GetArrayFromImage(moved)

    # the warped slices sit on a regular grid, so the stack is an image the 3D chain can pull from
    stack = sitk.GetImageFromArray(np.zeros(aligned.shape[:3], dtype=np.float32))
    stack.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])
    stack.SetOrigin([0.0, 0.0, float(kept[0] * spacing[2])])

    geom = sitk.ReadImage(str(geom_path))
    size, pts = grid_pts(geom, args.field_step)
    print(f"Resampling into {geom.GetSize()} voxels, from a {'x'.join(map(str, size))} field...",
          flush=True)
    transfo = field_transfo(pts, displacement(transfos_3d, pts), size, geom, args.field_step)

    ref = sitk.Cast(geom, sitk.sitkFloat32)
    out = []
    for c in range(nchan):
        vol = sitk.GetImageFromArray(np.ascontiguousarray(aligned[..., c]))
        vol.CopyInformation(stack)
        out.append(resample(vol, ref, transfo, background))

    polymorpheo.write_volume(out, out_path)
    print(f"Saved: {out_path}")


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
        if transfos_2d is None:
            return np.array(apply_transfo_chain(transfos_3d, pts)), None

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

    if transfos_2d is None:
        xy_moved = pts[:, :2]
        jac_2d = np.broadcast_to(np.eye(2), (len(pts), 2, 2))
    else:
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
    for start in tqdm(range(0, n, chunk_size)):
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
        if transfos_2d is None:
            fail("2D points live in a slice plane, so they need the in-plane chains: give "
                 "--transfos-2d, or 3D points the 3D chain can act on.")
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
        if transfos_2d is None:
            # No in-plane stage to address per slice, so every point goes straight to the 3D chain.
            chain_idx = np.zeros(len(pts), dtype=int)
        else:
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


def load_chains(args):
    # Either a single micro2mri pickle, or the align_slices and reg_surf ones, one or both. A
    # missing 2D chain means nothing acts in plane, a missing 3D one that nothing acts on the stack.

    if args.transfos is not None:
        if args.transfos_2d is not None or args.transfos_3d is not None:
            fail("give either the combined pickle or --transfos-2d/--transfos-3d, not both.")
        with open(check_file(args.transfos.resolve()), "rb") as f:
            t = pickle.load(f)
        return np.asarray(t["z_coords"]), t["transfos_2d"], t["transfos_3d"]

    if args.transfos_2d is None and args.transfos_3d is None:
        fail("no transform to apply, provide --transfos-2d and/or --transfos-3d.")

    z_coords_ref, transfos_2d, transfos_3d = None, None, []

    if args.transfos_2d is not None:
        with open(check_file(args.transfos_2d.resolve()), "rb") as f:
            t = pickle.load(f)
        z_coords_ref = np.asarray(t["z_coords"])
        transfos_2d = t["transfos_2d"]

    if args.transfos_3d is not None:
        with open(check_file(args.transfos_3d.resolve()), "rb") as f:
            t = pickle.load(f)
        transfos_3d = t["transfos"]

    return z_coords_ref, transfos_2d, transfos_3d


def main():
    args = parse_args()

    if args.covs is not None and args.pts is None:
        fail("--covs needs --pts.")
    if args.chunk_size is not None and args.chunk_size <= 0:
        fail("--chunk-size must be a positive integer.")
    if args.contours is None and args.pts is None and args.images is None:
        fail("nothing to transform, provide --contours, --pts and/or --images.")
    if args.images is not None and (args.geom is None or args.out_image is None):
        fail("--images needs --geom, the geometry to resample into, and --out-image.")
    if args.field_step < 1:
        fail("--field-step must be a positive integer.")
    if args.contours is not None and args.out is None:
        fail("--contours needs --out, the file to write the transformed contours to.")
    if args.pts is not None and args.outdir is None:
        fail("--pts needs --outdir, the directory to write its results to.")

    input_path = check_file(args.contours.resolve()) if args.contours else None
    pts_path = check_file(args.pts.resolve()) if args.pts else None
    covs_path = check_file(args.covs.resolve()) if args.covs else None
    z_coords_ref, transfos_2d, transfos_3d = load_chains(args)
    if args.images is not None and transfos_3d is None:
        fail("--images needs the 3D chain, to reach the geometry of another image.")

    spacing = np.array(args.spacing)

    if input_path:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        transform_contours(input_path, out_path, args, spacing, z_coords_ref, transfos_2d,
                           transfos_3d)

    if args.images is not None:
        imgdir = check_file(args.images.resolve())
        geom_path = check_file(args.geom.resolve())
        out_image = args.out_image.resolve()
        out_image.parent.mkdir(parents=True, exist_ok=True)
        transform_images(imgdir, geom_path, out_image, args, spacing, z_coords_ref, transfos_2d,
                         transfos_3d)

    if pts_path:
        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        transform_pts(pts_path, covs_path, outdir, args, spacing, z_coords_ref, transfos_2d,
                      transfos_3d)


if __name__ == "__main__":
    main()
