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
from polymorpheo.transfo import apply_transfo_chain


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a transform chain saved by micro2mri.py to a different contour series "
            "sharing the same slice/z layout, without re-running registration."
        )
    )
    parser.add_argument("input", type=Path,
                        help="Input NPZ file containing the contour series to transform "
                             "(same slice/z layout as the original micro2mri.py input).")
    parser.add_argument("transfos", type=Path,
                        help="Transform chain pickle produced by micro2mri.py (<name>_transfos.pkl).")
    parser.add_argument("--outdir", "-o", type=Path, required=True,
                        help="Output directory.")
    parser.add_argument("--spacing", "-s", type=float, nargs=3,
                        default=[0.1, 0.1, 1.25], metavar=("SX", "SY", "SZ"),
                        help="Pixel/voxel spacing in x, y, z (default: 0.1 0.1 1.25). "
                             "Must match the spacing used when the transforms were computed.")
    parser.add_argument("--npts", type=int, default=None,
                        help="Resample each contour to this many points before transforming "
                             "(default: no resampling).")
    parser.add_argument("--npts-min", type=int, default=5, dest="npts_min",
                        help="Minimum number of points to keep a contour (default: 5).")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    transfos_path = args.transfos.resolve()
    if not transfos_path.exists():
        print(f"Error: file not found: {transfos_path}", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    name = input_path.stem
    spacing = np.array(args.spacing)

    with open(transfos_path, "rb") as f:
        t = pickle.load(f)
    z_coords_ref = np.asarray(t["z_coords"])
    transfos_2d = t["transfos_2d"]
    transfos_3d = t["transfos_3d"]

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


if __name__ == "__main__":
    main()
