import os

import matplotlib.pyplot as plt
import numpy as np

import polymorpheo.plots as plots
import polymorpheo.utils as utils
from polymorpheo.log import get_logger

logger = get_logger(__name__)


class io:

    def __init__(self, datadir, names, spacing=None, npts=None, npts_min=1):
        self.npts = npts
        self.npts_min = npts_min
        self.nlabs = len(names)
        self.datadir = datadir
        self.names = names
        self.spacing = spacing

    def load(self, plot=False):
        files = [os.path.join(self.datadir, name + ".npz") for name in self.names]

        opts_lists = [np.load(file, allow_pickle=True)["registered_contours"] for file in files]
        nslice = len(opts_lists[0])
        self.nslice = nslice

        pts_all = np.vstack([opt for opts_list in opts_lists for opts in opts_list if opts is not None for opt in opts])

        if self.spacing is None:
            self.spacing = np.ones(pts_all.shape[1] + 1)
        else:
            self.spacing = np.array(self.spacing)

        scaled = pts_all * self.spacing[:2]
        self.xlim = np.min(scaled[:, 0]), np.max(scaled[:, 0])
        self.ylim = np.min(scaled[:, 1]), np.max(scaled[:, 1])

        polylines = []
        z_coords = []
        self.slice_idx = []
        new_z = 0
        for z in range(nslice):
            polyline = []
            for l in range(self.nlabs):
                opts = opts_lists[l][z]
                if opts is None:
                    continue

                opts = [opt * self.spacing[:2] for opt in opts if len(opt) >= self.npts_min]
                if len(opts) == 0:
                    continue

                polyline_l = utils.opts_to_contour(opts, npts=self.npts, get_simps=True, lab=l + 1)
                polyline.append(polyline_l)

            if len(polyline) == 0:
                logger.debug('slice %d: no contour, skipped', z)
                continue

            pts, simps, _, labs = utils.concat_contours(polyline)
            polyline = pts, simps, None, labs

            polylines.append(polyline)
            z_coords.append(z * self.spacing[2])
            self.slice_idx.append(z)
            logger.debug('slice %d -> %d', z, new_z)
            new_z += 1

            if plot:
                plots.plot_contour(polyline, xlim=self.xlim, ylim=self.ylim)
                plt.title('slice ' + str(z))
                plt.show()

        if polylines:
            logger.info('loaded %d of %d slices (raw %d..%d)', len(polylines), nslice,
                        self.slice_idx[0], self.slice_idx[-1])
        else:
            logger.warning('%s: no slice holds a contour of at least %d points',
                           self.names, self.npts_min)

        return polylines, z_coords

    def save(self, meshes, outdir, suffix):
        if suffix != "":
            suffix = "_" + suffix

        for l in range(self.nlabs):
            pts, simps = meshes[l]

            poly = utils.vtkpoly(pts, simps)
            poly = utils.fix_normals_vtkpoly(poly)
            out_file = os.path.join(outdir, self.names[l] + suffix + ".obj")
            utils.write_vtkpoly(poly, out_file)


def _load_array(file, key):
    # Read a single array out of a .csv (comma separated, '#' comment lines skipped) or .npy file,
    # or out of a .npz holding one array (or, failing that, one named key).

    name = str(file)
    if name.endswith(".csv"):
        return np.loadtxt(file, delimiter=",", ndmin=2)
    if name.endswith(".npy"):
        return np.load(file)

    data = np.load(file)
    if len(data.files) == 1:
        return data[data.files[0]]
    if key in data:
        return data[key]

    raise ValueError(f"{file} holds {len(data.files)} arrays {data.files}, expected a single one "
                     f"or a '{key}' key.")


def load_pts(file, spacing=None, key="pts"):
    # Load a point array, (npts, 2) or (npts, 3). In 3D the third column is the raw slice index,
    # so scaling by spacing turns it into the same physical z as io.load (z * spacing[2]).

    pts = np.asarray(_load_array(file, key), dtype=float)

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"points must have shape (npts, 2) or (npts, 3), got {pts.shape}.")

    if spacing is not None:
        pts = pts * np.asarray(spacing)[:pts.shape[1]]

    return pts


def load_covs(file, spacing=None, key="covs"):
    # Load packed symmetric covariances, (n, 3) in 2D or (n, 6) in 3D (see utils.pack_sym), and
    # return them as full (n, ndims, ndims) matrices. Scaling is S @ covs @ S.T with
    # S = diag(spacing), matching the point scaling.

    flat = np.asarray(_load_array(file, key), dtype=float)

    if flat.ndim != 2:
        raise ValueError(f"covariances must have shape (n, 3) or (n, 6), got {flat.shape}.")

    covs = utils.unpack_sym(flat)

    # eigh, used by the PPD reorientation, assumes a symmetric input.
    covs = 0.5 * (covs + np.swapaxes(covs, -1, -2))

    if spacing is not None:
        scale = np.asarray(spacing)[:covs.shape[-1]]
        covs = covs * np.outer(scale, scale)

    return covs
