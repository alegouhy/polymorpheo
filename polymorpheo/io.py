import os

import matplotlib.pyplot as plt
import numpy as np

import polymorpheo.utils as utils

from .log import get_logger

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

        pts_all = np.vstack([opt for opts_list in opts_lists for opts in opts_list if opts is not None for opt in opts])
        self.xlim = np.min(pts_all[:, 0]), np.max(pts_all[:, 0])
        self.ylim = np.min(pts_all[:, 1]), np.max(pts_all[:, 1])

        if self.spacing is None:
            self.spacing = np.ones(pts_all.shape[1] + 1)
        else:
            self.spacing = np.array(self.spacing)

        logger.info("Loading contours from %s", files)
        polylines = []
        z_coords = []
        for z in range(nslice):
            logger.debug("processing slice index=%d", z)
            polyline = []
            for l in range(self.nlabs):
                opts = opts_lists[l][z]
                if opts is None:
                    continue

                opts = [opt * self.spacing[:2] for opt in opts if len(opt) >= self.npts_min]

                polyline_l = utils.opts_to_contour(opts, npts=self.npts, get_simps=True, lab=l + 1)
                polyline.append(polyline_l)

            if len(polyline) == 0:
                continue

            pts, simps, _, labs = utils.concat_contours(polyline)
            polyline = pts, simps, None, labs

            polylines.append(polyline)
            z_coords.append(z * self.spacing[2])

            if plot:
                logger.info("Plotting slice %d/%d", z + 1, nslice)
                utils.plot_contour(polyline, xlim=self.xlim, ylim=self.ylim)
                plt.title('slice ' + str(z))
                plt.show()

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
