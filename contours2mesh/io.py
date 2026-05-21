import os

import matplotlib.pyplot as plt
import numpy as np

import contours2mesh.utils as utils

from .log import get_logger

logger = get_logger(__name__)


class io:
    def __init__(
        self, datadir, names, spacing=np.ones(3), npts=None, npts_min=1, normalise=False
    ):
        self.npts = npts
        self.npts_min = npts_min
        self.nlabs = len(names)
        self.datadir = datadir
        self.names = names
        self.spacing = np.array(spacing)
        self.permut = [0, 1, 2]
        self.normalise = normalise

    def load(self, plot=False):
        files = [os.path.join(self.datadir, name + ".npz") for name in self.names]

        opts_lists = [
            np.load(file, allow_pickle=True)["registered_contours"] for file in files
        ]
        nslice = len(opts_lists[0])

        pts_all = np.vstack(
            [
                opt
                for opts_list in opts_lists
                for opts in opts_list
                if opts is not None
                for opt in opts
            ]
        )
        if self.normalise:
            self.pts_mu = np.mean(pts_all, axis=0)
            self.pts_amp = np.max(np.abs(pts_all - self.pts_mu))
        else:
            self.pts_mu = np.zeros(pts_all.shape[1])
            self.pts_amp = 1

        self.xlim = np.min(pts_all[:, 0]), np.max(pts_all[:, 0])
        self.ylim = np.min(pts_all[:, 1]), np.max(pts_all[:, 1])

        logger.info("Loading contours from %s", files)
        ii = 0
        polylines = []
        for i in range(nslice):
            logger.debug("processing slice index=%d output_index=%d", i, ii)
            polyline = []
            for j in range(self.nlabs):
                opts = opts_lists[j][i]
                if opts is None:
                    continue
                opts = [opt for opt in opts if len(opt) >= self.npts_min]
                polyline_l = utils.opts_to_contour(
                    opts, npts=self.npts, get_simps=True, lab=j + 1
                )
                polyline.append(polyline_l)
            if len(polyline) == 0:
                continue

            pts, simps, _, labs = utils.concat_contours(polyline)
            pts = pts * self.spacing[:2]

            if self.normalise:
                pts = (pts - self.pts_mu) / self.pts_amp

            polyline = pts, simps, None, labs
            polylines.append(polyline)
            ii += 1
            if plot:
                logger.info("Plotting slice %d/%d", i + 1, nslice)
                utils.plot_contour(polyline, xlim=self.xlim, ylim=self.ylim)
                plt.title(str(i) + ", " + str(ii))
                plt.show()

        return polylines

    def load2(self, axis, nslice, ext="obj", plot=False):
        poly = utils.read_vtkpoly(os.path.join(self.datadir, self.names[0] + "." + ext))
        bounds = np.reshape(poly.GetBounds(), (3, 2))
        mu = np.array(poly.GetCenter())

        self.pts_amp = np.max(np.diff(np.delete(bounds, axis, axis=0))) / 2
        self.pts_mu = np.delete(mu, axis)
        pos_sli = np.linspace(bounds[axis, 0], bounds[axis, 1], nslice + 2)[1:-1]
        self.spacing = np.ones(3)
        self.spacing[axis] = pos_sli[1] - pos_sli[0]
        self.permut = list(range(axis)) + [2] + list(range(axis, 2))

        polylines = []
        for i in range(nslice):
            poly_sli = utils.slice_vtkpoly(poly, pos_sli[i], axis=axis)
            pts = np.array(poly_sli.GetPoints().GetData())
            pts = np.delete(pts, axis, axis=1)
            if self.normalise:
                pts = (pts - self.pts_mu) / self.pts_amp
            nsimps = poly_sli.GetNumberOfLines()
            simps = np.array(poly_sli.GetLines().GetData()).reshape((nsimps, -1))
            simps = np.array(simps, dtype=np.int32).reshape((nsimps, -1))[:, 1:]

            opts = utils.contours2opts(pts, simps)
            polyline = utils.opts_to_contour(opts, npts=self.npts)
            polylines.append(polyline)

            if plot:
                logger.info("Plotting vtk slice %d/%d", i + 1, nslice)
                utils.plot_contour(polyline)
                plt.title(str(i))
                plt.show()

        return polylines

    def save(self, meshes, outdir, suffix):
        if suffix != "":
            suffix = "_" + suffix

        for i in range(self.nlabs):
            pts, simps = meshes[i]

            if self.normalise:
                pts[:, :2] = pts[:, :2] * self.pts_amp + self.pts_mu
            pts = pts[:, self.permut]
            pts = pts * self.spacing

            poly = utils.vtkpoly(pts, simps)
            poly = utils.fix_normals_vtkpoly(poly)
            out_file = os.path.join(outdir, self.names[i] + suffix + ".obj")
            utils.write_vtkpoly(poly, out_file)
