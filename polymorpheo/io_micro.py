import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image

import polymorpheo.plots as plots
import polymorpheo.utils as utils
from polymorpheo.log import get_logger

logger = get_logger(__name__)

Image.MAX_IMAGE_PIXELS = None


class io_micro:

    def __init__(self, datadir, names, spacing=None, npts=None, npts_min=1, regions=None):
        self.npts = npts
        self.npts_min = npts_min
        self.datadir = datadir
        self.names = names
        self.spacing = spacing
        # regions gives the label order: the first is label 1, the second label 2 and so on, which
        # is what decides the correspondence with another labelled surface. Left out, the region
        # names found in the files are sorted, and a series without them is labelled by source.
        self.regions = regions
        self.labels = list(regions) if regions else list(names)
        self.nlabs = len(self.labels)

    def load(self, plot=False):
        series = [load_series(os.path.join(self.datadir, name), name) for name in self.names]
        nslice = max(len(serie) for serie in series)
        self.nslice = nslice

        # region names label the contours when they carry one, input sources otherwise
        named = {lab for serie in series for slce in serie if slce for _, lab in slce}
        if self.regions:
            self.labels = list(self.regions)
            unlisted = sorted(named - set(self.labels))
            if unlisted:
                logger.warning('region(s) %s are not in the given order, their contours are left '
                               'out', ', '.join(unlisted))
        else:
            self.labels = sorted(named) if named - set(self.names) else list(self.names)
        self.nlabs = len(self.labels)
        lab_of = {lab: i + 1 for i, lab in enumerate(self.labels)}

        pts_all = np.vstack([opt for serie in series for slce in serie if slce
                             for opt, lab in slce if lab in lab_of])

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
            opts, labs = [], []
            for serie in series:
                for opt, lab in serie[z] if z < len(serie) and serie[z] else []:
                    if len(opt) < self.npts_min or lab not in lab_of:
                        continue
                    opts.append(opt * self.spacing[:2])
                    labs.append(lab_of[lab])

            opts, labs = self._drop_unresolved(opts, labs, z)

            if len(opts) == 0:
                logger.debug('slice %d: no contour, skipped', z)
                continue

            # one call for the whole slice, so npts is shared out over all its contours
            pts, simps, _, labs = utils.opts_to_contour(opts, npts=self.npts, get_simps=True,
                                                        lab=labs)
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
            logger.info('loaded %d of %d slices (raw %d..%d), %d label(s): %s',
                        len(polylines), nslice, self.slice_idx[0], self.slice_idx[-1],
                        self.nlabs, ', '.join(self.labels))
        else:
            logger.warning('%s: no slice holds a contour of at least %d points',
                           self.names, self.npts_min)

        return polylines, z_coords

    def _drop_unresolved(self, opts, labs, z):
        # npts is shared out between the contours of a slice in proportion to their length, so one
        # far shorter than the others is left with too few points to bound an area. Resampling it
        # to a minimum of 3 only turns it into a sliver of a triangle, whose local similarity fit
        # is singular and takes the deformable energies to NaN, so it is dropped instead.
        if self.npts is None:
            return opts, labs

        lengths = np.array([np.linalg.norm(np.diff(opt, axis=0), axis=1).sum() for opt in opts])
        keep = [int(self.npts * length / lengths.sum()) >= 3 for length in lengths]
        for k, (opt, lab) in enumerate(zip(opts, labs)):
            if not keep[k]:
                logger.debug('slice %d, label %d: contour of %d points too short to resample, '
                             'dropped', z, lab, len(opt))

        return [opt for opt, k in zip(opts, keep) if k], [lab for lab, k in zip(labs, keep) if k]

    def save(self, meshes, outdir, suffix):
        if suffix != "":
            suffix = "_" + suffix

        for l in range(self.nlabs):
            pts, simps = meshes[l]

            poly = utils.vtkpoly(pts, simps)
            poly = utils.fix_normals_vtkpoly(poly)
            out_file = os.path.join(outdir, self.labels[l] + suffix + ".obj")
            utils.write_vtkpoly(poly, out_file)


def load_series(path, name):
    # Normalise both contour layouts to one list over slices, each None or a list of
    # (contour, label name): a directory of per-slice NPZ files holding region names, or a single
    # NPZ holding the whole series under "registered_contours", which carries no name of its own.
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "slice_*.npz")))
        idx = [int(os.path.basename(file)[len("slice_"):-len(".npz")]) for file in files]
        series = [None] * (max(idx) + 1)
        for file, z in zip(files, idx):
            contours = load_contour(file)
            if contours:
                series[z] = [(opt, region) for region, opts in contours.items() for opt in opts]

        return series

    opts_list = np.load(path + ".npz", allow_pickle=True)["registered_contours"]

    return [[(opt, name) for opt in opts] if opts is not None and len(opts) else None
            for opts in opts_list]


def write_volume(channels, out_path):
    # Three channels written to a nifti as a vector image are read back as a displacement field,
    # that being what the intent code of one says. Colour has a datatype of its own there, RGB24,
    # which sitk has no pixel type for, so it writes the geometry and the voxels go back in under
    # the right datatype afterwards.
    rgb = len(channels) == 3 and str(out_path).endswith((".nii", ".nii.gz"))
    image = channels[0] if len(channels) == 1 else sitk.Compose(
        [sitk.Cast(sitk.Clamp(c, sitk.sitkFloat32, 0, 255), sitk.sitkUInt8) for c in channels])
    sitk.WriteImage(image, str(out_path))

    if rgb:
        import nibabel as nib

        volume = nib.load(str(out_path))
        arr = np.squeeze(np.asanyarray(volume.dataobj)).astype(np.uint8)
        voxels = np.empty(arr.shape[:3], dtype=[(c, "u1") for c in "RGB"])
        for k, c in enumerate("RGB"):
            voxels[c] = arr[..., k]
        nib.save(nib.Nifti1Image(voxels, volume.affine), str(out_path))


def load_meta(datadir, project):
    with open(f"{datadir}/{project}.json") as f:
        return json.load(f)


def load_img(path):
    # one image, or every image of a directory in slice order
    if os.path.isdir(path):
        return [load_img(file) for file in sorted(glob.glob(f"{path}/*.jpg"))]

    return np.asarray(Image.open(path))


def load_contour(path):
    # one slice as {region name: list of (npts, 2) arrays}, or every slice of a directory;
    # a slice holding no contour reads as None
    if os.path.isdir(path):
        return [load_contour(file) for file in sorted(glob.glob(f"{path}/*.npz"))]

    npz = np.load(path, allow_pickle=True)
    contours = {}
    for name, opt in zip(npz["names"], npz["contours"]):
        contours.setdefault(str(name), []).append(opt)

    return contours or None


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
