import copy

import numpy as np
from scipy.linalg import expm, logm

import polymorpheo.energy as energy
import polymorpheo.register as register
import polymorpheo.transfo as transfo_ops
import polymorpheo.utils as utils


class bridge_contours:
    def __init__(self, thr_conn=1/3, greedy=False, sealed=True):
        self.thr_conn = thr_conn
        self.greedy = greedy
        self.sealed = sealed

    def compute(self, polylines, z_coords=None):
        nslices = len(polylines)
        is_lab = polylines[0][3] is not None
        if not is_lab:
            labs = [1]
        else:
            labs = np.unique(np.concatenate([polyline[3] for polyline in polylines]))
        if z_coords is None:
            z_coords = np.arange(nslices)

        if isinstance(self.thr_conn, (list, tuple, np.ndarray)):
            thr_conn = self.thr_conn
        else:
            thr_conn = [self.thr_conn] * len(labs)

        meshes = []
        for l in range(len(labs)):
            if is_lab:
                polylines_l = [utils.extract_polyline(polyline, polyline[3] == l + 1) for polyline in polylines]
            else:
                polylines_l = polylines
            opts_list = []

            for i in range(nslices):
                polyline_l = polylines_l[i]
                opts = utils.contours2opts(np.array(polyline_l[0]), np.array(polyline_l[1]))
                opts_list.append(opts)

            pts, simps = utils.bridge_contours(
                opts_list,
                z_coords,
                greedy=self.greedy,
                thr_conn=thr_conn[l],
                sealed=self.sealed,
            )
            meshes.append([pts, simps])

        return meshes


def register_contour_slices(
    polylines,
    xlim,
    ylim,
    propag="jacobi",
    multi="simultaneous",
    bidir=True,
    no_deformable=False,
    plot=False,
    verbose=True,
):
    """Rigid -> affine -> (optional) deformable registration of a contour slice series."""

    reg = register_slices(
        "rigid", propag=propag, multi=multi,
        init="centroid", bidir=bidir,
        xlim=xlim, ylim=ylim, verbose=verbose,
    )
    polylines, _ = reg.compute(polylines)

    reg = register_slices(
        "affine", propag=propag, multi=multi,
        init="identity", bidir=bidir,
        xlim=xlim, ylim=ylim, verbose=verbose,
    )
    polylines, _ = reg.compute(polylines)

    if not no_deformable:
        fit_fun = energy.point2point(agg="mean", bidir=bidir)
        regul_fun = energy.grad_disp(l_norm=2)
        reg = register_slices(
            "deformable", propag=propag, multi=multi,
            fit_fun=fit_fun, regul_fun=regul_fun,
            niter=1, icp_niter=50, lr=1e-2, wreg=5e-1, sigma=1e-1,
            int_steps=16, tol=1e-5,
            xlim=xlim, ylim=ylim, plot=plot, verbose=verbose,
        )
        polylines, _ = reg.compute(polylines)

    return polylines


class register_slices:
    def __init__(
        self,
        transfo_type,
        propag="gs",          # "gs" (Gauss-Seidel) | "jacobi"
        start="middle",       # "middle" | "first"  (GS only; ignored for jacobi)
        multi="simultaneous", # "independent_avg" | "simultaneous"
        n_neigh=2,            # 1 | 2
        init="identity",
        niter=1,
        degree=2,
        icp_niter=30,
        bidir=True,
        fit_fun=None,
        regul_fun=None,
        lr=1e-2,
        wreg=1e-1,
        sigma=1e-1,
        int_steps=16,
        tol=1e-6,
        plot=False,
        xlim=None,
        ylim=None,
        verbose=True,
    ):
        self.propag  = propag
        self.start   = start
        self.multi   = multi
        self.n_neigh = n_neigh
        self.niter   = niter
        self.verbose = verbose

        if transfo_type in ("rig", "rigid", "aff", "affine"):
            self.transfo_type = "linear"
            self.reg = register.reg_linear(
                niter=icp_niter,
                transfo=transfo_type,
                init=init,
                se=True,
                bidir=bidir,
                plot=plot,
                xlim=xlim,
                ylim=ylim,
                verbose=verbose,
            )

        elif transfo_type in ("poly", "polynomial"):
            self.transfo_type = "polynomial"
            self.reg = register.reg_polynom(
                niter=icp_niter,
                degree=degree,
                init=init,
                se=True,
                bidir=bidir,
                plot=plot,
                xlim=xlim,
                ylim=ylim,
                verbose=verbose,
            )

        elif transfo_type == "deformable":
            self.transfo_type = "deformable"
            self.reg = register.reg_deformable(
                niter=icp_niter,
                fit_fun=fit_fun,
                regul_fun=regul_fun,
                lr=lr,
                wreg=wreg,
                sigma=sigma,
                int_steps=int_steps,
                tol=tol,
                plot=plot,
                xlim=xlim,
                ylim=ylim,
                verbose=verbose,
            )

    def _sweep_passes(self, nslice):
        mid = nslice // 2
        if self.propag == "jacobi":
            even = [(i, "fwd") for i in range(0, nslice, 2)]
            odd  = [(i, "fwd") for i in range(1, nslice, 2)]
            return [even, odd]
        elif self.start == "first":
            return [[(i, "fwd") for i in range(1, nslice)]]
        else:  # middle
            fwd = [(i, "fwd") for i in range(mid + 1, nslice)]
            bwd = [(i, "bwd") for i in reversed(range(0, mid))]
            return [fwd, bwd]

    def _get_refs(self, polylines, polylines0, i, nslice, direction):
        def prev_ref(j):
            if j <= 0: return None
            return polylines[j-1] if self.propag == "gs" and direction == "fwd" else polylines0[j-1]

        def next_ref(j):
            if j >= nslice - 1: return None
            return polylines[j+1] if self.propag == "gs" and direction == "bwd" else polylines0[j+1]

        if self.n_neigh == 1:
            ref = prev_ref(i) if direction == "fwd" else next_ref(i)
            return [ref] if ref is not None else []
        else:
            return [r for r in [prev_ref(i), next_ref(i)] if r is not None]

    def _run_reg(self, refs, mov_polyline):
        ref_arg = refs[0] if len(refs) == 1 else refs
        if self.transfo_type == "linear":
            transfo, moved = self.reg.compute(ref_arg, mov_polyline)
            return moved, transfo
        elif self.transfo_type == "polynomial":
            _, moved = self.reg.compute(ref_arg, mov_polyline)
            return moved, None
        else:
            transfo, moved, _ = self.reg.compute(ref_arg, mov_polyline)
            return moved, transfo

    def _independent_avg(self, refs, mov_polyline):
        mov_pts, mov_simps, _, mov_labs = mov_polyline
        ndims = int(mov_pts.shape[1])
        ref_a = refs[0]
        ref_b = refs[1] if len(refs) > 1 else None

        if self.transfo_type == "linear":
            aff_a, _ = self.reg.compute(ref_a, mov_polyline)
            T_a = utils.aff_hmgn(aff_a.lin, aff_a.trans)
            if ref_b is not None:
                aff_b, _ = self.reg.compute(ref_b, mov_polyline)
                T_b = utils.aff_hmgn(aff_b.lin, aff_b.trans)
            else:
                T_b = np.eye(ndims + 1)
            aff = np.real(expm((logm(T_a, disp=False)[0] + logm(T_b, disp=False)[0]) / 2))
            lin, trans = utils.aff_dehmgn(aff)
            moved_pts = mov_pts @ lin.T + trans
            transfo = transfo_ops.affine()
            transfo.set_params(lin, trans)
            return (moved_pts, mov_simps, None, mov_labs), transfo

        elif self.transfo_type == "polynomial":
            _, moved_a = self.reg.compute(ref_a, mov_polyline)
            if ref_b is not None:
                _, moved_b = self.reg.compute(ref_b, mov_polyline)
                moved_pts = (np.array(moved_a[0]) + np.array(moved_b[0])) / 2
            else:
                moved_pts = (np.array(moved_a[0]) + np.array(mov_pts)) / 2
            return (moved_pts, mov_simps, None, mov_labs), None

        else:  # deformable
            poly_a, _, _ = self.reg.compute(ref_a, mov_polyline)
            if ref_b is not None:
                poly_b, _, _ = self.reg.compute(ref_b, mov_polyline)
                theta_avg = (poly_a.theta_trans + poly_b.theta_trans) / 2
            else:
                theta_avg = poly_a.theta_trans / 2
            poly_a.set_params(poly_a.cpts, theta_trans=theta_avg)
            moved_pts = poly_a.transform(mov_pts)
            return (moved_pts, mov_simps, None, mov_labs), poly_a

    def _register(self, refs, mov_polyline):
        if self.n_neigh == 1 or (self.multi == "simultaneous" and len(refs) >= 2):
            return self._run_reg(refs, mov_polyline)
        else:
            return self._independent_avg(refs, mov_polyline)

    def compute(self, polylines):
        nslice = len(polylines)
        polylines  = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos   = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            for pass_ in self._sweep_passes(nslice):
                for i, direction in pass_:
                    refs = self._get_refs(polylines, polylines0, i, nslice, direction)
                    if not refs:
                        continue
                    self.reg.title = f"slice: {i}"
                    moved, transfo = self._register(refs, polylines0[i])
                    polylines[i] = moved
                    if transfo is not None:
                        transfos[i].append(transfo)
                if self.propag == "jacobi":
                    polylines0 = copy.deepcopy(polylines)
            polylines0 = copy.deepcopy(polylines)

        return polylines, transfos

