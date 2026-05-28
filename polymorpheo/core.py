import copy
import time

import numpy as np
from scipy.linalg import expm, logm

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


class register_slices:
    def __init__(
        self,
        method,
        transfo_type,
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
        plot=False,
        xlim=None,
        ylim=None,
    ):
        self.method = method
        self.niter = niter

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
                plot=plot,
            )

    def _identity_transfo(self, ndims):
        transfo = transfo_ops.affine()
        transfo.set_params(np.eye(ndims), np.zeros(ndims))
        return transfo

    def _linear_transfo(self, T):
        lin, trans = utils.aff_dehmgn(T)
        transfo = transfo_ops.affine()
        transfo.set_params(lin, trans)
        return transfo

    def _deformable_transfo(self):
        return copy.deepcopy(self.reg.polytransfo_out)

    def compute(self, polylines):
        if self.method == 0:
            return self._method_0(polylines)
        elif self.method == 1:
            return self._method_1(polylines)
        elif self.method == 2:
            return self._method_2(polylines)
        elif self.method == 3:
            return self._method_3(polylines)
        elif self.method == 4:
            return self._method_4(polylines)
        elif self.method == 5:
            return self._method_5(polylines)

    def _method_0(self, polylines):
        nslice = len(polylines)
        midslice = int(nslice / 2)
        ndims = polylines[0][0].shape[1]
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            for i in range(midslice, nslice - 1):
                ref_polyline = polylines[i]
                mov_polyline = polylines0[i + 1]
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    T, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i + 1].append(self._linear_transfo(T))
                elif self.transfo_type == "polynomial":
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == "deformable":
                    theta, moved_polyline, _ = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i + 1].append(self._deformable_transfo())

                polylines[i + 1] = moved_polyline

            for i in reversed(range(1, midslice + 1)):
                ref_polyline = polylines[i]
                mov_polyline = polylines[i - 1]
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    T, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i - 1].append(self._linear_transfo(T))
                elif self.transfo_type == "polynomial":
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == "deformable":
                    theta, moved_polyline, _ = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i - 1].append(self._deformable_transfo())

                polylines[i - 1] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

        return polylines, transfos

    def _method_1(self, polylines):
        nslice = len(polylines)
        ndims = polylines[0][0].shape[1]
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            for i in range(1, nslice - 1):
                prev_polyline = polylines[i - 1]
                next_polyline = polylines0[i + 1]
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    transfo = transfo_ops.affine()
                    transfo.set_params(lin, trans)
                    transfos[i].append(transfo)
                elif self.transfo_type == "polynomial":
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                elif self.transfo_type == "deformable":
                    theta_prev, _, _ = self.reg.compute(prev_polyline, mov_polyline)
                    theta_next, _, _ = self.reg.compute(next_polyline, mov_polyline)
                    theta = (theta_prev + theta_next) / 2
                    moved_pts = self.reg.polytransfo.transform(mov_pts, mov_pts, theta_lin=None, theta_trans=theta)
                    transfos[i].append(self._deformable_transfo())

                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

        return polylines, transfos

    def _method_2(self, polylines):
        nslice = len(polylines)
        midslice = int(nslice / 2)
        ndims = polylines[0][0].shape[1]
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            for i in range(midslice, nslice - 1):
                prev_polyline = polylines[i - 1]
                next_polyline = polylines0[i + 1]
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    transfo = transfo_ops.affine()
                    transfo.set_params(lin, trans)
                    transfos[i].append(transfo)
                elif self.transfo_type == "polynomial":
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                elif self.transfo_type == "deformable":
                    theta_prev, _, _ = self.reg.compute(prev_polyline, mov_polyline)
                    theta_next, _, _ = self.reg.compute(next_polyline, mov_polyline)
                    theta = (theta_prev + theta_next) / 2
                    moved_pts = self.reg.polytransfo.transform(mov_pts, mov_pts, theta_lin=None, theta_trans=theta)
                    transfos[i].append(self._deformable_transfo())

                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline

            for i in reversed(range(1, midslice + 1)):
                prev_polyline = polylines[i + 1]
                next_polyline = polylines0[i - 1]
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    transfo = transfo_ops.affine()
                    transfo.set_params(lin, trans)
                    transfos[i].append(transfo)
                elif self.transfo_type == "polynomial":
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                elif self.transfo_type == "deformable":
                    theta_prev, _, _ = self.reg.compute(prev_polyline, mov_polyline)
                    theta_next, _, _ = self.reg.compute(next_polyline, mov_polyline)
                    theta = (theta_prev + theta_next) / 2
                    moved_pts = self.reg.polytransfo.transform(mov_pts, mov_pts, theta_lin=None, theta_trans=theta)
                    transfos[i].append(self._deformable_transfo())

                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

        return polylines, transfos

    def _method_3(self, polylines):
        nslice = len(polylines)
        ndims = polylines[0][0].shape[1]
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            for i in range(1, nslice, 2):
                mov_polyline = polylines0[i]
                prev_polyline = polylines0[i - 1]
                next_polyline = polylines0[i + 1] if i < nslice - 1 else None
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else (np.eye(3), None)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    transfo = transfo_ops.affine()
                    transfo.set_params(lin, trans)
                    transfos[i].append(transfo)
                elif self.transfo_type == "polynomial":
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else mov_polyline
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                elif self.transfo_type == "deformable":
                    theta_prev, _, _ = self.reg.compute(prev_polyline, mov_polyline)
                    theta_next, _, _ = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else (np.zeros_like(mov_pts), None, None)
                    theta = (theta_prev + theta_next) / 2
                    moved_pts = self.reg.polytransfo.transform(mov_pts, mov_pts, theta_lin=None, theta_trans=theta)
                    transfos[i].append(self._deformable_transfo())

                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

            for i in range(0, nslice, 2):
                mov_polyline = polylines0[i]
                prev_polyline = polylines0[i - 1] if i > 0 else None
                next_polyline = polylines0[i + 1] if i < nslice - 1 else None
                mov_pts, mov_simps, _, mov_labs = mov_polyline

                if self.transfo_type == "linear":
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline) if i > 0 else (np.eye(3), None)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else (np.eye(3), None)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    transfo = transfo_ops.affine()
                    transfo.set_params(lin, trans)
                    transfos[i].append(transfo)
                elif self.transfo_type == "polynomial":
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline) if i > 0 else mov_polyline
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else mov_polyline
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                elif self.transfo_type == "deformable":
                    theta_prev, _, _ = self.reg.compute(prev_polyline, mov_polyline) if i > 0 else (np.zeros_like(mov_pts), None, None)
                    theta_next, _, _ = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else (np.zeros_like(mov_pts), None, None)
                    theta = (theta_prev + theta_next) / 2
                    moved_pts = self.reg.polytransfo.transform(mov_pts, mov_pts, theta_lin=None, theta_trans=theta)
                    transfos[i].append(self._deformable_transfo())

                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

        return polylines, transfos

    def _method_4(self, polylines):
        nslice = len(polylines)
        ndims = polylines[0][0].shape[1]
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            for i in range(1, nslice - 1, 2):
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                ref_polyline = [polylines0[i - 1], polylines0[i + 1]]

                if self.transfo_type == "linear":
                    T, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._linear_transfo(T))
                elif self.transfo_type == "polynomial":
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == "deformable":
                    theta, moved_polyline, _ = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._deformable_transfo())

                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

            for i in range(2, nslice - 1, 2):
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                ref_polyline = [polylines0[i - 1], polylines0[i + 1]]

                if self.transfo_type == "linear":
                    T, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._linear_transfo(T))
                elif self.transfo_type == "polynomial":
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == "deformable":
                    theta, moved_polyline, _ = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._deformable_transfo())

                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

        return polylines, transfos

    def _method_5(self, polylines):
        nslice = len(polylines)
        midslice = int(nslice / 2)
        ndims = polylines[0][0].shape[1]
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        transfos = [[] for _ in range(nslice)]

        for _ in range(self.niter):
            t = time.time()
            for i in range(midslice + 1, nslice - 1):
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                ref_polyline = [polylines[i - 1], polylines0[i + 1]]

                if self.transfo_type == "linear":
                    T, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._linear_transfo(T))
                elif self.transfo_type == "polynomial":
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == "deformable":
                    theta, moved_polyline, _ = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._deformable_transfo())
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)

            for i in reversed(range(1, midslice)):
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                ref_polyline = [polylines0[i - 1], polylines[i + 1]]

                if self.transfo_type == "linear":
                    T, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._linear_transfo(T))
                elif self.transfo_type == "polynomial":
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == "deformable":
                    theta, moved_polyline, _ = self.reg.compute(ref_polyline, mov_polyline)
                    transfos[i].append(self._deformable_transfo())
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)
            print("done in: ", time.time() - t)

        return polylines, transfos
