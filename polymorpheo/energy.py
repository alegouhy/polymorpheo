import jax
import jax.numpy as jnp

import polymorpheo.transfo as transfo_ops
import polymorpheo.utils as utils


class point2point:
    def __init__(self, agg="mean", alpha=2, scale=1, bidir=True):
        self.alpha = alpha
        self.scale = scale
        self.bidir = bidir
        if agg == "max":
            self.agg_fun = jnp.max
        elif agg == "mean":
            self.agg_fun = jnp.mean

    def compute(self, ref_pts, mov_pts, ref_labs=None, mov_labs=None, ref_normals=None):
        ref_pts_list = [ref_pts] if hasattr(ref_pts, "shape") else ref_pts
        ref_labs_list = [ref_labs] if hasattr(ref_pts, "shape") else ref_labs
        use_labs = (ref_labs_list is not None) and (mov_labs is not None)

        pts_dist = 0.0
        if not use_labs:
            for ref_pts in ref_pts_list:
                dist = ref_pts[:, None, :] - mov_pts[None, :, :]
                dist = jnp.sum(dist**2, axis=-1)

                dist_nn = jnp.min(dist, axis=0)
                dist_nn = robust_rho(dist_nn, alpha=self.alpha, scale=self.scale)
                pts_dist += self.agg_fun(dist_nn)

                if self.bidir:
                    dist_nn = jnp.min(dist, axis=1)
                    dist_nn = robust_rho(dist_nn, alpha=self.alpha, scale=self.scale)
                    pts_dist += self.agg_fun(dist_nn)

        else:
            for ref_pts, ref_labs in zip(ref_pts_list, ref_labs_list):
                labs = jnp.intersect1d(mov_labs, ref_labs)
                for lab in labs:
                    ref_pts_lab = ref_pts[ref_labs == lab, :]
                    mov_ind_lab = mov_labs == lab
                    mov_pts_lab = mov_pts[mov_ind_lab, :]

                    dist = ref_pts_lab[:, None, :] - mov_pts_lab[None, :, :]
                    dist = jnp.sum(dist**2, axis=-1)

                    dist_nn = jnp.min(dist, axis=0)
                    dist_nn = robust_rho(dist_nn, alpha=self.alpha, scale=self.scale)
                    pts_dist += self.agg_fun(dist_nn)
                    if self.bidir:
                        dist_nn = jnp.min(dist, axis=1)
                        dist_nn = robust_rho(dist_nn, alpha=self.alpha, scale=self.scale)
                        pts_dist += self.agg_fun(dist_nn)

        return pts_dist


class point2plane:
    def __init__(self, agg="mean", alpha=2, scale=1, bidir=True):
        self.alpha = alpha
        self.scale = scale
        self.bidir = bidir
        if agg == "max":
            self.agg_fun = jnp.max
        elif agg == "mean":
            self.agg_fun = jnp.mean

    def compute(self, ref_pts, mov_pts, ref_labs=None, mov_labs=None, ref_normals=None):
        dist = ref_pts[:, None, :] - mov_pts[None, :, :]   # (nref, nmov, 3)
        sq_dist = jnp.sum(dist**2, axis=-1)                # (nref, nmov)

        # forward: each moving point → its nearest reference point
        nn_idx = jnp.argmin(sq_dist, axis=0)               # (nmov,)
        if ref_normals is not None:
            diff = mov_pts - ref_pts[nn_idx]                # (nmov, 3)
            proj = jnp.sum(diff * ref_normals[nn_idx], axis=-1) ** 2
        else:
            proj = jnp.min(sq_dist, axis=0)
        proj = robust_rho(proj, alpha=self.alpha, scale=self.scale)
        total = self.agg_fun(proj)

        if self.bidir:
            # backward: each reference point → nearest moving point (point-to-point)
            back = robust_rho(jnp.min(sq_dist, axis=1), alpha=self.alpha, scale=self.scale)
            total = total + self.agg_fun(back)

        return total


class grad_disp:
    def __init__(self, l_norm=2, eps=1e-9):
        self.l_norm = l_norm
        self.eps = eps

    def compute(self, disp, pts, simps):
        grad = self.compute_map(disp, pts, simps)

        return jnp.mean(grad)

    def compute_map(self, disp, pts, simps):
        # 1 value per simplex

        ndims = simps.shape[1]

        if ndims == 2:
            grad = self.edge_grad(disp, pts, simps[:, 0], simps[:, 1])

        elif ndims == 3:
            grad1 = self.edge_grad(disp, pts, simps[:, 0], simps[:, 1])
            grad2 = self.edge_grad(disp, pts, simps[:, 1], simps[:, 2])
            grad3 = self.edge_grad(disp, pts, simps[:, 2], simps[:, 0])
            grad = (grad1 + grad2 + grad3) / 3

        return grad

    def edge_grad(self, disp, pts, idx0, idx1):
        disp_diff = disp[idx0] - disp[idx1]
        pts_diff = pts[idx0] - pts[idx1]

        if self.l_norm == 2:
            disp_norm = jnp.sum(disp_diff**2, axis=-1)
            pts_norm = jnp.sum(pts_diff**2, axis=-1)
        elif self.l_norm == 1:
            disp_norm = jnp.sum(jnp.abs(disp_diff), axis=-1)
            pts_norm = jnp.sum(jnp.abs(pts_diff), axis=-1)

        return disp_norm / (pts_norm + self.eps)


class alap:
    """
    as linear as possible
    """

    def __init__(self, transfo="rigid", l_norm=2, eps=1e-9):
        self.l_norm = l_norm  # l1 or l2 norm
        self.eps = eps
        self.transfo = transfo  # 'rigid', 'similarity' or 'affine'
        self.opti_transfo_fun = transfo_ops.opti_linear_transfo(transfo, se=True)
        self.neighborhoods = None
        self.n_neighbors = None

    def set_neighs(self, simps, npts=None):
        self.neighborhoods, self.n_neighbors = utils.neighs_from_simps(simps, npts)

    def compute(self, disp, pts, simps=None):
        if self.neighborhoods is None:
            raise ValueError("Must call set_neighs() before compute()!")
        alap_map = self.compute_map(disp, pts)

        return jnp.mean(alap_map)

    def compute_map(self, disp, pts, simps=None):
        if self.neighborhoods is None:
            raise ValueError("Must call set_neighs() before compute_map()!")

        alap_map = jax.vmap(self._vertex_alap, in_axes=(0, None, None))(jnp.arange(len(pts)), disp, pts)

        return alap_map

    def _vertex_alap(self, vertex_idx, disp, pts):
        neigh_idx = self.neighborhoods[vertex_idx]
        mask = neigh_idx >= 0
        n_neigh = jnp.sum(mask)
        mask = mask.astype(jnp.float32)

        safe_idx = jnp.maximum(neigh_idx, 0)

        pts_neigh = pts[safe_idx]
        disp_neigh = disp[safe_idx]
        moved_neigh = pts_neigh + disp_neigh

        lin, trans = self.opti_transfo_fun.fit(moved_neigh, pts_neigh, weights=mask)

        reconstructed = (pts_neigh @ lin.T) + trans
        diff = moved_neigh - reconstructed

        if self.l_norm == 2:
            energy_per_neighbor = jnp.sum(diff**2, axis=-1)
        elif self.l_norm == 1:
            energy_per_neighbor = jnp.sum(jnp.abs(diff), axis=-1)

        masked_energy = energy_per_neighbor * mask
        total_energy = jnp.sum(masked_energy)

        return jnp.where(n_neigh > 0, total_energy / n_neigh, 0.0)


def energy_total_fn(theta, cpts, mov_mesh, ref_mesh_list, fit_fun, regul_fun, wreg, polytransfo):
    # regularization on the smoothed field or on theta?

    mov_pts, mov_simps, _, mov_labs = mov_mesh

    # regul = regul_fun.compute(theta, mov_pts, mov_simps)
    if polytransfo.sigma is not None:
        svf = polytransfo.interp(mov_pts, cpts, theta_lin=None, theta_trans=theta)
        moved_pts = polytransfo.transform(mov_pts, cpts, theta_lin=None, theta_trans=theta)
    else:
        svf = theta
        moved_pts = mov_pts + theta

    regul = regul_fun.compute(svf, mov_pts, mov_simps)


    fit = 0.0
    for ref_mesh in ref_mesh_list:
        ref_pts, ref_simps, ref_normals, ref_labs = ref_mesh
        fit += fit_fun.compute(ref_pts, moved_pts, ref_normals=ref_normals)

    return fit + wreg * regul


def robust_rho(x_sq, alpha, scale, eps=1e-9):
    # https://arxiv.org/pdf/1701.03077

    x_sq_scal = x_sq / (scale**2)

    if alpha == 2.0:
        return x_sq_scal / 2
    elif alpha == 0.0:
        return jnp.log1p(x_sq_scal / 2)
    elif alpha < -10:
        return -jnp.expm1(-x_sq_scal / 2)
    else:
        beta = jnp.maximum(eps, jnp.abs(alpha - 2.0))
        alpha_safe = jnp.where(alpha == 0, eps, jnp.sign(alpha) * jnp.maximum(eps, jnp.abs(alpha)))
        return (beta / alpha_safe) * (jnp.power(x_sq_scal / beta + 1.0, 0.5 * alpha) - 1.0)


def m_estimator(x_sq, est="l2", sigma=1.0):
    if est == "l1":
        # ρ(x) = |x|
        dist = jnp.sqrt(x_sq)

    elif est == "l2":
        # ρ(x) = x²
        dist = x_sq

    elif est == "welsch":
        # ρ(x) = σ²(1 - exp(-x²/(2σ²)))
        dist = sigma**2 * (1 - jnp.exp(-x_sq / (2 * sigma**2)))

    elif est == "geman-mcclure":
        # ρ(x) = x²/(σ² + x²)
        dist = sigma**2 * x_sq / (sigma**2 + x_sq)

    elif est == "cauchy":
        # ρ(x) = (σ²/2) * log(1 + (x/σ)²)
        dist = (sigma**2 / 2) * jnp.log(1 + x_sq / sigma**2)

    elif est == "tukey":
        # ρ(x) = (σ²/6)(1 - (1-(x/σ)²)³) for |x| ≤ σ, σ²/6 for |x| > σ
        x_abs = jnp.sqrt(x_sq)
        dist = jnp.where(
            x_abs <= sigma,
            (sigma**2 / 6) * (1 - (1 - x_sq / sigma**2) ** 3),
            sigma**2 / 6,
        )

    else:
        raise ValueError(
            f"Unknown estimator: {est}. Choose from 'l1', 'l2', 'welsch', 'geman-mcclure', 'cauchy', 'tukey'"
        )

    return dist
