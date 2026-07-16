from itertools import product

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from scipy.linalg import expm, logm
from scipy.stats.qmc import Sobol

import polymorpheo.utils as utils


class polytransfo:
    """
    should investigate: https://github.com/dodgebc/jaxkd
    """

    def __init__(self, sigma, int_steps=64, int_step_max=None, eps=1e-14, rk=1):
        self.sigma = sigma
        self.int_steps = int_steps
        self.int_step_max = int_steps if int_step_max is None else int_step_max
        self.eps = eps
        self.rk = rk
        self.cpts = None
        self.theta_trans = None
        self.theta_lin = None

    def set_params(self, cpts, theta_trans=None, theta_lin=None):
        self.cpts = cpts
        self.theta_trans = theta_trans
        self.theta_lin = theta_lin

    def transform(self, pts, cpts=None, theta_lin=None, theta_trans=None):
        # pts (npts, ndims):                query points where we want to evaluate the disp.
        # cpts (ncpts, ndims):              control points where theta is known.
        # theta_trans (ncpts, ndims):       transfo params (translation part) at cpts.
        # theta_lin (ncpts, ndims, ndims):  transfo params (linear part) at cpts.

        if cpts is None:
            cpts = self.cpts
        if theta_trans is None:
            theta_trans = self.theta_trans
        if theta_lin is None:
            theta_lin = self.theta_lin

        if cpts is None:
            raise ValueError("cpts must be provided either via transform() or set_params().")
        if theta_trans is None and theta_lin is None:
            raise ValueError("At least one of theta_trans or theta_lin must be provided via transform() or set_params().")

        if self.int_steps == 0:
            disp = self.interp(pts, cpts, theta_lin, theta_trans)
        else:
            disp = self.lie_exp(pts, cpts, theta_lin, theta_trans)

        return pts + disp

    def jacobian(self, pts, cpts=None, theta_lin=None, theta_trans=None):
        # Returns the Jacobian of T(x) = x + d(x) at each point in pts.
        # Output shape: (npts, ndims, ndims), where jac[i, k, l] = ∂T_k(x_i)/∂x_l

        if cpts is None:
            cpts = self.cpts
        if theta_trans is None:
            theta_trans = self.theta_trans
        if theta_lin is None:
            theta_lin = self.theta_lin

        if cpts is None:
            raise ValueError("cpts must be provided either via jacobian() or set_params().")
        if theta_trans is None and theta_lin is None:
            raise ValueError("At least one of theta_trans or theta_lin must be provided via jacobian() or set_params().")

        def _transform_single(pt):
            return self.transform(pt[None], cpts, theta_lin, theta_trans)[0]

        return jax.vmap(jax.jacobian(_transform_single))(pts)

    def transform_ellipsoids(self, centers, covs, cpts=None, theta_lin=None, theta_trans=None, orientation_only=False):
        # centers (n, ndims):         ellipsoid centers
        # covs    (n, ndims, ndims):  covariance matrices (symmetric positive definite)
        # orientation_only:           if True, use PPD reorientation
        # Returns new_centers (n, ndims), new_covs (n, ndims, ndims)

        new_centers = self.transform(centers, cpts, theta_lin, theta_trans)
        jac = self.jacobian(centers, cpts, theta_lin, theta_trans)  # (n, ndims, ndims)

        return new_centers, utils.transform_ellipsoids(jac, covs, orientation_only)

    def lie_exp(self, pts, cpts, theta_lin, theta_trans):

        dt = 1.0 / self.int_steps
        d = jnp.zeros_like(pts)

        for _ in range(self.int_step_max):
            if self.rk == 1:
                v = self.interp(pts + d, cpts, theta_lin, theta_trans)
                d = d + dt * v

            elif self.rk == 2:
                x = pts + d
                k1 = self.interp(x, cpts, theta_lin, theta_trans)
                k2 = self.interp(x + dt * k1, cpts, theta_lin, theta_trans)
                d = d + dt * (k1 + k2) / 2.0

            elif self.rk == 4:
                x = pts + d
                k1 = self.interp(x, cpts, theta_lin, theta_trans)
                k2 = self.interp(x + 0.5 * dt * k1, cpts, theta_lin, theta_trans)
                k3 = self.interp(x + 0.5 * dt * k2, cpts, theta_lin, theta_trans)
                k4 = self.interp(x + dt * k3, cpts, theta_lin, theta_trans)
                d = d + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return d

    def interp(self, pts, cpts, theta_lin, theta_trans):

        sqdist = utils.pts_dist(pts, cpts)  # (npts, ncpts)
        weight = jnp.exp(-sqdist / (2 * self.sigma**2))

        weight = weight / (jnp.sum(weight, axis=1)[..., None] + self.eps)

        if theta_trans is not None and theta_lin is None:
            disp = weight @ theta_trans

        else:
            disp = jnp.zeros_like(pts)
            if theta_lin is not None:
                disp += jnp.einsum("ij,jkl,il->ik", weight, theta_lin, pts)
            if theta_trans is not None:
                disp += weight @ theta_trans

        return disp

    def sigma_silverman(self, pts):

        iqr = jnp.mean(jnp.quantile(pts, (3 / 4), axis=0) - jnp.quantile(pts, (1 / 4), axis=0))
        std = jnp.mean(jnp.std(pts, axis=0))
        sigma = 0.9 * jnp.min(jnp.stack((std, iqr / 1.349))) * pts.shape[0] ** (-1 / 5)

        return sigma

    def invert(self):
        if self.theta_trans is not None:
            self.theta_trans = -self.theta_trans
        if self.theta_lin is not None:
            self.theta_lin = -self.theta_lin


class affine:
    def __init__(self):
        self.lin = None
        self.trans = None

    def set_params(self, lin, trans):
        self.lin = lin
        self.trans = trans

    def transform(self, pts, lin=None, trans=None):

        if lin is None:
            lin = self.lin
        if trans is None:
            trans = self.trans

        if lin is None:
            raise ValueError("lin must be provided either via transform() or set_params().")
        if trans is None:
            raise ValueError("trans must be provided either via transform() or set_params().")

        return pts @ lin.T + trans

    def jacobian(self, pts, lin=None, trans=None):
        # Returns the Jacobian of T(x) = lin x + trans at each point in pts.
        # For an affine map the Jacobian is constant (= lin), broadcast to (npts, ndims, ndims).

        if lin is None:
            lin = self.lin

        return jnp.broadcast_to(lin[None], (len(pts), *lin.shape))

    def transform_ellipsoids(self, centers, covs, lin=None, trans=None, orientation_only=False):
        # centers (n, ndims):         ellipsoid centers
        # covs    (n, ndims, ndims):  covariance matrices (symmetric positive definite)
        # orientation_only:           if True, use PPD reorientation
        # Returns new_centers (n, ndims), new_covs (n, ndims, ndims)

        if lin is None:
            lin = self.lin
        if trans is None:
            trans = self.trans

        new_centers = self.transform(centers, lin, trans)
        jac = self.jacobian(centers, lin)

        return new_centers, utils.transform_ellipsoids(jac, covs, orientation_only)

    def invert(self):
        lin_inv = jnp.linalg.inv(self.lin)
        self.trans = -lin_inv @ self.trans
        self.lin = lin_inv


def apply_transfo_chain(transfo_list, pts, invert=False):
    # Evaluates transfo_list[-1] o ... o transfo_list[0].
    # With invert=True, evaluates the inverse chain (reversed order, each transfo inverted then restored).
    #  - transfo_list:      list of polytransfo or affine instances with parameters set via set_params().
    #  - pts (npts, ndims): points at which to evaluate the composed transformation.
    # Returns the final moved points (npts, ndims).

    pts_moved = pts
    if invert:
        for transfo in reversed(transfo_list):
            transfo.invert()
            pts_moved = transfo.transform(pts_moved)
            transfo.invert()
    else:
        for transfo in transfo_list:
            pts_moved = transfo.transform(pts_moved)

    return pts_moved


class init_transfo:
    def __init__(self, init="identity"):
        """
        init: init: 'identity', 'centroids', 'similarity', 'ellipsoid'
        """

        self.init = init

    def transform(self, ref_pts, mov_pts):
        # init: 'identity', 'centroids', 'similarity', 'ellipsoid'

        ndims = ref_pts.shape[1]

        if self.init in (None, "identity"):
            return jnp.eye(ndims + 1), mov_pts

        else:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
            mov_pts_mu = jnp.mean(mov_pts, axis=0)

            if self.init in ("similarity", "ellipsoid"):
                s = jnp.std(ref_pts, axis=0) / jnp.std(mov_pts, axis=0)
                if self.init == "ellipsoid":
                    A = jnp.diag(s)
                elif self.init == "similarity":
                    A = jnp.mean(s) * jnp.eye(ndims)
            else:
                A = jnp.eye(ndims)

            t = ref_pts_mu - (mov_pts_mu @ A.T)
            moved_pts = (mov_pts @ A.T) + t

            T = utils.aff_hmgn(A, t)

            return T, moved_pts


class opti_linear_transfo:
    def __init__(self, transfo, se=True):  #  gamma=1e-5
        """
        transfo: 'translation', 'rigid', 'similarity' or 'affine'
        """

        self.transfo = transfo
        self.se = se
        # self.gamma = gamma

    def fit(self, ref_pts, mov_pts, ref_pts_mu=None, mov_pts_mu=None, weights=None, eps=1e-9, tol=1e-6):
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """

        ndims = ref_pts.shape[1]

        if weights is None:
            weights = jnp.ones(len(ref_pts))
        weights = weights / (jnp.sum(weights) + eps)

        if ref_pts_mu is None:
            ref_pts_mu = jnp.sum(ref_pts * weights[:, None], axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.sum(mov_pts * weights[:, None], axis=0)


        if self.transfo == "translation":
            return jnp.eye(ndims), ref_pts_mu - mov_pts_mu

        sqrt_weights = jnp.sqrt(weights[:, None])
        ref_pts_wbar = (ref_pts - ref_pts_mu) * sqrt_weights
        mov_pts_wbar = (mov_pts - mov_pts_mu) * sqrt_weights

        if self.transfo in ("rigid", "similarity"):
            cov = ref_pts_wbar.T @ mov_pts_wbar
            U, s, Vt = self.stable_svd(cov, ndims, eps, tol)
            S = jnp.eye(ndims)

            if self.transfo == "similarity":
                mov_norm_sq = jnp.sum(mov_pts_wbar**2)
                s = jnp.trace(cov @ Vt.T @ U.T) / (mov_norm_sq + eps)
                S = s * S

        elif self.transfo == "affine":
            MtM = mov_pts_wbar.T @ mov_pts_wbar
            A = jnp.linalg.solve(MtM + tol * jnp.eye(ndims), mov_pts_wbar.T @ ref_pts_wbar)
            U, S, Vt = self.stable_svd(A.T, ndims, eps, tol)
            S = jnp.diag(S)

        if self.se:
            det_sign = jnp.sign(jnp.linalg.det(U @ Vt))
            S = S.at[-1, -1].set(S[-1, -1] * det_sign)

        A = (U @ S) @ Vt
        t = ref_pts_mu - (mov_pts_mu @ A.T)

        return A, t

    def transform(self, A, t, mov_pts):
        return (mov_pts @ A.T) + t

    def stable_svd(self, mat, ndims, eps, tol):
        # guard against zero / repeated singular values (NaN-free backward pass)
        healthy = jnp.linalg.norm(mat) > eps
        mat = jnp.where(healthy, mat, jnp.eye(ndims))
        mat = mat + tol * jnp.diag(1.0 + jnp.arange(ndims, dtype=mat.dtype))

        return jnp.linalg.svd(mat, full_matrices=False)


class polynom:
    def __init__(self):
        self._opti = None
        self.coeffs = None

    def set_params(self, coeffs, mov_pts_mu, ref_pts_mu, degree):
        self.coeffs = coeffs
        self._opti = opti_polynom_transfo(degree)
        self._opti.mov_pts_mu = mov_pts_mu
        self._opti.ref_pts_mu = ref_pts_mu

    def transform(self, pts):
        return self._opti.transform(self.coeffs, jnp.array(pts))


class opti_polynom_transfo:
    def __init__(self, degree, se=True):
        self.degree = degree

    def fit(self, ref_pts, mov_pts, ref_pts_mu=None, mov_pts_mu=None):
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """

        if ref_pts_mu is None:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.mean(mov_pts, axis=0)
        self.ref_pts_mu = ref_pts_mu
        self.mov_pts_mu = mov_pts_mu
        ref_pts_bar = ref_pts - ref_pts_mu
        mov_pts_bar = mov_pts - mov_pts_mu

        X, _ = self.design_mat(mov_pts_bar)

        coeffs, _, _, _ = jnp.linalg.lstsq(X, ref_pts_bar, rcond=None)

        return coeffs

    def design_mat(self, x):
        npts, ndims = x.shape
        exponents = []
        for total_deg in range(self.degree + 1):
            for exponent in product(range(total_deg + 1), repeat=ndims):
                if sum(exponent) == total_deg:
                    exponents.append(exponent)

        X = jnp.ones((npts, len(exponents)))
        for j, e in enumerate(exponents):
            X = X.at[:, j].set(jnp.prod(x ** jnp.array(e), axis=1))

        return X, exponents

    def transform(self, coeffs, mov_pts, X=None):
        mov_pts_bar = mov_pts - self.mov_pts_mu

        if X is None:
            X, _ = self.design_mat(mov_pts_bar)

        return X @ coeffs + self.ref_pts_mu


def random_locAff(
    ndims,
    ncpts=300,
    seed=None,
    liealg=True,
    so_corner=[0],
    ne_corner=[1],
    bound_scal=1.1,
    trans_bounds=0.01,
    rot_bounds=np.pi / 2,
    scalDir_bounds=np.pi / 4,
    scal_bounds=3,
):

    if len(so_corner) == 1:
        so_corner *= ndims
    if len(ne_corner) == 1:
        ne_corner *= ndims
    so_corner = np.array(so_corner)
    ne_corner = np.array(ne_corner)

    center = (ne_corner + so_corner) / 2
    ne_corner = np.expand_dims(bound_scal * (ne_corner - center) + center, axis=-1)
    so_corner = np.expand_dims(bound_scal * (so_corner - center) + center, axis=-1)

    r = Sobol(ndims, seed=seed).random(ncpts)
    cpts = np.expand_dims(r * ne_corner.T + (1 - r) * so_corner.T, axis=-1)

    if ndims == 2:
        rot_angles = 2 * rot_bounds * (np.random.rand(ncpts) - 0.5)
        rot = np.stack(
            [np.stack([np.zeros(ncpts), rot_angles], axis=1), np.stack([-rot_angles, np.zeros(ncpts)], axis=1)], axis=2
        )

        scalDir_angles = 2 * scalDir_bounds * (np.random.rand(ncpts) - 0.5)
        scalDir = np.stack(
            [np.stack([np.zeros(ncpts), scalDir_angles], axis=1), np.stack([-scalDir_angles, np.zeros(ncpts)], axis=1)],
            axis=2,
        )

    elif ndims == 3:
        rot_angles = 2 * rot_bounds * (np.random.rand(ncpts, 3) - 0.5)
        rot = np.stack(
            [
                np.stack([np.zeros(ncpts), rot_angles[..., 2], -rot_angles[..., 1]], axis=1),
                np.stack([-rot_angles[..., 2], np.zeros(ncpts), rot_angles[..., 0]], axis=1),
                np.stack([rot_angles[..., 1], -rot_angles[..., 0], np.zeros(ncpts)], axis=1),
            ],
            axis=2,
        )

        scalDir_angles = 2 * scalDir_bounds * (np.random.rand(ncpts, 3) - 0.5)
        scalDir = np.stack(
            [
                np.stack([np.zeros(ncpts), scalDir_angles[..., 2], -scalDir_angles[..., 1]], axis=1),
                np.stack([-scalDir_angles[..., 2], np.zeros(ncpts), scalDir_angles[..., 0]], axis=1),
                np.stack([scalDir_angles[..., 1], -scalDir_angles[..., 0], np.zeros(ncpts)], axis=1),
            ],
            axis=2,
        )

    rot = expm(rot)
    scalDir = expm(scalDir)

    scal_factors = np.exp(2 * np.log(scal_bounds) * (np.random.rand(ncpts, ndims, 1) - 0.5))
    scal = np.eye(ndims) * scal_factors

    mats = np.matmul(rot, np.matmul(scalDir, np.matmul(scal, np.transpose(scalDir, [0, 2, 1]))))
    trans = 2 * trans_bounds * (np.random.rand(ncpts, ndims, 1) - 0.5)
    trans = np.matmul(mats, -cpts) + trans + cpts

    if not liealg:
        return cpts[..., 0], mats, trans[..., 0]

    affs = np.concatenate((mats, trans), axis=2)
    affs = np.concatenate((affs, np.concatenate((np.zeros((ncpts, 1, ndims)), np.ones((ncpts, 1, 1))), axis=2)), axis=1)

    log_affs = np.stack([logm(affs[i, ...]) for i in range(ncpts)], axis=0)
    log_mats = log_affs[:, :ndims, :ndims]
    log_trans = log_affs[:, :ndims, ndims]

    return cpts[..., 0], log_mats.real, log_trans.real
