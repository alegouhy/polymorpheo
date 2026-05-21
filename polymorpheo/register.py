import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import lax

import polymorpheo.energy as energy
import polymorpheo.transfo as transfo_ops
import polymorpheo.utils as utils


# %%

def load_meshes(ref_mesh, mov_mesh, normalise=True):
    ref_meshes = [ref_mesh] if hasattr(ref_mesh[0], "shape") else ref_mesh

    ref_pts_list = []
    ref_labs_list = []
    ref_mesh_list = []
    for mesh in ref_meshes:
        mesh = [jnp.array(cont) if cont is not None else None for cont in mesh]
        ref_pts, ref_simps, ref_normals, ref_labs = mesh
        ref_pts_list.append(ref_pts)
        ref_labs_list.append(ref_labs)
        ref_mesh_list.append(mesh)

    mov_mesh = [jnp.array(cont) if cont is not None else None for cont in mov_mesh]

    mu = 0
    amp = 1
    if normalise:
        meshes = ref_meshes + [mov_mesh]
        meshes, mu, amp = utils.normalise_meshes(meshes)
        mov_mesh = meshes[-1]
        ref_mesh_list = meshes[:-1]

    return ref_mesh_list, mov_mesh, mu, amp


# %%


def init_affcube(ref_pts, mov_pts, do_scale=True, decimals=10):
    mov_pts_mu = jnp.mean(mov_pts, axis=0)
    ref_pts_mu = jnp.mean(ref_pts, axis=0)
    if do_scale:
        ref_pts_amp = jnp.max(ref_pts, axis=0) - jnp.min(ref_pts, axis=0)

    angles = [0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
    dist_best = jnp.inf
    lin_best = None
    moved_pts_best = None

    lins = []
    for angx in angles:
        rotx = utils.rot_mat(angx, 0, 3)
        for angy in angles:
            roty = utils.rot_mat(angy, 1, 3)
            for angz in angles:
                rotz = utils.rot_mat(angz, 2, 3)
                for axx in [None, 0]:
                    reflx = utils.refl_mat(axx, 3)
                    for axy in [None, 1]:
                        refly = utils.refl_mat(axy, 3)
                        for axz in [None, 2]:
                            reflz = utils.refl_mat(axz, 3)

                            lin = rotx @ roty @ rotz @ reflx @ refly @ reflz
                            lins.append(lin)

    lins = jnp.stack(lins, axis=-1)
    lins = jnp.round(lins, decimals=decimals)
    lins = jnp.unique(lins, axis=-1)

    for i in range(lins.shape[-1]):
        lin = lins[:, :, i]
        trans = -mov_pts_mu @ lin.T + mov_pts_mu
        moved_pts = mov_pts @ lin.T + trans.T

        if do_scale:
            moved_pts_amp = jnp.max(moved_pts, axis=0) - jnp.min(moved_pts, axis=0)
            scal = jnp.diag(ref_pts_amp / moved_pts_amp)
            lin = lin @ scal
            trans = -mov_pts_mu @ lin.T + ref_pts_mu
            moved_pts = mov_pts @ lin.T + trans

        dist = utils.chamfer(moved_pts, ref_pts)

        if dist < dist_best:
            dist_best = dist
            lin_best = lin
            trans_best = trans
            moved_pts_best = moved_pts

    return moved_pts_best, lin_best, trans_best


# %%


class reg_linear:
    def __init__(
        self,
        niter,
        transfo="rigid",
        init="identity",
        se=True,
        bidir=False,
        tol=1e-6,
        plot=False,
        xlim=None,
        ylim=None,
    ):
        """
        transfo: 'rigid', 'rigid2' or 'affine'
        init: 'identity', 'centroids', 'similarity' or 'ellipsoid'
        """
        self.init = init
        self.niter = niter
        self.bidir = bidir
        self.tol = tol

        self.plot = plot
        self.xlim = xlim
        self.ylim = ylim

        self.opti_transfo_fun = transfo_ops.opti_linear_transfo(transfo, se=se)
        self.init_transfo = transfo_ops.init_transfo(init)

    def compute(self, ref_mesh, mov_mesh, T0=None):
        """
        T0 has priority over init.
        ref_mesh can be a single polyline of a list of polylines
        """

        ref_mesh_list, mov_mesh, _, _ = load_meshes(ref_mesh, mov_mesh, False)
        ref_pts_list = [mesh[0] for mesh in ref_mesh_list]
        ref_labs_list = [mesh[3] for mesh in ref_mesh_list]
        mov_pts, mov_simps, _, mov_labs = mov_mesh

        if T0 is None:
            ref_pts = jnp.concatenate(ref_pts_list, axis=0)
            T, moved_pts = self.init_transfo.compute(ref_pts, mov_pts)
        else:
            T = T0
            A, t = utils.aff_dehmgn(T)
            moved_pts = (mov_pts @ A.T) + t

        for k in range(self.niter):
            ref_nn_pts, mov_nn_pts = utils.nearest_neighbors(
                ref_pts_list, moved_pts, ref_labs_list, mov_labs, self.bidir
            )
            lin, trans = self.opti_transfo_fun.fit(ref_nn_pts, mov_nn_pts)

            moved_pts_prev = moved_pts
            moved_pts = self.opti_transfo_fun.transform(lin, trans, moved_pts)

            moved_mesh = moved_pts, mov_simps, None, mov_labs

            T = utils.aff_hmgn(lin, trans) @ T

            if self.tol is not None:
                delta = jnp.mean(jnp.sum((moved_pts - moved_pts_prev) ** 2, axis=1))
                if delta < self.tol**2:
                    break

            if self.plot:
                if k % self.plot == 0:
                    for ref_mesh in ref_mesh_list:
                        utils.plot_contour(ref_mesh, col=[1, 0, 0])
                    utils.plot_contour(moved_mesh, col=[0, 0, 1])
                    plt.xlim(self.xlim)
                    plt.ylim(self.ylim)
                    plt.title(f"it: {k}", fontsize=7)
                    plt.show()

        return T, moved_mesh


# %%


class reg_polynom:
    def __init__(
        self,
        niter,
        degree=2,
        init="identity",
        se=True,
        bidir=False,
        tol=1e-6,
        plot=False,
        xlim=None,
        ylim=None,
    ):
        """
        init: 'identity', 'centroids', 'similarity' or 'ellipsoid'
        """
        self.init = init
        self.niter = niter
        self.bidir = bidir
        self.tol = tol

        self.plot = plot
        self.xlim = xlim
        self.ylim = ylim

        self.opti_transfo_fun = transfo_ops.opti_polynom_transfo(degree, se=se)
        self.init_transfo = transfo_ops.init_transfo(init)

    def compute(self, ref_mesh, mov_mesh, disp0=None):
        """
        disp0 has priority over init.
        """

        ref_mesh_list, mov_mesh, _, _ = load_meshes(ref_mesh, mov_mesh, False)
        ref_pts_list = [mesh[0] for mesh in ref_mesh_list]
        ref_labs_list = [mesh[3] for mesh in ref_mesh_list]
        mov_pts, mov_simps, _, mov_labs = mov_mesh

        if disp0 is None:
            ref_pts = jnp.concatenate(ref_pts_list, axis=0)
            _, moved_pts = self.init_transfo.compute(ref_pts, mov_pts)
        else:
            moved_pts = mov_pts + disp0

        for k in range(self.niter):
            ref_nn_pts, mov_nn_pts = utils.nearest_neighbors(
                ref_pts_list, moved_pts, ref_labs_list, mov_labs, self.bidir
            )
            coeffs = self.opti_transfo_fun.fit(ref_nn_pts, mov_nn_pts)

            moved_pts_prev = moved_pts
            moved_pts = self.opti_transfo_fun.transform(coeffs, moved_pts)

            moved_mesh = moved_pts, mov_simps, None, mov_labs

            if self.tol is not None:
                delta = jnp.mean(jnp.sum((moved_pts - moved_pts_prev) ** 2, axis=1))
                if delta < self.tol**2:
                    break

            if self.plot:
                if k % self.plot == 0:
                    for ref_mesh in ref_mesh_list:
                        utils.plot_contour(ref_mesh, col=[1, 0, 0])
                    utils.plot_mesh(moved_mesh, col=[0, 0, 1])
                    plt.xlim(self.xlim)
                    plt.ylim(self.ylim)
                    plt.title(f"it: {k}", fontsize=7)
                    plt.show()

        return moved_mesh


# %%


class reg_deformable:
    def __init__(
        self,
        niter,
        fit_fun,
        regul_fun,
        cpts_ratio=1,
        lr=1e-2,
        wreg=0,
        sigma=None,
        int_steps=64,
        normalise=True,
        rk=1,
        tol=None,
        warmup_steps=5,
        plot=False,
        xlim=None,
        ylim=None,
    ):
        self.niter = niter
        self.lr = lr
        self.wreg = wreg
        self.sigma = sigma
        self.int_steps = int_steps
        self.normalise = normalise
        self.polytransfo = transfo_ops.polytransfo(
            sigma=sigma, int_steps=int_steps, rk=rk
        )
        self.fit_fun = fit_fun
        self.regul_fun = regul_fun
        self.cpts_ratio = cpts_ratio
        self.tol = tol
        self.plot = plot
        self.xlim = xlim
        self.ylim = ylim

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 1e-5,
            end_value=lr * 1e-2,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=niter,
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(eps=1e-8),
            optax.scale_by_schedule(schedule),
            optax.scale(-1.0),
        )

        opti_step = self.make_opti_step(
            fit_fun, regul_fun, wreg, self.polytransfo, self.optimizer
        )
        if not self.plot:
            self.opti_loop = self.make_opti_loop(opti_step, tol, niter, warmup_steps)
        else:
            self.opti_loop = self.make_opti_plot_loop(
                opti_step, tol, niter, warmup_steps
            )

    def compute(self, ref_mesh, mov_mesh):
        
        ref_mesh_list, mov_mesh_n, mu, amp = load_meshes(ref_mesh, mov_mesh, self.normalise)
        mov_pts_n, mov_simps, _, mov_labs = mov_mesh_n

        if self.cpts_ratio == 1:
            cpts = mov_pts_n
        else:
            ncpts = int(mov_pts_n.shape[0] * self.cpts_ratio)
            _, cpts = utils.farthest_point_sampling(mov_pts_n, ncpts)

        theta0 = jnp.zeros_like(cpts)
        opt_state = self.optimizer.init(theta0)
        theta, losses = self.opti_loop(theta0, opt_state, cpts, mov_mesh_n, ref_mesh_list)

        disp = self.polytransfo.compute(mov_pts_n, cpts, theta_lin=None, theta_trans=theta)
        moved_mesh = utils.denormalise_meshes(
            [(mov_pts_n + disp, mov_simps, None, mov_labs)], mu, amp
        )[0]

        self.polytransfo.sigma = self.sigma * amp
        self.polytransfo.set_params(cpts * amp + mu, theta_trans=theta * amp)

        return theta * amp, moved_mesh, losses



    def make_opti_step(self, fit_fun, regul_fun, wreg, polytransfo, optimizer):
        
        @jax.jit
        def opti_step(theta, opt_state, cpts, mov_mesh, ref_mesh_list):
            loss, grads = jax.value_and_grad(energy.energy_total_fn)(
                theta,
                cpts,
                mov_mesh,
                ref_mesh_list,
                fit_fun,
                regul_fun,
                wreg,
                polytransfo,
            )
            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
            return theta, opt_state, loss

        return opti_step

    def make_opti_loop(self, opti_step, tol, niter, warmup_steps):
        tol_val = -1.0 if tol is None else float(tol)

        @jax.jit
        def opti_loop(theta0, opt_state, cpts, mov_mesh, ref_mesh_list):
            def cond_fn(state):
                _, _, _, k, loss_prev, loss_curr = state
                delta = jnp.abs(loss_prev - loss_curr)
                return (k < niter) & ((k < warmup_steps + 2) | (delta > tol_val))

            def body_fn(state):
                theta, opt_state, loss_hist, k, loss_prev, loss_curr = state
                theta, opt_state, loss = opti_step(
                    theta, opt_state, cpts, mov_mesh, ref_mesh_list
                )
                loss_hist = loss_hist.at[k].set(loss)
                return theta, opt_state, loss_hist, k + 1, loss_curr, loss

            init_state = (
                theta0,
                opt_state,
                jnp.full(niter, jnp.nan),
                jnp.zeros((), dtype=jnp.int32),
                jnp.inf,
                jnp.inf,
            )
            theta, opt_state, losses, _, _, _ = lax.while_loop(
                cond_fn, body_fn, init_state
            )
            return theta, losses

        return opti_loop

    def make_opti_plot_loop(self, opti_step, tol, niter, warmup_steps):
        tol_val = -1.0 if tol is None else float(tol)

        def plot_loop(theta, opt_state, cpts, mov_mesh, ref_mesh_list):
            mov_pts, mov_simps, _, mov_labs = mov_mesh

            losses = []
            for k in range(niter):
                theta, opt_state, loss = opti_step(
                    theta, opt_state, cpts, mov_mesh, ref_mesh_list
                )
                losses.append(loss)

                if tol_val > 0 and len(losses) >= warmup_steps + 2:
                    delta = abs(losses[-2] - losses[-1])
                    if delta <= tol_val:
                        break

                if k % self.plot == 0:
                    disp = self.polytransfo.compute(
                        mov_pts, cpts, theta_lin=None, theta_trans=theta
                    )
                    moved_mesh_plot = (mov_pts + disp, mov_simps, None, mov_labs)
                    for ref_m in ref_mesh_list:
                        utils.plot_contour(ref_m, col=[1, 0, 0])
                    utils.plot_contour(moved_mesh_plot, col=[0, 0, 1])
                    plt.xlim(self.xlim)
                    plt.ylim(self.ylim)
                    plt.title(f"it: {k}", fontsize=7)
                    plt.show()

            return theta, jnp.array(losses)

        return plot_loop
