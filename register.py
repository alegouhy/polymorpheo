import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

import utils
import energy
import transfo as transfo_ops


#%%

class reg_linear():
    
    def __init__(self, niter, transfo='rigid', init='identity', se=True, gamma=1e-5, plot=False):
        """
        transfo: 'rigid', 'rigid2' or 'affine'
        init: 'identity', 'centroids', 'similarity' or 'ellipsoid'
        """
        self.init = init
        self.niter = niter
        self.plot = plot
        
        self.opti_transfo_fun = transfo_ops.opti_linear_transfo(transfo, gamma=gamma, se=se)
        self.init_transfo = transfo_ops.init_transfo(init)
        
        
    def compute(self, ref_contour, mov_contour, T0=None):
        """
        T0 has priority over init.
        """
        
        ref_contour = [jnp.array(cont) if cont is not None else None for cont in ref_contour]
        mov_contour = [jnp.array(cont) if cont is not None else None for cont in mov_contour]
        ref_pts, ref_simps, ref_normals = ref_contour
        mov_pts, mov_simps, mov_normals = mov_contour
        
        if T0 is None:
            T, moved_pts = self.init_transfo.compute(ref_pts, mov_pts)
        else:
            T = T0
            A, t = utils.aff_dehmgn(T)
            moved_pts = (mov_pts @ A.T) + t 
        
        for k in range(self.niter):

            dist = utils.pts_sqdist(ref_pts, moved_pts)
            ref_nn_ind = jnp.argmin(dist, axis=0)
            ref_nn_pts = ref_pts[ref_nn_ind, :]
            
            lin, trans = self.opti_transfo_fun.compute(ref_nn_pts, moved_pts)

            moved_pts = (moved_pts @ lin.T) + trans
            moved_contour = moved_pts, mov_simps, mov_normals    # rotate normals !!!
        
            T = utils.aff_hmgn(lin, trans) @ T
            
            if self.plot:
                if k % self.plot == 0:
                    utils.plot_contour(ref_contour, col=[1,0,0])
                    utils.plot_contour(moved_contour, col=[0,0,1])
                    plt.title(f"it: {k}", fontsize=7) 
                    plt.show()
            
        return T, moved_contour
        
        
#%%

class reg_polynom():
    
    def __init__(self, niter, degree=2, init='identity', se=True, plot=False):
        """
        init: 'identity', 'centroids', 'similarity' or 'ellipsoid'
        """
        self.init = init
        self.niter = niter
        self.plot = plot
        
        self.opti_transfo_fun = transfo_ops.opti_polynom_transfo(degree, se=se)
        self.init_transfo = transfo_ops.init_transfo(init)
        
        
    def compute(self, ref_contour, mov_contour, disp0=None):
        """
        disp0 has priority over init.
        """
        
        ref_pts, ref_simps, ref_normals = ref_contour
        mov_pts, mov_simps, mov_normals = mov_contour
        
        if disp0 is None:
            _, moved_pts = self.init_transfo.compute(ref_pts, mov_pts)
        else:
            moved_pts = mov_pts + disp0
        
        for k in range(self.niter):

            dist = utils.pts_sqdist(ref_pts, moved_pts)
            ref_nn_ind = jnp.argmin(dist, axis=0)
            ref_nn_pts = ref_pts[ref_nn_ind, :]
            
            _, moved_pts = self.opti_transfo_fun.compute(ref_nn_pts, moved_pts)

            moved_contour = moved_pts, mov_simps, mov_normals    # rotate normals !!!
            
            if self.plot:
                if k % self.plot == 0:
                    utils.plot_contour(ref_contour, col=[1,0,0])
                    utils.plot_contour(moved_contour, col=[0,0,1])
                    plt.title(f"it: {k}", fontsize=7) 
                    plt.show()
            
        return moved_contour
    
        
#%%

class reg_deformable():
    
    def __init__(self, niter, fit_fun, regul_fun, lr=1e-2, wreg=0, sigma=None, int_steps=64, eps=1e-9, plot=False):
        
        self.niter = niter
        self.lr = lr
        self.wreg = wreg
        self.sigma = sigma
        self.int_steps = int_steps
        self.eps = eps
        self.plot = plot
        
        kernel_fun = transfo_ops.kernel_disp(sigma=sigma, int_steps=int_steps)
        
        self.energy_fun = energy.energy_total(fit_fun=fit_fun, regul_fun=regul_fun, 
                                              wreg=wreg, kernel_fun=kernel_fun).compute
        
    def compute(self, ref_contour, mov_contour, disp0=None):
        
        mov_pts, mov_simps, mov_normals = mov_contour
        
        if disp0 is None:
            disp0 = jnp.zeros_like(mov_pts)

        moved0_pts = mov_pts + disp0
        moved0_contour = moved0_pts, mov_simps, mov_normals
        
        optimizer = optax.adam(learning_rate=self.lr)
        opt_state = optimizer.init(disp0)
        
        disp = jnp.zeros_like(mov_pts)
        losses = []
        for k in range(self.niter):

            loss, grads = jax.value_and_grad(self.energy_fun)(disp, moved0_contour, ref_contour)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            disp = optax.apply_updates(disp, updates)
            
            if self.plot:
                if k % self.plot == 0:
                    moved_pts = moved0_pts + disp
                    moved_contour = moved_pts, mov_simps, mov_normals
                    utils.plot_contour(ref_contour, col=[1,0,0])
                    utils.plot_contour(moved_contour, col=[0,0,1])
                    plt.title(f"it: {k}, energy = {loss:.6f}", fontsize=7) 
                    plt.show()
            
            losses.append(loss)
        
        disp = disp0 + disp
        moved_pts = mov_pts + disp
        moved_contour = moved_pts, mov_simps, mov_normals
    
        return disp, moved_contour, losses
# #%%

# class reg_deformable():
    
#     def __init__(self, niter, fit_fun, regul_fun, lr=1e-2, wreg=0, sigma=None, int_steps=64, eps=1e-9, plot=False):
        
#         self.niter = niter
#         self.lr = lr
#         self.wreg = wreg
#         self.sigma = sigma
#         self.int_steps = int_steps
#         self.eps = eps
#         self.plot = plot
        
#         kernel_fun = transfo.kernel_disp(sigma=sigma, int_steps=int_steps)
        
#         self.energy_fun = energy.energy_total(fit_fun=fit_fun, regul_fun=regul_fun, 
#                                               wreg=wreg, kernel_fun=kernel_fun).compute
        
#     def compute(self, ref_contour, mov_contour, disp0=None):
        
#         mov_pts, mov_simps, mov_normals = mov_contour
        
#         if disp0 is None: 
#             disp = jnp.zeros_like(mov_pts)
#         else:  
#             disp = disp0
            
#         optimizer = optax.adam(learning_rate=self.lr)
#         opt_state = optimizer.init(disp)
        
#         losses = []
#         for k in range(self.niter):

#             loss, grads = jax.value_and_grad(self.energy_fun)(disp, mov_contour, ref_contour)
            
#             updates, opt_state = optimizer.update(grads, opt_state)
#             disp = optax.apply_updates(disp, updates)
            
#             if self.plot:
#                 if k % self.plot == 0:
#                     moved_pts = mov_pts + disp
#                     moved_contour = moved_pts, mov_simps, mov_normals
#                     utils.plot_contour(ref_contour, col=[1,0,0])
#                     utils.plot_contour(moved_contour, col=[0,0,1])
#                     plt.title(f"it: {k}, energy = {loss:.6f}", fontsize=7) 
#                     plt.show()
            
#             losses.append(loss)
        
#         moved_pts = mov_pts + disp
#         moved_contour = moved_pts, mov_simps, mov_normals
    
#         return disp, moved_contour, losses
