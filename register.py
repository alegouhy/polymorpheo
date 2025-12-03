import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

import utils
import energy
import transfo as transfo_ops


#%%


def load_contours(ref_contour, mov_contour):
        
    ref_contours = [ref_contour] if hasattr(ref_contour[0], 'shape') else ref_contour
    
    ref_pts_list = []
    ref_labs_list = []
    ref_contour_list = []
    for ref_contour in ref_contours:
        ref_contour = [jnp.array(cont) if cont is not None else None for cont in ref_contour]
        ref_pts, ref_simps, ref_normals, ref_labs = ref_contour
        ref_pts_list.append(ref_pts)
        ref_labs_list.append(ref_labs)
        ref_contour_list.append(ref_contour)
        
    mov_contour = [jnp.array(cont) if cont is not None else None for cont in mov_contour]
    
    return ref_contour_list, mov_contour
    
    
class reg_linear():
    
    def __init__(self, niter, transfo='rigid', init='identity', se=True, bidir=False, plot=False):
        """
        transfo: 'rigid', 'rigid2' or 'affine'
        init: 'identity', 'centroids', 'similarity' or 'ellipsoid'
        """
        self.init = init
        self.niter = niter
        self.bidir = bidir
        self.plot = plot
        
        self.opti_transfo_fun = transfo_ops.opti_linear_transfo(transfo, se=se)
        self.init_transfo = transfo_ops.init_transfo(init)
        
        
    def compute(self, ref_contour, mov_contour, T0=None, use_labs=None):
        """
        T0 has priority over init.
        ref_contour can be a single polyline of a list of polylines
        """
        
        ref_contour_list, mov_contour = load_contours(ref_contour, mov_contour)
        ref_pts_list  = [contour[0] for contour in ref_contour_list]
        ref_labs_list = [contour[3] for contour in ref_contour_list]
        mov_pts, mov_simps, _, mov_labs = mov_contour
        
        if T0 is None:
            ref_pts = jnp.concatenate(ref_pts_list, axis=0)
            T, moved_pts = self.init_transfo.compute(ref_pts, mov_pts)
        else:
            T = T0
            A, t = utils.aff_dehmgn(T)
            moved_pts = (mov_pts @ A.T) + t
        
        for k in range(self.niter):
            
            ref_nn_pts, mov_nn_pts = utils.nearest_neighbors(ref_pts_list, moved_pts, ref_labs_list, mov_labs, self.bidir)
            lin, trans = self.opti_transfo_fun.fit(ref_nn_pts, mov_nn_pts)  
                
            moved_pts = self.opti_transfo_fun.transform(lin, trans, moved_pts)
            
            moved_contour = moved_pts, mov_simps, None, mov_labs
        
            T = utils.aff_hmgn(lin, trans) @ T
            
            if self.plot:
                if k % self.plot == 0:
                    for ref_contour in ref_contour_list:
                        utils.plot_contour(ref_contour, col=[1,0,0])
                    utils.plot_contour(moved_contour, col=[0,0,1])
                    plt.title(f"it: {k}", fontsize=7) 
                    plt.show()
            
        return T, moved_contour

        
#%%

class reg_polynom():
    
    def __init__(self, niter, degree=2, init='identity', se=True, bidir=False, plot=False):
        """
        init: 'identity', 'centroids', 'similarity' or 'ellipsoid'
        """
        self.init = init
        self.niter = niter
        self.bidir = bidir
        self.plot = plot
        
        self.opti_transfo_fun = transfo_ops.opti_polynom_transfo(degree, se=se)
        self.init_transfo = transfo_ops.init_transfo(init)
        
        
    def compute(self, ref_contour, mov_contour, disp0=None, use_labs=None):
        """
        disp0 has priority over init.
        """
        
        ref_contour_list, mov_contour = load_contours(ref_contour, mov_contour)
        ref_pts_list  = [contour[0] for contour in ref_contour_list]
        ref_labs_list = [contour[3] for contour in ref_contour_list]
        mov_pts, mov_simps, _, mov_labs = mov_contour
            
        if disp0 is None:
            ref_pts = jnp.concatenate(ref_pts_list, axis=0)
            _, moved_pts = self.init_transfo.compute(ref_pts, mov_pts)
        else:
            moved_pts = mov_pts + disp0
        
        for k in range(self.niter):
                
            ref_nn_pts, mov_nn_pts = utils.nearest_neighbors(ref_pts_list, moved_pts, ref_labs_list, mov_labs, self.bidir)
            coeffs = self.opti_transfo_fun.fit(ref_nn_pts, mov_nn_pts)  
                
            moved_pts = self.opti_transfo_fun.transform(coeffs, moved_pts)

            moved_contour = moved_pts, mov_simps, None, mov_labs
            
            if self.plot:
                if k % self.plot == 0:
                    for ref_contour in ref_contour_list:
                        utils.plot_contour(ref_contour, col=[1,0,0])
                    utils.plot_contour(moved_contour, col=[0,0,1])
                    plt.title(f"it: {k}", fontsize=7) 
                    plt.show()
            
        return moved_contour
    
 
#%%

class reg_deformable():
    
    def __init__(self, niter, fit_fun, regul_fun, lr=1e-2, wreg=0, sigma=None, int_steps=64, plot=False):
        
        self.niter = niter
        self.lr = lr
        self.wreg = wreg
        self.sigma = sigma
        self.int_steps = int_steps
        self.plot = plot
        
        self.kernel_fun = transfo_ops.kernel_disp(sigma=sigma, int_steps=int_steps)
        
        self.energy_fun = energy.energy_total(fit_fun=fit_fun, regul_fun=regul_fun, 
                                              wreg=wreg, kernel_fun=self.kernel_fun)
        
        
    def compute(self, ref_contour, mov_contour):
        
        ref_contour_list, mov_contour = load_contours(ref_contour, mov_contour)
        ref_pts_list  = [contour[0] for contour in ref_contour_list]
        ref_labs_list = [contour[3] for contour in ref_contour_list]
        mov_pts, mov_simps, _, mov_labs = mov_contour
        
        theta0 = jnp.zeros_like(mov_pts)
        moved_pts = mov_pts
        moved_contour = moved_pts, mov_simps, None, mov_labs
        
        self.energy_fun.set_contours(moved_contour, ref_contour_list)
        
        optimizer = optax.adam(learning_rate=self.lr)
        opt_state = optimizer.init(theta0)
        
        theta = jnp.zeros_like(mov_pts)
        losses = []
        for k in range(self.niter):

            loss, grads = jax.value_and_grad(self.energy_fun.compute)(theta)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            
            if self.plot:
                if k % self.plot == 0:
                    disp = self.kernel_fun.compute(mov_pts, mov_pts, theta)
                    moved_pts = mov_pts + disp
                    moved_contour = moved_pts, mov_simps, None, mov_labs
                    for ref_contour in ref_contour_list:
                        utils.plot_contour(ref_contour, col=[1,0,0])
                    utils.plot_contour(moved_contour, col=[0,0,1])
                    plt.title(f"it: {k}, energy = {loss:.6f}", fontsize=7) 
                    plt.show()
            
            losses.append(loss)
        
        disp = self.kernel_fun.compute(mov_pts, mov_pts, theta)
        moved_pts = mov_pts + disp
        moved_contour = moved_pts, mov_simps, None, mov_labs
    
        return theta, moved_contour, losses
    
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
