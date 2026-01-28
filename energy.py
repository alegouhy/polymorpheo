import jax
import jax.numpy as jnp

import transfo as transfo_ops
import utils


class pointdist():
    
    def __init__(self, agg='mean', l=2, bidir=True): 
    
        self.l = l      # l1 or l2 norm
        self.bidir = bidir
        if agg == 'max': self.agg_fun = jnp.max     
        elif agg == 'mean': self.agg_fun = jnp.mean
        

    def compute(self, ref_pts, mov_pts, ref_labs=None, mov_labs=None): 
   
        ref_pts_list = [ref_pts] if hasattr(ref_pts, 'shape') else ref_pts
        ref_labs_list = [ref_labs] if hasattr(ref_pts, 'shape') else ref_labs
        use_labs = (ref_labs_list is not None) and (mov_labs is not None)                   

        dist_nn = 0.
        if not use_labs:
            for ref_pts in ref_pts_list:
                dist = utils.pts_dist(ref_pts, mov_pts, l=self.l)
                dist_nn += self.agg_fun(jnp.min(dist, axis=0))
                if self.bidir:
                    dist_nn += self.agg_fun(jnp.min(dist, axis=1))
            
        else:  
            for ref_pts, ref_labs in zip(ref_pts_list, ref_labs_list):
                
                labs = jnp.intersect1d(mov_labs, ref_labs)               
                for lab in labs:
                    
                    ref_pts_lab = ref_pts[ref_labs == lab, :]
                    mov_ind_lab = mov_labs == lab
                    mov_pts_lab = mov_pts[mov_ind_lab, :]
                    
                    dist = utils.pts_dist(ref_pts_lab, mov_pts_lab, l=self.l)
                    dist_nn += self.agg_fun(jnp.min(dist, axis=0))
                    if self.bidir:
                        dist_nn += self.agg_fun(jnp.min(dist, axis=1))
            
        return dist_nn


class grad_disp():

    def __init__(self, l=2, eps=1e-9):
        
        self.l = l
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

        if self.l == 2:
            disp_norm = jnp.sum(disp_diff ** 2, axis=-1)
            pts_norm = jnp.sum(pts_diff ** 2, axis=-1)
        elif self.l == 1:
            disp_norm = jnp.sum(jnp.abs(disp_diff), axis=-1)
            pts_norm = jnp.sum(jnp.abs(pts_diff), axis=-1)
        
        return disp_norm / (pts_norm + self.eps)


class alap():
    """
    as linear as possible
    """
    
    def __init__(self, transfo='rigid', l=2, eps=1e-9):
        self.l = l      # l1 or l2 norm
        self.eps = eps
        self.transfo = transfo     # 'rigid', 'similarity' or 'affine'
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
        
        alap_map = jax.vmap(self._vertex_energy, in_axes=(0, None, None))(jnp.arange(len(pts)), disp, pts)
        
        return alap_map


    def _vertex_energy(self, vertex_idx, disp, pts):

        neigh_idx = self.neighborhoods[vertex_idx]  # (max_valence,)
        mask = neigh_idx >= 0 
        n_neigh = jnp.sum(mask)
        
        safe_idx = jnp.maximum(neigh_idx, 0)
        
        pts_neigh = pts[safe_idx]
        disp_neigh = disp[safe_idx]
        moved_neigh = pts_neigh + disp_neigh
    
        weights = mask.astype(jnp.float32)

        lin, trans = self.opti_transfo_fun.fit(moved_neigh, pts_neigh, weights=weights)
        
        reconstructed = (pts_neigh @ lin.T) + trans
        diff = moved_neigh - reconstructed
        
        if self.l == 2: energy_per_neighbor = jnp.sum(diff ** 2, axis=-1)  # (max_valence,)
        elif self.l == 1: energy_per_neighbor = jnp.sum(jnp.abs(diff), axis=-1)
 
        masked_energy = energy_per_neighbor * weights
        total_energy = jnp.sum(masked_energy)
        
        return jnp.where(n_neigh > 0, total_energy / n_neigh, 0.0)
    
    
    
    
    
    # def compute_map(self, disp, pts):

    #     if self.neighborhoods is None:
    #         raise ValueError("Must call set_neighs() before compute_map()!")
    
    #     pts_neighs = pts[self.neighs]          # (nneighs, nptsneigh, ndims)
    #     disp_neighs = disp[self.neighs]        # (nneighs, nptsneigh, ndims)

    #     arap_neighs = jax.vmap(self.arap_neigh)(disp_neighs, pts_neighs)
        
    #     return arap_neighs
    
    
    # def arap_neigh(self, disp_neigh, pts_neigh):
        
    #     moved_pts_neigh = pts_neigh + disp_neigh
        
    #     lin, trans = self.opti_transfo_fun.fit(moved_pts_neigh, pts_neigh)
        
    #     diff = moved_pts_neigh - ((pts_neigh @ lin.T) + trans)
        
    #     if self.l == 2:
    #         arap = jnp.sum(diff ** 2, axis=-1)
    #     elif self.l == 1:
    #         arap = jnp.sum(jnp.abs(diff), axis=-1)
            
    #     return arap



def energy_total_fn(theta, cpts, mov_mesh, ref_mesh_list,
                    fit_fun, regul_fun, wreg, kernel_fun):
    # regularization on the smoothed field or on theta?
    
    mov_pts, mov_simps, _, mov_labs = mov_mesh

    # regul = regul_fun.compute(theta, mov_pts, mov_simps)
    if kernel_fun.sigma is not None:
        svf = kernel_fun.interp(mov_pts, cpts, theta_lin=None, theta_trans=theta)
        disp = kernel_fun.compute(mov_pts, cpts, theta_lin=None, theta_trans=theta)
    else:
        svf = theta
        disp = theta
        
    regul = regul_fun.compute(svf, mov_pts, mov_simps)

    moved_pts = mov_pts + disp

    fit = 0.0
    for ref_mesh in ref_mesh_list:
        ref_pts, ref_simps, ref_normals, ref_labs = ref_mesh
        fit += fit_fun.compute(ref_pts, moved_pts)

    return fit + wreg * regul


# class energy_total():

#     def __init__(self, fit_fun, regul_fun, wreg, kernel_fun):
        
#         self.fit_fun = fit_fun
#         self.regul_fun = regul_fun
#         self.wreg = wreg
#         self.kernel_fun = kernel_fun
        
        
#     def set_meshs(self, mov_mesh, ref_mesh_list):
        
#         self.mov_mesh = mov_mesh
#         self.ref_mesh_list = ref_mesh_list
    
    
#     def compute(self, theta):
        
#         mov_pts, mov_simps, mov_normal, mov_labs = self.mov_mesh
            
#         regul = self.regul_fun.compute(theta, mov_pts, mov_simps)
        
#         if self.kernel_fun.sigma not in (0, None):
#             disp = self.kernel_fun.compute(mov_pts, mov_pts, theta_lin=None, theta_trans=theta)
#         else: disp = theta
#         moved_pts = mov_pts + disp
        
#         fit = 0
#         for ref_mesh in self.ref_mesh_list:
#             ref_pts, ref_simps, ref_normals, ref_labs = ref_mesh
#             fit += self.fit_fun.compute(ref_pts, moved_pts)    
            
#         # jax.debug.print("{}", [fit, regul])

#         return fit + self.wreg * regul
    
    