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
    
    def __init__(self, transfo='rigid', normtype='l2', eps=1e-9):
        self.normtype = normtype   # l1 or l2
        self.eps = eps
        self.transfo = transfo     # 'rigid', 'similarity' or 'affine'
        self.opti_transfo_fun = transfo_ops.opti_linear_transfo(transfo, se=True)
        self.neighs = None
    
    def compute(self, disp, pts, simps=None):
        
        if self.neighs is None:
            self.neighs = utils.neighs_contour(simps, pts.shape[0])
        grad = self.compute_map(disp, pts)

        return jnp.mean(grad)
    
    def set_neighs(self, neighs):
        
        self.neighs = neighs
    
    def compute_map(self, disp, pts):
        # 1 value per neighborhood
            
        pts_neighs = pts[self.neighs]          # (nneighs, nptsneigh, ndims)
        disp_neighs = disp[self.neighs]        # (nneighs, nptsneigh, ndims)

        arap_neighs = jax.vmap(self.arap_neigh)(disp_neighs, pts_neighs)
        
        return arap_neighs
    
    def arap_neigh(self, disp_neigh, pts_neigh):
        
        moved_pts_neigh = pts_neigh + disp_neigh
        
        lin, trans = self.opti_transfo_fun.fit(moved_pts_neigh, pts_neigh)
        
        diff = moved_pts_neigh - ((pts_neigh @ lin.T) + trans)
        
        if self.normtype == 'l2':
            arap = jnp.sum(diff ** 2, axis=-1)
        elif self.normtype == 'l1':
            arap = jnp.sum(jnp.abs(diff), axis=-1)
            
        return arap


class energy_total():

    def __init__(self, fit_fun, regul_fun, wreg, kernel_fun):
        
        self.fit_fun = fit_fun
        self.regul_fun = regul_fun
        self.wreg = wreg
        self.kernel_fun = kernel_fun
        
        
    def set_contours(self, mov_contour, ref_contour_list):
        
        self.mov_contour = mov_contour
        self.ref_contour_list = ref_contour_list
    
    
    def compute(self, theta):
        
        mov_pts, mov_simps, mov_normal, mov_labs = self.mov_contour
            
        regul = self.regul_fun.compute(theta, mov_pts, mov_simps)
        
        if self.kernel_fun.sigma not in (0, None):
            disp = self.kernel_fun.compute(mov_pts, mov_pts, theta)
        else: disp = theta
        moved_pts = mov_pts + disp
        
        fit = 0
        for ref_contour in self.ref_contour_list:
            ref_pts, ref_simps, ref_normals, ref_labs = ref_contour
            fit += self.fit_fun.compute(ref_pts, moved_pts)    
            
        # jax.debug.print("{}", [fit, regul])

        return fit + self.wreg * regul
    
    