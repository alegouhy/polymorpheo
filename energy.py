import jax
import jax.numpy as jnp

import transfo as transfo_ops


class pointdist():
    
    def __init__(self, dist='chamfer', normtype='l2'): 
    
        self.dist = dist              # 'chamfer' or 'hausdorff'
        self.normtype = normtype      # l1 or l2
    
    def compute(self, ref_pts, mov_pts): 
        
        ref_pts = jnp.expand_dims(ref_pts, axis=1)                             # (npts1, 1, ndims)
        mov_pts = jnp.expand_dims(mov_pts, axis=0)                             # (1, npts2, ndims)
        
        if self.normtype == 'l1':
            dist = jnp.abs(ref_pts - mov_pts)                                  # (npts1, npts2, ndims)
        elif self.normtype == 'l2':
            dist = (ref_pts - mov_pts) ** 2                                    # (npts1, npts2, ndims)
        dist = jnp.sum(dist, axis=-1)                                          # (npts1, npts2)
        
        nn_ref2mov = jnp.min(dist, axis=1)                                     # (npts1)
        nn_mov2ref = jnp.min(dist, axis=0)                                     # (npts2)
        
        if self.dist == 'chamfer':
            dist_ref2mov = jnp.mean(nn_ref2mov, axis=-1)
            dist_mov2ref = jnp.mean(nn_mov2ref, axis=-1)
        
        elif self.dist == 'hausdorff':
            dist_ref2mov = jnp.max(nn_ref2mov, axis=-1)
            dist_mov2ref = jnp.max(nn_mov2ref, axis=-1)
            
        return dist_ref2mov + dist_mov2ref


class grad_disp():

    def __init__(self, normtype='l2', eps=1e-9):
        self.normtype = normtype   # l1 or l2
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

        if self.normtype == 'l2':
            disp_norm = jnp.sum(disp_diff ** 2, axis=-1)
            pts_norm = jnp.sum(pts_diff ** 2, axis=-1)
        elif self.normtype == 'l1':
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
    
    def compute(self, disp, pts):

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
        
        lin, trans = self.opti_transfo_fun.compute(moved_pts_neigh, pts_neigh)
        
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
    
    def compute(self, disp, mov_contour, ref_contour):
        
        mov_pts, mov_simps, mov_normals = mov_contour
        ref_pts, ref_simps, ref_normals = ref_contour
        
        regul = self.regul_fun.compute(disp, mov_pts) #, mov_simps)
        
        if self.kernel_fun.sigma not in (0, None):
            disp = self.kernel_fun.compute(mov_pts, mov_pts, disp)
        moved_pts = mov_pts + disp
        
        fit = self.fit_fun.compute(ref_pts, moved_pts)    

        return fit + self.wreg * regul
    
    