import jax.numpy as jnp
from itertools import product

import utils


class kernel_disp():
    """
    should investigate: https://github.com/dodgebc/jaxkd
    """
    def __init__(self, sigma='silverman', int_steps=64, int_step_max=None, eps=1e-14):

        self.sigma = sigma
        self.int_steps = int_steps
        self.int_step_max = int_steps if int_step_max is None else int_step_max
        self.eps = eps
        
    def compute(self, pts, c_pts, log_c_disp, invert=False):
        # pts:        (npts, ndims),   points where we want to evaluate the disp.
        # c_pts:      (ncpts, ndims),  points where the disp is known.              
        # log_c_disp: (ncpts, ndims),  disp in log form at points c_pts.
        
        if invert:
            if log_c_disp is not None: log_c_disp *= -1
        self.invert = invert

        if self.sigma == 'silverman':
            self.sigma = self.sigma_silverman(c_pts)        
        
        disp = self.get_disp(pts, c_pts, log_c_disp)

        if self.int_steps > 0:
            disp = self.exp_disp(disp, pts, c_pts, log_c_disp)

        return disp

    
    def exp_disp(self, disp, pts, c_pts, log_c_disp):
        
        disp = disp / self.int_steps
        for i in range(self.int_step_max):
            disp += self.get_disp(pts + disp, c_pts, log_c_disp) / self.int_steps

        return disp

    
    def get_disp(self, pts, c_pts, log_c_disp):
            
        sqdist = utils.pts_sqdist(pts, c_pts)                                  # (npts, ncpts)
        weight = jnp.exp(-sqdist / (2*self.sigma**2))                            
        weight = weight / (jnp.sum(weight, axis=1)[...,None] + self.eps)

        disp = weight @ log_c_disp                                             # (npts, ndims)
            
        return disp
        
        
    def sigma_silverman(self, pts):
        
        iqr = jnp.mean(jnp.quantile(pts, (3/4), axis=0) - jnp.quantile(pts, (1/4),axis=0))
        std = jnp.mean(jnp.std(pts, axis=0))
        sigma = 0.9 * jnp.min(jnp.stack((std, iqr / 1.349))) * pts.shape[0] ** (-1/5)
        
        return sigma


class init_transfo():
    
    def __init__(self, init='identity'):
        """
        init: init: 'identity', 'centroids', 'similarity', 'ellipsoid'
        """
        
        self.init = init
    
    def compute(self, ref_pts, mov_pts):
        # init: 'identity', 'centroids', 'similarity', 'ellipsoid'
        
        ndims = ref_pts.shape[1]
    
        if self.init == 'identity':
            return jnp.eye(ndims+1), mov_pts
        
        else:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
            mov_pts_mu = jnp.mean(mov_pts, axis=0)    
        
            if self.init in ('similarity', 'ellipsoid'):
                s = jnp.std(ref_pts, axis=0) / jnp.std(mov_pts, axis=0)
                if self.init == 'ellipsoid':
                    A = jnp.diag(s)
                elif self.init == 'similarity':
                    A = jnp.mean(s) * jnp.eye(ndims)
            else:
                A = jnp.eye(ndims)
                
            t = ref_pts_mu - (mov_pts_mu @ A.T)
            moved_pts = (mov_pts @ A.T) + t 
            
            T = utils.aff_hmgn(A, t)
                
            return T, moved_pts
    
    
class opti_linear_transfo():
    
    def __init__(self, transfo, gamma=1e-5, se=True):
        """ 
        transfo: 'rigid' or 'affine'
        """
        
        self.transfo = transfo
        self.se = se
        self.gamma = gamma
        

    def compute(self, ref_pts, mov_pts,
                      ref_pts_mu=None, mov_pts_mu=None):
        
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """

        ndims = ref_pts.shape[1]
        
        if ref_pts_mu is None:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.mean(mov_pts, axis=0)
        ref_pts_bar = ref_pts - ref_pts_mu
        mov_pts_bar = mov_pts - mov_pts_mu
            
        # nu = jnp.unique(ref_pts_bar, axis=0).shape[0]
        # if nu <= ndims + 2:   # Tikhonov regularization
        #     mov_pts_bar = jnp.vstack([mov_pts_bar, self.gamma * jnp.eye(ndims)])
        #     ref_pts_bar = jnp.vstack([ref_pts_bar, jnp.zeros((ndims, ndims))])
        
        if self.transfo in ('rigid', 'similarity'):   
            cov = ref_pts_bar.T @ mov_pts_bar
            U, _, Vt = jnp.linalg.svd(cov, full_matrices=False)
            S = jnp.eye(ndims)
            
        if self.transfo == 'similarity':
            mov_norm_sq = jnp.sum(mov_pts_bar ** 2)
            s = jnp.trace(cov @ Vt.T @ U.T) / mov_norm_sq
            S = s * S
 
        elif self.transfo == 'affine':
            A, _, _, _ = jnp.linalg.lstsq(mov_pts_bar, ref_pts_bar, rcond=None)
            U, S, Vt = jnp.linalg.svd(A.T, full_matrices=False)                      
            S = jnp.diag(S)

        if self.se:
            det_sign = jnp.sign(jnp.linalg.det(U @ Vt))
            S = S.at[-1,-1].set(S[-1,-1]*det_sign)

        A = (U @ S) @ Vt
         
        t = ref_pts_mu - (mov_pts_mu @ A.T)

        return A, t



class opti_polynom_transfo():
    
    def __init__(self, degree, gamma=1e-5, se=True):

        self.degree = degree
    

    def compute(self, ref_pts, mov_pts,
                      ref_pts_mu=None, mov_pts_mu=None):
        
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """
        
        if ref_pts_mu is None:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.mean(mov_pts, axis=0)
        ref_pts_bar = ref_pts - ref_pts_mu
        mov_pts_bar = mov_pts - mov_pts_mu         
        
        X, exponents = self.design_mat(mov_pts_bar)
        
        coeffs, _, _, _ = jnp.linalg.lstsq(X, ref_pts_bar, rcond=None)
        
        moved_pts = X @ coeffs + ref_pts_mu

        return coeffs, moved_pts
    
    
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
    