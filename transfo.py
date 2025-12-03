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
        
    def compute(self, pts, cpts, theta):
        # pts:   (npts, ndims),   points where we want to evaluate the disp.
        # cpts:  (ncpts, ndims),  points where theta is known.              
        # theta: (ncpts, ndims),  transfo params.
        
        if self.sigma == 'silverman':
            self.sigma = self.sigma_silverman(cpts)        
        
        disp = self.interp(pts, cpts, theta)

        if self.int_steps > 0:
            disp = self.lie_exp(disp, pts, cpts, theta)

        return disp

    
    def lie_exp(self, disp, pts, cpts, theta):
        
        disp = disp / self.int_steps
        for i in range(self.int_step_max):
            disp += self.interp(pts + disp, cpts, theta) / self.int_steps

        return disp

    
    def interp(self, pts, cpts, theta):
            
        sqdist = utils.pts_dist(pts, cpts)                                  # (npts, ncpts)
        weight = jnp.exp(-sqdist / (2*self.sigma**2))                            
        weight = weight / (jnp.sum(weight, axis=1)[...,None] + self.eps)

        disp = weight @ theta                                               # (npts, ndims)
            
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
    
        if self.init in (None, 'identity'):
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
    
    def __init__(self, transfo, se=True): #  gamma=1e-5
        """ 
        transfo: 'rigid' or 'affine'
        """
        
        self.transfo = transfo
        self.se = se
        # self.gamma = gamma
        

    def fit(self, ref_pts, mov_pts,
                  ref_pts_mu=None, mov_pts_mu=None):
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """

        ndims = ref_pts.shape[1]
        
        if ref_pts_mu is None:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.mean(mov_pts, axis=0)
        self.ref_pts_mu = ref_pts_mu
        self.mov_pts_mu = mov_pts_mu
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


    def transform(self, A, t, mov_pts):
        
        return (mov_pts @ A.T) + t
    
    

class opti_polynom_transfo():
    
    def __init__(self, degree, se=True):

        self.degree = degree
    

    def fit(self, ref_pts, mov_pts,
                  ref_pts_mu=None, mov_pts_mu=None):
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
        
    